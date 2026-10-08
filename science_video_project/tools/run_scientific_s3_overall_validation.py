"""S3 fixed-split replacement validation for the Scientific Branch.

All three arms retain main_v3-D3 + KEEP_P2_S2 + T0.  B/C replace only the
scientific branch.  Their 1540-D attribute predictor is exactly the S1
pre-registered structure; a small representation adapter merely preserves the
128-D interface required by the unchanged main-model fusion layers.
"""
from __future__ import annotations

import argparse, hashlib, json, logging, math, random, sys, time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pipeline.config import CFG
from pipeline.utils_io import ensure_dir, load_metadata, load_pt
from tools.run_progressive_training import (
    branch_rank_loss, collate_unique, make_train_loader, model_config, prepare_batch,
    prepare_data, safe_srcc, train_epoch,
)
from tools.run_scientific_s0 import ATTRS, FEATURES, HP, SEEDS, SPLIT_HASH, sha, table, write_json
from training.dataloader_pair import collate_pair
from training.dataset_pair import PairwiseVideoDataset
from training.losses import branch_consistency_loss, branch_supervision_loss, ranknet_pairwise_loss
from training.model_mvp import MultiModalQualityModel
from training.utils_train import get_device, set_seed

MODELS = ("s3_a_original", "s3_b_shared_head", "s3_c_four_adapter")
LABELS = {"s3_a_original": "S3-A Original Scientific Branch", "s3_b_shared_head": "S3-B Shared Head", "s3_c_four_adapter": "S3-C Four-Adapter"}
ATTR_WEIGHT, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED = .02, 1000, 7007


class AttributeScientificBranch(nn.Module):
    """S1 predictor plus the 128-D adapter required by the fixed main fusion."""
    def __init__(self, kind: str, hidden_dim: int = 128):
        super().__init__(); self.kind = kind; input_dim = sum(FEATURES.values())
        if kind == "shared":
            self.projection, self.activation, self.dropout = nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(.2)
            self.head = nn.Linear(256, 4); self.fusion_adapter = nn.Sequential(nn.Linear(256, hidden_dim), nn.ReLU())
        elif kind == "adapter":
            self.projection, self.activation, self.dropout = nn.Linear(input_dim, 224), nn.ReLU(), nn.Dropout(.2)
            self.adapters = nn.ModuleList([nn.Sequential(nn.Linear(224, 32), nn.ReLU(), nn.Linear(32, 1)) for _ in ATTRS])
            self.fusion_adapter = nn.Sequential(nn.Linear(224, hidden_dim), nn.ReLU())
        else: raise ValueError(kind)

    def forward(self, text_feat, meta_feat, llm_knowledge_feat=None, llm_analysis_feat=None, sci_hand_feat=None, return_gate=False):
        del meta_feat, sci_hand_feat
        if llm_analysis_feat is None or llm_knowledge_feat is None: raise ValueError("S3 requires cached analysis and score features")
        raw = torch.cat([text_feat, llm_analysis_feat, llm_knowledge_feat], dim=-1)
        h = self.dropout(self.activation(self.projection(raw)))
        logits = self.head(h) if self.kind == "shared" else torch.cat([head(h) for head in self.adapters], dim=-1)
        attrs = torch.sigmoid(logits)
        sci_prob = attrs.mean(dim=-1, keepdim=True).clamp(1e-5, 1 - 1e-5)
        sci_logit = torch.logit(sci_prob)
        gate = torch.full((h.size(0), 1), .5, device=h.device, dtype=h.dtype)
        return self.fusion_adapter(h), sci_logit, gate

    def attribute_scores(self, text_feat, llm_analysis_feat, llm_knowledge_feat):
        raw = torch.cat([text_feat, llm_analysis_feat, llm_knowledge_feat], dim=-1)
        h = self.dropout(self.activation(self.projection(raw)))
        logits = self.head(h) if self.kind == "shared" else torch.cat([head(h) for head in self.adapters], dim=-1)
        return torch.sigmoid(logits)


def build_model(kind: str, seed: int, device: torch.device) -> MultiModalQualityModel:
    set_seed(seed); model = MultiModalQualityModel(**model_config())
    if kind == "s3_b_shared_head": model.scientific_branch = AttributeScientificBranch("shared")
    elif kind == "s3_c_four_adapter": model.scientific_branch = AttributeScientificBranch("adapter")
    elif kind != "s3_a_original": raise ValueError(kind)
    return model.to(device)


def attr_prediction(model, batch):
    branch = model.scientific_branch
    if not isinstance(branch, AttributeScientificBranch): return None
    return branch.attribute_scores(batch["text_feat"], batch["llm_analysis_feat"], batch["llm_knowledge_feat"])


def attribute_loss(model, batch) -> torch.Tensor:
    prediction = attr_prediction(model, batch)
    if prediction is None: return batch["overall_score"].sum() * 0.0 if "overall_score" in batch else batch["text_feat"].sum() * 0.0
    target = batch["attribute_targets"][:, :4]; valid = target >= 0
    return F.mse_loss(prediction[valid], target[valid]) if valid.any() else prediction.sum() * 0.0


def losses(model, pos_out, neg_out, pos, neg, attr_enabled):
    rank = ranknet_pairwise_loss(pos_out["overall_score"], neg_out["overall_score"])
    point = (F.binary_cross_entropy_with_logits(pos_out["overall_score"], torch.ones_like(pos_out["overall_score"])) + F.binary_cross_entropy_with_logits(neg_out["overall_score"], torch.zeros_like(neg_out["overall_score"]))) / 2
    consistency = (branch_consistency_loss(pos_out["overall_score"], pos_out["scientific_score"], pos_out["technical_score"], pos_out["aesthetic_score"]) + branch_consistency_loss(neg_out["overall_score"], neg_out["scientific_score"], neg_out["technical_score"], neg_out["aesthetic_score"])) / 2
    branch = (branch_supervision_loss(pos_out["scientific_score"], pos_out["technical_score"], pos_out["aesthetic_score"], pos["sci_target"], pos["tech_target"], pos["aes_target"]) + branch_supervision_loss(neg_out["scientific_score"], neg_out["technical_score"], neg_out["aesthetic_score"], neg["sci_target"], neg["tech_target"], neg["aes_target"])) / 2
    brank = branch_rank_loss(pos_out, neg_out, pos, neg, .05)
    attr = ((attribute_loss(model, pos) + attribute_loss(model, neg)) / 2) if attr_enabled else rank.new_zeros(())
    return rank, point, consistency, branch, brank, attr


def train_one_epoch(model, loader, optimizer, device, stage, attr_enabled):
    model.train(); sums = np.zeros(6, dtype=float)
    for pos, neg in loader:
        pos, neg = prepare_batch(pos, device), prepare_batch(neg, device)
        po, no = model(**pos), model(**neg); terms = losses(model, po, no, pos, neg, attr_enabled)
        rank, point, consistency, branch, brank, attr = terms
        total = branch + .05 * brank + ATTR_WEIGHT * attr if stage == "stage1" else rank + .1 * point + .1 * consistency + .05 * branch + .05 * brank + ATTR_WEIGHT * attr
        optimizer.zero_grad(set_to_none=True); total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        sums += [x.detach().item() for x in (rank, point, consistency, branch, brank, attr)]
    keys = ("rank", "point", "consistency", "branch", "branch_rank", "attribute")
    return {key: float(value / max(len(loader), 1)) for key, value in zip(keys, sums)}


def evaluate(model, samples, device):
    model.eval(); records = []
    with torch.inference_mode():
        for start in range(0, len(samples), 32):
            batch = prepare_batch(collate_unique(samples[start:start + 32]), device); out = model(**batch); attr = attr_prediction(model, batch)
            for index, video_id in enumerate(batch["video_id"]):
                row = {"video_id": str(video_id), "label": float(batch["label_target"][index, 0]), "probability": float(out["probability"][index, 0]), "overall_q": float(out["overall_score"][index, 0]), "pred_scientific_score": float(torch.sigmoid(out["scientific_score"][index, 0])), "technical_score": float(torch.sigmoid(out["technical_score"][index, 0])), "aesthetic_score": float(torch.sigmoid(out["aesthetic_score"][index, 0])), "semantic_score": float(torch.sigmoid(out["scientific_score"][index, 0])), "true_technical_target": float(batch["tech_target"][index, 0]), "true_aesthetic_target": float(batch["aes_target"][index, 0]), "science_weight": float(out["branch_weights"][index, 0]), "technical_weight": float(out["branch_weights"][index, 1]), "aesthetic_weight": float(out["branch_weights"][index, 2])}
                for j, name in enumerate(ATTRS): row[f"true_{name}"] = float(batch["attribute_targets"][index, j]); row[f"pred_{name}"] = float(attr[index, j]) if attr is not None else np.nan
                row["true_scientific_aggregate"] = float(batch["attribute_targets"][index, :4].mean())
                records.append(row)
    frame = pd.DataFrame(records); y, p = frame.label.to_numpy(), frame.probability.to_numpy(); threshold = max((f1_score(y, p >= t), t) for t in np.arange(.1, .95, .05))[1]
    positive, negative = p[y == 1], p[y == 0]
    metric = {"auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)), "accuracy": float((y == (p >= threshold)).mean()), "f1": float(f1_score(y, p >= threshold)), "ranking_accuracy": float((positive[:, None] > negative[None, :]).mean()), "best_threshold": float(threshold)}
    for pred, truth, name in [(frame.pred_scientific_score, frame.true_scientific_aggregate, "sci"), (frame.technical_score, frame.true_technical_target, "tech"), (frame.aesthetic_score, frame.true_aesthetic_target, "aes")]: metric[f"{name}_srcc"] = float(spearmanr(pred, truth).statistic)
    metric["branch_macro_srcc"] = float(np.mean([metric["sci_srcc"], metric["tech_srcc"], metric["aes_srcc"]]))
    metric["validation_loss"] = float(F.softplus(-(torch.from_numpy(frame.loc[y == 1, "overall_q"].to_numpy())[:, None] - torch.from_numpy(frame.loc[y == 0, "overall_q"].to_numpy())[None, :])).mean())
    if frame.pred_science_info.notna().all():
        attribute = {}
        for name in ATTRS:
            pred, truth = frame[f"pred_{name}"].to_numpy(), frame[f"true_{name}"].to_numpy()
            attribute[name] = {"srcc": float(spearmanr(pred, truth).statistic), "plcc": float(pearsonr(pred, truth).statistic), "mae": float(np.abs(pred-truth).mean()), "mse": float(np.square(pred-truth).mean())}
        attribute["Macro"] = {key: float(np.mean([attribute[name][key] for name in ATTRS])) for key in ("srcc", "plcc", "mae", "mse")}
    else: attribute = None
    sci_p, sci_y = frame.pred_scientific_score.to_numpy(), frame.true_scientific_aggregate.to_numpy()
    aggregate = {"srcc": float(spearmanr(sci_p, sci_y).statistic), "plcc": float(pearsonr(sci_p, sci_y).statistic), "mae": float(np.abs(sci_p-sci_y).mean()), "mse": float(np.square(sci_p-sci_y).mean())}
    return metric, attribute, aggregate, frame


def train_model(kind, seed, train_ds, val_ds, device, out, logger, split_hash):
    model = build_model(kind, seed, device); attr_enabled = kind != "s3_a_original"; history = []
    for stage, max_epochs, patience in (("stage1", 15, 5), ("stage2", 30, 10)):
        optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=CFG.weight_decay); scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=CFG.scheduler_factor, patience=CFG.scheduler_patience, min_lr=1e-6); best, stale, best_payload = -float("inf"), 0, None
        loader = make_train_loader(train_ds, seed); started = time.perf_counter()
        for epoch in range(1, max_epochs + 1):
            train_stats = train_one_epoch(model, loader, optimizer, device, stage, attr_enabled); metric, attributes, aggregate, frame = evaluate(model, val_ds.samples, device); selection = metric["branch_macro_srcc"] if stage == "stage1" else metric["ranking_accuracy"]; scheduler.step(metric["validation_loss"])
            history.append({"model": kind, "seed": seed, "stage": stage, "epoch": epoch, **train_stats, **metric, "scientific_aggregate_srcc": aggregate["srcc"], "attribute_macro_srcc": attributes["Macro"]["srcc"] if attributes else np.nan, "learning_rate": optimizer.param_groups[0]["lr"]})
            logger.info("%s seed=%d %s epoch=%d AUC=%.4f PR=%.4f rank=%.4f sci=%.4f", kind, seed, stage, epoch, metric["auc"], metric["pr_auc"], metric["ranking_accuracy"], aggregate["srcc"])
            if selection > best:
                best, stale = selection, 0; best_payload = {"model_state_dict": model.state_dict(), "model": kind, "seed": seed, "stage": stage, "epoch": epoch, "metrics": metric, "attributes": attributes, "aggregate": aggregate, "split_hash": split_hash, "config": model_config(), "attribute_loss_weight": ATTR_WEIGHT if attr_enabled else 0.0, "train_seconds": time.perf_counter()-started}
            else:
                stale += 1
                if stale >= patience: break
        path = out / kind / f"seed_{seed}" / f"{stage}_best.pt"; path.parent.mkdir(parents=True, exist_ok=True); torch.save(best_payload, path); model.load_state_dict(best_payload["model_state_dict"], strict=True)
    metric, attributes, aggregate, frame = evaluate(model, val_ds.samples, device); final_path = out / kind / f"seed_{seed}" / "stage2_best.pt"; return model, metric, attributes, aggregate, frame, history, final_path


def bootstrap(predictions):
    pairs = {"s3_c_minus_s3_a": ("s3_c_four_adapter", "s3_a_original"), "s3_c_minus_s3_b": ("s3_c_four_adapter", "s3_b_shared_head")}; rng = np.random.default_rng(BOOTSTRAP_SEED); result = {}
    for name, (candidate, base) in pairs.items():
        values = {"auc": [], "pr_auc": [], "sci_aggregate_srcc": [], "attribute_macro_srcc": []}
        for _ in range(BOOTSTRAP_SAMPLES):
            per = {key: [] for key in values}
            for seed in SEEDS:
                c, b = predictions[candidate][seed], predictions[base][seed]; index = rng.integers(0, len(c), len(c)); yc, pc, pb = c.label.to_numpy()[index], c.probability.to_numpy()[index], b.probability.to_numpy()[index]
                per["auc"].append(roc_auc_score(yc, pc)-roc_auc_score(yc, pb)); per["pr_auc"].append(average_precision_score(yc, pc)-average_precision_score(yc, pb)); per["sci_aggregate_srcc"].append(spearmanr(c.pred_scientific_score.to_numpy()[index], c.true_scientific_aggregate.to_numpy()[index]).statistic-spearmanr(b.pred_scientific_score.to_numpy()[index], b.true_scientific_aggregate.to_numpy()[index]).statistic)
                if candidate != "s3_a_original" and base != "s3_a_original": per["attribute_macro_srcc"].append(np.mean([spearmanr(c[f"pred_{a}"].to_numpy()[index], c[f"true_{a}"].to_numpy()[index]).statistic-spearmanr(b[f"pred_{a}"].to_numpy()[index], b[f"true_{a}"].to_numpy()[index]).statistic for a in ATTRS]))
            for key, value in per.items():
                if value: values[key].append(float(np.mean(value)))
        result[name] = {key: {"mean": float(np.mean(value)), "ci95_percentile": [float(np.quantile(value,.025)), float(np.quantile(value,.975))], "positive_fraction": float((np.asarray(value)>0).mean())} if value else None for key, value in values.items()}
    return {"method": "1000 paired bootstrap samples over fixed 123 validation videos; mean delta across three seeds", "repetitions": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "comparisons": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--metadata", type=Path, default=Path(r"D:\Projects\science_video_ranker_mvp\science_video_project\data\parsed_metadata_filtered.csv")); parser.add_argument("--feature-dir", type=Path, default=Path(r"C:\Users\10465\.codex\worktrees\1b6b\science_video_ranker_mvp\science_video_project\outputs\features")); parser.add_argument("--output-dir", type=Path, default=PROJECT/"outputs/scientific_s3_overall_validation"); args=parser.parse_args(); out=args.output_dir.resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True); logger=logging.getLogger("scientific_s3"); logger.setLevel(logging.INFO); logger.handlers.clear(); handler=logging.FileHandler(out/"s3.log",encoding="utf-8"); handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s")); logger.addHandler(handler)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True); device=get_device(CFG.device); prepared=type("Args",(),{"metadata":args.metadata,"feature_dir":args.feature_dir,"split_seed":42})(); train_ds,val_ds,split_hash=prepare_data(prepared,logger)
    if split_hash != SPLIT_HASH: raise ValueError("Unexpected fixed split")
    protected={str(args.metadata.resolve()):sha(args.metadata)}
    for path in args.feature_dir.glob("*.pt"): protected[str(path.resolve())]=sha(path)
    config={"experiment":"S3", "created_at":datetime.now(timezone(timedelta(hours=8))).isoformat(), "split_hash":split_hash,"seeds":SEEDS,"metadata":str(args.metadata.resolve()),"feature_dir":str(args.feature_dir.resolve()),"base_model":"main_v3-D3 + KEEP_P2_S2 + T0", "fixed_training":{"stage1_epochs":15,"stage1_patience":5,"stage2_epochs":30,"stage2_patience":10,"optimizer":"AdamW(lr=5e-5, weight_decay=1e-3)","loss":"P2-S2 RankNet + pointwise + consistency + branch + branch RankNet", "attribute_loss":"B/C add required S3 four-attribute MSE at fixed lambda=0.02; A has no attribute outputs"},"models":{"s3_a_original":"official original ScientificBranch", "s3_b_shared_head":"S1 1540->256->4 sigmoid plus 256->128 fixed-fusion interface adapter", "s3_c_four_adapter":"S1 1540->224->four(224->32->1) sigmoid plus 224->128 fixed-fusion interface adapter"},"script_sha256":sha(__file__)}; write_json(out/"experiment_manifest.json",config)
    rows=[]; attributes=[]; aggregates=[]; paired=[]; histories=[]; predictions={}
    for kind in MODELS:
        predictions[kind]={}
        for seed in SEEDS:
            model,metric,attr,aggregate,frame,history,path=train_model(kind,seed,train_ds,val_ds,device,out,logger,split_hash); predictions[kind][seed]=frame; rows.append({"model":kind,"model_label":LABELS[kind],"seed":seed,**metric,"scientific_aggregate_srcc":aggregate["srcc"],"scientific_aggregate_plcc":aggregate["plcc"],"scientific_aggregate_mae":aggregate["mae"],"scientific_aggregate_mse":aggregate["mse"],"scientific_params":sum(x.numel() for x in model.scientific_branch.parameters()),"total_params":sum(x.numel() for x in model.parameters()),"checkpoint":str(path.resolve())}); histories+=history
            if attr: attributes += [{"model":kind,"seed":seed,"attribute":name,**values} for name,values in attr.items()]
            aggregates.append({"model":kind,"seed":seed,**aggregate})
            for _,record in frame.iterrows(): paired.append({"model":kind,"seed":seed,**record.to_dict(),"predicted_label":int(record.probability>=metric["best_threshold"])})
    pd.DataFrame(rows).to_csv(out/"per_seed_results.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(histories).to_csv(out/"training_history.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(attributes).to_csv(out/"scientific_attribute_results.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(aggregates).to_csv(out/"scientific_aggregate_results.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(paired).to_csv(out/"paired_predictions.csv",index=False,encoding="utf-8-sig")
    summary=pd.DataFrame(rows).groupby(["model","model_label"],as_index=False)[["auc","pr_auc","accuracy","f1","ranking_accuracy","scientific_aggregate_srcc","scientific_aggregate_plcc","scientific_aggregate_mae","scientific_aggregate_mse","scientific_params","total_params"]].agg(["mean","std"]); summary.to_csv(out/"overall_results.csv",encoding="utf-8-sig")
    boot=bootstrap(predictions); write_json(out/"bootstrap_results.json",boot); same=all(sha(path)==digest for path,digest in protected.items());
    if not same: raise RuntimeError("Input changed during S3")
    write_json(out/"data_verification.json",{"status":"passed","total":615,"train":492,"validation":123,"split_hash":split_hash,"source_files":len(protected),"source_integrity":same})
    (out/"model_architecture_audit.md").write_text("# S3 Architecture Audit\n\nS3-A uses the unmodified formal ScientificBranch. S3-B/C accept exactly the S1 1540D feature concatenation and produce four sigmoid attributes. Their equal mean is converted to a logit only because the fixed P2-S2 branch losses and fusion consume logits. The only added module is the documented 256/224-to-128 interface adapter needed by unchanged main fusion.\n",encoding="utf-8")
    mean=pd.DataFrame(rows).groupby("model").mean(numeric_only=True); a,b,c=(mean.loc[k] for k in MODELS); decision="A/B" if c.scientific_aggregate_srcc>=max(a.scientific_aggregate_srcc,b.scientific_aggregate_srcc) and c.auc>=a.auc and c.pr_auc>=a.pr_auc else "C" if c.scientific_aggregate_srcc>=a.scientific_aggregate_srcc else "D"
    summary_lines=["# S3 Scientific Branch Overall Validation Summary","","## 1. Experiment Setup","",f"Fixed 615 videos, 492/123 split, three seeds {SEEDS}; main_v3-D3 + KEEP_P2_S2 + T0 fixed.","","## 2. Pre-registered Models","",table(pd.DataFrame([{ "Model":LABELS[k],"Scientific structure":config["models"][k]} for k in MODELS])),"","## 3. Data and Split Verification","",json.dumps(json.loads((out/"data_verification.json").read_text(encoding="utf-8")),ensure_ascii=False,indent=2),"","## 4. Scientific Branch Architecture","",(out/"model_architecture_audit.md").read_text(encoding="utf-8"),"","## 5. Initialization and Training Audit","","All models use identical seeds, split, DataLoader construction, non-scientific configuration, P2-S2 stages, optimizer and scheduler. B/C alone expose the pre-registered four attributes and use the S3-mandated attribute MSE.","","## 6. Scientific Attribute Results","",table(pd.DataFrame(attributes).groupby(["model","attribute"])[["srcc","plcc","mae","mse"]].mean().reset_index()),"","## 7. Scientific Aggregate Results","",table(pd.DataFrame(aggregates).groupby("model")[["srcc","plcc","mae","mse"]].mean().reset_index()),"","## 8. Overall Main-task Results","",table(pd.DataFrame(rows).groupby("model")[["auc","pr_auc","accuracy","f1","ranking_accuracy"]].mean().reset_index()),"","## 9. Multi-seed Stability","",table(pd.DataFrame(rows)[["model","seed","auc","pr_auc","scientific_aggregate_srcc"]]),"","## 10. Paired Bootstrap","",json.dumps(boot,ensure_ascii=False,indent=2),"","## 11. Parameter / Complexity Comparison","",table(pd.DataFrame(rows).groupby("model")[["scientific_params","total_params"]].mean().reset_index()),"","## 12. Key Findings","",f"S3-C decision category: {decision}. Bootstrap describes only the fixed validation sample uncertainty.","","## 13. Final Scientific Branch Decision","",{"A/B":"Adopt Four-Adapter subject to the pre-registered metrics above.","C":"Record the attribute benefit but do not replace the formal main branch without a new locked split.","D":"Keep the original formal ScientificBranch; Four-Adapter remains an attribute-analysis module."}[decision]]; (out/"S3_SUMMARY.md").write_text("\n\n".join(summary_lines)+"\n",encoding="utf-8"); write_json(out/"completion.json",{"status":"completed","decision":decision,"split_hash":split_hash,"source_integrity":same}); print(f"Completed {out} decision={decision}")

if __name__ == "__main__": main()
