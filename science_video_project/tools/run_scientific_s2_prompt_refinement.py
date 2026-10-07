"""S2-P2: final fixed engagement-oriented prompt-space refinement."""

from __future__ import annotations

import argparse
import json
import random
import struct
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizerFast

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pipeline.utils_io import load_metadata, load_pt
from tools.run_scientific_s0 import ATTRS, FEATURES, HP, SEEDS, SPLIT_HASH, metrics, sha, table, write_json
from tools.run_scientific_s2_multimodal import TextOnlyFourAdapters, aggregate_metrics
from tools.run_scientific_s2_prompt_interest import PromptInterestFusion

PROMPTS = [
    {"name": "animation", "positive": "a science explanation using vivid animation or simulation to make the content engaging", "negative": "a science explanation without animated or simulated visual presentation"},
    {"name": "storytelling", "positive": "a science video using visual storytelling to make the explanation engaging and easy to follow", "negative": "a science video presenting information without visual storytelling"},
    {"name": "humor", "positive": "a science video using playful or humorous visual presentation to make the content entertaining", "negative": "a science video using a serious plain and non-humorous visual presentation"},
    {"name": "interaction", "positive": "a science presenter directly engaging the audience through questions gestures or interactive presentation", "negative": "a science presenter delivering information without audience-directed interaction"},
    {"name": "emotion", "positive": "a science video with expressive emotional and energetic visual presentation", "negative": "a science video with neutral restrained and emotionally flat visual presentation"},
]
FEATURE_NAMES = [f"{item['name']}_{stat}" for item in PROMPTS for stat in ("mean", "top20")]
TEXT_DIM, GENERIC_DIM, PROMPT_DIM, ADAPTER_DIM = 224, 64, 16, 32
MODELS = ("q0_text", "q1_old_prompt", "q2_engagement_prompt", "q3_generic_engagement")
LABELS = {"q0_text": "Q0 Text-only", "q1_old_prompt": "Q1 Old Prompt Fusion", "q2_engagement_prompt": "Q2 Engagement Prompt Fusion", "q3_generic_engagement": "Q3 Generic CLIP + Engagement Prompt"}
BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED = 1000, 7005


class EngagementPromptFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_projection = nn.Linear(sum(FEATURES.values()), TEXT_DIM)
        self.prompt_projection = nn.Linear(len(FEATURE_NAMES), PROMPT_DIM)
        self.activation, self.dropout = nn.ReLU(), nn.Dropout(HP["dropout"])
        self.text_adapters = nn.ModuleList([nn.Sequential(nn.Linear(TEXT_DIM, ADAPTER_DIM), nn.ReLU(), nn.Linear(ADAPTER_DIM, 1)) for _ in ATTRS[:3]])
        self.interest_adapter = nn.Sequential(nn.Linear(TEXT_DIM + PROMPT_DIM, ADAPTER_DIM), nn.ReLU(), nn.Linear(ADAPTER_DIM, 1))

    def forward(self, text, visual, prompt):
        del visual
        h_text = self.dropout(self.activation(self.text_projection(text)))
        h_prompt = self.dropout(self.activation(self.prompt_projection(prompt)))
        return torch.sigmoid(torch.cat([*[adapter(h_text) for adapter in self.text_adapters], self.interest_adapter(torch.cat([h_text, h_prompt], dim=-1))], dim=-1))


class GenericEngagementPromptFusion(nn.Module):
    def __init__(self, visual_dim):
        super().__init__()
        self.text_projection = nn.Linear(sum(FEATURES.values()), TEXT_DIM)
        self.visual_projection = nn.Linear(visual_dim, GENERIC_DIM)
        self.prompt_projection = nn.Linear(len(FEATURE_NAMES), PROMPT_DIM)
        self.activation, self.dropout = nn.ReLU(), nn.Dropout(HP["dropout"])
        self.text_adapters = nn.ModuleList([nn.Sequential(nn.Linear(TEXT_DIM, ADAPTER_DIM), nn.ReLU(), nn.Linear(ADAPTER_DIM, 1)) for _ in ATTRS[:3]])
        self.interest_adapter = nn.Sequential(nn.Linear(TEXT_DIM + GENERIC_DIM + PROMPT_DIM, ADAPTER_DIM), nn.ReLU(), nn.Linear(ADAPTER_DIM, 1))

    def forward(self, text, visual, prompt):
        h_text = self.dropout(self.activation(self.text_projection(text)))
        h_visual = self.dropout(self.activation(self.visual_projection(visual)))
        h_prompt = self.dropout(self.activation(self.prompt_projection(prompt)))
        return torch.sigmoid(torch.cat([*[adapter(h_text) for adapter in self.text_adapters], self.interest_adapter(torch.cat([h_text, h_visual, h_prompt], dim=-1))], dim=-1))


def make_model(kind, seed, visual_dim):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if kind == "q0_text": return TextOnlyFourAdapters()
    if kind == "q1_old_prompt": return PromptInterestFusion()
    if kind == "q2_engagement_prompt": return EngagementPromptFusion()
    if kind == "q3_generic_engagement": return GenericEngagementPromptFusion(visual_dim)
    raise ValueError(kind)


def forward(model, kind, text, visual, prompt):
    if kind == "q0_text": return model(text, visual)
    return model(text, visual, prompt)


def encode_text(model_path):
    tokenizer = CLIPTokenizerFast.from_pretrained(model_path, local_files_only=True)
    # Loading this local checkpoint through from_pretrained intermittently triggers
    # a Windows mmap access violation.  Load only the matching text-tower tensors,
    # the same explicit safetensors pattern used by the project vision loader.
    config = CLIPTextConfig.from_pretrained(model_path, local_files_only=True)
    model = CLIPTextModelWithProjection(config)
    checkpoint = Path(model_path) / "model.safetensors"
    with checkpoint.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
        data_start = 8 + header_size
        own = model.state_dict()
        expected = {key for key in own if not key.endswith(".position_ids")}
        loaded = set()
        for key, spec in header.items():
            if key not in expected:
                continue
            if spec["dtype"] != "F32":
                raise RuntimeError(f"Unsupported text tensor dtype for {key}: {spec['dtype']}")
            start, end = spec["data_offsets"]
            handle.seek(data_start + start)
            raw = handle.read(end - start)
            value = torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy()).reshape(spec["shape"])
            own[key].copy_(value)
            loaded.add(key)
    missing = sorted(expected - loaded)
    if missing:
        raise RuntimeError(f"Missing streamed text tensors: {missing[:3]}")
    model.eval()
    texts = [item[key] for item in PROMPTS for key in ("positive", "negative")]
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
    with torch.inference_mode():
        embeddings = F.normalize(model(**tokens).text_embeds, dim=-1)
    return embeddings[0::2].cpu(), embeddings[1::2].cpu(), tokenizer.__class__.__name__, model.config.to_dict()


def make_prompt_feature(frames, positive, negative):
    response = F.normalize(torch.as_tensor(frames, dtype=torch.float32), dim=-1) @ positive.T - F.normalize(torch.as_tensor(frames, dtype=torch.float32), dim=-1) @ negative.T
    top = response.topk(max(1, int(np.ceil(len(response) * .2))), dim=0).values.mean(dim=0)
    return torch.stack([response.mean(dim=0), top], dim=1).flatten().numpy()


def train(kind, text, visual, prompt, target, indices, seed, epochs, monitor=None):
    model = make_model(kind, seed, visual.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    loader = DataLoader(TensorDataset(text[indices], visual[indices], prompt[indices], target[indices]), batch_size=HP["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    best, chosen, stale, history = float("inf"), 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0
        for xb, vb, pb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(forward(model, kind, xb, vb, pb), yb)
            if not torch.isfinite(loss): raise ValueError(f"Nonfinite {kind} loss")
            loss.backward(); optimizer.step(); total += loss.item() * len(xb)
        row = {"model": kind, "seed": seed, "phase": "selection" if monitor is not None else "refit", "epoch": epoch, "train_mse": total / len(indices)}
        if monitor is not None:
            model.eval()
            with torch.inference_mode(): value = F.mse_loss(forward(model, kind, text[monitor], visual[monitor], prompt[monitor]), target[monitor]).item()
            row["inner_val_mse"] = value
            if value < best: best, chosen, stale = value, epoch, 0
            else: stale += 1
        history.append(row)
        if epoch == 1 or epoch % 10 == 0: print(f"{kind} seed={seed} {row['phase']} epoch={epoch} mse={row['train_mse']:.6f}", flush=True)
        if monitor is not None and stale >= HP["patience"]: break
    return model, chosen if monitor is not None else epochs, history


def bootstrap(predictions, target):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = {"q2_minus_q1": ("q2_engagement_prompt", "q1_old_prompt"), "q2_minus_q0": ("q2_engagement_prompt", "q0_text"), "q3_minus_q2": ("q3_generic_engagement", "q2_engagement_prompt")}
    output = {}
    for name, (candidate, baseline) in comparisons.items():
        values = []
        for _ in range(BOOTSTRAP_SAMPLES):
            sample = rng.integers(0, len(target), size=len(target))
            values.append(float(np.mean([metrics(predictions[candidate][seed][sample], target[sample])["content_interest"]["srcc"] - metrics(predictions[baseline][seed][sample], target[sample])["content_interest"]["srcc"] for seed in SEEDS])))
        output[name] = {"interest_srcc": {"mean": float(np.mean(values)), "ci95_percentile": [float(np.quantile(values, .025)), float(np.quantile(values, .975))], "positive_fraction": float(np.mean(np.asarray(values) > 0))}}
    return {"method": "paired nonparametric bootstrap over 123 validation videos; mean delta across three fixed training seeds", "repetitions": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "comparisons": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s0-dir", type=Path, required=True); parser.add_argument("--s2p-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/scientific_s2_prompt_refinement")
    parser.add_argument("--clip-model", type=Path, default=Path(r"D:\Projects\science_video_ranker_mvp\openaiclip-vit-large-patch14"))
    args = parser.parse_args(); out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True); (out / "checkpoints").mkdir(); torch.set_num_threads(4); torch.use_deterministic_algorithms(True); started = time.perf_counter()
    s0, s2p = json.loads((args.s0_dir / "config.json").read_text(encoding="utf-8")), json.loads((args.s2p_dir / "config.json").read_text(encoding="utf-8"))
    if s0["split_hash"] != SPLIT_HASH or s2p["split_hash"] != SPLIT_HASH or s0["features"] != FEATURES: raise ValueError("Prior protocol mismatch")
    metadata, feature_dir, reference = map(Path, [s0["metadata"], s0["feature_dir"], s0["reference_checkpoint"]]); protected = {str(p.resolve()): sha(p) for p in [metadata, reference]}
    manifest = pd.read_csv(args.s0_dir / "split_manifest.csv", dtype={"video_id": str}); data = load_metadata(metadata); data = data[data.video_id.isin(set(manifest.video_id))].copy().reset_index(drop=True)
    train_idx, val_idx = train_test_split(np.arange(len(data)), test_size=.2, random_state=42, stratify=data.label); inner_train, inner_val = train_test_split(train_idx, test_size=HP["inner_val_ratio"], random_state=HP["inner_split_seed"], stratify=data.iloc[train_idx].label)
    if len(data) != 615 or (len(train_idx), len(val_idx), len(inner_train), len(inner_val)) != (492, 123, 393, 99): raise ValueError("Unexpected fixed split")
    positive, negative, tokenizer_name, text_config = encode_text(args.clip_model)
    if positive.shape != (5, 768): raise ValueError(f"Text encoder mismatch: {positive.shape}")
    write_json(out / "prompt_v2_definitions.json", {"version": "s2p2_engagement_v1", "score": "cosine(pos)-cosine(neg)", "top_fraction": .2, "prompts": PROMPTS, "feature_order": FEATURE_NAMES})
    torch.save({"positive": positive, "negative": negative, "feature_names": FEATURE_NAMES, "clip_model": str(args.clip_model.resolve()), "tokenizer": tokenizer_name}, out / "prompt_v2_text_embeddings.pt")
    (out / "prompt_v1_definitions.json").write_text((args.s2p_dir / "prompt_definitions.json").read_text(encoding="utf-8"), encoding="utf-8")
    old_table = pd.read_csv(args.s2p_dir / "interest_prompt_features.csv", dtype={"video_id": str}).set_index("video_id")
    old_cols = list(old_table.columns); texts=[]; visuals=[]; prompt_rows=[]; audit_rows=[]
    for row in data.itertuples(index=False):
        path = feature_dir / f"{row.video_id}.pt"; protected[str(path.resolve())] = sha(path); sample = load_pt(path)
        text = np.concatenate([np.asarray(sample[name], dtype=np.float32) for name in FEATURES]); visual = np.asarray(sample["clip_video_feat"], dtype=np.float32); frames=np.asarray(sample["frame_features"], dtype=np.float32)
        if text.shape != (1540,) or visual.shape != (768,) or frames.ndim != 2 or frames.shape[1] != 768 or len(frames) < 1: raise ValueError(f"Invalid cached feature {path}")
        feature = make_prompt_feature(frames, positive, negative); texts.append(text); visuals.append(visual); prompt_rows.append(feature); audit_rows.append({"video_id": row.video_id, **{name: float(feature[i]) for i, name in enumerate(FEATURE_NAMES)}})
    text, visual, new_prompt = torch.from_numpy(np.stack(texts)), torch.from_numpy(np.stack(visuals)), torch.from_numpy(np.stack(prompt_rows)); old_prompt = torch.from_numpy(old_table.loc[data.video_id, old_cols].to_numpy(dtype=np.float32)); target = torch.from_numpy(data[ATTRS].to_numpy(dtype=np.float32) / 5.0)
    pd.DataFrame(audit_rows).to_csv(out / "engagement_prompt_features.csv", index=False, encoding="utf-8-sig")
    if not np.isfinite(new_prompt.numpy()).all() or np.any(new_prompt.numpy().std(axis=0) == 0): raise ValueError("Invalid engagement prompt feature")
    stats = pd.DataFrame({"dimension": FEATURE_NAMES, "mean": new_prompt.numpy().mean(0), "std": new_prompt.numpy().std(0), "min": new_prompt.numpy().min(0), "max": new_prompt.numpy().max(0), "nan_count": np.isnan(new_prompt.numpy()).sum(0)}); stats.to_csv(out / "prompt_feature_stats.csv", index=False, encoding="utf-8-sig")
    diagnostic = pd.DataFrame([{ "dimension": name, "srcc_content_interest": float(spearmanr(new_prompt[:, i].numpy(), data.content_interest).statistic), "srcc_video_aesthetics": float(spearmanr(new_prompt[:, i].numpy(), data.video_aesthetics).statistic)} for i, name in enumerate(FEATURE_NAMES)]); diagnostic.to_csv(out / "prompt_interest_aesthetic_correlation.csv", index=False, encoding="utf-8-sig")
    corr = pd.DataFrame(new_prompt.numpy(), columns=FEATURE_NAMES).corr(method="spearman"); corr.to_csv(out / "prompt_feature_srcc_matrix.csv", encoding="utf-8-sig")
    params={kind: sum(p.numel() for p in make_model(kind, 1, 768).parameters()) for kind in MODELS}; parameter_frame=pd.DataFrame([{ "Model": LABELS[k], "Text": True, "Old Prompt": k=="q1_old_prompt", "New Prompt": k in {"q2_engagement_prompt", "q3_generic_engagement"}, "Generic CLIP": k=="q3_generic_engagement", "Params": params[k]} for k in MODELS]); parameter_frame.to_csv(out / "parameter_comparison.csv", index=False, encoding="utf-8-sig")
    config={"experiment":"S2-P2", "created_at":datetime.now(timezone(timedelta(hours=8))).isoformat(), "s0_dir":str(args.s0_dir.resolve()), "s2p_dir":str(args.s2p_dir.resolve()), "metadata":str(metadata.resolve()), "feature_dir":str(feature_dir.resolve()), "reference_checkpoint":str(reference.resolve()), "clip_model":str(args.clip_model.resolve()), "clip_text_encoder":"CLIPTextModelWithProjection", "clip_text_config":text_config, "split_hash":SPLIT_HASH, "seeds":SEEDS, "features":FEATURES, "input_dim":1540, "new_prompt_feature_dim":10, "old_prompt_feature_dim":14, "generic_visual_dim":768, "prompt_projection_dim":PROMPT_DIM, "generic_projection_dim":GENERIC_DIM, "top_fraction":.2, "hyperparameters":HP, "parameter_counts":params, "selection":"S0 fixed original order; 393/99 minimum MSE; refit all 492; validation final only", "bootstrap":{"repetitions":BOOTSTRAP_SAMPLES,"seed":BOOTSTRAP_SEED}, "torch":torch.__version__, "numpy":np.__version__, "scipy":scipy.__version__, "sklearn":sklearn.__version__, "script_sha256":sha(__file__)}; write_json(out / "config.json", config)
    for kind in MODELS: write_json(out / f"config_{kind}.json", {**config, "active_model":kind, "active_model_label":LABELS[kind]})
    prompts={"q0_text":old_prompt, "q1_old_prompt":old_prompt, "q2_engagement_prompt":new_prompt, "q3_generic_engagement":new_prompt}; official_target=target[val_idx].numpy(); predictions={k:{} for k in MODELS}; rows=[]; histories=[]; prediction_rows=[]
    for seed in SEEDS:
        seed_dir=out/f"seed_{seed}"; seed_dir.mkdir()
        for kind in MODELS:
            _, epoch, select_history=train(kind,text,visual,prompts[kind],target,inner_train,seed,HP["max_epochs"],inner_val); model,_,refit_history=train(kind,text,visual,prompts[kind],target,train_idx,seed,epoch)
            model.eval()
            with torch.inference_mode(): pred=forward(model,kind,text[val_idx],visual[val_idx],prompts[kind][val_idx]).numpy()
            result, aggregate=metrics(pred,official_target),aggregate_metrics(pred,official_target); predictions[kind][seed]=pred
            write_json(seed_dir/f"{kind}_metrics.json", {"seed":seed,"model":LABELS[kind],"selected_epoch":epoch,"split_hash":SPLIT_HASH,"attributes":result,"aggregate":aggregate})
            torch.save({"model_state_dict":model.state_dict(),"model":kind,"seed":seed,"selected_epoch":epoch,"split_hash":SPLIT_HASH,"config":config},out/"checkpoints"/f"{kind}_seed_{seed}.pt")
            rows.extend({"model":kind,"model_label":LABELS[kind],"seed":seed,"attribute":attr,**values} for attr,values in result.items()); histories.extend(select_history+refit_history)
            for pos, video_id in enumerate(data.iloc[val_idx].video_id):
                row={"video_id":video_id,"seed":seed,"model":kind}; [row.update({f"true_{attr}":official_target[pos,i],f"pred_{attr}":pred[pos,i]}) for i,attr in enumerate(ATTRS)]; prediction_rows.append(row)
            print(f"completed {kind} seed={seed} epoch={epoch} interest_srcc={result['content_interest']['srcc']:.4f}",flush=True)
    frame=pd.DataFrame(rows); frame.to_csv(out/"metrics_all_seeds.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(histories).to_csv(out/"training_history.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(prediction_rows).to_csv(out/"predictions_by_model_seed.csv",index=False,encoding="utf-8-sig")
    mean=frame.groupby(["model","attribute"])[["srcc","plcc","mae","mse"]].mean(); std=frame.groupby(["model","attribute"])[["srcc","plcc","mae","mse"]].std()
    content=pd.DataFrame([{ "Model":LABELS[k], **{m.upper():f"{mean.loc[(k,'content_interest'),m]:.4f} ± {std.loc[(k,'content_interest'),m]:.4f}" for m in ("srcc","plcc","mae","mse")} } for k in MODELS]); content.to_csv(out/"content_interest_results.csv",index=False,encoding="utf-8-sig")
    aggregate=pd.DataFrame([{ "model":k,"model_label":LABELS[k],"seed":s,**aggregate_metrics(predictions[k][s],official_target)} for k in MODELS for s in SEEDS]); aggregate.to_csv(out/"aggregate_metrics_all_seeds.csv",index=False,encoding="utf-8-sig"); agg=aggregate.groupby("model")[["srcc","plcc"]].agg(["mean","std"])
    macro=pd.DataFrame([{ "Model":LABELS[k], "Macro SRCC":f"{mean.loc[(k,'Macro'),'srcc']:.4f} ± {std.loc[(k,'Macro'),'srcc']:.4f}","Macro PLCC":f"{mean.loc[(k,'Macro'),'plcc']:.4f} ± {std.loc[(k,'Macro'),'plcc']:.4f}","Aggregate SRCC":f"{agg.loc[k,('srcc','mean')]:.4f} ± {agg.loc[k,('srcc','std')]:.4f}","Aggregate PLCC":f"{agg.loc[k,('plcc','mean')]:.4f} ± {agg.loc[k,('plcc','std')]:.4f}"} for k in MODELS]); macro.to_csv(out/"aggregate_results.csv",index=False,encoding="utf-8-sig")
    per_seed=pd.DataFrame({"seed":SEEDS,**{k:[metrics(predictions[k][s],official_target)["content_interest"]["srcc"] for s in SEEDS] for k in MODELS}}); per_seed["q2_minus_q0"]=per_seed.q2_engagement_prompt-per_seed.q0_text; per_seed.to_csv(out/"per_seed_interest_srcc.csv",index=False,encoding="utf-8-sig")
    paired=[]
    for seed in SEEDS:
        for pos,video_id in enumerate(data.iloc[val_idx].video_id):
            row={"video_id":video_id,"seed":seed}
            for i,attr in enumerate(ATTRS):
                row[f"true_{attr}"]=official_target[pos,i]
                for kind in MODELS: row[f"pred_{attr}_{kind}"]=predictions[kind][seed][pos,i]
            paired.append(row)
    pd.DataFrame(paired).to_csv(out/"paired_predictions.csv",index=False,encoding="utf-8-sig"); boot=bootstrap(predictions,official_target); write_json(out/"bootstrap_results.json",boot)
    s2p_metrics=pd.read_csv(args.s2p_dir/"metrics_all_seeds.csv"); checks=[]
    for label,kind,source in (("q0_vs_p0","q0_text","p0_text"),("q1_vs_p2","q1_old_prompt","p2_prompt")):
        left=frame[frame.model==kind].set_index(["seed","attribute"])[["srcc","plcc","mae","mse"]].sort_index(); right=s2p_metrics[s2p_metrics.model==source].set_index(["seed","attribute"])[["srcc","plcc","mae","mse"]].sort_index(); checks.append({"check":label,"max_abs_error":float(np.abs((left-right).to_numpy()).max())})
    checks=pd.DataFrame(checks); checks.to_csv(out/"s2p_reproduction_checks.csv",index=False,encoding="utf-8-sig")
    if checks.max_abs_error.max()>1e-10: raise RuntimeError(f"Q0/Q1 reproduction failed: {checks}")
    unchanged=all(sha(p)==d for p,d in protected.items())
    if not unchanged: raise RuntimeError("Source inputs changed")
    q0,q1,q2,q3=(mean.loc[(k,"content_interest"),"srcc"] for k in MODELS); positive=int((per_seed.q2_minus_q0>0).sum()); macro_ok=mean.loc[("q2_engagement_prompt","Macro"),"srcc"]>=mean.loc[("q0_text","Macro"),"srcc"]-.005; aggregate_ok=aggregate[aggregate.model=="q2_engagement_prompt"].srcc.mean()>=aggregate[aggregate.model=="q0_text"].srcc.mean()-.005
    if q2>q1 and q2>q0 and positive>=2 and macro_ok and aggregate_ok and boot["comparisons"]["q2_minus_q0"]["interest_srcc"]["ci95_percentile"][0]>0: decision="A"
    elif q3>q2 and q3>q0 and mean.loc[("q3_generic_engagement","Macro"),"srcc"]>=mean.loc[("q0_text","Macro"),"srcc"]-.005 and aggregate[aggregate.model=="q3_generic_engagement"].srcc.mean()>=aggregate[aggregate.model=="q0_text"].srcc.mean()-.005: decision="C"
    elif q2>q1 and q2<=q0: decision="B"
    else: decision="D"
    summary=["# S2-P2 Engagement-oriented Interest Prompt Refinement Summary","","## 1. Experiment Setup","",f"615 videos; fixed 492/123 split; seeds={SEEDS}; identical S0/S1/S2 protocol.","","## 2. Motivation from S2-P","","A pre-defined final prompt-space redesign tests engagement-oriented rather than knowledge-presentation semantics.","","## 3. Prompt Space Redesign","",table(pd.DataFrame(PROMPTS)),"","## 4. Visual Feature Audit","","Existing cached Shared ViT-L/14 N×768 frame embeddings only; no image extraction or DeepSeek call.","","## 5. Prompt Feature Statistics","",table(stats),"","## 6. Prompt vs Interest / Aesthetic Diagnostic","",table(diagnostic),"","## 7. Model Architecture","",table(parameter_frame),"","## 8. Parameter Comparison","",f"Q2 has {params['q2_engagement_prompt']} parameters, Q1 has {params['q1_old_prompt']}; the difference is {(params['q2_engagement_prompt']-params['q1_old_prompt']):+d}.","","## 9. Content Interest Results","",table(content),"","## 10. Multi-seed Results","",table(per_seed),"","## 11. Macro and Aggregate Results","",table(macro),"","## 12. Paired Bootstrap","",json.dumps(boot,ensure_ascii=False,indent=2),"","## 13. Key Findings","",f"1. Q2/Q1/Q0/Q3 Interest SRCC={q2:.4f}/{q1:.4f}/{q0:.4f}/{q3:.4f}.",f"2. Q2-Q0 positive Interest SRCC in {positive}/3 seeds; Macro/aggregate hold={macro_ok}/{aggregate_ok}.",f"3. Official decision: Case {decision}.","","## 14. Final Decision","",{"A":"Adopt engagement prompt feature for Interest Adapter and enter S3.","B":"Keep text-only formal model; record semantic redesign as prompt-space improvement only.","C":"Evaluate Q3 only in a dedicated stability study before formal adoption.","D":"End visual Prompt exploration and retain text/knowledge-only Four-Adapter for S3."}[decision]]
    (out/"S2P2_SUMMARY.md").write_text("\n\n".join(summary)+"\n",encoding="utf-8"); write_json(out/"integrity_check.json",{"status":"passed","source_files":len(protected),"source_sha256":protected}); write_json(out/"completion.json",{"status":"completed","elapsed_seconds":time.perf_counter()-started,"split_hash":SPLIT_HASH,"source_integrity":unchanged,"decision_case":decision,"reproduction_max_error":float(checks.max_abs_error.max()),"params":params}); print(f"Completed: {out}",flush=True)

if __name__ == "__main__": main()
