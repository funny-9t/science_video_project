"""S2-P2-Fix: isolated content-interest Prompt increment test.

The Q0 shared projection and the first three attribute adapters are loaded from
the matching S2-P2 Q0 checkpoint and kept frozen/eval.  Only the interest
increment path is trainable, so any change in macro/aggregate quality is
strictly attributable to content_interest.
"""

from __future__ import annotations

import argparse
import json
import random
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
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pipeline.utils_io import load_metadata, load_pt
from tools.run_scientific_s0 import ATTRS, FEATURES, HP, SEEDS, SPLIT_HASH, metrics, sha, table, write_json
from tools.run_scientific_s2_multimodal import TextOnlyFourAdapters, aggregate_metrics

MODELS = ("f0_frozen_text", "f1_old_prompt", "f2_new_prompt", "f3_generic_new_prompt")
LABELS = {
    "f0_frozen_text": "F0 Frozen Q0 Text-only Reference",
    "f1_old_prompt": "F1 Frozen Q0 + Old 14D Prompt",
    "f2_new_prompt": "F2 Frozen Q0 + New 10D Engagement Prompt",
    "f3_generic_new_prompt": "F3 Frozen Q0 + Generic CLIP + New Prompt",
}
PROMPT_DIM, GENERIC_DIM = 16, 64
BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED = 1000, 7006


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class FrozenInterestIncrement(nn.Module):
    """Warm-started Q0 interest head with an explicitly frozen text backbone."""

    def __init__(self, base: TextOnlyFourAdapters, prompt_dim: int, generic: bool) -> None:
        super().__init__()
        self.base = base
        self.generic = generic
        self.prompt_projection = nn.Linear(prompt_dim, PROMPT_DIM)
        if generic:
            self.generic_projection = nn.Linear(768, GENERIC_DIM)
        increment_dim = PROMPT_DIM + (GENERIC_DIM if generic else 0)
        self.fusion = nn.Linear(224 + increment_dim, 224)
        self.activation = nn.ReLU()
        self.increment_dropout = nn.Dropout(HP["dropout"])
        self.freeze_q0_modules()

    def freeze_q0_modules(self) -> None:
        for module in [self.base.projection, self.base.dropout, *self.base.adapters[:3]]:
            module.eval()
        for parameter in self.base.projection.parameters():
            parameter.requires_grad_(False)
        for adapter in self.base.adapters[:3]:
            for parameter in adapter.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # Calling .train() recursively must never reactivate Dropout in frozen Q0.
        self.freeze_q0_modules()
        return self

    def forward(self, text: torch.Tensor, visual: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            h_text = self.base.dropout(self.base.activation(self.base.projection(text)))
            fixed_logits = [adapter(h_text) for adapter in self.base.adapters[:3]]
        pieces = [h_text, self.increment_dropout(self.activation(self.prompt_projection(prompt)))]
        if self.generic:
            pieces.append(self.increment_dropout(self.activation(self.generic_projection(visual))))
        fused = self.activation(self.fusion(torch.cat(pieces, dim=-1)))
        interest_logit = self.base.adapters[3](fused)
        return torch.sigmoid(torch.cat(fixed_logits + [interest_logit], dim=-1))


def load_q0(path: Path, seed: int) -> TextOnlyFourAdapters:
    set_seed(seed)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model") != "q0_text" or payload.get("seed") != seed or payload.get("split_hash") != SPLIT_HASH:
        raise ValueError(f"Invalid matching Q0 checkpoint: {path}")
    model = TextOnlyFourAdapters()
    model.load_state_dict(payload["model_state_dict"])
    return model


def build_increment(q0_path: Path, seed: int, prompt_dim: int, generic: bool) -> FrozenInterestIncrement:
    set_seed(seed)
    model = FrozenInterestIncrement(TextOnlyFourAdapters(), prompt_dim, generic)
    payload = torch.load(q0_path, map_location="cpu", weights_only=False)
    model.base.load_state_dict(payload["model_state_dict"])
    model.freeze_q0_modules()
    return model


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if not trainable_only or p.requires_grad)


def train_interest(model, text, visual, prompt, target, indices, seed, epochs, monitor=None):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=HP["lr"], weight_decay=HP["weight_decay"])
    loader = DataLoader(TensorDataset(text[indices], visual[indices], prompt[indices], target[indices]), batch_size=HP["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    best, selected_epoch, stale, history = float("inf"), 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0
        for xb, vb, pb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(xb, vb, pb)[:, 3], yb[:, 3])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite isolated-interest loss")
            loss.backward(); optimizer.step(); total += loss.item() * len(xb)
        row = {"seed": seed, "phase": "selection" if monitor is not None else "refit", "epoch": epoch, "train_interest_mse": total / len(indices)}
        if monitor is not None:
            model.eval()
            with torch.inference_mode():
                value = F.mse_loss(model(text[monitor], visual[monitor], prompt[monitor])[:, 3], target[monitor, 3]).item()
            row["inner_interest_mse"] = value
            if value < best:
                best, selected_epoch, stale = value, epoch, 0
            else:
                stale += 1
        history.append(row)
        if epoch == 1 or epoch % 10 == 0:
            print(f"increment seed={seed} {row['phase']} epoch={epoch} mse={row['train_interest_mse']:.6f}", flush=True)
        if monitor is not None and stale >= HP["patience"]:
            break
    return model, (selected_epoch if monitor is not None else epochs), history


def bootstrap(predictions, target):
    comparisons = {"f2_minus_f0": ("f2_new_prompt", "f0_frozen_text"), "f2_minus_f1": ("f2_new_prompt", "f1_old_prompt"), "f3_minus_f2": ("f3_generic_new_prompt", "f2_new_prompt")}
    rng, result = np.random.default_rng(BOOTSTRAP_SEED), {}
    for name, (candidate, baseline) in comparisons.items():
        values = []
        for _ in range(BOOTSTRAP_SAMPLES):
            sampled = rng.integers(0, len(target), size=len(target))
            values.append(float(np.mean([spearmanr(predictions[candidate][seed][sampled, 3], target[sampled, 3]).statistic - spearmanr(predictions[baseline][seed][sampled, 3], target[sampled, 3]).statistic for seed in SEEDS])))
        values = np.asarray(values)
        result[name] = {"interest_srcc": {"mean": float(values.mean()), "ci95_percentile": [float(np.quantile(values, .025)), float(np.quantile(values, .975))], "positive_fraction": float((values > 0).mean())}}
    return {"method": "paired nonparametric bootstrap over 123 fixed validation videos; statistic is mean interest SRCC delta across three fixed training seeds", "repetitions": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "comparisons": result}


def frozen_audit(model: FrozenInterestIncrement):
    frozen_prefixes = ("base.projection", "base.adapters.0", "base.adapters.1", "base.adapters.2")
    rows = []
    for name, parameter in model.named_parameters():
        rows.append({"parameter": name, "numel": parameter.numel(), "requires_grad": bool(parameter.requires_grad), "must_be_frozen": name.startswith(frozen_prefixes)})
        if name.startswith(frozen_prefixes) and parameter.requires_grad:
            raise RuntimeError(f"Frozen parameter is trainable: {name}")
    if model.base.projection.training or model.base.dropout.training or any(adapter.training for adapter in model.base.adapters[:3]):
        raise RuntimeError("Frozen Q0 module is not in eval mode")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s0-dir", type=Path, default=PROJECT / "outputs/scientific_s0")
    parser.add_argument("--s2p-dir", type=Path, default=PROJECT / "outputs/scientific_s2_prompt_interest")
    parser.add_argument("--s2p2-dir", type=Path, default=PROJECT / "outputs/scientific_s2_prompt_refinement")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/scientific_s2_prompt_isolated")
    args = parser.parse_args(); out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing experiment: {out}")
    out.mkdir(parents=True, exist_ok=True); (out / "checkpoints").mkdir()
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True); started = time.perf_counter()

    s0 = json.loads((args.s0_dir / "config.json").read_text(encoding="utf-8"))
    s2p2 = json.loads((args.s2p2_dir / "config.json").read_text(encoding="utf-8"))
    if s0["split_hash"] != SPLIT_HASH or s2p2["split_hash"] != SPLIT_HASH or s0["features"] != FEATURES:
        raise ValueError("Prior experiment protocol mismatch")
    metadata, feature_dir, reference = map(Path, [s0["metadata"], s0["feature_dir"], s0["reference_checkpoint"]])
    protected = {str(path.resolve()): sha(path) for path in [metadata, reference, args.s2p_dir / "interest_prompt_features.csv", args.s2p2_dir / "engagement_prompt_features.csv"]}
    q0_paths = {seed: args.s2p2_dir / "checkpoints" / f"q0_text_seed_{seed}.pt" for seed in SEEDS}
    for path in q0_paths.values():
        protected[str(path.resolve())] = sha(path)

    manifest = pd.read_csv(args.s0_dir / "split_manifest.csv", dtype={"video_id": str})
    data = load_metadata(metadata); data = data[data.video_id.isin(set(manifest.video_id))].copy().reset_index(drop=True)
    train_idx, val_idx = train_test_split(np.arange(len(data)), test_size=.2, random_state=42, stratify=data.label)
    inner_train, inner_val = train_test_split(train_idx, test_size=HP["inner_val_ratio"], random_state=HP["inner_split_seed"], stratify=data.iloc[train_idx].label)
    if len(data) != 615 or (len(train_idx), len(val_idx), len(inner_train), len(inner_val)) != (492, 123, 393, 99):
        raise ValueError("Unexpected fixed split size")
    for name, indices, expected in [("validation", val_idx, "validation"), ("inner_train", inner_train, "inner_train"), ("inner_validation", inner_val, "inner_validation")]:
        field = "split" if name == "validation" else "inner_split"
        if set(data.iloc[indices].video_id) != set(manifest.loc[manifest[field] == expected, "video_id"]):
            raise ValueError(f"Fixed {name} membership mismatch")

    old = pd.read_csv(args.s2p_dir / "interest_prompt_features.csv", dtype={"video_id": str}).set_index("video_id")
    new = pd.read_csv(args.s2p2_dir / "engagement_prompt_features.csv", dtype={"video_id": str}).set_index("video_id")
    texts, visuals = [], []
    for video_id in data.video_id:
        path = feature_dir / f"{video_id}.pt"; protected[str(path.resolve())] = sha(path)
        sample = load_pt(path)
        text = np.concatenate([np.asarray(sample[name], dtype=np.float32) for name in FEATURES])
        visual = np.asarray(sample["clip_video_feat"], dtype=np.float32)
        if text.shape != (1540,) or visual.shape != (768,) or not np.isfinite(text).all() or not np.isfinite(visual).all():
            raise ValueError(f"Invalid cached feature: {path}")
        texts.append(text); visuals.append(visual)
    text, visual = torch.from_numpy(np.stack(texts)), torch.from_numpy(np.stack(visuals))
    old_prompt = torch.from_numpy(old.loc[data.video_id].to_numpy(dtype=np.float32))
    new_prompt = torch.from_numpy(new.loc[data.video_id].to_numpy(dtype=np.float32))
    target = torch.from_numpy(data[ATTRS].to_numpy(dtype=np.float32) / 5.0)
    if old_prompt.shape != (615, 14) or new_prompt.shape != (615, 10):
        raise ValueError(f"Prompt cache dimensions mismatch: {tuple(old_prompt.shape)}, {tuple(new_prompt.shape)}")

    model_spec = {"f0_frozen_text": {"prompt": "none", "generic": False}, "f1_old_prompt": {"prompt": "old_14d", "generic": False}, "f2_new_prompt": {"prompt": "new_10d", "generic": False}, "f3_generic_new_prompt": {"prompt": "new_10d", "generic": True}}
    parameter_rows, audit = [], []
    for kind in MODELS[1:]:
        prototype = build_increment(q0_paths[42], 42, 14 if kind == "f1_old_prompt" else 10, kind == "f3_generic_new_prompt")
        audit.extend({"model": kind, **row} for row in frozen_audit(prototype))
        parameter_rows.append({"model": kind, "model_label": LABELS[kind], "total_parameters": count_parameters(prototype), "trainable_parameters": count_parameters(prototype, True), "frozen_parameters": count_parameters(prototype) - count_parameters(prototype, True)})
    parameter_rows.insert(0, {"model": "f0_frozen_text", "model_label": LABELS["f0_frozen_text"], "total_parameters": 374116, "trainable_parameters": 0, "frozen_parameters": 374116})
    pd.DataFrame(parameter_rows).to_csv(out / "parameter_comparison.csv", index=False, encoding="utf-8-sig")
    write_json(out / "frozen_module_audit.json", {"frozen_modules": ["base.projection", "base.dropout", "base.adapters.0", "base.adapters.1", "base.adapters.2"], "forced_eval": True, "allowed_trainable_modules": ["base.adapters.3", "prompt_projection", "fusion", "generic_projection (F3 only)"], "parameters": audit})
    config = {"experiment": "S2-P2-FIX", "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(), "s0_dir": str(args.s0_dir.resolve()), "s2p_dir": str(args.s2p_dir.resolve()), "s2p2_dir": str(args.s2p2_dir.resolve()), "metadata": str(metadata.resolve()), "feature_dir": str(feature_dir.resolve()), "reference_checkpoint": str(reference.resolve()), "q0_checkpoints": {str(k): str(v.resolve()) for k, v in q0_paths.items()}, "split_hash": SPLIT_HASH, "seeds": SEEDS, "features": FEATURES, "attributes": ATTRS, "input_dim": 1540, "old_prompt_dim": 14, "new_prompt_dim": 10, "generic_clip_dim": 768, "hyperparameters": HP, "loss": "MSE(content_interest) only", "selection": "fixed 393/99 minimum internal content_interest MSE; refit all 492; evaluate 123 once", "freeze": "Q0 projection + first three adapters eval/frozen; original interest adapter warm-started and trainable", "bootstrap": {"repetitions": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED}, "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "script_sha256": sha(__file__)}
    write_json(out / "config.json", config)
    for kind in MODELS:
        write_json(out / f"config_{kind}.json", {**config, "active_model": kind, "active_model_label": LABELS[kind], "model_spec": model_spec[kind]})

    prompts = {"f1_old_prompt": old_prompt, "f2_new_prompt": new_prompt, "f3_generic_new_prompt": new_prompt}
    all_rows, histories, pred_rows, check_rows = [], [], [], []
    predictions = {kind: {} for kind in MODELS}; official_target = target[val_idx].numpy()
    for seed in SEEDS:
        seed_dir = out / f"seed_{seed}"; seed_dir.mkdir()
        q0 = load_q0(q0_paths[seed], seed).eval()
        with torch.inference_mode():
            f0 = q0(text[val_idx], visual[val_idx]).numpy()
        predictions["f0_frozen_text"][seed] = f0
        write_json(seed_dir / "f0_frozen_text_metrics.json", {"seed": seed, "source_checkpoint": str(q0_paths[seed].resolve()), "attributes": metrics(f0, official_target), "aggregate": aggregate_metrics(f0, official_target)})
        check_rows.append({"seed": seed, "model": "f0_frozen_text", "max_abs_diff_info": 0.0, "max_abs_diff_topic": 0.0, "max_abs_diff_access": 0.0, "pass": True})
        for kind in MODELS[1:]:
            prompt = prompts[kind]; generic = kind == "f3_generic_new_prompt"; prompt_dim = int(prompt.shape[1])
            selection = build_increment(q0_paths[seed], seed, prompt_dim, generic)
            _, selected_epoch, selection_history = train_interest(selection, text, visual, prompt, target, inner_train, seed, HP["max_epochs"], inner_val)
            refit = build_increment(q0_paths[seed], seed, prompt_dim, generic)
            model, _, refit_history = train_interest(refit, text, visual, prompt, target, train_idx, seed, selected_epoch)
            model.eval()
            with torch.inference_mode():
                pred = model(text[val_idx], visual[val_idx], prompt[val_idx]).numpy()
            diffs = np.abs(pred[:, :3] - f0[:, :3]).max(axis=0)
            passed = bool((diffs < 1e-7).all())
            check_rows.append({"seed": seed, "model": kind, "max_abs_diff_info": float(diffs[0]), "max_abs_diff_topic": float(diffs[1]), "max_abs_diff_access": float(diffs[2]), "pass": passed})
            if not passed:
                raise RuntimeError(f"Isolation failed for {kind}/seed={seed}: {diffs}")
            predictions[kind][seed] = pred
            result, aggregate = metrics(pred, official_target), aggregate_metrics(pred, official_target)
            write_json(seed_dir / f"{kind}_metrics.json", {"seed": seed, "model": LABELS[kind], "selected_epoch": selected_epoch, "split_hash": SPLIT_HASH, "attributes": result, "aggregate": aggregate})
            torch.save({"model_state_dict": model.state_dict(), "model": kind, "seed": seed, "selected_epoch": selected_epoch, "split_hash": SPLIT_HASH, "q0_source_checkpoint": str(q0_paths[seed].resolve()), "config": config}, out / "checkpoints" / f"{kind}_seed_{seed}.pt")
            histories.extend([{**item, "model": kind} for item in selection_history + refit_history])
            print(f"completed {kind} seed={seed} epoch={selected_epoch} interest_srcc={result['content_interest']['srcc']:.4f}", flush=True)
        for kind in MODELS:
            pred = predictions[kind][seed]; result, aggregate = metrics(pred, official_target), aggregate_metrics(pred, official_target)
            all_rows.extend({"model": kind, "model_label": LABELS[kind], "seed": seed, "attribute": attr, **value} for attr, value in result.items())
            for pos, video_id in enumerate(data.iloc[val_idx].video_id):
                row = {"video_id": video_id, "seed": seed, "model": kind}
                for i, attr in enumerate(ATTRS): row[f"true_{attr}"], row[f"pred_{attr}"] = official_target[pos, i], pred[pos, i]
                pred_rows.append(row)
    check = pd.DataFrame(check_rows); check.to_csv(out / "frozen_branch_prediction_check.csv", index=False, encoding="utf-8-sig")
    if not check["pass"].all(): raise RuntimeError("At least one frozen branch isolation check failed")
    frame = pd.DataFrame(all_rows); frame.to_csv(out / "metrics_all_seeds.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(histories).to_csv(out / "training_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pred_rows).to_csv(out / "predictions_by_model_seed.csv", index=False, encoding="utf-8-sig")
    mean, std = frame.groupby(["model", "attribute"])[["srcc", "plcc", "mae", "mse"]].mean(), frame.groupby(["model", "attribute"])[["srcc", "plcc", "mae", "mse"]].std()
    interest = pd.DataFrame([{ "model": kind, "model_label": LABELS[kind], **{metric: float(mean.loc[(kind, "content_interest"), metric]) for metric in ("srcc", "plcc", "mae", "mse")}, **{f"{metric}_std": float(std.loc[(kind, "content_interest"), metric]) for metric in ("srcc", "plcc", "mae", "mse")} } for kind in MODELS]); interest.to_csv(out / "interest_metrics.csv", index=False, encoding="utf-8-sig")
    per_seed = pd.DataFrame({"seed": SEEDS, **{kind: [metrics(predictions[kind][seed], official_target)["content_interest"]["srcc"] for seed in SEEDS] for kind in MODELS}}); per_seed.to_csv(out / "per_seed_interest_srcc.csv", index=False, encoding="utf-8-sig")
    delta_rows = []
    for name, candidate, baseline in [("F1 - F0", "f1_old_prompt", "f0_frozen_text"), ("F2 - F0", "f2_new_prompt", "f0_frozen_text"), ("F2 - F1", "f2_new_prompt", "f1_old_prompt"), ("F3 - F2", "f3_generic_new_prompt", "f2_new_prompt")]:
        delta_rows.append({"comparison": name, **{f"delta_{metric}": float(mean.loc[(candidate, "content_interest"), metric] - mean.loc[(baseline, "content_interest"), metric]) for metric in ("srcc", "plcc", "mae", "mse")}})
    pd.DataFrame(delta_rows).to_csv(out / "delta_results.csv", index=False, encoding="utf-8-sig")
    aggregate_rows = []
    for kind in MODELS:
        values = [aggregate_metrics(predictions[kind][seed], official_target) for seed in SEEDS]
        aggregate_rows.append({"model": kind, "model_label": LABELS[kind], "macro_srcc": float(mean.loc[(kind, "Macro"), "srcc"]), "macro_srcc_std": float(std.loc[(kind, "Macro"), "srcc"]), "aggregate_srcc": float(np.mean([v["srcc"] for v in values])), "aggregate_srcc_std": float(np.std([v["srcc"] for v in values], ddof=1))})
    pd.DataFrame(aggregate_rows).to_csv(out / "derived_macro_aggregate.csv", index=False, encoding="utf-8-sig")
    paired = []
    for seed in SEEDS:
        for pos, video_id in enumerate(data.iloc[val_idx].video_id):
            row = {"video_id": video_id, "seed": seed}
            for i, attr in enumerate(ATTRS):
                row[f"true_{attr}"] = official_target[pos, i]
                for kind in MODELS: row[f"pred_{attr}_{kind}"] = predictions[kind][seed][pos, i]
            paired.append(row)
    pd.DataFrame(paired).to_csv(out / "paired_predictions.csv", index=False, encoding="utf-8-sig")
    boot = bootstrap(predictions, official_target); write_json(out / "bootstrap_results.json", boot)
    unchanged = all(sha(path) == digest for path, digest in protected.items())
    if not unchanged: raise RuntimeError("A protected source input changed during experiment")
    f0, f1, f2, f3 = [float(interest.loc[interest.model == kind, "srcc"].iloc[0]) for kind in MODELS]
    positive_f2 = int((per_seed.f2_new_prompt - per_seed.f0_frozen_text > 0).sum())
    boot_f2 = boot["comparisons"]["f2_minus_f0"]["interest_srcc"]
    if f2 > f0 and f2 > f1 and positive_f2 >= 2 and boot_f2["positive_fraction"] > .5: decision = "A"
    elif f1 > f0 and f2 > f1: decision = "B"
    elif f3 > f2 > f0: decision = "C"
    else: decision = "D"
    summary = ["# S2-P2-Fix Interest-Isolated Prompt Increment Test Summary", "", "## 1. Experiment Setup", "", "615 videos; fixed 492/123 outer split; 393/99 internal split; seeds [42, 123, 2026]. Only content_interest MSE trained.", "", "## 2. Why Isolation Was Necessary", "", "S2-P2 jointly retrained the shared representation. This run freezes Q0 projection plus info/topic/access adapters, so the three non-interest predictions cannot change.", "", "## 3. Base Q0 Checkpoint", "", "Each seed uses its matching S2-P2 Q0 checkpoint; F0 is the direct loaded checkpoint output and is not retrained.", "", "## 4. Frozen Module Audit", "", "Projection, Dropout, and the first three adapters are requires_grad=False and forced to eval in every train epoch.", "", "## 5. Model Architecture", "", "F1/F2: [frozen h_text; projected prompt] -> trainable 224-D fusion -> warm-started Q0 interest adapter. F3 additionally uses 768->64 generic CLIP projection.", "", "## 6. Isolation Verification", "", table(check), "", "## 7. Content Interest Results", "", table(interest), "", "## 8. Multi-seed Results", "", table(per_seed), "", "## 9. Prompt Increment Comparison", "", table(pd.DataFrame(delta_rows)), "", "## 10. Derived Macro / Aggregate", "", table(pd.DataFrame(aggregate_rows)), "", "## 11. Paired Bootstrap", "", json.dumps(boot, ensure_ascii=False, indent=2), "", "## 12. Key Findings", "", f"1. All frozen prediction checks passed: {bool(check['pass'].all())}.", f"2. Interest SRCC F0/F1/F2/F3: {f0:.4f}/{f1:.4f}/{f2:.4f}/{f3:.4f}.", f"3. F2-F0 is positive in {positive_f2}/3 seeds; bootstrap mean delta={boot_f2['mean']:.4f}, 95% CI={boot_f2['ci95_percentile']}.", f"4. Primary decision: Case {decision}.", "", "## 13. Final Decision", "", {"A": "New Prompt has an isolated increment; evaluate formal adoption only in a pre-specified follow-up.", "B": "Both prompts help but the new Prompt is stronger; retain the frozen control result and avoid validation-driven redesign.", "C": "Generic CLIP is best in this isolated test; evaluate its cost/stability before adoption.", "D": "No stable isolated Prompt increment. Retain the text/knowledge-only Four-Adapter and proceed to S3."}[decision]]
    (out / "S2P2_FIX_SUMMARY.md").write_text("\n\n".join(summary) + "\n", encoding="utf-8")
    write_json(out / "integrity_check.json", {"status": "passed", "source_files": len(protected), "source_sha256": protected})
    write_json(out / "completion.json", {"status": "completed", "elapsed_seconds": time.perf_counter() - started, "split_hash": SPLIT_HASH, "source_integrity": unchanged, "isolation_passed": True, "decision_case": decision, "interest_srcc": {kind: float(interest.loc[interest.model == kind, "srcc"].iloc[0]) for kind in MODELS}})
    print(f"Completed: {out}", flush=True)


if __name__ == "__main__":
    main()
