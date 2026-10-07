"""Finalize an already-trained S2-P2-Fix run from its per-seed checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pipeline.utils_io import load_metadata, load_pt
from tools.run_scientific_s0 import ATTRS, FEATURES, SEEDS, SPLIT_HASH, metrics, sha, table, write_json
from tools.run_scientific_s2_multimodal import aggregate_metrics
from tools.run_scientific_s2_prompt_isolated import LABELS, MODELS, bootstrap, build_increment, load_q0


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(); out = args.output_dir.resolve()
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    metadata, feature_dir = Path(config["metadata"]), Path(config["feature_dir"])
    manifest = pd.read_csv(Path(config["s0_dir"]) / "split_manifest.csv", dtype={"video_id": str})
    data = load_metadata(metadata); data = data[data.video_id.isin(set(manifest.video_id))].copy().reset_index(drop=True)
    _, val_idx = train_test_split(np.arange(len(data)), test_size=.2, random_state=42, stratify=data.label)
    if (len(data), len(val_idx)) != (615, 123): raise ValueError("Fixed split mismatch")
    old = pd.read_csv(Path(config["s2p_dir"]) / "interest_prompt_features.csv", dtype={"video_id": str}).set_index("video_id")
    new = pd.read_csv(Path(config["s2p2_dir"]) / "engagement_prompt_features.csv", dtype={"video_id": str}).set_index("video_id")
    prompts = {"f1_old_prompt": torch.from_numpy(old.loc[data.video_id].to_numpy(dtype=np.float32)), "f2_new_prompt": torch.from_numpy(new.loc[data.video_id].to_numpy(dtype=np.float32)), "f3_generic_new_prompt": torch.from_numpy(new.loc[data.video_id].to_numpy(dtype=np.float32))}
    texts, visuals, protected = [], [], {str(metadata.resolve()): sha(metadata), str((Path(config["s2p_dir"]) / "interest_prompt_features.csv").resolve()): sha(Path(config["s2p_dir"]) / "interest_prompt_features.csv"), str((Path(config["s2p2_dir"]) / "engagement_prompt_features.csv").resolve()): sha(Path(config["s2p2_dir"]) / "engagement_prompt_features.csv")}
    for video_id in data.video_id:
        path = feature_dir / f"{video_id}.pt"; protected[str(path.resolve())] = sha(path); sample = load_pt(path)
        texts.append(np.concatenate([np.asarray(sample[key], dtype=np.float32) for key in FEATURES])); visuals.append(np.asarray(sample["clip_video_feat"], dtype=np.float32))
    text, visual = torch.from_numpy(np.stack(texts)), torch.from_numpy(np.stack(visuals)); target = data[ATTRS].to_numpy(dtype=np.float32)[val_idx] / 5.0
    predictions, all_rows, pred_rows, checks = {kind: {} for kind in MODELS}, [], [], []
    for seed in SEEDS:
        q0_path = Path(config["q0_checkpoints"][str(seed)]); protected[str(q0_path.resolve())] = sha(q0_path)
        q0 = load_q0(q0_path, seed).eval()
        with torch.inference_mode(): f0 = q0(text[val_idx], visual[val_idx]).numpy()
        predictions["f0_frozen_text"][seed] = f0
        checks.append({"seed": seed, "model": "f0_frozen_text", "max_abs_diff_info": 0.0, "max_abs_diff_topic": 0.0, "max_abs_diff_access": 0.0, "pass": True})
        for kind in MODELS[1:]:
            payload = torch.load(out / "checkpoints" / f"{kind}_seed_{seed}.pt", map_location="cpu", weights_only=False)
            generic = kind == "f3_generic_new_prompt"; model = build_increment(q0_path, seed, int(prompts[kind].shape[1]), generic); model.load_state_dict(payload["model_state_dict"]); model.eval()
            with torch.inference_mode(): pred = model(text[val_idx], visual[val_idx], prompts[kind][val_idx]).numpy()
            diff = np.abs(pred[:, :3] - f0[:, :3]).max(axis=0); passed = bool((diff < 1e-7).all())
            if not passed: raise RuntimeError(f"Isolation failed: {kind}/{seed}: {diff}")
            predictions[kind][seed] = pred; checks.append({"seed": seed, "model": kind, "max_abs_diff_info": float(diff[0]), "max_abs_diff_topic": float(diff[1]), "max_abs_diff_access": float(diff[2]), "pass": passed})
        for kind in MODELS:
            result = metrics(predictions[kind][seed], target)
            all_rows.extend({"model": kind, "model_label": LABELS[kind], "seed": seed, "attribute": attr, **value} for attr, value in result.items())
            for pos, video_id in enumerate(data.iloc[val_idx].video_id):
                row = {"video_id": video_id, "seed": seed, "model": kind}
                for index, attr in enumerate(ATTRS): row[f"true_{attr}"], row[f"pred_{attr}"] = target[pos, index], predictions[kind][seed][pos, index]
                pred_rows.append(row)
    check = pd.DataFrame(checks); check.to_csv(out / "frozen_branch_prediction_check.csv", index=False, encoding="utf-8-sig")
    frame = pd.DataFrame(all_rows); frame.to_csv(out / "metrics_all_seeds.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(pred_rows).to_csv(out / "predictions_by_model_seed.csv", index=False, encoding="utf-8-sig")
    mean, std = frame.groupby(["model", "attribute"])[["srcc", "plcc", "mae", "mse"]].mean(), frame.groupby(["model", "attribute"])[["srcc", "plcc", "mae", "mse"]].std()
    interest = pd.DataFrame([{ "model": kind, "model_label": LABELS[kind], **{metric: float(mean.loc[(kind, "content_interest"), metric]) for metric in ("srcc", "plcc", "mae", "mse")}, **{f"{metric}_std": float(std.loc[(kind, "content_interest"), metric]) for metric in ("srcc", "plcc", "mae", "mse")} } for kind in MODELS]); interest.to_csv(out / "interest_metrics.csv", index=False, encoding="utf-8-sig")
    per_seed = pd.DataFrame({"seed": SEEDS, **{kind: [metrics(predictions[kind][seed], target)["content_interest"]["srcc"] for seed in SEEDS] for kind in MODELS}}); per_seed.to_csv(out / "per_seed_interest_srcc.csv", index=False, encoding="utf-8-sig")
    delta_rows = []
    for name, candidate, baseline in [("F1 - F0", "f1_old_prompt", "f0_frozen_text"), ("F2 - F0", "f2_new_prompt", "f0_frozen_text"), ("F2 - F1", "f2_new_prompt", "f1_old_prompt"), ("F3 - F2", "f3_generic_new_prompt", "f2_new_prompt")]:
        delta_rows.append({"comparison": name, **{f"delta_{metric}": float(mean.loc[(candidate, "content_interest"), metric] - mean.loc[(baseline, "content_interest"), metric]) for metric in ("srcc", "plcc", "mae", "mse")}})
    delta = pd.DataFrame(delta_rows); delta.to_csv(out / "delta_results.csv", index=False, encoding="utf-8-sig")
    aggregate_rows = []
    for kind in MODELS:
        values = [aggregate_metrics(predictions[kind][seed], target) for seed in SEEDS]
        aggregate_rows.append({"model": kind, "model_label": LABELS[kind], "macro_srcc": float(mean.loc[(kind, "Macro"), "srcc"]), "macro_srcc_std": float(std.loc[(kind, "Macro"), "srcc"]), "aggregate_srcc": float(np.mean([x["srcc"] for x in values])), "aggregate_srcc_std": float(np.std([x["srcc"] for x in values], ddof=1))})
    aggregate = pd.DataFrame(aggregate_rows); aggregate.to_csv(out / "derived_macro_aggregate.csv", index=False, encoding="utf-8-sig")
    paired = []
    for seed in SEEDS:
        for position, video_id in enumerate(data.iloc[val_idx].video_id):
            row = {"video_id": video_id, "seed": seed}
            for index, attr in enumerate(ATTRS):
                row[f"true_{attr}"] = target[position, index]
                for kind in MODELS: row[f"pred_{attr}_{kind}"] = predictions[kind][seed][position, index]
            paired.append(row)
    pd.DataFrame(paired).to_csv(out / "paired_predictions.csv", index=False, encoding="utf-8-sig")
    boot = bootstrap(predictions, target); write_json(out / "bootstrap_results.json", boot)
    f0, f1, f2, f3 = [float(interest.loc[interest.model == kind, "srcc"].iloc[0]) for kind in MODELS]
    positive_f2 = int((per_seed.f2_new_prompt - per_seed.f0_frozen_text > 0).sum()); boot_f2 = boot["comparisons"]["f2_minus_f0"]["interest_srcc"]
    if f2 > f0 and f2 > f1 and positive_f2 >= 2 and boot_f2["positive_fraction"] > .5: decision = "A"
    elif f1 > f0 and f2 > f1: decision = "B"
    elif f3 > f2 > f0: decision = "C"
    else: decision = "D"
    summary = ["# S2-P2-Fix Interest-Isolated Prompt Increment Test Summary", "", "## 1. Experiment Setup", "", "615 videos; fixed 492/123 outer split; 393/99 internal split; seeds [42, 123, 2026]. Only content_interest MSE trained.", "", "## 2. Why Isolation Was Necessary", "", "S2-P2 jointly retrained the shared representation. This run freezes Q0 projection plus info/topic/access adapters, so the three non-interest predictions cannot change.", "", "## 3. Base Q0 Checkpoint", "", "Each seed uses its matching S2-P2 Q0 checkpoint; F0 is the direct loaded checkpoint output and is not retrained.", "", "## 4. Frozen Module Audit", "", "Projection, Dropout, and the first three adapters are requires_grad=False and forced to eval in every train epoch.", "", "## 5. Model Architecture", "", "F1/F2: [frozen h_text; projected prompt] -> trainable 224-D fusion -> warm-started Q0 interest adapter. F3 additionally uses 768->64 generic CLIP projection.", "", "## 6. Isolation Verification", "", table(check), "", "## 7. Content Interest Results", "", table(interest), "", "## 8. Multi-seed Results", "", table(per_seed), "", "## 9. Prompt Increment Comparison", "", table(delta), "", "## 10. Derived Macro / Aggregate", "", table(aggregate), "", "## 11. Paired Bootstrap", "", json.dumps(boot, ensure_ascii=False, indent=2), "", "## 12. Key Findings", "", f"1. All frozen prediction checks passed: {bool(check['pass'].all())}.", f"2. Interest SRCC F0/F1/F2/F3: {f0:.4f}/{f1:.4f}/{f2:.4f}/{f3:.4f}.", f"3. F2-F0 is positive in {positive_f2}/3 seeds; bootstrap mean delta={boot_f2['mean']:.4f}, 95% CI={boot_f2['ci95_percentile']}.", f"4. Primary decision: Case {decision}.", "", "## 13. Final Decision", "", {"A": "New Prompt has an isolated increment; evaluate formal adoption only in a pre-specified follow-up.", "B": "Both prompts help but the new Prompt is stronger; retain the frozen control result and avoid validation-driven redesign.", "C": "Generic CLIP is best in this isolated test; evaluate its cost/stability before adoption.", "D": "No stable isolated Prompt increment. Retain the text/knowledge-only Four-Adapter and proceed to S3."}[decision]]
    (out / "S2P2_FIX_SUMMARY.md").write_text("\n\n".join(summary) + "\n", encoding="utf-8")
    source_integrity = all(sha(path) == digest for path, digest in protected.items())
    if not source_integrity: raise RuntimeError("A protected source input changed")
    write_json(out / "integrity_check.json", {"status": "passed", "source_files": len(protected), "source_sha256": protected})
    write_json(out / "completion.json", {"status": "completed", "split_hash": SPLIT_HASH, "source_integrity": source_integrity, "isolation_passed": bool(check["pass"].all()), "decision_case": decision, "interest_srcc": {kind: float(interest.loc[interest.model == kind, "srcc"].iloc[0]) for kind in MODELS}})
    print(f"Completed: {out}; decision={decision}")


if __name__ == "__main__": main()
