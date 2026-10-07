"""Reload S2-P2-Fix checkpoints and verify frozen branches and saved predictions."""

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
from tools.run_scientific_s0 import ATTRS, FEATURES, SEEDS, SPLIT_HASH, sha, write_json
from tools.run_scientific_s2_prompt_isolated import build_increment, load_q0


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(); out = args.output_dir.resolve()
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    if config["split_hash"] != SPLIT_HASH or config["features"] != FEATURES: raise ValueError("Protocol mismatch")
    metadata, feature_dir = Path(config["metadata"]), Path(config["feature_dir"])
    data = load_metadata(metadata); manifest = pd.read_csv(Path(config["s0_dir"]) / "split_manifest.csv", dtype={"video_id": str}); data = data[data.video_id.isin(set(manifest.video_id))].copy().reset_index(drop=True)
    _, val_idx = train_test_split(np.arange(len(data)), test_size=.2, random_state=42, stratify=data.label)
    if (len(data), len(val_idx)) != (615, 123): raise ValueError("Fixed split mismatch")
    old = pd.read_csv(Path(config["s2p_dir"]) / "interest_prompt_features.csv", dtype={"video_id": str}).set_index("video_id")
    new = pd.read_csv(Path(config["s2p2_dir"]) / "engagement_prompt_features.csv", dtype={"video_id": str}).set_index("video_id")
    prompts = {"f1_old_prompt": torch.from_numpy(old.loc[data.video_id].to_numpy(dtype=np.float32)), "f2_new_prompt": torch.from_numpy(new.loc[data.video_id].to_numpy(dtype=np.float32)), "f3_generic_new_prompt": torch.from_numpy(new.loc[data.video_id].to_numpy(dtype=np.float32))}
    texts, visuals = [], []
    protected = {str(metadata.resolve()): sha(metadata)}
    for video_id in data.video_id:
        path = feature_dir / f"{video_id}.pt"; protected[str(path.resolve())] = sha(path); sample = load_pt(path)
        texts.append(np.concatenate([np.asarray(sample[key], dtype=np.float32) for key in FEATURES])); visuals.append(np.asarray(sample["clip_video_feat"], dtype=np.float32))
    text, visual = torch.from_numpy(np.stack(texts)), torch.from_numpy(np.stack(visuals))
    saved = pd.read_csv(out / "predictions_by_model_seed.csv", dtype={"video_id": str}); checks = []
    for seed in SEEDS:
        q0_path = Path(config["q0_checkpoints"][str(seed)]); protected[str(q0_path.resolve())] = sha(q0_path)
        q0 = load_q0(q0_path, seed).eval()
        with torch.inference_mode(): f0 = q0(text[val_idx], visual[val_idx]).numpy()
        expected = saved[(saved.model == "f0_frozen_text") & (saved.seed == seed)].set_index("video_id").loc[data.iloc[val_idx].video_id][[f"pred_{x}" for x in ATTRS]].to_numpy()
        error = float(np.abs(f0 - expected).max()); checks.append({"model": "f0_frozen_text", "seed": seed, "checkpoint_prediction_max_error": error})
        if error > 4e-7: raise ValueError(f"F0 mismatch {seed}: {error}")
        for kind in ("f1_old_prompt", "f2_new_prompt", "f3_generic_new_prompt"):
            payload = torch.load(out / "checkpoints" / f"{kind}_seed_{seed}.pt", map_location="cpu", weights_only=False)
            generic = kind == "f3_generic_new_prompt"; model = build_increment(q0_path, seed, int(prompts[kind].shape[1]), generic); model.load_state_dict(payload["model_state_dict"]); model.eval()
            with torch.inference_mode(): pred = model(text[val_idx], visual[val_idx], prompts[kind][val_idx]).numpy()
            expected = saved[(saved.model == kind) & (saved.seed == seed)].set_index("video_id").loc[data.iloc[val_idx].video_id][[f"pred_{x}" for x in ATTRS]].to_numpy()
            error, fixed_error = float(np.abs(pred - expected).max()), float(np.abs(pred[:, :3] - f0[:, :3]).max())
            if error > 4e-7 or fixed_error >= 1e-7: raise ValueError(f"Checkpoint/isolation mismatch {kind}/{seed}: {error}, {fixed_error}")
            checks.append({"model": kind, "seed": seed, "checkpoint_prediction_max_error": error, "frozen_branch_max_error": fixed_error})
    frozen = pd.read_csv(out / "frozen_branch_prediction_check.csv")
    if not frozen["pass"].all(): raise ValueError("Saved isolation check failed")
    if not all(sha(path) == digest for path, digest in protected.items()): raise RuntimeError("Source input changed")
    result = {"status": "passed", "split_hash": SPLIT_HASH, "source_integrity": True, "source_files": len(protected), "frozen_prediction_max_error": float(frozen[["max_abs_diff_info", "max_abs_diff_topic", "max_abs_diff_access"]].to_numpy().max()), "checkpoints": checks}
    write_json(out / "verification.json", result); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
