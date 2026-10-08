"""Verify saved S2 checkpoints, fixed protocol inputs, and S1 reproduction."""

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
from tools.run_scientific_s0 import ATTRS, FEATURES, HP, SEEDS, SPLIT_HASH, sha, write_json
from tools.run_scientific_s2_multimodal import MODELS, make_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    if config["split_hash"] != SPLIT_HASH or config["text_features"] != FEATURES:
        raise ValueError("S2 configuration does not match the fixed S0/S1 protocol")
    metadata, features, checkpoint = map(Path, [config["metadata"], config["feature_dir"], config["reference_checkpoint"]])
    protected = {str(path.resolve()): sha(path) for path in [metadata, checkpoint]}
    data = load_metadata(metadata)
    data = data[data.video_id.isin({path.stem for path in features.glob("*.pt")})].copy().reset_index(drop=True)
    train_idx, val_idx = train_test_split(np.arange(len(data)), test_size=.2, random_state=42, stratify=data.label)
    if (len(data), len(train_idx), len(val_idx)) != (615, 492, 123):
        raise ValueError("Unexpected official sample count or split")
    text_rows, visual_rows = [], []
    for video_id in data.video_id:
        path = features / f"{video_id}.pt"
        protected[str(path.resolve())] = sha(path)
        sample = load_pt(path)
        text_rows.append(np.concatenate([np.asarray(sample[key], dtype=np.float32) for key in FEATURES]))
        visual_rows.append(np.asarray(sample["clip_video_feat"], dtype=np.float32))
    text, visual = torch.from_numpy(np.stack(text_rows)), torch.from_numpy(np.stack(visual_rows))
    saved = pd.read_csv(out / "predictions_by_model_seed.csv", dtype={"video_id": str})
    rows = []
    for kind in MODELS:
        for seed in SEEDS:
            payload = torch.load(out / "checkpoints" / f"{kind}_seed_{seed}.pt", map_location="cpu", weights_only=False)
            if payload["model"] != kind or payload["seed"] != seed or payload["split_hash"] != SPLIT_HASH:
                raise ValueError(f"Checkpoint metadata mismatch: {kind}/{seed}")
            model = make_model(kind, seed, config["visual_input_dim"])
            model.load_state_dict(payload["model_state_dict"])
            model.eval()
            with torch.inference_mode():
                prediction = model(text[val_idx], visual[val_idx]).numpy()
            reference = saved[(saved.model == kind) & (saved.seed == seed)].set_index("video_id").loc[data.iloc[val_idx].video_id]
            expected = reference[[f"pred_{attr}" for attr in ATTRS]].to_numpy()
            error = float(np.abs(prediction - expected).max())
            if error > 4e-7:
                raise ValueError(f"Checkpoint prediction mismatch {kind}/{seed}: {error}")
            rows.append(dict(model=kind, seed=seed, checkpoint_prediction_max_error=error, selected_epoch=payload["selected_epoch"]))
    reproduction = pd.read_csv(out / "s1_text_only_reproduction_delta.csv")
    reproduction_error = float(np.abs(reproduction[["srcc", "plcc", "mae", "mse"]].to_numpy()).max())
    if reproduction_error > 1e-10:
        raise ValueError(f"S2-A differs from S1-B: {reproduction_error}")
    unchanged = all(sha(path) == value for path, value in protected.items())
    if not unchanged:
        raise RuntimeError("Source metadata, feature cache, or reference checkpoint changed")
    result = dict(status="passed", split_hash=SPLIT_HASH, source_files=len(protected), source_integrity=True,
                  text_only_reproduction_max_error=reproduction_error, checkpoints=rows,
                  official_decision="D", decision_note="DECISION_NOTE.md")
    write_json(out / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
