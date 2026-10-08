"""Verify saved S2-R checkpoints and R0's exact S1-B reproduction."""

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
from tools.run_scientific_s2_interest_rank import MODELS, make_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    if config["split_hash"] != SPLIT_HASH or config["features"] != FEATURES:
        raise ValueError("S2-R configuration mismatch")
    metadata, feature_dir, reference = map(Path, [config["metadata"], config["feature_dir"], config["reference_checkpoint"]])
    protected = {str(path.resolve()): sha(path) for path in [metadata, reference]}
    data = load_metadata(metadata)
    data = data[data.video_id.isin({path.stem for path in feature_dir.glob("*.pt")})].copy().reset_index(drop=True)
    _, val_idx = train_test_split(np.arange(len(data)), test_size=.2, random_state=42, stratify=data.label)
    if (len(data), len(val_idx)) != (615, 123):
        raise ValueError("Unexpected fixed split")
    values = []
    for video_id in data.video_id:
        path = feature_dir / f"{video_id}.pt"
        protected[str(path.resolve())] = sha(path)
        sample = load_pt(path)
        values.append(np.concatenate([np.asarray(sample[key], dtype=np.float32) for key in FEATURES]))
    text = torch.from_numpy(np.stack(values))
    saved = pd.read_csv(out / "predictions_by_model_seed.csv", dtype={"video_id": str})
    rows = []
    for kind in MODELS:
        for seed in SEEDS:
            payload = torch.load(out / "checkpoints" / f"{kind}_seed_{seed}.pt", map_location="cpu", weights_only=False)
            if payload["model"] != kind or payload["seed"] != seed or payload["split_hash"] != SPLIT_HASH:
                raise ValueError(f"Checkpoint metadata mismatch for {kind}/{seed}")
            model = make_model(seed)
            model.load_state_dict(payload["model_state_dict"])
            model.eval()
            with torch.inference_mode():
                prediction = model(text[val_idx], torch.empty(0)).numpy()
            expected = saved[(saved.model == kind) & (saved.seed == seed)].set_index("video_id").loc[data.iloc[val_idx].video_id]
            max_error = float(np.abs(prediction - expected[[f"pred_{attr}" for attr in ATTRS]].to_numpy()).max())
            if max_error > 4e-7:
                raise ValueError(f"Checkpoint prediction mismatch for {kind}/{seed}: {max_error}")
            rows.append({"model": kind, "seed": seed, "checkpoint_prediction_max_error": max_error, "selected_epoch": payload["selected_epoch"]})
    reproduction = pd.read_csv(out / "s1_r0_reproduction_delta.csv")
    reproduction_error = float(np.abs(reproduction[["srcc", "plcc", "mae", "mse"]].to_numpy()).max())
    if reproduction_error > 1e-10:
        raise ValueError("R0 does not reproduce S1-B")
    unchanged = all(sha(path) == digest for path, digest in protected.items())
    if not unchanged:
        raise RuntimeError("Source input changed")
    result = {"status": "passed", "split_hash": SPLIT_HASH, "source_files": len(protected), "source_integrity": True,
              "r0_reproduction_max_error": reproduction_error, "checkpoints": rows, "official_decision": "D"}
    write_json(out / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
