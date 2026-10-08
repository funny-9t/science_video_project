"""Reload and verify all saved S2-P prompt-probing checkpoints."""

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
from tools.run_scientific_s2_prompt_interest import MODELS, forward, make_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    if config["split_hash"] != SPLIT_HASH or config["text_features"] != FEATURES:
        raise ValueError("S2-P configuration does not match S0/S1")
    metadata, features, reference = map(Path, [config["metadata"], config["feature_dir"], config["reference_checkpoint"]])
    protected = {str(path.resolve()): sha(path) for path in [metadata, reference]}
    data = load_metadata(metadata)
    data = data[data.video_id.isin({path.stem for path in features.glob("*.pt")})].copy().reset_index(drop=True)
    _, val_idx = train_test_split(np.arange(len(data)), test_size=.2, random_state=42, stratify=data.label)
    if (len(data), len(val_idx)) != (615, 123):
        raise ValueError("Unexpected fixed data protocol")
    prompt_table = pd.read_csv(out / "interest_prompt_features.csv", dtype={"video_id": str}).set_index("video_id")
    feature_columns = [column for column in prompt_table.columns]
    original_prompt = torch.from_numpy(prompt_table.loc[data.video_id, feature_columns].to_numpy(dtype=np.float32))
    shuffled_map = pd.read_csv(out / "shuffled_prompt_mapping.csv", dtype=str).set_index("video_id")
    shuffled_prompt = torch.from_numpy(prompt_table.loc[shuffled_map.loc[data.video_id, "source_video_id"], feature_columns].to_numpy(dtype=np.float32))
    if torch.equal(original_prompt, shuffled_prompt):
        raise ValueError("Prompt control is not shuffled")
    texts, visuals = [], []
    for video_id in data.video_id:
        path = features / f"{video_id}.pt"
        protected[str(path.resolve())] = sha(path)
        sample = load_pt(path)
        texts.append(np.concatenate([np.asarray(sample[key], dtype=np.float32) for key in FEATURES]))
        visuals.append(np.asarray(sample["clip_video_feat"], dtype=np.float32))
    text, visual = torch.from_numpy(np.stack(texts)), torch.from_numpy(np.stack(visuals))
    saved = pd.read_csv(out / "predictions_by_model_seed.csv", dtype={"video_id": str})
    rows = []
    for kind in MODELS:
        prompt = shuffled_prompt if kind == "p4_shuffled_prompt" else original_prompt
        for seed in SEEDS:
            payload = torch.load(out / "checkpoints" / f"{kind}_seed_{seed}.pt", map_location="cpu", weights_only=False)
            model = make_model(kind, seed, int(visual.shape[1]))
            model.load_state_dict(payload["model_state_dict"])
            model.eval()
            with torch.inference_mode():
                prediction = forward(model, kind, text[val_idx], visual[val_idx], prompt[val_idx]).numpy()
            reference_prediction = saved[(saved.model == kind) & (saved.seed == seed)].set_index("video_id").loc[data.iloc[val_idx].video_id]
            expected = reference_prediction[[f"pred_{attr}" for attr in ATTRS]].to_numpy()
            error = float(np.abs(prediction - expected).max())
            if error > 4e-7:
                raise ValueError(f"Prediction mismatch for {kind}/{seed}: {error}")
            rows.append({"model": kind, "seed": seed, "checkpoint_prediction_max_error": error, "selected_epoch": payload["selected_epoch"]})
    reproduction = pd.read_csv(out / "s2_reproduction_checks.csv")
    reproduction_error = float(reproduction.max_abs_error.max())
    if reproduction_error > 1e-10:
        raise ValueError("P0/P1 does not reproduce S2")
    unchanged = all(sha(path) == digest for path, digest in protected.items())
    if not unchanged:
        raise RuntimeError("Source input changed")
    result = {"status": "passed", "split_hash": SPLIT_HASH, "source_files": len(protected), "source_integrity": True,
              "s2_reproduction_max_error": reproduction_error, "checkpoints": rows, "official_decision": "D"}
    write_json(out / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
