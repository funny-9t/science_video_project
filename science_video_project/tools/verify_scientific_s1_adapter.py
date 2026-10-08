"""Verify saved S1 predictions, checkpoints, split protocol, and bootstrap output."""

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
from tools.run_scientific_s0 import ATTRS, FEATURES, HP, SEEDS, SPLIT_HASH, metrics, sha, write_json
from tools.run_scientific_s1_adapter import (
    ADAPTER_DIM, ADAPTER_SHARED_DIM, BASELINE_HIDDEN, FourAttributeAdapters,
    SharedHead, aggregate_metrics, bootstrap_delta, parameter_count,
)


def model_from_kind(kind):
    if kind == "shared":
        return SharedHead()
    if kind == "adapter":
        return FourAttributeAdapters()
    raise ValueError(kind)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    torch.set_num_threads(4)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    if config["split_hash"] != SPLIT_HASH or config["features"] != FEATURES or config["hyperparameters"] != HP:
        raise ValueError("Saved configuration mismatch")
    s0_dir = Path(config["s0_dir"])
    manifest = pd.read_csv(s0_dir / "split_manifest.csv", dtype={"video_id": str})
    data = load_metadata(config["metadata"])
    data = data[data.video_id.isin(set(manifest.video_id))].copy().reset_index(drop=True)
    train_idx, val_idx = train_test_split(np.arange(len(data)), test_size=.2, random_state=42, stratify=data.label)
    inner_train, inner_val = train_test_split(
        train_idx, test_size=HP["inner_val_ratio"], random_state=HP["inner_split_seed"],
        stratify=data.iloc[train_idx].label,
    )
    assert len(data) == 615 and len(train_idx) == 492 and len(val_idx) == 123
    assert set(data.iloc[train_idx].video_id) == set(manifest.loc[manifest.split == "train", "video_id"])
    assert set(data.iloc[val_idx].video_id) == set(manifest.loc[manifest.split == "validation", "video_id"])
    assert set(data.iloc[inner_train].video_id) == set(manifest.loc[manifest.inner_split == "inner_train", "video_id"])
    assert set(data.iloc[inner_val].video_id) == set(manifest.loc[manifest.inner_split == "inner_validation", "video_id"])
    sources = json.loads((out / "integrity_check.json").read_text(encoding="utf-8"))["source_sha256"]
    assert all(sha(path) == digest for path, digest in sources.items())
    values = []
    for video_id in data.video_id:
        sample = load_pt(Path(config["feature_dir"]) / f"{video_id}.pt")
        values.append(np.concatenate([np.asarray(sample[key], dtype=np.float32) for key in FEATURES]))
    x = torch.from_numpy(np.stack(values))
    target = data.iloc[val_idx][ATTRS].to_numpy(dtype=np.float32) / 5.0
    stored_predictions = pd.read_csv(out / "predictions_by_model_seed.csv", dtype={"video_id": str})
    paired = pd.read_csv(out / "paired_predictions.csv", dtype={"video_id": str})
    history = pd.read_csv(out / "training_history.csv")
    all_metrics = pd.read_csv(out / "metrics_all_seeds.csv")
    s0_metrics = pd.read_csv(s0_dir / "metrics_all_seeds.csv")
    predictions = {"shared": {}, "adapter": {}}
    checks = []
    for kind in ("shared", "adapter"):
        for seed in SEEDS:
            selected = json.loads((out / f"seed_{seed}" / f"{kind}_metrics.json").read_text(encoding="utf-8"))
            selection_history = history[(history.model == kind) & (history.seed == seed) & (history.phase == "selection")]
            refit_history = history[(history.model == kind) & (history.seed == seed) & (history.phase == "refit")]
            assert selected["selected_epoch"] == int(selection_history.loc[selection_history.inner_val_mse.idxmin(), "epoch"])
            assert len(refit_history) == selected["selected_epoch"]
            checkpoint = torch.load(out / "checkpoints" / f"{kind}_seed_{seed}.pt", map_location="cpu", weights_only=False)
            assert checkpoint["model"] == kind and checkpoint["seed"] == seed and checkpoint["split_hash"] == SPLIT_HASH
            model = model_from_kind(kind)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()
            with torch.inference_mode():
                prediction = model(x[val_idx]).numpy()
            predictions[kind][seed] = prediction
            stored = stored_predictions[(stored_predictions.model == kind) & (stored_predictions.seed == seed)]
            assert list(stored.video_id) == list(data.iloc[val_idx].video_id)
            maximum_error = 0.0
            for index, attr in enumerate(ATTRS):
                np.testing.assert_allclose(stored[f"pred_{attr}"].to_numpy(), prediction[:, index], atol=1e-6)
                np.testing.assert_allclose(stored[f"true_{attr}"].to_numpy(), target[:, index], atol=1e-7)
                recomputed = metrics(prediction, target)[attr]
                for name, value in recomputed.items():
                    np.testing.assert_allclose(selected["attributes"][attr][name], value, atol=2e-6)
                    value_in_table = all_metrics[(all_metrics.model == kind) & (all_metrics.seed == seed) & (all_metrics.attribute == attr)].iloc[0][name]
                    np.testing.assert_allclose(value_in_table, value, atol=2e-6)
                paired_values = paired[paired.seed == seed]
                np.testing.assert_allclose(paired_values[f"pred_{attr}_{kind}"].to_numpy(), prediction[:, index], atol=1e-6)
                maximum_error = max(maximum_error, float(np.max(np.abs(stored[f"pred_{attr}"].to_numpy() - prediction[:, index]))))
            aggregate = aggregate_metrics(prediction, target)
            for name, value in aggregate.items():
                np.testing.assert_allclose(selected["aggregate"][name], value, atol=2e-6)
            checks.append(dict(model=kind, seed=seed, checkpoint_prediction_max_error=maximum_error,
                               selected_epoch=selected["selected_epoch"]))
    assert parameter_count(SharedHead()) == 395524
    assert parameter_count(FourAttributeAdapters()) == 374116
    for seed in SEEDS:
        for attr in ATTRS + ["Macro"]:
            current = all_metrics[(all_metrics.model == "shared") & (all_metrics.seed == seed) & (all_metrics.attribute == attr)].iloc[0]
            previous = s0_metrics[(s0_metrics.seed == seed) & (s0_metrics.attribute == attr)].iloc[0]
            for name in ("srcc", "plcc", "mae", "mse"):
                np.testing.assert_allclose(current[name], previous[name], atol=2e-6)
    bootstrap = bootstrap_delta(predictions["shared"], predictions["adapter"], target)
    saved_bootstrap = json.loads((out / "bootstrap_results.json").read_text(encoding="utf-8"))
    for statistic in ("macro_srcc", "aggregate_srcc"):
        np.testing.assert_allclose(bootstrap["adapter_minus_shared"][statistic]["mean"], saved_bootstrap["adapter_minus_shared"][statistic]["mean"], atol=1e-12)
        np.testing.assert_allclose(bootstrap["adapter_minus_shared"][statistic]["ci95_percentile"], saved_bootstrap["adapter_minus_shared"][statistic]["ci95_percentile"], atol=1e-12)
    report = dict(status="passed", split_hash=SPLIT_HASH, source_files=len(sources), checkpoints=checks,
                  shared_head_exactly_reproduces_s0=True, bootstrap_recomputed=True)
    write_json(out / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
