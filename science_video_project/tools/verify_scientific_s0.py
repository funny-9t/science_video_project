"""Verify saved S0 splits, predictions, checkpoint inference, and source integrity."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from run_scientific_s0 import ATTRS, FEATURES, SEEDS, SPLIT_HASH, new_model, sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    out = parser.parse_args().output_dir
    torch.set_num_threads(4)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(out / "split_manifest.csv", dtype={"video_id": str})
    expected = manifest.loc[manifest.split == "validation"].set_index("video_id")
    assert len(manifest) == manifest.video_id.nunique() == 615
    assert len(expected) == 123 and sum(manifest.split == "train") == 492
    assert config["split_hash"] == SPLIT_HASH
    assert (manifest.loc[manifest.split == "validation", "inner_split"] == "not_used_for_selection").all()
    hashes = json.loads((out / "integrity_check.json").read_text(encoding="utf-8"))["source_sha256"]
    assert all(sha(path) == digest for path, digest in hashes.items())
    history = pd.read_csv(out / "training_history.csv")
    all_metrics = pd.read_csv(out / "metrics_all_seeds.csv")
    checks = []
    for seed in SEEDS:
        predictions = pd.read_csv(out / f"attribute_predictions_seed{seed}.csv", dtype={"video_id": str})
        assert len(predictions) == predictions.video_id.nunique() == 123
        assert set(predictions.video_id) == set(expected.index)
        target = expected.loc[predictions.video_id, ATTRS].to_numpy() / 5.0
        saved_target = predictions[[a + "_target" for a in ATTRS]].to_numpy()
        pred = predictions[[a + "_pred" for a in ATTRS]].to_numpy()
        np.testing.assert_allclose(saved_target, target, atol=1e-7)
        assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
        saved = json.loads((out / f"seed_{seed}_metrics.json").read_text(encoding="utf-8"))
        selection = history[(history.seed == seed) & (history.phase == "selection")]
        assert saved["selected_epoch"] == int(selection.loc[selection.inner_val_mse.idxmin(), "epoch"])
        refit = history[(history.seed == seed) & (history.phase == "refit")]
        assert len(refit) == saved["selected_epoch"]
        for i, attr in enumerate(ATTRS):
            calculated = dict(srcc=float(spearmanr(pred[:, i], target[:, i]).statistic),
                              plcc=float(pearsonr(pred[:, i], target[:, i]).statistic),
                              mae=float(np.abs(pred[:, i] - target[:, i]).mean()),
                              mse=float(np.square(pred[:, i] - target[:, i]).mean()))
            for name, value in calculated.items():
                np.testing.assert_allclose(saved["metrics"][attr][name], value, atol=2e-6)
                stored = all_metrics[(all_metrics.seed == seed) & (all_metrics.attribute == attr)].iloc[0]
                np.testing.assert_allclose(stored[name], value, atol=2e-6)
        for name in ("srcc", "plcc", "mae", "mse"):
            np.testing.assert_allclose(saved["metrics"]["Macro"][name],
                                       np.mean([saved["metrics"][a][name] for a in ATTRS]), atol=1e-8)
        arrays = []
        for video_id in predictions.video_id:
            sample = torch.load(Path(config["feature_dir"]) / f"{video_id}.pt",
                                map_location="cpu", weights_only=False)
            arrays.append(np.concatenate([np.asarray(sample[k], dtype=np.float32) for k in FEATURES]))
        checkpoint = torch.load(out / "checkpoints" / f"seed_{seed}.pt", map_location="cpu", weights_only=False)
        assert checkpoint["split_hash"] == SPLIT_HASH
        model = new_model(seed)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        with torch.inference_mode():
            restored = model(torch.from_numpy(np.stack(arrays))).numpy()
        np.testing.assert_allclose(restored, pred, atol=1e-6)
        checks.append(dict(seed=seed, predictions=123,
                           max_checkpoint_prediction_error=float(np.max(np.abs(restored - pred))),
                           selected_epoch=saved["selected_epoch"]))
    report = dict(status="passed", split_hash=SPLIT_HASH, protected_files=len(hashes), seeds=checks)
    write_json(out / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
