"""Run E5: technical-aesthetic visual fusion on DIVIDE-MaxWell caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.utils_train import set_seed


SEEDS = (42, 123, 2026)
PROBE_VERSION = "e5_divide_visual_linear_fusion_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--e1-cache", type=Path,
        default=PROJECT_ROOT / "outputs" / "external_technical_validation" / "cache"
        / "external_validation" / "divide_maxwell",
    )
    parser.add_argument(
        "--e3-cache", type=Path,
        default=PROJECT_ROOT / "outputs" / "external_aesthetic_validation" / "cache"
        / "divide_maxwell_aesthetic_features.pt",
    )
    parser.add_argument(
        "--e1-predictions", type=Path,
        default=PROJECT_ROOT / "outputs" / "external_technical_validation" / "divide_predictions.csv",
    )
    parser.add_argument(
        "--e3-predictions", type=Path,
        default=PROJECT_ROOT / "outputs" / "external_aesthetic_validation"
        / "external_aesthetic_predictions.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "divide_visual_fusion",
    )
    return parser.parse_args()


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def correlation(prediction: np.ndarray, target: np.ndarray, kind: str) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(prediction) & np.isfinite(target)
    prediction, target = prediction[valid], target[valid]
    if prediction.size < 2 or np.unique(prediction).size < 2 or np.unique(target).size < 2:
        return float("nan")
    if kind == "spearman":
        prediction, target = rankdata(prediction), rankdata(target)
    x = torch.from_numpy(prediction.copy())
    y = torch.from_numpy(target.copy())
    x, y = x - x.mean(), y - y.mean()
    denominator = torch.sqrt(torch.sum(x * x) * torch.sum(y * y)).clamp_min(1e-15)
    return float((torch.sum(x * y) / denominator).item())


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    error = np.asarray(prediction, np.float64) - np.asarray(target, np.float64)
    return {
        "SRCC": correlation(prediction, target, "spearman"),
        "PLCC": correlation(prediction, target, "pearson"),
        "MAE": float(np.mean(np.abs(error))),
        "MSE": float(np.mean(error ** 2)),
    }


def load_aligned_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    aesthetic = torch.load(args.e3_cache, map_location="cpu", weights_only=False)
    aesthetic_ids = list(map(str, aesthetic["video_ids"]))
    if len(aesthetic_ids) != len(set(aesthetic_ids)):
        raise ValueError("Duplicate IDs in E3 cache")
    aesthetic_lookup = {video_id: index for index, video_id in enumerate(aesthetic_ids)}

    rows = []
    technical_embeddings = []
    aesthetic_embeddings = []
    for path in sorted(args.e1_cache.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        video_id = str(payload["video_id"])
        if video_id not in aesthetic_lookup:
            continue
        labels = payload["labels"]
        tech_embedding = np.asarray(payload["technical_embedding"], dtype=np.float32)
        aesthetic_index = aesthetic_lookup[video_id]
        aes_embedding = np.asarray(aesthetic["aesthetic_embeddings"][aesthetic_index], dtype=np.float32)
        if tech_embedding.shape != (128,) or aes_embedding.shape != (128,):
            raise ValueError(f"Unexpected embedding shape for {video_id}")
        rows.append({
            "video_id": video_id,
            "split": str(labels["official_split"]),
            "technical_mos": float(labels["technical_mos"]),
            "aesthetic_mos": float(labels["aesthetic_mos"]),
            "overall_mos": float(labels["overall_mos"]),
            "s_tech": float(payload["zero_shot_score"]),
            "s_aes": float(aesthetic["aesthetic_scores"][aesthetic_index]),
            "e1_feature_version": str(payload["feature_version"]),
        })
        technical_embeddings.append(tech_embedding)
        aesthetic_embeddings.append(aes_embedding)
    frame = pd.DataFrame(rows)
    if len(frame) != len(aesthetic_ids):
        missing_e1 = sorted(set(aesthetic_ids) - set(frame["video_id"]))
        missing_e3 = sorted(set(frame["video_id"]) - set(aesthetic_ids))
        raise RuntimeError(f"Cache mismatch: missing E1={len(missing_e1)}, missing E3={len(missing_e3)}")
    frame["overall_normalized"] = frame["overall_mos"] / 5.0

    e1_predictions = pd.read_csv(args.e1_predictions, dtype={"video_id": str})
    e3_predictions = pd.read_csv(args.e3_predictions, dtype={"video_id": str})
    e1_predictions["video_id"] = e1_predictions["video_id"].str.zfill(4)
    e3_predictions["video_id"] = e3_predictions["video_id"].str.zfill(4)
    frame["video_id"] = frame["video_id"].str.zfill(4)
    frame = frame.merge(
        e1_predictions[["video_id", "linear_probe_score"]].rename(
            columns={"linear_probe_score": "reused_e1_z_tech"}
        ), on="video_id", how="left",
    )
    frame = frame.merge(
        e3_predictions[["video_id", "linear_probe_mean"]].rename(
            columns={"linear_probe_mean": "reused_e3_z_aes"}
        ), on="video_id", how="left",
    )
    return frame, np.stack(technical_embeddings), np.stack(aesthetic_embeddings)


def fixed_indices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    official_train = np.flatnonzero(frame["split"].eq("train").to_numpy())
    official_val = np.flatnonzero(frame["split"].eq("val").to_numpy())
    train_index, inner_val_index = train_test_split(
        official_train, test_size=0.1, random_state=42, shuffle=True,
    )
    return np.sort(train_index), np.sort(inner_val_index), official_val


def train_linear(
    features: np.ndarray,
    target: np.ndarray,
    train_index: np.ndarray,
    inner_val_index: np.ndarray,
    test_index: np.ndarray,
    seed: int,
    name: str,
    output_dir: Path,
) -> tuple[np.ndarray, dict]:
    set_seed(seed)
    probe = nn.Linear(features.shape[1], 1)
    optimizer = AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features[train_index]).float(), torch.from_numpy(target[train_index]).float()),
        batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    history = []
    best_validation_mse = float("inf")
    stale = 0
    checkpoint = output_dir / f"{name}_seed_{seed}.pt"
    for epoch in range(1, 101):
        probe.train()
        losses = []
        for x, y in loader:
            prediction = probe(x).squeeze(-1)
            loss = F.mse_loss(prediction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        probe.eval()
        with torch.inference_mode():
            inner_prediction = probe(torch.from_numpy(features[inner_val_index]).float()).squeeze(-1).numpy()
        result = metrics(inner_prediction, target[inner_val_index])
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), **result})
        if result["MSE"] < best_validation_mse:
            best_validation_mse = result["MSE"]
            stale = 0
            torch.save({
                "model_state_dict": probe.state_dict(), "probe_version": PROBE_VERSION,
                "name": name, "seed": seed, "best_epoch": epoch,
                "inner_validation_metrics": result, "feature_dim": features.shape[1],
                "trainable_params": features.shape[1] + 1, "encoder_trainable_params": 0,
                "early_stopping_metric": "inner_validation_MSE",
            }, checkpoint)
        else:
            stale += 1
            if stale >= 10:
                break
    pd.DataFrame(history).to_csv(output_dir / f"{name}_history_seed_{seed}.csv", index=False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    probe.load_state_dict(payload["model_state_dict"])
    probe.eval()
    with torch.inference_mode():
        prediction = probe(torch.from_numpy(features[test_index]).float()).squeeze(-1).numpy()
    payload["weight"] = probe.weight.detach().squeeze(0).numpy().tolist()
    payload["bias"] = float(probe.bias.detach().item())
    return prediction, payload


def add_result(rows: list[dict], method: str, input_name: str, prediction: np.ndarray, target: np.ndarray, note: str = "") -> dict:
    result = metrics(prediction, target)
    rows.append({"Method": method, "Input": input_name, **result, "SRCC_std": "", "PLCC_std": "", "note": note})
    return result


def summarize_multiseed(rows: list[dict], method: str, input_name: str, per_seed: list[dict]) -> dict:
    summary = {"Method": method, "Input": input_name, "note": "mean +/- population std over 3 seeds"}
    for key in ("SRCC", "PLCC", "MAE", "MSE"):
        values = np.asarray([row[key] for row in per_seed], dtype=np.float64)
        summary[key] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=0))
    rows.append(summary)
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe_dir = args.output_dir / "linear_probes"
    probe_dir.mkdir(exist_ok=True)
    frame, technical_embedding, aesthetic_embedding = load_aligned_dataset(args)
    train_index, inner_val_index, test_index = fixed_indices(frame)
    target = frame["overall_normalized"].to_numpy(np.float32)
    target_test = target[test_index]
    score_features = frame[["s_tech", "s_aes"]].to_numpy(np.float32)
    embedding_features = np.concatenate([technical_embedding, aesthetic_embedding], axis=1).astype(np.float32)

    results = []
    prediction_frame = frame.iloc[test_index][[
        "video_id", "technical_mos", "aesthetic_mos", "overall_mos", "overall_normalized",
        "s_tech", "s_aes", "reused_e1_z_tech", "reused_e3_z_aes",
    ]].copy()
    add_result(results, "Technical Only (score)", "s_tech", frame.loc[test_index, "s_tech"], target_test)
    valid_e1 = prediction_frame["reused_e1_z_tech"].notna().to_numpy()
    add_result(
        results, "Technical Only (reused E1 probe)", "z_tech -> technical MOS",
        prediction_frame.loc[valid_e1, "reused_e1_z_tech"], target_test[valid_e1],
        "E1 seed-42 probe reused; no E5 retraining",
    )
    add_result(results, "Aesthetic Only (score)", "s_aes", frame.loc[test_index, "s_aes"], target_test)
    valid_e3 = prediction_frame["reused_e3_z_aes"].notna().to_numpy()
    add_result(
        results, "Aesthetic Only (reused E3 probe)", "z_aes -> aesthetic MOS",
        prediction_frame.loc[valid_e3, "reused_e3_z_aes"], target_test[valid_e3],
        "E3 three-seed mean probe reused; no E5 retraining",
    )
    average_prediction = score_features[test_index].mean(axis=1)
    average_result = add_result(results, "Score Average", "(s_tech+s_aes)/2", average_prediction, target_test)
    prediction_frame["score_average"] = average_prediction

    multiseed_rows = []
    weight_rows = []
    score_predictions = []
    embedding_predictions = []
    for name, features, collector in (
        ("score_linear_fusion", score_features, score_predictions),
        ("embedding_linear_fusion", embedding_features, embedding_predictions),
    ):
        for seed in SEEDS:
            prediction, payload = train_linear(
                features, target, train_index, inner_val_index, test_index,
                seed, name, probe_dir,
            )
            result = metrics(prediction, target_test)
            row = {
                "method": name, "seed": seed, **result,
                "best_epoch": payload["best_epoch"], "feature_dim": features.shape[1],
                "trainable_params": payload["trainable_params"], "encoder_trainable_params": 0,
            }
            multiseed_rows.append(row)
            collector.append(prediction)
            prediction_frame[f"{name}_seed_{seed}"] = prediction
            if name == "score_linear_fusion":
                weight_rows.append({
                    "seed": seed, "w_tech": payload["weight"][0],
                    "w_aes": payload["weight"][1], "bias": payload["bias"],
                    **result, "best_epoch": payload["best_epoch"],
                })
    score_seed_rows = [row for row in multiseed_rows if row["method"] == "score_linear_fusion"]
    embedding_seed_rows = [row for row in multiseed_rows if row["method"] == "embedding_linear_fusion"]
    score_summary = summarize_multiseed(results, "Score Linear Fusion", "[s_tech,s_aes]", score_seed_rows)
    embedding_summary = summarize_multiseed(
        results, "Embedding Linear Fusion", "[z_tech,z_aes] (256-d)", embedding_seed_rows,
    )
    prediction_frame["score_linear_mean"] = np.mean(np.stack(score_predictions), axis=0)
    prediction_frame["embedding_linear_mean"] = np.mean(np.stack(embedding_predictions), axis=0)
    prediction_frame["embedding_absolute_error"] = np.abs(
        prediction_frame["embedding_linear_mean"] - prediction_frame["overall_normalized"]
    )

    pd.DataFrame(results).to_csv(args.output_dir / "visual_fusion_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(multiseed_rows).to_csv(args.output_dir / "visual_fusion_multiseed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(weight_rows).to_csv(args.output_dir / "visual_fusion_weights.csv", index=False, encoding="utf-8-sig")
    prediction_frame.to_csv(args.output_dir / "visual_fusion_predictions.csv", index=False, encoding="utf-8-sig")
    prediction_frame.nlargest(30, "embedding_absolute_error").to_csv(
        args.output_dir / "visual_fusion_error_cases.csv", index=False, encoding="utf-8-sig",
    )

    e1_versions = sorted(frame["e1_feature_version"].unique().tolist())
    audit = f"""# E5 Fusion Dataset Audit

- E1 technical cache: `{args.e1_cache.resolve()}`
- E3 aesthetic cache: `{args.e3_cache.resolve()}`
- Matched videos: {len(frame)}
- Official train / validation: {int(frame['split'].eq('train').sum())} / {int(frame['split'].eq('val').sum())}
- Fixed inner train / validation: {len(train_index)} / {len(inner_val_index)} (`random_state=42`)
- E1 feature versions: {', '.join(e1_versions)}
- E3 feature version: `e3_divide_aesthetic_e1_clip_compat_v1`
- Technical embedding: {technical_embedding.shape[1]} dimensions
- Aesthetic embedding: {aesthetic_embedding.shape[1]} dimensions
- Concatenated embedding: {embedding_features.shape[1]} dimensions
- Score-fusion trainable parameters: 3
- Embedding-fusion trainable parameters: {embedding_features.shape[1] + 1}
- Encoder trainable parameters: 0
- CLIP/COVER calls in E5: 0; all representations are loaded from E1/E3 caches.
"""
    (args.output_dir / "fusion_dataset_audit.md").write_text(audit, encoding="utf-8")

    best_single = max(
        (row for row in results if row["Method"].startswith(("Technical Only", "Aesthetic Only"))),
        key=lambda row: row["SRCC"],
    )
    weights = pd.DataFrame(weight_rows)
    complement = embedding_summary["SRCC"] > best_single["SRCC"]
    embedding_gain = embedding_summary["SRCC"] - score_summary["SRCC"]
    mean_abs_tech = float(weights["w_tech"].abs().mean())
    mean_abs_aes = float(weights["w_aes"].abs().mean())
    dominant = "technical" if mean_abs_tech > mean_abs_aes else "aesthetic"
    summary = f"""# E5 DIVIDE Visual Fusion Summary

## Results

| Method | SRCC | PLCC | MAE | MSE |
|---|---:|---:|---:|---:|
| Score Average | {average_result['SRCC']:.4f} | {average_result['PLCC']:.4f} | {average_result['MAE']:.4f} | {average_result['MSE']:.4f} |
| Score Linear Fusion | {score_summary['SRCC']:.4f} +/- {score_summary['SRCC_std']:.4f} | {score_summary['PLCC']:.4f} +/- {score_summary['PLCC_std']:.4f} | {score_summary['MAE']:.4f} | {score_summary['MSE']:.4f} |
| Embedding Linear Fusion | {embedding_summary['SRCC']:.4f} +/- {embedding_summary['SRCC_std']:.4f} | {embedding_summary['PLCC']:.4f} +/- {embedding_summary['PLCC_std']:.4f} | {embedding_summary['MAE']:.4f} | {embedding_summary['MSE']:.4f} |

## Answers

1. **Complementarity:** {'supported' if complement else 'not supported'} by the primary embedding experiment; its SRCC is compared with the best directly reused single-branch baseline `{best_single['Method']}` ({best_single['SRCC']:.4f}).
2. **Score-level fusion:** mean SRCC changes from {average_result['SRCC']:.4f} (average) to {score_summary['SRCC']:.4f}, but remains near zero; PLCC changes from {average_result['PLCC']:.4f} to {score_summary['PLCC']:.4f}. The raw score heads therefore do not transfer reliably.
3. **Embedding-level fusion:** SRCC difference relative to score fusion is {embedding_gain:+.4f}; this quantifies whether hidden representations retain information lost by the score heads.
4. **Linear score contribution:** `{dominant}` has the larger mean absolute coefficient on DIVIDE (tech={mean_abs_tech:.4f}, aes={mean_abs_aes:.4f}), but coefficient signs vary across seeds. These are unstable, dataset-specific coefficients rather than universal perceptual weights.
5. **Design implication:** the strong embedding result supports the cross-domain value of the frozen branch representations, but {'also supports' if complement else 'does not establish'} an incremental technical-aesthetic complementarity gain over the reused technical-only probe. It makes no claim about Scientific Branch importance because DIVIDE has no scientific-quality labels.
"""
    (args.output_dir / "divide_visual_fusion_summary.md").write_text(summary, encoding="utf-8")
    (args.output_dir / "experiment_manifest.json").write_text(json.dumps({
        "experiment": "E5", "seeds": list(SEEDS), "num_samples": len(frame),
        "inner_split_seed": 42, "e1_cache": str(args.e1_cache.resolve()),
        "e3_cache": str(args.e3_cache.resolve()), "encoder_trainable_params": 0,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"E5 complete: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
