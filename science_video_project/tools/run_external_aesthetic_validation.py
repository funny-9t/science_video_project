"""Run E3: KEEP_P2_S2 aesthetic validation on DIVIDE-MaxWell.

The experiment reuses E1's cached 512-d CLIP vectors.  A1 text embeddings are
projected with the same deterministic 768->512 adaptive pooling operation so
that prompt scores can be computed without decoding or encoding videos again.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import CFG
from pipeline.step_aesthetic_clip import AESTHETIC_PROMPT_SETS
from pipeline.step_video import VideoEncoder
from training.model_mvp import MultiModalQualityModel
from training.utils_train import set_seed


SEEDS = (42, 123, 2026)
EXPECTED_SPLIT_HASH = "79413d21918a956716f230456b80aec037be51bfd1c6a412a8c0a30087a2a58f"
FEATURE_VERSION = "e3_divide_aesthetic_e1_clip_compat_v1"
PROBE_VERSION = "e3_aesthetic_linear_probe_inner_val_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "outputs" / "progressive_training" / "P2" / "stage2_best.pt",
    )
    parser.add_argument(
        "--e1-cache", type=Path,
        default=PROJECT_ROOT / "outputs" / "external_technical_validation" / "cache"
        / "external_validation" / "divide_maxwell",
    )
    parser.add_argument(
        "--train-labels", type=Path,
        default=Path(r"D:\Projects\COVER\examplar_data_labels\DIVIDE_MaxWell\train_labels.txt"),
    )
    parser.add_argument(
        "--val-labels", type=Path,
        default=Path(r"D:\Projects\COVER\examplar_data_labels\DIVIDE_MaxWell\val_labels.txt"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "external_aesthetic_validation",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict] | pd.DataFrame) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_labels(path: Path, split: str) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            filename, aesthetic, technical, overall = [part.strip() for part in line.split(",")]
            rows.append({
                "video_id": Path(filename).stem,
                "filename": filename,
                "split": split,
                "aesthetic_mos": float(aesthetic),
                "technical_mos": float(technical),
                "overall_mos": float(overall),
            })
    return pd.DataFrame(rows)


def correlation(prediction: np.ndarray, target: np.ndarray, kind: str) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.size < 2 or np.unique(prediction).size < 2 or np.unique(target).size < 2:
        return float("nan")
    result = spearmanr(prediction, target).statistic if kind == "spearman" else pearsonr(prediction, target).statistic
    return float(result)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = prediction - target
    return {
        "SRCC": correlation(prediction, target, "spearman"),
        "PLCC": correlation(prediction, target, "pearson"),
        "MSE": float(np.mean(error ** 2)),
        "MAE": float(np.mean(np.abs(error))),
    }


def load_model(checkpoint: Path, device: torch.device) -> tuple[MultiModalQualityModel, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("split_hash") != EXPECTED_SPLIT_HASH:
        raise ValueError(f"Unexpected checkpoint split hash: {payload.get('split_hash')}")
    model = MultiModalQualityModel(**payload["config"]).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, payload


def load_records(metadata: pd.DataFrame, cache_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    clip_features = []
    for row in metadata.itertuples(index=False):
        path = cache_dir / f"{row.video_id}.pt"
        if not path.is_file():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        feature = np.asarray(payload.get("clip_feature", []), dtype=np.float32)
        if feature.shape != (512,) or not np.isfinite(feature).all():
            continue
        rows.append(row._asdict())
        clip_features.append(feature)
    if not rows:
        raise RuntimeError(f"No valid E1 cache entries found in {cache_dir}")
    return pd.DataFrame(rows), np.stack(clip_features)


def projected_prompt_features(device: torch.device) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    prompts = AESTHETIC_PROMPT_SETS["a1_original"]
    tokenizer = CLIPTokenizerFast.from_pretrained(CFG.clip_model_name, local_files_only=True)
    text_model = CLIPTextModelWithProjection.from_pretrained(
        CFG.clip_model_name, local_files_only=True,
    ).to(device)
    text_model.eval()
    texts = [item["positive"] for item in prompts] + [item["negative"] for item in prompts]
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
    tokens = {key: value.to(device) for key, value in tokens.items()}
    with torch.inference_mode():
        native = F.normalize(text_model(**tokens).text_embeds.float(), dim=-1)
        projected = torch.stack([VideoEncoder.legacy_project(vector, 512) for vector in native])
        projected = F.normalize(projected, dim=-1)
    split = len(prompts)
    names = [item["name"] for item in prompts]
    del text_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return names, projected[:split].cpu(), projected[split:].cpu()


def build_aesthetic_cache(
    frame: pd.DataFrame,
    clip_features: np.ndarray,
    model: MultiModalQualityModel,
    device: torch.device,
    output_path: Path,
) -> dict:
    names, positive, negative = projected_prompt_features(device)
    clip = F.normalize(torch.from_numpy(clip_features).float(), dim=-1)
    prompt_scores = clip @ positive.T - clip @ negative.T
    embeddings = []
    scores = []
    with torch.inference_mode():
        for start in range(0, len(frame), 512):
            stop = min(start + 512, len(frame))
            video = torch.from_numpy(clip_features[start:stop]).float().to(device)
            aesthetic = prompt_scores[start:stop].float().to(device)
            batch = stop - start
            embedding, logit = model.aesthetic_branch(
                video,
                torch.zeros(batch, 768, device=device),
                torch.zeros(batch, 384, device=device),
                aesthetic,
            )
            embeddings.append(embedding.cpu())
            scores.append(torch.sigmoid(logit).squeeze(-1).cpu())
    payload = {
        "feature_version": FEATURE_VERSION,
        "video_ids": frame["video_id"].tolist(),
        "splits": frame["split"].tolist(),
        "aesthetic_mos": torch.tensor(frame["aesthetic_mos"].to_numpy(), dtype=torch.float32),
        "technical_mos": torch.tensor(frame["technical_mos"].to_numpy(), dtype=torch.float32),
        "overall_mos": torch.tensor(frame["overall_mos"].to_numpy(), dtype=torch.float32),
        "clip_features": torch.from_numpy(clip_features),
        "prompt_names": names,
        "prompt_scores": prompt_scores.float(),
        "aesthetic_embeddings": torch.cat(embeddings),
        "aesthetic_scores": torch.cat(scores),
        "metadata": {
            "prompt_version": "a1_original",
            "prompt_space": "ViT-L/14 text and E1 image vectors projected 768->512 by adaptive_avg_pool1d",
            "missing_context_policy": "zero_text_and_audio",
            "source_cache": str(output_path.parent.parent.parent / "external_technical_validation"),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return payload


def fixed_inner_split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    official_train = np.flatnonzero(frame["split"].eq("train").to_numpy())
    official_val = np.flatnonzero(frame["split"].eq("val").to_numpy())
    inner_train, inner_val = train_test_split(official_train, test_size=0.1, random_state=42, shuffle=True)
    return np.sort(inner_train), np.sort(inner_val), official_val


def train_probe(
    x: np.ndarray,
    y: np.ndarray,
    train_index: np.ndarray,
    inner_val_index: np.ndarray,
    test_index: np.ndarray,
    seed: int,
    checkpoint: Path,
    history_path: Path,
) -> tuple[np.ndarray, dict]:
    set_seed(seed)
    probe = nn.Linear(x.shape[1], 1)
    optimizer = AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x[train_index]).float(), torch.from_numpy(y[train_index]).float()),
        batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    best_srcc = -float("inf")
    stale = 0
    history = []
    for epoch in range(1, 101):
        probe.train()
        losses = []
        for features, target in loader:
            prediction = probe(features).squeeze(-1)
            loss = F.mse_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        probe.eval()
        with torch.inference_mode():
            inner_prediction = probe(torch.from_numpy(x[inner_val_index]).float()).squeeze(-1).numpy()
        result = metrics(inner_prediction, y[inner_val_index])
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), **result})
        if result["SRCC"] > best_srcc:
            best_srcc = result["SRCC"]
            stale = 0
            torch.save({
                "model_state_dict": probe.state_dict(), "probe_version": PROBE_VERSION,
                "seed": seed, "best_epoch": epoch, "inner_validation_metrics": result,
                "feature_dim": x.shape[1], "trainable_params": x.shape[1] + 1,
                "inner_split_seed": 42,
            }, checkpoint)
        else:
            stale += 1
            if stale >= 10:
                break
    write_csv(history_path, history)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    probe.load_state_dict(payload["model_state_dict"])
    probe.eval()
    with torch.inference_mode():
        prediction = probe(torch.from_numpy(x[test_index]).float()).squeeze(-1).numpy()
    return prediction, payload


def create_reports(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    cache: dict,
    model: MultiModalQualityModel,
) -> None:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    prompt_names = cache["prompt_names"]
    prompt_scores = cache["prompt_scores"].numpy()
    embeddings = cache["aesthetic_embeddings"].numpy()
    aesthetic_scores = cache["aesthetic_scores"].numpy()
    y = frame["aesthetic_mos"].to_numpy(np.float32) / 5.0
    train_index, inner_val_index, test_index = fixed_inner_split(frame)
    frozen_params = sum(parameter.numel() for parameter in model.aesthetic_branch.parameters())

    prompt_frame = frame[["video_id", "split", "aesthetic_mos"]].copy()
    for index, name in enumerate(prompt_names):
        prompt_frame[name] = prompt_scores[:, index]
    write_csv(output / "aesthetic_prompt_scores.csv", prompt_frame)

    diagnostics = []
    for index, name in enumerate(prompt_names):
        result = metrics(prompt_scores[test_index, index], y[test_index])
        diagnostics.append({"prompt": name, **result})
    write_csv(output / "aesthetic_prompt_diagnostics.csv", diagnostics)

    results = []
    zero_metrics = metrics(aesthetic_scores[test_index], y[test_index])
    results.append({
        "protocol": "Zero-shot s_aes", "seed": "", "num_train": 0,
        "num_inner_validation": 0, "num_test": len(test_index), **zero_metrics,
        "feature_dim": 128, "trainable_params": 0, "frozen_branch_params": frozen_params,
    })
    predictions = frame.iloc[test_index][["video_id", "split", "aesthetic_mos", "overall_mos"]].copy()
    predictions["target_normalized"] = y[test_index]
    predictions["zero_shot_score"] = aesthetic_scores[test_index]
    probe_predictions = []
    probe_metrics = []
    probe_dir = output / "linear_probes"
    probe_dir.mkdir(exist_ok=True)
    for seed in SEEDS:
        prediction, payload = train_probe(
            embeddings, y, train_index, inner_val_index, test_index, seed,
            probe_dir / f"linear_probe_seed_{seed}.pt",
            probe_dir / f"history_seed_{seed}.csv",
        )
        result = metrics(prediction, y[test_index])
        probe_metrics.append(result)
        probe_predictions.append(prediction)
        predictions[f"linear_probe_seed_{seed}"] = prediction
        results.append({
            "protocol": "Frozen Linear Probe", "seed": seed,
            "num_train": len(train_index), "num_inner_validation": len(inner_val_index),
            "num_test": len(test_index), **result, "feature_dim": embeddings.shape[1],
            "trainable_params": payload["trainable_params"],
            "frozen_branch_params": frozen_params, "best_epoch": payload["best_epoch"],
        })
    mean_prediction = np.mean(np.stack(probe_predictions), axis=0)
    predictions["linear_probe_mean"] = mean_prediction
    predictions["absolute_error"] = np.abs(mean_prediction - y[test_index])
    write_csv(output / "external_aesthetic_predictions.csv", predictions)

    summary_row = {
        "protocol": "Frozen Linear Probe mean+-std", "seed": "3 seeds",
        "num_train": len(train_index), "num_inner_validation": len(inner_val_index),
        "num_test": len(test_index), "feature_dim": embeddings.shape[1],
        "trainable_params": embeddings.shape[1] + 1, "frozen_branch_params": frozen_params,
    }
    for key in ("SRCC", "PLCC", "MSE", "MAE"):
        values = np.array([item[key] for item in probe_metrics])
        summary_row[key] = float(values.mean())
        summary_row[f"{key}_std"] = float(values.std(ddof=0))
    results.append(summary_row)
    write_csv(output / "external_aesthetic_results.csv", results)

    errors = predictions.nlargest(20, "absolute_error")
    error_lines = [
        "# E3 DIVIDE-MaxWell Error Analysis", "",
        "Top-20 absolute errors use the mean prediction from three frozen linear probes.", "",
        "| video_id | MOS | prediction | absolute error |", "|---|---:|---:|---:|",
    ]
    for row in errors.itertuples(index=False):
        error_lines.append(
            f"| {row.video_id} | {row.target_normalized:.4f} | {row.linear_probe_mean:.4f} | {row.absolute_error:.4f} |"
        )
    (output / "external_aesthetic_error_analysis.md").write_text("\n".join(error_lines) + "\n", encoding="utf-8")

    audit = f"""# KEEP_P2_S2 Aesthetic Branch Audit

- Checkpoint: `{args.checkpoint.resolve()}`
- Branch inputs: 512-d legacy CLIP video feature, 768-d text feature, 384-d audio feature, 7-d A1 prompt feature.
- External policy: text/audio are unavailable and are set to zero; visual CLIP and A1 prompts remain active.
- Branch output: 128-d `aesthetic_embedding` and sigmoid score `s_aes`.
- Frozen branch parameters: {frozen_params:,}.
- Prompt set: `a1_original` (7 dimensions); A2 is not used.
- E1 cache compatibility: cached images are 512-d vectors produced by adaptive pooling native ViT-L/14 features. The same 768->512 deterministic projection is applied to A1 text embeddings before cosine-difference scoring.
- Limitation: projected-cache prompt scores are a deterministic compatibility approximation, not the native 768-d per-frame prompt aggregation used by the internal pipeline.
"""
    (output / "aesthetic_branch_audit.md").write_text(audit, encoding="utf-8")
    dataset_audit = f"""# DIVIDE-MaxWell Audit for E3

- Label source: `{args.train_labels}` and `{args.val_labels}`
- E1 feature cache: `{args.e1_cache}`
- Parsed rows: {len(frame)}
- Official train: {int(frame['split'].eq('train').sum())}
- Official validation: {int(frame['split'].eq('val').sum())}
- Fixed inner split: {len(train_index)} train / {len(inner_val_index)} validation (`random_state=42`).
- Missing or invalid E1 cache rows: 0 after alignment.
- Targets: Aesthetic MOS for E3; Overall MOS is retained only for later E5 reuse.
"""
    (output / "divide_maxwell_audit.md").write_text(dataset_audit, encoding="utf-8")

    best_prompt = max(diagnostics, key=lambda item: item["SRCC"])
    summary = f"""# E3 External Aesthetic Validation Summary

## Protocol

KEEP_P2_S2 is frozen. E3-A evaluates `s_aes` zero-shot; E3-B trains only a `Linear(128, 1)` probe on the fixed inner split for seeds 42/123/2026. Official DIVIDE validation is touched only for final reporting.

## Results

| Method | SRCC | PLCC | MSE | MAE |
|---|---:|---:|---:|---:|
| Zero-shot `s_aes` | {zero_metrics['SRCC']:.4f} | {zero_metrics['PLCC']:.4f} | {zero_metrics['MSE']:.4f} | {zero_metrics['MAE']:.4f} |
| Frozen probe (mean) | {summary_row['SRCC']:.4f} +/- {summary_row['SRCC_std']:.4f} | {summary_row['PLCC']:.4f} +/- {summary_row['PLCC_std']:.4f} | {summary_row['MSE']:.4f} +/- {summary_row['MSE_std']:.4f} | {summary_row['MAE']:.4f} +/- {summary_row['MAE_std']:.4f} |

Best individual A1 prompt on official validation is `{best_prompt['prompt']}` (SRCC={best_prompt['SRCC']:.4f}, PLCC={best_prompt['PLCC']:.4f}). The branch transfer result must be read with the documented zero-text/audio and projected-prompt domain-shift limitations.
"""
    (output / "external_aesthetic_summary.md").write_text(summary, encoding="utf-8")
    write_json(output / "experiment_manifest.json", {
        "experiment": "E3", "feature_version": FEATURE_VERSION, "seeds": list(SEEDS),
        "checkpoint": str(args.checkpoint.resolve()), "e1_cache": str(args.e1_cache.resolve()),
        "official_validation_size": len(test_index), "frozen_branch_params": frozen_params,
    })


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.concat([
        parse_labels(args.train_labels, "train"), parse_labels(args.val_labels, "val")
    ], ignore_index=True)
    frame, clip_features = load_records(metadata, args.e1_cache)
    if len(frame) != len(metadata):
        missing = sorted(set(metadata["video_id"]) - set(frame["video_id"]))
        raise RuntimeError(f"E1 cache incomplete: {len(missing)} missing rows")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.checkpoint, device)
    cache_path = args.output_dir / "cache" / "divide_maxwell_aesthetic_features.pt"
    if cache_path.exists() and not args.force:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache.get("feature_version") != FEATURE_VERSION or cache.get("video_ids") != frame["video_id"].tolist():
            cache = build_aesthetic_cache(frame, clip_features, model, device, cache_path)
    else:
        cache = build_aesthetic_cache(frame, clip_features, model, device, cache_path)
    create_reports(args, frame, cache, model)
    print(f"E3 complete: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
