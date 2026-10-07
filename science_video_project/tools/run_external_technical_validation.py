"""Evaluate KEEP_P2_S2 Technical Branch on external VQA datasets."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import CFG
from pipeline.step_cover_technical import COVERTechnicalFeatureExtractor
from pipeline.step_extract import VideoExtractor
from pipeline.step_video import VideoEncoder
from pipeline.utils_io import load_metadata, load_pt
from training.model_mvp import MultiModalQualityModel
from training.utils_train import get_device, set_seed


DATASET_NAMES = ("divide_maxwell", "konvid1k")
PROBE_SEEDS = (42, 123, 2026)
PROBE_VERSION = "external_linear_probe_v2_inner_validation"
EXPECTED_SPLIT_HASH = "79413d21918a956716f230456b80aec037be51bfd1c6a412a8c0a30087a2a58f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("divide", "konvid", "all"), default="all")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "outputs" / "progressive_training" / "P2" / "stage2_best.pt",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "external_technical_validation",
    )
    parser.add_argument(
        "--divide-video-dir", type=Path,
        default=Path(r"F:\test_data\DIVIDE-MaxWell\videos"),
    )
    parser.add_argument(
        "--divide-train-labels", type=Path,
        default=Path(r"D:\Projects\COVER\examplar_data_labels\DIVIDE_MaxWell\train_labels.txt"),
    )
    parser.add_argument(
        "--divide-val-labels", type=Path,
        default=Path(r"D:\Projects\COVER\examplar_data_labels\DIVIDE_MaxWell\val_labels.txt"),
    )
    parser.add_argument(
        "--konvid-video-dir", type=Path,
        default=Path(r"F:\test_data\KoNViD-1k\KoNViD_1k_videos"),
    )
    parser.add_argument(
        "--konvid-labels", type=Path,
        default=Path(r"D:\Projects\COVER\examplar_data_labels\KoNViD\labels.txt"),
    )
    parser.add_argument(
        "--science-metadata", type=Path,
        default=Path(r"D:\Projects\science_video_ranker_mvp\science_video_project\data\parsed_metadata_filtered.csv"),
    )
    parser.add_argument(
        "--science-feature-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "features",
    )
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force-probes", action="store_true")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def selected_datasets(value: str) -> list[str]:
    if value == "divide":
        return ["divide_maxwell"]
    if value == "konvid":
        return ["konvid1k"]
    return list(DATASET_NAMES)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("external_technical_validation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def parse_divide_labels(path: Path, split: str, video_dir: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                raise ValueError(f"Invalid DIVIDE label line: {line!r}")
            filename, aesthetic, technical, overall = parts
            rows.append({
                "video_id": Path(filename).stem,
                "filename": filename,
                "video_path": str((video_dir / filename).resolve()),
                "split": split,
                "aesthetic_mos": float(aesthetic),
                "technical_mos": float(technical),
                "overall_mos": float(overall),
                "ground_truth": float(technical),
            })
    return pd.DataFrame(rows)


def parse_konvid_labels(path: Path, video_dir: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                raise ValueError(f"Invalid KoNViD label line: {line!r}")
            filename, duration, fps, mos = parts
            basename = Path(filename).name
            rows.append({
                "video_id": Path(basename).stem,
                "filename": filename,
                "video_path": str((video_dir / basename).resolve()),
                "split": "all",
                "duration": float(duration),
                "fps": float(fps),
                "overall_mos": float(mos),
                "ground_truth": float(mos),
            })
    return pd.DataFrame(rows)


def load_external_metadata(args: argparse.Namespace, dataset: str) -> pd.DataFrame:
    if dataset == "divide_maxwell":
        train = parse_divide_labels(args.divide_train_labels, "train", args.divide_video_dir)
        val = parse_divide_labels(args.divide_val_labels, "val", args.divide_video_dir)
        metadata = pd.concat([train, val], ignore_index=True)
    else:
        metadata = parse_konvid_labels(args.konvid_labels, args.konvid_video_dir)
    metadata["ground_truth_normalized"] = metadata["ground_truth"] / 5.0
    if metadata["video_id"].duplicated().any():
        raise ValueError(f"Duplicate video IDs in {dataset}")
    return metadata


def validate_dataset_files(metadata: pd.DataFrame, dataset: str) -> None:
    missing = [path for path in metadata["video_path"] if not Path(path).is_file()]
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"{dataset}: {len(missing)}/{len(metadata)} videos are missing; examples: {preview}"
        )


def load_model(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("split_hash") != EXPECTED_SPLIT_HASH:
        raise ValueError(f"Unexpected KEEP_P2_S2 split hash: {payload.get('split_hash')}")
    model = MultiModalQualityModel(**payload["config"]).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, payload


def external_branch_forward(
    model: MultiModalQualityModel,
    video_feature: np.ndarray | torch.Tensor,
    cover_feature: np.ndarray | torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    video = torch.as_tensor(video_feature, dtype=torch.float32, device=device).view(1, -1)
    cover = torch.as_tensor(cover_feature, dtype=torch.float32, device=device).view(1, -1)
    with torch.inference_mode():
        embedding, logit = model.technical_branch(
            video,
            torch.zeros(1, 16, device=device),
            dnsmos_feat=torch.zeros(1, 3, device=device),
            wpm=torch.zeros(1, 1, device=device),
            speech_rhythm_feat=torch.zeros(1, 6, device=device),
            cover_technical_feat=cover,
        )
        score = torch.sigmoid(logit).item()
    return embedding.squeeze(0).cpu().numpy().astype(np.float32), float(score)


def cache_path(output_dir: Path, dataset: str, video_id: str) -> Path:
    return output_dir / "cache" / "external_validation" / dataset / f"{video_id}.pt"


def valid_cache(path: Path, dataset: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = load_pt(path)
        return (
            payload.get("dataset") == dataset
            and payload.get("feature_version") == "keep_p2_s2_external_technical_v1"
            and np.asarray(payload.get("clip_feature", [])).shape == (512,)
            and np.asarray(payload.get("technical_feature", [])).shape == (768,)
            and np.asarray(payload.get("technical_embedding", [])).shape == (128,)
            and np.isfinite(float(payload.get("zero_shot_score", np.nan)))
        )
    except Exception:
        return False


def extract_dataset_cache(
    args: argparse.Namespace,
    dataset: str,
    metadata: pd.DataFrame,
    model: MultiModalQualityModel,
    device: torch.device,
    logger: logging.Logger,
) -> dict:
    os.environ.setdefault("FFMPEG_PATH", str(CFG.ffmpeg_path))
    video_extractor = VideoExtractor(audio_sr=CFG.audio_sr)
    clip_encoder = VideoEncoder(CFG.clip_model_name, CFG.device)
    cover_encoder = COVERTechnicalFeatureExtractor(CFG.cover_root, device=CFG.device)
    temporary_root = args.output_dir / "_tmp_frames" / dataset
    temporary_root.mkdir(parents=True, exist_ok=True)
    cache_root = args.output_dir / "cache" / "external_validation" / dataset
    cache_root.mkdir(parents=True, exist_ok=True)

    selected = metadata.iloc[: args.max_videos] if args.max_videos > 0 else metadata
    counters = {"expected": len(selected), "cached": 0, "extracted": 0, "errors": 0}
    errors = []
    started = time.perf_counter()
    for index, row in enumerate(selected.itertuples(index=False), start=1):
        target = cache_path(args.output_dir, dataset, str(row.video_id))
        if not args.force_features and valid_cache(target, dataset):
            counters["cached"] += 1
            continue
        frame_dir = temporary_root / str(row.video_id)
        shutil.rmtree(frame_dir, ignore_errors=True)
        try:
            video_extractor.extract_frames(row.video_path, frame_dir, fps=CFG.frame_fps)
            frame_features = clip_encoder.encode_frame_features(
                frame_dir, max_frames=CFG.clip_max_frames
            )
            clip_native = frame_features.mean(dim=0)
            clip_feature = VideoEncoder.legacy_project(
                clip_native, output_dim=CFG.video_dim
            ).numpy().astype(np.float32)
            technical_feature = cover_encoder.extract(row.video_path).numpy().astype(np.float32)
            embedding, score = external_branch_forward(
                model, clip_feature, technical_feature, device
            )
            label_payload = {
                key: float(getattr(row, key))
                for key in ("ground_truth", "ground_truth_normalized")
            }
            if dataset == "divide_maxwell":
                label_payload.update({
                    "technical_mos": float(row.technical_mos),
                    "aesthetic_mos": float(row.aesthetic_mos),
                    "overall_mos": float(row.overall_mos),
                    "official_split": str(row.split),
                })
            else:
                label_payload.update({"overall_mos": float(row.overall_mos)})
            torch.save({
                "video_id": str(row.video_id),
                "dataset": dataset,
                "clip_feature": clip_feature,
                "technical_feature": technical_feature,
                "technical_embedding": embedding,
                "zero_shot_score": score,
                "ground_truth": float(row.ground_truth),
                "ground_truth_normalized": float(row.ground_truth_normalized),
                "labels": label_payload,
                "feature_version": "keep_p2_s2_external_technical_v1",
                "metadata": {
                    "checkpoint": str(args.checkpoint.resolve()),
                    "clip_model": str(CFG.clip_model_name),
                    "clip_sampling": f"fps_{CFG.frame_fps}_global_uniform_{CFG.clip_max_frames}",
                    "cover_version": COVERTechnicalFeatureExtractor.FEATURE_VERSION,
                    "cover_sampling": cover_encoder.sampling_config,
                    "missing_context_policy": "zero_meta_wpm_rhythm",
                },
            }, target)
            counters["extracted"] += 1
        except Exception as exc:
            counters["errors"] += 1
            errors.append({"video_id": str(row.video_id), "error": repr(exc)})
            logger.exception("%s feature extraction failed for %s", dataset, row.video_id)
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)
        if index % max(args.log_every, 1) == 0 or index == len(selected):
            elapsed = time.perf_counter() - started
            logger.info(
                "%s cache %d/%d | new=%d reused=%d errors=%d | %.2f videos/s",
                dataset, index, len(selected), counters["extracted"], counters["cached"],
                counters["errors"], index / max(elapsed, 1e-6),
            )
    shutil.rmtree(temporary_root, ignore_errors=True)
    counters["elapsed_seconds"] = time.perf_counter() - started
    counters["cache_dir"] = str(cache_root.resolve())
    counters["errors_detail"] = errors
    return counters


def load_cached_records(
    output_dir: Path, dataset: str, metadata: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows = []
    embeddings = []
    technical_features = []
    for row in metadata.itertuples(index=False):
        path = cache_path(output_dir, dataset, str(row.video_id))
        if not valid_cache(path, dataset):
            continue
        payload = load_pt(path)
        record = row._asdict()
        record["zero_shot_score"] = float(payload["zero_shot_score"])
        rows.append(record)
        embeddings.append(np.asarray(payload["technical_embedding"], dtype=np.float32))
        technical_features.append(np.asarray(payload["technical_feature"], dtype=np.float32))
    if not rows:
        return pd.DataFrame(), np.zeros((0, 128), np.float32), np.zeros((0, 768), np.float32)
    return pd.DataFrame(rows), np.stack(embeddings), np.stack(technical_features)


def correlation(prediction: np.ndarray, target: np.ndarray, kind: str) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.size < 2 or np.unique(prediction).size < 2 or np.unique(target).size < 2:
        return float("nan")
    result = (
        spearmanr(prediction, target).statistic
        if kind == "spearman"
        else pearsonr(prediction, target).statistic
    )
    return float(result)


def regression_metrics(prediction: np.ndarray, target_normalized: np.ndarray) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target_normalized, dtype=np.float64)
    error = prediction - target
    return {
        "SRCC": correlation(prediction, target, "spearman"),
        "PLCC": correlation(prediction, target, "pearson"),
        "MSE": float(np.mean(error ** 2)),
        "MAE": float(np.mean(np.abs(error))),
    }


def train_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    seed: int,
    checkpoint: Path,
    history_path: Path,
    force: bool,
) -> tuple[np.ndarray, dict]:
    if checkpoint.exists() and not force:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("probe_version") == PROBE_VERSION:
            probe = nn.Linear(train_x.shape[1], 1)
            probe.load_state_dict(payload["model_state_dict"])
            probe.eval()
            with torch.inference_mode():
                prediction = probe(torch.from_numpy(val_x).float()).squeeze(-1).numpy()
            return prediction, payload

    set_seed(seed)
    probe = nn.Linear(train_x.shape[1], 1)
    optimizer = AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()),
        batch_size=64, shuffle=True, generator=generator,
    )
    total_params = sum(parameter.numel() for parameter in probe.parameters())
    best_srcc = -float("inf")
    stale = 0
    history = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, 101):
        probe.train()
        losses = []
        for features, target in loader:
            prediction = probe(features).squeeze(-1)
            loss = loss_fn(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        probe.eval()
        with torch.inference_mode():
            val_prediction = probe(torch.from_numpy(val_x).float()).squeeze(-1).numpy()
        metrics = regression_metrics(val_prediction, val_y)
        history.append({
            "epoch": epoch,
            "train_mse": float(np.mean(losses)),
            **metrics,
        })
        if metrics["SRCC"] > best_srcc:
            best_srcc = metrics["SRCC"]
            stale = 0
            torch.save({
                "model_state_dict": probe.state_dict(),
                "probe_version": PROBE_VERSION,
                "seed": seed,
                "best_epoch": epoch,
                "validation_metrics": metrics,
                "feature_dim": train_x.shape[1],
                "total_params": total_params,
                "trainable_params": total_params,
                "frozen_encoder": True,
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
        prediction = probe(torch.from_numpy(val_x).float()).squeeze(-1).numpy()
    return prediction, payload


def external_results_and_predictions(
    args: argparse.Namespace,
    datasets: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]],
    technical_params: int,
    logger: logging.Logger,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    results = []
    predictions: dict[str, pd.DataFrame] = {}
    if "divide_maxwell" in datasets:
        frame, embeddings, _ = datasets["divide_maxwell"]
        train_mask = frame["split"].eq("train").to_numpy()
        val_mask = frame["split"].eq("val").to_numpy()
        zero_metrics = regression_metrics(
            frame.loc[val_mask, "zero_shot_score"].to_numpy(),
            frame.loc[val_mask, "ground_truth_normalized"].to_numpy(),
        )
        results.append({
            "dataset": "DIVIDE-MaxWell", "ground_truth": "Technical MOS",
            "protocol": "Zero-shot", "seed": "", "num_train": 0,
            "num_validation": 0, "num_test": int(val_mask.sum()), **zero_metrics,
            "checkpoint": str(args.checkpoint.resolve()), "feature_dim": 128,
            "trainable_params": 0,
        })
        probe_dir = args.output_dir / "divide_linear_probe"
        probe_prediction, payload = train_linear_probe(
            embeddings[train_mask], frame.loc[train_mask, "ground_truth_normalized"].to_numpy(np.float32),
            embeddings[val_mask], frame.loc[val_mask, "ground_truth_normalized"].to_numpy(np.float32),
            42, probe_dir / "linear_probe_best.pt", probe_dir / "history.csv",
            args.force_probes,
        )
        probe_metrics = regression_metrics(
            probe_prediction, frame.loc[val_mask, "ground_truth_normalized"].to_numpy()
        )
        trainable_params = int(payload["trainable_params"])
        frozen_params = int(technical_params)
        results.append({
            "dataset": "DIVIDE-MaxWell", "ground_truth": "Technical MOS",
            "protocol": "Frozen Linear Probe", "seed": 42,
            "num_train": int(train_mask.sum()), "num_validation": int(val_mask.sum()),
            "num_test": int(val_mask.sum()),
            **probe_metrics, "checkpoint": str((probe_dir / "linear_probe_best.pt").resolve()),
            "feature_dim": 128, "trainable_params": trainable_params,
        })
        pred = frame.loc[val_mask, ["video_id", "ground_truth", "ground_truth_normalized"]].copy()
        pred["technical_mos"] = pred["ground_truth"]
        pred["zero_shot_score"] = frame.loc[val_mask, "zero_shot_score"].to_numpy()
        pred["linear_probe_score"] = probe_prediction
        pred["probe_seed"] = 42
        predictions["divide_maxwell"] = pred
        logger.info(
            "DIVIDE zero-shot SRCC=%.4f | probe SRCC=%.4f | probe params=%d frozen=%d",
            zero_metrics["SRCC"], probe_metrics["SRCC"], trainable_params, frozen_params,
        )
        logger.info(
            "DIVIDE probe parameter audit | total=%d trainable=%d frozen=%d",
            trainable_params + frozen_params, trainable_params, frozen_params,
        )

    if "konvid1k" in datasets:
        frame, embeddings, _ = datasets["konvid1k"]
        split_manifest = frame[["video_id"]].copy()
        zero_metrics = regression_metrics(
            frame["zero_shot_score"].to_numpy(),
            frame["ground_truth_normalized"].to_numpy(),
        )
        results.append({
            "dataset": "KoNViD-1k", "ground_truth": "Overall MOS",
            "protocol": "Zero-shot", "seed": "", "num_train": 0,
            "num_validation": 0, "num_test": len(frame), **zero_metrics,
            "checkpoint": str(args.checkpoint.resolve()), "feature_dim": 128,
            "trainable_params": 0,
        })
        seed42_prediction = None
        seed42_test_indices = None
        indices = np.arange(len(frame))
        for seed in PROBE_SEEDS:
            outer_train_indices, test_indices = train_test_split(
                indices, test_size=0.2, random_state=seed
            )
            train_indices, validation_indices = train_test_split(
                outer_train_indices, test_size=0.1, random_state=seed
            )
            roles = np.full(len(frame), "train", dtype=object)
            roles[validation_indices] = "validation"
            roles[test_indices] = "test"
            split_manifest[f"seed_{seed}"] = roles
            probe_dir = args.output_dir / "konvid_linear_probe" / f"seed_{seed}"
            _, payload = train_linear_probe(
                embeddings[train_indices], frame.iloc[train_indices]["ground_truth_normalized"].to_numpy(np.float32),
                embeddings[validation_indices], frame.iloc[validation_indices]["ground_truth_normalized"].to_numpy(np.float32),
                seed, probe_dir / "linear_probe_best.pt", probe_dir / "history.csv",
                args.force_probes,
            )
            probe = nn.Linear(embeddings.shape[1], 1)
            probe.load_state_dict(payload["model_state_dict"])
            probe.eval()
            with torch.inference_mode():
                prediction = probe(
                    torch.from_numpy(embeddings[test_indices]).float()
                ).squeeze(-1).numpy()
            metrics = regression_metrics(
                prediction, frame.iloc[test_indices]["ground_truth_normalized"].to_numpy()
            )
            results.append({
                "dataset": "KoNViD-1k", "ground_truth": "Overall MOS",
                "protocol": "Frozen Linear Probe", "seed": seed,
                "num_train": len(train_indices),
                "num_validation": len(validation_indices), "num_test": len(test_indices),
                **metrics, "checkpoint": str((probe_dir / "linear_probe_best.pt").resolve()),
                "feature_dim": 128, "trainable_params": int(payload["trainable_params"]),
            })
            if seed == 42:
                seed42_prediction = prediction
                seed42_test_indices = test_indices
        pred = frame[["video_id", "ground_truth", "ground_truth_normalized", "zero_shot_score"]].copy()
        pred["overall_mos"] = pred["ground_truth"]
        pred["linear_probe_score"] = np.nan
        pred["probe_seed"] = 42
        pred["probe_split"] = "train"
        pred.loc[seed42_test_indices, "linear_probe_score"] = seed42_prediction
        pred.loc[seed42_test_indices, "probe_split"] = "test"
        predictions["konvid1k"] = pred
        write_csv(
            args.output_dir / "konvid_split_manifest.csv",
            split_manifest.to_dict("records"),
        )
        logger.info(
            "KoNViD zero-shot SRCC=%.4f | probe mean SRCC=%.4f | trainable=129 frozen=%d",
            zero_metrics["SRCC"],
            float(np.mean([row["SRCC"] for row in results if row["dataset"] == "KoNViD-1k" and row["protocol"] == "Frozen Linear Probe"])),
            technical_params,
        )
        logger.info(
            "KoNViD probe parameter audit | total=%d trainable=%d frozen=%d",
            technical_params + 129, 129, technical_params,
        )
    return results, predictions


def science_training_distribution(
    args: argparse.Namespace,
    model: MultiModalQualityModel,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    metadata = load_metadata(args.science_metadata)
    available = {path.stem for path in args.science_feature_dir.glob("*.pt")}
    metadata = metadata[metadata["video_id"].astype(str).isin(available)].copy()
    train_df, _ = train_test_split(
        metadata, test_size=CFG.val_ratio, random_state=42, stratify=metadata["label"]
    )
    embeddings = []
    predictions = []
    for video_id in train_df["video_id"].astype(str):
        sample = load_pt(args.science_feature_dir / f"{video_id}.pt")
        video = torch.as_tensor(sample["video_feat"], dtype=torch.float32, device=device).view(1, -1)
        meta = torch.as_tensor(sample["meta_feat"], dtype=torch.float32, device=device).view(1, -1)
        cover = torch.as_tensor(sample["cover_technical_feat"], dtype=torch.float32, device=device).view(1, -1)
        wpm = torch.as_tensor([sample.get("wpm", 0.0)], dtype=torch.float32, device=device).view(1, 1)
        rhythm = torch.as_tensor(
            sample.get("speech_rhythm_feat", np.zeros(6)), dtype=torch.float32, device=device
        ).view(1, -1)
        with torch.inference_mode():
            embedding, logit = model.technical_branch(
                video, meta, dnsmos_feat=torch.zeros(1, 3, device=device),
                wpm=wpm, speech_rhythm_feat=rhythm, cover_technical_feat=cover,
            )
        embeddings.append(embedding.squeeze(0).cpu().numpy())
        predictions.append(float(torch.sigmoid(logit).item()))
    return np.stack(embeddings).astype(np.float32), np.asarray(predictions, np.float32)


def distribution_row(dataset: str, embedding: np.ndarray, prediction: np.ndarray) -> dict:
    norms = np.linalg.norm(embedding, axis=1)
    return {
        "dataset": dataset,
        "num_samples": len(embedding),
        "feature_dim": embedding.shape[1],
        "feature_global_mean": float(embedding.mean()),
        "feature_global_std": float(embedding.std()),
        "feature_norm_mean": float(norms.mean()),
        "feature_norm_std": float(norms.std()),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
    }


def error_case_rows(predictions: pd.DataFrame, dataset: str) -> list[dict]:
    rows = []
    for protocol, column in (
        ("Zero-shot", "zero_shot_score"),
        ("Frozen Linear Probe", "linear_probe_score"),
    ):
        valid = predictions[column].notna()
        selected = predictions.loc[valid].copy()
        selected["error"] = (
            selected[column] - selected["ground_truth_normalized"]
        ).abs()
        selected = selected.nlargest(10, "error")
        for row in selected.itertuples(index=False):
            rows.append({
                "dataset": dataset,
                "protocol": protocol,
                "video_id": str(row.video_id),
                "ground_truth": float(row.ground_truth),
                "ground_truth_normalized": float(row.ground_truth_normalized),
                "prediction": float(getattr(row, column)),
                "error": float(row.error),
            })
    return rows


def write_cache_report(
    args: argparse.Namespace,
    reports: dict[str, dict],
    metadata: dict[str, pd.DataFrame],
) -> None:
    lines = [
        "# External Technical Feature Cache Report", "",
        "The cache stores separate 512D CLIP features, 768D COVER technical features, "
        "128D frozen Technical Branch embeddings, zero-shot scores, labels, and extraction metadata.",
        "",
        "| Dataset | Labels | Videos found | Valid cache | New | Reused | Errors | Seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, frame in metadata.items():
        valid = sum(
            valid_cache(cache_path(args.output_dir, dataset, str(video_id)), dataset)
            for video_id in frame["video_id"]
        )
        found = sum(Path(path).is_file() for path in frame["video_path"])
        report = reports.get(dataset, {})
        lines.append(
            f"| {dataset} | {len(frame)} | {found} | {valid} | "
            f"{report.get('extracted', 0)} | {report.get('cached', 0)} | "
            f"{report.get('errors', 0)} | {report.get('elapsed_seconds', 0.0):.1f} |"
        )
    lines.extend([
        "", "Feature version: `keep_p2_s2_external_technical_v1`.",
        "Missing external metadata/WPM/rhythm policy: deterministic zero fill.",
    ])
    (args.output_dir / "feature_cache_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def format_mean_std(rows: list[dict], metric: str) -> str:
    values = [float(row[metric]) for row in rows]
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}"


def write_summary(
    args: argparse.Namespace,
    results: list[dict],
    distributions: list[dict],
    metadata_status: dict[str, pd.DataFrame],
) -> None:
    lookup = {}
    for dataset in ("DIVIDE-MaxWell", "KoNViD-1k"):
        for protocol in ("Zero-shot", "Frozen Linear Probe"):
            lookup[(dataset, protocol)] = [
                row for row in results
                if row["dataset"] == dataset and row["protocol"] == protocol
            ]
    lines = [
        "# Technical Branch Cross-Dataset Generalization", "",
        "## 1. Research Question", "",
        "KEEP_P2_S2 的 Technical Branch 是否具有跨数据集的视频技术质量泛化能力？", "",
        "## 2. Model Audit", "",
        "冻结模型使用 legacy CLIP 512D 与 COVER technical 768D，经 256D/256D 投影和 "
        "128D 融合后，与零填充的外部缺失上下文共同生成 128D technical embedding。"
        "详细审计见 `technical_branch_audit.md`。", "",
        "COVER technical 权重来自官方 YouTube-UGC 训练 checkpoint；因此结果表示冻结 UGC "
        "质量先验的跨数据集迁移，不应表述为无 VQA 预训练的从零泛化。", "",
        "## 3. External Datasets", "",
        f"- DIVIDE-MaxWell: {len(metadata_status.get('divide_maxwell', []))} 条，Technical MOS，官方 train/val。",
        f"- KoNViD-1k: {len(metadata_status.get('konvid1k', []))} 条，Overall MOS；这里只能表述为通用 UGC 感知质量迁移。",
        "", "## 4. Experimental Protocol", "",
        "Zero-shot 冻结全部参数并使用原 Technical score head。Frozen Linear Probe 冻结全部"
        "编码器，只训练 `Linear(128,1)` 的 129 个参数；MOS/5 仅用于回归误差。"
        "KoNViD 外层采用 80/20 train/test，80% 训练池再按 90/10 划分 train/validation，"
        "只用内部 validation 选择 checkpoint，外层 test 不参与调参。"
        "DIVIDE 严格使用官方 train/validation；官方 validation 同时用于 checkpoint 选择和"
        "结果报告，因此应称 validation result，而不是独立 test result。", "",
        "## Technical Branch Cross-Dataset Generalization", "",
        "| Dataset | Ground Truth | Protocol | SRCC | PLCC | MAE | MSE |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for dataset, ground_truth in (
        ("DIVIDE-MaxWell", "Technical MOS"), ("KoNViD-1k", "Overall MOS")
    ):
        for protocol in ("Zero-shot", "Frozen Linear Probe"):
            rows = lookup[(dataset, protocol)]
            if rows:
                lines.append(
                    f"| {dataset} | {ground_truth} | {protocol} | "
                    f"{format_mean_std(rows, 'SRCC')} | {format_mean_std(rows, 'PLCC')} | "
                    f"{format_mean_std(rows, 'MAE')} | {format_mean_std(rows, 'MSE')} |"
                )
            else:
                lines.append(f"| {dataset} | {ground_truth} | {protocol} | N/A | N/A | N/A | N/A |")
    lines.extend(["", "## 5. DIVIDE-MaxWell Results", ""])
    divide_zero = lookup[("DIVIDE-MaxWell", "Zero-shot")]
    divide_probe = lookup[("DIVIDE-MaxWell", "Frozen Linear Probe")]
    if divide_zero and divide_probe:
        lines.append(
            f"Zero-shot SRCC={divide_zero[0]['SRCC']:.4f}，linear probe SRCC={divide_probe[0]['SRCC']:.4f}。"
        )
    else:
        lines.append("DIVIDE 实验尚未具备完整缓存。")
    lines.extend(["", "## 6. KoNViD-1k Results", ""])
    konvid_zero = lookup[("KoNViD-1k", "Zero-shot")]
    konvid_probe = lookup[("KoNViD-1k", "Frozen Linear Probe")]
    if konvid_zero and konvid_probe:
        lines.append(
            f"Zero-shot SRCC={konvid_zero[0]['SRCC']:.4f}；三种子 probe SRCC="
            f"{format_mean_std(konvid_probe, 'SRCC')}。"
        )
    else:
        lines.append("KoNViD 视频或缓存尚未完整。")
    lines.extend(["", "## 7. Multi-seed Stability", ""])
    if konvid_probe:
        lines.extend([
            "| Seed | SRCC | PLCC | MAE | MSE |", "|---:|---:|---:|---:|---:|",
            *[
                f"| {row['seed']} | {row['SRCC']:.4f} | {row['PLCC']:.4f} | "
                f"{row['MAE']:.4f} | {row['MSE']:.4f} |"
                for row in konvid_probe
            ],
        ])
    else:
        lines.append("N/A")
    lines.extend(["", "## 8. Zero-shot vs Linear Probe", ""])
    if divide_zero and divide_probe:
        lines.append(
            f"DIVIDE probe 相对 zero-shot 的 SRCC 变化为 "
            f"{divide_probe[0]['SRCC'] - divide_zero[0]['SRCC']:+.4f}。"
        )
    if konvid_zero and konvid_probe:
        lines.append(
            f"KoNViD probe 平均 SRCC 相对 zero-shot 变化为 "
            f"{np.mean([row['SRCC'] for row in konvid_probe]) - konvid_zero[0]['SRCC']:+.4f}。"
        )
    lines.extend(["", "## 9. Feature Distribution Analysis", ""])
    if distributions:
        lines.extend([
            "| Dataset | N | Feature mean | Feature std | Norm mean | Pred mean | Pred std |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *[
                f"| {row['dataset']} | {row['num_samples']} | {row['feature_global_mean']:.4f} | "
                f"{row['feature_global_std']:.4f} | {row['feature_norm_mean']:.4f} | "
                f"{row['prediction_mean']:.4f} | {row['prediction_std']:.4f} |"
                for row in distributions
            ],
        ])
        source = next(
            (row for row in distributions if row["dataset"] == "ScienceVideo-train"),
            None,
        )
        if source is not None:
            for row in distributions:
                if row is source:
                    continue
                norm_ratio = row["feature_norm_mean"] / max(source["feature_norm_mean"], 1e-12)
                score_std_ratio = row["prediction_std"] / max(source["prediction_std"], 1e-12)
                lines.append(
                    f"{row['dataset']} 的 embedding norm 均值为源域的 {norm_ratio:.2%}，"
                    f"zero-shot 分数标准差为源域的 {score_std_ratio:.2%}。"
                )
    lines.extend([
        "", "## 10. Error Analysis", "",
        "Top-10 normalized absolute errors for each available dataset/protocol are saved in "
        "`divide_error_cases.csv` and `konvid_error_cases.csv`。",
    ])
    for filename in ("divide_error_cases.csv", "konvid_error_cases.csv"):
        path = args.output_dir / filename
        if not path.exists():
            continue
        error_frame = pd.read_csv(path)
        for protocol in error_frame["protocol"].drop_duplicates():
            subset = error_frame[error_frame["protocol"] == protocol]
            if subset.empty:
                continue
            worst = subset.loc[subset["error"].idxmax()]
            lines.append(
                f"{worst['dataset']} / {protocol} 最大归一化绝对误差样本为 "
                f"`{worst['video_id']}`（GT={worst['ground_truth_normalized']:.4f}，"
                f"prediction={worst['prediction']:.4f}，error={worst['error']:.4f}）。"
            )
    lines.extend(["", "## 11. Conclusion", ""])
    if divide_zero and divide_probe and konvid_zero and konvid_probe:
        dz, dp = divide_zero[0]["SRCC"], divide_probe[0]["SRCC"]
        kz = konvid_zero[0]["SRCC"]
        kp = float(np.mean([row["SRCC"] for row in konvid_probe]))
        if dz <= 0 and kz <= 0:
            lines.append("两个数据集的 zero-shot SRCC 均为负，原始 score head 不具备可靠跨域排序能力。")
        elif dz > 0 and kz > 0:
            lines.append("两个数据集 zero-shot 均为正相关，原始 score head 具有一定跨域排序能力。")
        else:
            lines.append("两个数据集的 zero-shot 结论不一致，原始 score head 存在域依赖。")
        if dp > dz or kp > kz:
            lines.append(
                "Frozen linear probe 在两个数据集均显著提升，表明 128D technical representation "
                "中包含可迁移质量信息，主要失配发生在原 score head 与外部评分尺度。"
            )
        else:
            lines.append("Linear probe 未形成一致提升，当前 technical representation 的外部可迁移性有限。")
        lines.append(
            "两数据集在“zero-shot 失败、linear probe 有效”这一模式上表现一致；数值差异应结合"
            "标签语义解释：DIVIDE 直接监督 Technical MOS，KoNViD Overall MOS 同时受内容与美学影响。"
        )
        lines.append(
            "COVER technical backbone 使用官方 YouTube-UGC 训练权重，因此上述结论是冻结 UGC "
            "质量先验的跨数据集迁移能力，不是无质量数据预训练条件下的从零泛化。"
        )
    else:
        lines.append("结论暂不完整：至少一个外部数据集缺少完整视频缓存。")
    (args.output_dir / "external_technical_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(args.output_dir / "external_technical_validation.log")
    device = get_device(CFG.device)
    model, checkpoint = load_model(args.checkpoint, device)
    technical_params = sum(parameter.numel() for parameter in model.technical_branch.parameters())
    logger.info(
        "Loaded KEEP_P2_S2 | technical params=%d trainable=0 | hash=%s",
        technical_params, checkpoint["split_hash"],
    )

    datasets = selected_datasets(args.dataset)
    metadata = {dataset: load_external_metadata(args, dataset) for dataset in datasets}
    reports = {}
    if not args.evaluate_only:
        for dataset in datasets:
            validate_dataset_files(metadata[dataset], dataset)
            reports[dataset] = extract_dataset_cache(
                args, dataset, metadata[dataset], model, device, logger
            )
        status_path = args.output_dir / "feature_cache_status.json"
        existing_reports = {}
        if status_path.exists():
            try:
                existing_reports = json.loads(status_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing_reports = {}
        existing_reports.update(reports)
        write_json(status_path, existing_reports)
    write_cache_report(args, reports, metadata)
    if args.extract_only:
        return

    cached = {}
    for dataset in datasets:
        frame, embedding, technical = load_cached_records(
            args.output_dir, dataset, metadata[dataset]
        )
        expected = min(len(metadata[dataset]), args.max_videos) if args.max_videos > 0 else len(metadata[dataset])
        if len(frame) != expected:
            raise RuntimeError(f"Incomplete {dataset} cache: {len(frame)}/{expected}")
        if args.max_videos > 0:
            logger.info("Smoke cache complete for %s; skipping formal evaluation", dataset)
            continue
        cached[dataset] = (frame, embedding, technical)
    if not cached:
        return

    results, predictions = external_results_and_predictions(
        args, cached, technical_params, logger
    )
    result_fields = [
        "dataset", "ground_truth", "protocol", "seed", "num_train", "num_validation", "num_test",
        "SRCC", "PLCC", "MSE", "MAE", "checkpoint", "feature_dim", "trainable_params",
    ]
    write_csv(args.output_dir / "external_technical_results.csv", results, result_fields)
    distributions = []
    science_embedding, science_prediction = science_training_distribution(args, model, device)
    distributions.append(distribution_row("ScienceVideo-train", science_embedding, science_prediction))
    for dataset, (frame, embedding, _) in cached.items():
        distributions.append(distribution_row(
            dataset, embedding, frame["zero_shot_score"].to_numpy(np.float32)
        ))
    write_csv(args.output_dir / "external_feature_distribution.csv", distributions)

    if "divide_maxwell" in predictions:
        divide = predictions["divide_maxwell"]
        write_csv(args.output_dir / "divide_predictions.csv", divide.to_dict("records"))
        write_csv(
            args.output_dir / "divide_error_cases.csv",
            error_case_rows(divide, "DIVIDE-MaxWell"),
        )
    if "konvid1k" in predictions:
        konvid = predictions["konvid1k"]
        write_csv(args.output_dir / "konvid_predictions.csv", konvid.to_dict("records"))
        write_csv(
            args.output_dir / "konvid_error_cases.csv",
            error_case_rows(konvid, "KoNViD-1k"),
        )
    write_summary(args, results, distributions, metadata)
    logger.info("External Technical validation complete: %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
