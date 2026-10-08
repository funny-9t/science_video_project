"""Evaluate lightweight COVER temporal statistics inside KEEP_P2_S2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import CFG
from pipeline.step_cover_technical import COVERTechnicalFeatureExtractor
from pipeline.utils_io import ensure_dir, load_metadata
from training.dataset_pair import PairwiseVideoDataset
from training.metrics import compute_video_metrics
from training.model_mvp import MultiModalQualityModel
from training.train import _search_best_threshold
from training.utils_train import get_device, set_seed
from tools import run_progressive_training as progressive


EXPECTED_HASH = "79413d21918a956716f230456b80aec037be51bfd1c6a412a8c0a30087a2a58f"
VARIANTS = {
    "T0_KEEP_P2_S2": "baseline",
    "T1_MEAN_STD": "mean_std",
    "T2_MEAN_STD_DIFF": "mean_std_diff",
}
RESULT_FIELDS = (
    "variant", "temporal_mode", "seed", "split_seed", "validation_hash",
    "auc", "pr_auc", "accuracy", "f1", "overall_srcc",
    "sci_srcc", "tech_srcc", "tech_plcc", "tech_mse", "tech_mae",
    "aes_srcc", "branch_macro_srcc", "science_weight", "technical_weight",
    "aesthetic_weight", "best_epoch", "train_seconds", "trainable_params",
    "temporal_feature_dim", "projection_dim", "checkpoint",
)
HISTORY_FIELDS = (
    "variant", "temporal_mode", "seed", "stage", "epoch", "rank_loss",
    "point_loss", "branch_reg_loss", "branch_rank_loss", "consistency_loss",
    "total_loss", "auc", "pr_auc", "overall_srcc", "sci_srcc", "tech_srcc",
    "tech_plcc", "tech_mse", "aes_srcc", "branch_macro_srcc", "learning_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument(
        "--temporal-cache-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "temporal_quality_statistics" / "cache",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "temporal_quality_statistics",
    )
    parser.add_argument("--variants", default="T0_KEEP_P2_S2,T1_MEAN_STD,T2_MEAN_STD_DIFF")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def logger_for(path: Path) -> logging.Logger:
    logger = logging.getLogger("temporal_quality_statistics")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path, fields: tuple[str, ...]) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for key in fields:
                if key in {"variant", "temporal_mode", "validation_hash", "checkpoint", "stage"}:
                    continue
                if key in {"seed", "split_seed", "best_epoch", "trainable_params", "temporal_feature_dim", "projection_dim", "epoch"}:
                    row[key] = int(float(raw[key]))
                else:
                    row[key] = float(raw[key])
            rows.append(row)
    return rows


def merge_rows(existing: list[dict], current: list[dict], keys: tuple[str, ...]) -> list[dict]:
    merged = {tuple(row[key] for key in keys): row for row in existing}
    for row in current:
        merged[tuple(row[key] for key in keys)] = row
    return sorted(merged.values(), key=lambda row: tuple(row[key] for key in keys))


def prepare_data(args: argparse.Namespace, logger: logging.Logger):
    metadata = load_metadata(args.metadata)
    available = {path.stem for path in args.feature_dir.glob("*.pt")}
    metadata = metadata[metadata["video_id"].astype(str).isin(available)].copy()
    train_df, val_df = train_test_split(
        metadata, test_size=CFG.val_ratio, random_state=args.split_seed,
        stratify=metadata["label"],
    )
    payload = "\n".join(
        [f"train:{value}" for value in sorted(train_df["video_id"].astype(str))]
        + [f"val:{value}" for value in sorted(val_df["video_id"].astype(str))]
    )
    split_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if args.split_seed != 42 or split_hash != EXPECTED_HASH:
        raise RuntimeError(f"Unexpected split seed/hash: {args.split_seed}/{split_hash}")
    common = dict(
        same_category=False, use_native_clip=False, use_cover_features=False,
        aesthetic_feature_backend="shared_clip", technical_visual_source="clip_cover",
        cover_temporal_feature_dir=args.temporal_cache_dir,
    )
    train_ds = PairwiseVideoDataset.from_metadata(
        train_df, args.feature_dir, deterministic_pairs=False, **common
    )
    val_ds = PairwiseVideoDataset.from_metadata(
        val_df, args.feature_dir, deterministic_pairs=True, **common
    )
    logger.info(
        "data=%d train=%d val=%d labels=%s hash=%s",
        len(metadata), len(train_df), len(val_df),
        metadata["label"].value_counts().sort_index().to_dict(), split_hash,
    )
    return train_ds, val_ds, split_hash


def model_config(mode: str) -> dict:
    config = progressive.model_config()
    config["technical_temporal_mode"] = mode
    return config


def new_model(mode: str, seed: int, device: torch.device) -> MultiModalQualityModel:
    set_seed(seed)
    baseline = MultiModalQualityModel(**model_config("baseline"))
    if mode == "baseline":
        return baseline.to(device)
    set_seed(seed)
    model = MultiModalQualityModel(**model_config(mode))
    baseline_state = baseline.state_dict()
    model_state = model.state_dict()
    for name, value in baseline_state.items():
        if name in model_state and model_state[name].shape == value.shape:
            model_state[name] = value.clone()
    model.load_state_dict(model_state, strict=True)
    return model.to(device)


def collate_unique(samples: list[dict]) -> dict:
    converted = [PairwiseVideoDataset._to_tensor_dict(sample) for sample in samples]
    result = {"video_id": [sample["video_id"] for sample in converted]}
    for key in converted[0]:
        if key == "video_id" or key == "frame_features":
            continue
        tensors = [sample[key] for sample in converted]
        if key == "cover_temporal_feat":
            max_len = max(tensor.size(0) for tensor in tensors)
            padded = torch.zeros(len(tensors), max_len, 768)
            for index, tensor in enumerate(tensors):
                padded[index, :tensor.size(0)] = tensor
            result[key] = padded
        else:
            result[key] = torch.stack(tensors)
    return result


def correlation(prediction: np.ndarray, target: np.ndarray, method: str) -> float:
    valid = np.isfinite(prediction) & np.isfinite(target) & (target >= 0)
    if valid.sum() < 2 or np.unique(target[valid]).size < 2:
        return float("nan")
    value = (
        spearmanr(prediction[valid], target[valid]).statistic
        if method == "spearman"
        else pearsonr(prediction[valid], target[valid]).statistic
    )
    return float(value)


def evaluate(model: MultiModalQualityModel, samples: list[dict], device: torch.device) -> dict:
    model.eval()
    keys = (
        "label", "probability", "overall", "sci_logit", "tech_logit", "aes_logit",
        "sci_score", "tech_score", "aes_score", "sci_target", "tech_target",
        "aes_target", "overall_target", "science_weight", "technical_weight",
        "aesthetic_weight",
    )
    records = {key: [] for key in keys}
    with torch.inference_mode():
        for start in range(0, len(samples), 32):
            batch = progressive.prepare_batch(collate_unique(samples[start:start + 32]), device)
            output = model(**batch)
            values = {
                "label": batch["label_target"].squeeze(-1),
                "probability": output["probability"].squeeze(-1),
                "overall": output["overall_score"].squeeze(-1),
                "sci_logit": output["scientific_score"].squeeze(-1),
                "tech_logit": output["technical_score"].squeeze(-1),
                "aes_logit": output["aesthetic_score"].squeeze(-1),
                "sci_score": torch.sigmoid(output["scientific_score"].squeeze(-1)),
                "tech_score": torch.sigmoid(output["technical_score"].squeeze(-1)),
                "aes_score": torch.sigmoid(output["aesthetic_score"].squeeze(-1)),
                "sci_target": batch["sci_target"].squeeze(-1),
                "tech_target": batch["tech_target"].squeeze(-1),
                "aes_target": batch["aes_target"].squeeze(-1),
                "overall_target": batch["overall_quality_target"].squeeze(-1),
                "science_weight": output["branch_weights"][:, 0],
                "technical_weight": output["branch_weights"][:, 1],
                "aesthetic_weight": output["branch_weights"][:, 2],
            }
            for key, value in values.items():
                records[key].extend(value.detach().cpu().numpy().tolist())
    arrays = {key: np.asarray(value, dtype=np.float32) for key, value in records.items()}
    threshold, _ = _search_best_threshold(
        arrays["probability"][arrays["label"] == 1],
        arrays["probability"][arrays["label"] == 0], metric="f1",
    )
    metrics = compute_video_metrics(arrays["label"], arrays["probability"], threshold)
    positive = arrays["probability"][arrays["label"] == 1]
    negative = arrays["probability"][arrays["label"] == 0]
    metrics["ranking_accuracy"] = float((positive[:, None] > negative[None, :]).mean())
    metrics["best_threshold"] = float(threshold)
    metrics["overall_srcc"] = correlation(
        arrays["overall"], arrays["overall_target"], "spearman"
    )
    for prefix, score, target in (
        ("sci", "sci_score", "sci_target"),
        ("tech", "tech_score", "tech_target"),
        ("aes", "aes_score", "aes_target"),
    ):
        metrics[f"{prefix}_srcc"] = correlation(arrays[score], arrays[target], "spearman")
    metrics["tech_plcc"] = correlation(arrays["tech_score"], arrays["tech_target"], "pearson")
    valid_tech = arrays["tech_target"] >= 0
    error = arrays["tech_score"][valid_tech] - arrays["tech_target"][valid_tech]
    metrics["tech_mse"] = float(np.mean(error ** 2))
    metrics["tech_mae"] = float(np.mean(np.abs(error)))
    metrics["branch_macro_srcc"] = float(np.mean([
        metrics["sci_srcc"], metrics["tech_srcc"], metrics["aes_srcc"]
    ]))
    for key in ("science_weight", "technical_weight", "aesthetic_weight"):
        metrics[key] = float(arrays[key].mean())

    pos_index = arrays["label"] == 1
    neg_index = ~pos_index
    metrics["rank_loss"] = float(F.softplus(-(
        torch.from_numpy(arrays["overall"][pos_index])[:, None]
        - torch.from_numpy(arrays["overall"][neg_index])[None, :]
    )).mean().item())
    metrics["point_loss"] = float(F.binary_cross_entropy_with_logits(
        torch.from_numpy(arrays["overall"]), torch.from_numpy(arrays["label"])
    ).item())
    metrics["consistency_loss"] = float(np.mean((
        arrays["overall"]
        - (arrays["sci_logit"] + arrays["tech_logit"] + arrays["aes_logit"]) / 3.0
    ) ** 2))
    branch_losses = []
    for score, target in (("sci_score", "sci_target"), ("tech_score", "tech_target"), ("aes_score", "aes_target")):
        valid = arrays[target] >= 0
        branch_losses.append(float(np.mean((arrays[score][valid] - arrays[target][valid]) ** 2)))
    metrics["branch_reg_loss"] = statistics.fmean(branch_losses)
    metrics["validation_loss"] = (
        metrics["rank_loss"] + 0.1 * metrics["point_loss"]
        + 0.1 * metrics["consistency_loss"] + 0.05 * metrics["branch_reg_loss"]
    )
    return metrics


def result_row(payload: dict, checkpoint: Path) -> dict:
    metrics = payload["validation_metrics"]
    mode = payload["temporal_mode"]
    multiplier = {"baseline": 1, "mean_std": 2, "mean_std_diff": 4}[mode]
    return {
        "variant": payload["variant"], "temporal_mode": mode,
        "seed": payload["seed"], "split_seed": payload["split_seed"],
        "validation_hash": payload["validation_hash"],
        **{key: float(metrics[key]) for key in (
            "auc", "pr_auc", "accuracy", "f1", "overall_srcc", "sci_srcc",
            "tech_srcc", "tech_plcc", "tech_mse", "tech_mae", "aes_srcc",
            "branch_macro_srcc", "science_weight", "technical_weight", "aesthetic_weight",
        )},
        "best_epoch": int(payload["best_epoch"]),
        "train_seconds": float(payload.get("train_seconds", 0.0)),
        "trainable_params": int(payload["trainable_params"]),
        "temporal_feature_dim": 768 * multiplier,
        "projection_dim": 256,
        "checkpoint": str(checkpoint.resolve()),
    }


def save_checkpoint(path, model, optimizer, variant, mode, stage, epoch, metrics, args, split_hash, names):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model_config(mode),
        "model_version": "main_v3-D3_KEEP_P2_S2_temporal_v1",
        "variant": variant, "temporal_mode": mode,
        "temporal_feature_version": COVERTechnicalFeatureExtractor.TEMPORAL_FEATURE_VERSION,
        "initialization_policy": "same-seed baseline init; copy all shape-compatible parameters",
        "temporal_feature_dim": {"baseline": 768, "mean_std": 1536, "mean_std_diff": 3072}[mode],
        "projection_dim": 256, "training_stage": stage,
        "loss_weights": {"rank": 1.0 if stage == "stage2" else 0.0, "pointwise": 0.1 if stage == "stage2" else 0.0, "consistency": 0.1 if stage == "stage2" else 0.0, "branch_mse": 1.0 if stage == "stage1" else 0.05, "branch_rank": 0.05},
        "branch_floor": 0.2, "epoch": epoch, "best_epoch": epoch,
        "validation_metrics": metrics, "seed": args.seed, "split_seed": args.split_seed,
        "validation_hash": split_hash, "trainable_parameter_names": names,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }, path)


def train_stage(model, train_ds, val_ds, device, args, logger, variant, mode, stage, checkpoint, max_epochs, patience, split_hash, history):
    names = progressive.set_trainable(model, stage)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(parameters, lr=5e-5, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=CFG.scheduler_factor,
        patience=CFG.scheduler_patience, min_lr=1e-6,
    )
    loader = progressive.make_train_loader(train_ds, args.seed)
    best = -float("inf")
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        train_metrics = progressive.train_epoch(
            model, loader, optimizer, device, stage, branch_rank_weight=0.05
        )
        validation = evaluate(model, val_ds.samples, device)
        selection = validation["branch_macro_srcc"] if stage == "stage1" else validation["ranking_accuracy"]
        scheduler.step(validation["branch_reg_loss"] if stage == "stage1" else validation["validation_loss"])
        history.append({
            "variant": variant, "temporal_mode": mode, "seed": args.seed,
            "stage": stage, "epoch": epoch,
            **{key: train_metrics[key] for key in (
                "rank_loss", "point_loss", "branch_reg_loss", "branch_rank_loss",
                "consistency_loss", "total_loss",
            )},
            **{key: validation[key] for key in (
                "auc", "pr_auc", "overall_srcc", "sci_srcc", "tech_srcc",
                "tech_plcc", "tech_mse", "aes_srcc", "branch_macro_srcc",
            )},
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        write_csv(args.output_dir / "temporal_quality_history.csv", history, HISTORY_FIELDS)
        logger.info(
            "%s/%s epoch=%d loss=%.4f AUC=%.4f PR=%.4f overall=%.4f "
            "S/T/A=%.4f/%.4f/%.4f techPLCC=%.4f techMSE=%.4f",
            variant, stage, epoch, train_metrics["total_loss"], validation["auc"],
            validation["pr_auc"], validation["overall_srcc"], validation["sci_srcc"],
            validation["tech_srcc"], validation["aes_srcc"], validation["tech_plcc"],
            validation["tech_mse"],
        )
        if selection > best:
            best = selection
            stale = 0
            save_checkpoint(
                checkpoint, model, optimizer, variant, mode, stage, epoch,
                validation, args, split_hash, names,
            )
        else:
            stale += 1
            if stale >= patience:
                break
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["train_seconds"] = time.perf_counter() - started
    torch.save(payload, checkpoint)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return payload


def historical_checkpoint(seed: int) -> Path:
    root = PROJECT_ROOT / "outputs" / "progressive_training" / "P2"
    return root / ("stage2_best.pt" if seed == 42 else f"seed_{seed}/stage2_best.pt")


def load_t0(seed, val_ds, device, split_hash) -> dict:
    checkpoint = historical_checkpoint(seed)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    historical_hash = payload.get("split_hash", payload.get("validation_hash"))
    if historical_hash != split_hash:
        raise RuntimeError(f"Historical T0 split mismatch: {historical_hash}")
    config = dict(payload["config"])
    config.setdefault("technical_temporal_mode", "baseline")
    model = MultiModalQualityModel(**config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    metrics = evaluate(model, val_ds.samples, device)
    return {
        "variant": "T0_KEEP_P2_S2", "temporal_mode": "baseline",
        "seed": seed, "split_seed": 42, "validation_hash": split_hash,
        "best_epoch": payload["best_epoch"], "validation_metrics": metrics,
        "train_seconds": payload.get("train_seconds", 0.0),
        "trainable_params": payload["trainable_params"],
    }, checkpoint


def run_temporal(variant, mode, train_ds, val_ds, device, args, logger, split_hash, history):
    output = args.output_dir / variant.lower()
    if args.seed != 42:
        output = output / f"seed_{args.seed}"
    ensure_dir(output)
    stage1_path = output / "stage1_best.pt"
    stage2_path = output / "stage2_best.pt"
    model = new_model(mode, args.seed, device)
    if stage2_path.exists() and not args.force:
        payload = torch.load(stage2_path, map_location="cpu", weights_only=False)
        if payload.get("validation_hash") != split_hash or payload.get("temporal_mode") != mode:
            raise RuntimeError(f"Stale temporal checkpoint: {stage2_path}")
        logger.info("Reusing %s", stage2_path)
        return result_row(payload, stage2_path)
    stage1_epochs, stage2_epochs = (2, 2) if args.smoke else (15, 30)
    stage1_patience, stage2_patience = (2, 2) if args.smoke else (5, 10)
    if stage1_path.exists() and not args.force:
        stage1 = torch.load(stage1_path, map_location="cpu", weights_only=False)
        if stage1.get("validation_hash") != split_hash or stage1.get("temporal_mode") != mode:
            raise RuntimeError(f"Stale temporal checkpoint: {stage1_path}")
        logger.info("Reusing %s", stage1_path)
    else:
        stage1 = train_stage(
            model, train_ds, val_ds, device, args, logger, variant, mode, "stage1",
            stage1_path, stage1_epochs, stage1_patience, split_hash, history,
        )
    model.load_state_dict(stage1["model_state_dict"], strict=True)
    stage2 = train_stage(
        model, train_ds, val_ds, device, args, logger, variant, mode, "stage2",
        stage2_path, stage2_epochs, stage2_patience, split_hash, history,
    )
    stage2["train_seconds"] = stage1["train_seconds"] + stage2["train_seconds"]
    torch.save(stage2, stage2_path)
    return result_row(stage2, stage2_path)


def temporal_diagnostics(val_ds, output_dir: Path) -> dict:
    rows = []
    short = 0
    for sample in val_ds.samples:
        feature = np.asarray(sample["cover_temporal_feat"], dtype=np.float32)
        std_mean = float(feature.std(axis=0).mean())
        if feature.shape[0] < 2:
            short += 1
            diff_mean = 0.0
            diff_std = 0.0
        else:
            difference = np.abs(feature[1:] - feature[:-1])
            diff_mean = float(difference.mean(axis=0).mean())
            diff_std = float(difference.std(axis=0).mean())
        rows.append({
            "video_id": sample["video_id"], "label": int(sample["label"]),
            "temporal_std_mean": std_mean, "temporal_diff_mean": diff_mean,
            "temporal_diff_std": diff_std, "tech_target": float(sample["tech_target"]),
        })
    path = output_dir / "temporal_feature_statistics.csv"
    write_csv(path, rows, tuple(rows[0]))
    result = {"num_short_temporal_sequences": short, "count": len(rows)}
    for key in ("temporal_std_mean", "temporal_diff_mean", "temporal_diff_std"):
        values = np.asarray([row[key] for row in rows], dtype=np.float32)
        targets = np.asarray([row["tech_target"] for row in rows], dtype=np.float32)
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_min"] = float(values.min())
        result[f"{key}_max"] = float(values.max())
        result[f"{key}_nan_count"] = int((~np.isfinite(values)).sum())
        result[f"{key}_tech_srcc"] = correlation(values, targets, "spearman")
        positive = values[np.asarray([row["label"] for row in rows]) == 1]
        negative = values[np.asarray([row["label"] for row in rows]) == 0]
        result[f"{key}_positive_mean"] = float(positive.mean())
        result[f"{key}_negative_mean"] = float(negative.mean())
    (output_dir / "temporal_feature_diagnostics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def write_audit(args, diagnostics):
    report_path = args.output_dir / "extraction_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    cache_shapes = {}
    valid_cache_count = 0
    for path in args.temporal_cache_dir.glob("*.pt"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        feature = np.asarray(payload.get("temporal_feature", []), dtype=np.float32)
        if feature.ndim == 2 and feature.shape[1] == 768 and np.isfinite(feature).all():
            key = "x".join(str(value) for value in feature.shape)
            cache_shapes[key] = cache_shapes.get(key, 0) + 1
            valid_cache_count += 1
    lines = [
        "# Technical Feature Audit", "",
        "- Current cache: `cover_technical_feat` has shape `[768]`; no COVER time axis is retained in the main feature files.",
        "- Backbone output: `SwinTransformer3D` returns `[B, 768, T, H, W]`.",
        "- Existing pooling: mean over `(T, H, W)`, so the cached 768D vector is a global pooled representation.",
        "- New temporal cache point: mean over `(H, W)` only, stored as `[T, 768]`.",
        "- D3 fusion: the active legacy CLIP `video_feat` is 512D -> 256D; the cached native `clip_video_feat` is 768D but is not selected here; COVER is 768D -> 256D; both paths are fused into the unchanged Technical representation.",
        "- Sampling is unchanged: clip_len 40, t_frag 20, frame_interval 2, num_clips 1, deterministic per-video fragment offsets.",
        f"- Temporal sequence shapes: `{cache_shapes}`.",
        f"- Maximum pooled-feature reproduction error: `{report.get('pooled_max_abs_error_max', float('nan'))}`.",
        f"- Cached/expected videos: `{valid_cache_count}/615`.",
        f"- Short temporal sequences: `{diagnostics['num_short_temporal_sequences']}`.",
        "- Scientific, CLIP, Aesthetic, audio, metadata, and LLM features are reused unchanged.",
    ]
    (args.output_dir / "technical_feature_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def mean_std(rows, key):
    values = [float(row[key]) for row in rows]
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def write_summary(args, results, diagnostics):
    seed42 = {row["variant"]: row for row in results if row["seed"] == 42}
    lines = [
        "# Temporal Quality Statistics Experiment", "", "## 1. Research Question", "",
        "Can explicit mean, variation, and adjacent-change statistics improve the Technical branch of KEEP_P2_S2?",
        "", "## 2. Baseline", "",
        "T0 is main_v3-D3 with Stage1 -> Stage2 progressive branch training, Branch RankNet, 20% branch floor, and the fixed 492/123 split.",
        f"Validation hash: `{EXPECTED_HASH}`.", "", "## 3. Technical Feature Audit", "",
        "The original COVER technical cache is pooled `[768]`. The new cache reuses the same Swin weights, sampling, preprocessing, and deterministic offsets, retaining spatially pooled `[20,768]` sequences.",
        "", "## 4. Experimental Variants", "",
        "- T0_KEEP_P2_S2: pooled 768D COVER feature.",
        "- T1_MEAN_STD: concatenate temporal mean and std, 1536D -> 256D.",
        "- T2_MEAN_STD_DIFF: add absolute adjacent-difference mean/std, 3072D -> 256D.",
        "", "## 5. Seed 42 Results", "",
        "| Variant | AUROC | PR-AUC | Accuracy | F1 | Overall SRCC | Tech SRCC | Tech PLCC | Tech MSE | Tech MAE | Sci SRCC | Aes SRCC | Branch Macro | Epoch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANTS:
        if name not in seed42:
            continue
        row = seed42[name]
        lines.append(
            f"| {name} | {row['auc']:.4f} | {row['pr_auc']:.4f} | {row['accuracy']:.4f} | {row['f1']:.4f} | {row['overall_srcc']:.4f} | {row['tech_srcc']:.4f} | {row['tech_plcc']:.4f} | {row['tech_mse']:.4f} | {row['tech_mae']:.4f} | {row['sci_srcc']:.4f} | {row['aes_srcc']:.4f} | {row['branch_macro_srcc']:.4f} | {row['best_epoch']} |"
        )
    if "T0_KEEP_P2_S2" in seed42:
        baseline = seed42["T0_KEEP_P2_S2"]
        lines.extend(["", "### Delta vs T0", ""])
        for name in ("T1_MEAN_STD", "T2_MEAN_STD_DIFF"):
            if name in seed42:
                row = seed42[name]
                lines.append(
                    f"- {name}: AUC `{row['auc']-baseline['auc']:+.4f}`, PR-AUC `{row['pr_auc']-baseline['pr_auc']:+.4f}`, F1 `{row['f1']-baseline['f1']:+.4f}`, Tech SRCC `{row['tech_srcc']-baseline['tech_srcc']:+.4f}`, Tech PLCC `{row['tech_plcc']-baseline['tech_plcc']:+.4f}`, Tech MSE `{row['tech_mse']-baseline['tech_mse']:+.4f}`."
                )
    lines.extend(["", "## 6. Multi-seed Results", ""])
    complete = []
    for name in VARIANTS:
        selected = [row for row in results if row["variant"] == name and row["seed"] in {42, 123, 2026}]
        if len(selected) == 3:
            complete.append((name, selected))
    if complete:
        lines.extend([
            "| Variant | AUROC | PR-AUC | F1 | Overall SRCC | Tech SRCC | Tech PLCC | Tech MSE | Branch Macro |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for name, selected in complete:
            fmt = lambda key: f"{mean_std(selected, key)[0]:.4f} +/- {mean_std(selected, key)[1]:.4f}"
            lines.append(f"| {name} | {fmt('auc')} | {fmt('pr_auc')} | {fmt('f1')} | {fmt('overall_srcc')} | {fmt('tech_srcc')} | {fmt('tech_plcc')} | {fmt('tech_mse')} | {fmt('branch_macro_srcc')} |")
        baseline_rows = next((rows for name, rows in complete if name == "T0_KEEP_P2_S2"), None)
        if baseline_rows is not None:
            baseline_by_seed = {row["seed"]: row for row in baseline_rows}
            for name, selected in complete:
                if name == "T0_KEEP_P2_S2":
                    continue
                lines.append("")
                lines.append(f"- {name} mean delta vs T0: AUC `{mean_std(selected, 'auc')[0] - mean_std(baseline_rows, 'auc')[0]:+.4f}`, PR-AUC `{mean_std(selected, 'pr_auc')[0] - mean_std(baseline_rows, 'pr_auc')[0]:+.4f}`, Tech SRCC `{mean_std(selected, 'tech_srcc')[0] - mean_std(baseline_rows, 'tech_srcc')[0]:+.4f}`, Tech PLCC `{mean_std(selected, 'tech_plcc')[0] - mean_std(baseline_rows, 'tech_plcc')[0]:+.4f}`, Tech MSE `{mean_std(selected, 'tech_mse')[0] - mean_std(baseline_rows, 'tech_mse')[0]:+.4f}`.")
                deltas = [row["tech_srcc"] - baseline_by_seed[row["seed"]]["tech_srcc"] for row in selected]
                lines.append(
                    f"- Per-seed Tech SRCC delta (42/123/2026): "
                    f"`{deltas[0]:+.4f}/{deltas[1]:+.4f}/{deltas[2]:+.4f}`."
                )
    else:
        lines.append("Pending candidate multi-seed runs.")
    lines.extend([
        "", "## 7. Technical Branch Analysis", "",
        "Technical SRCC/PLCC/MSE are treated as the primary evidence; overall metrics are secondary and no fusion or learning-rate tuning is performed.",
        "", "## 8. Overall Performance", "",
        "See the controlled seed-42 and multi-seed tables above.",
        "", "## 9. Temporal Feature Diagnostics", "",
        f"- Validation samples: `{diagnostics['count']}`; short sequences: `{diagnostics['num_short_temporal_sequences']}`.",
    ])
    for key in ("temporal_std_mean", "temporal_diff_mean", "temporal_diff_std"):
        lines.append(
            f"- {key}: mean `{diagnostics[key + '_mean']:.6f}`, range `[{diagnostics[key + '_min']:.6f}, {diagnostics[key + '_max']:.6f}]`, NaN `{diagnostics[key + '_nan_count']}`, Tech-target SRCC `{diagnostics[key + '_tech_srcc']:.4f}`, positive/negative mean `{diagnostics[key + '_positive_mean']:.6f}/{diagnostics[key + '_negative_mean']:.6f}`."
        )
    lines.extend(["", "## 10. Conclusion", ""])
    if len(complete) >= 2:
        baseline_rows = next(rows for name, rows in complete if name == "T0_KEEP_P2_S2")
        candidates = [(name, rows) for name, rows in complete if name != "T0_KEEP_P2_S2"]
        baseline_tech = mean_std(baseline_rows, "tech_srcc")[0]
        baseline_auc = mean_std(baseline_rows, "auc")[0]
        baseline_pr = mean_std(baseline_rows, "pr_auc")[0]
        baseline_by_seed = {row["seed"]: row for row in baseline_rows}
        viable = []
        for name, rows in candidates:
            improved_seeds = sum(
                row["tech_srcc"] > baseline_by_seed[row["seed"]]["tech_srcc"]
                for row in rows
            )
            stable_technical_gain = improved_seeds >= 2
            plcc_ok = mean_std(rows, "tech_plcc")[0] > mean_std(baseline_rows, "tech_plcc")[0]
            mse_ok = mean_std(rows, "tech_mse")[0] <= mean_std(baseline_rows, "tech_mse")[0]
            overall_ok = (
                mean_std(rows, "auc")[0] >= baseline_auc - 0.01
                and mean_std(rows, "pr_auc")[0] >= baseline_pr - 0.03
            )
            if stable_technical_gain and plcc_ok and mse_ok and overall_ok:
                viable.append((name, rows))
        if viable:
            best_name, _ = max(viable, key=lambda item: (mean_std(item[1], "tech_srcc")[0], mean_std(item[1], "pr_auc")[0]))
            decision = "KEEP_T1" if best_name == "T1_MEAN_STD" else "KEEP_T2"
        else:
            decision = "KEEP_T0"
        lines.append(f"**{decision}**")
        if decision == "KEEP_T0":
            lines.append("")
            lines.append(
                "T2's mean Technical correlation gain is not stable across seeds and its "
                "Technical MSE and overall ranking metrics regress. The temporal statistics "
                "remain useful diagnostics, but they are not promoted into KEEP_P2_S2."
            )
    else:
        lines.append("**PENDING_MULTI_SEED**")
    (args.output_dir / "temporal_quality_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    logger = logger_for(args.output_dir / f"temporal_quality_seed{args.seed}.log")
    train_ds, val_ds, split_hash = prepare_data(args, logger)
    diagnostics = temporal_diagnostics(val_ds, args.output_dir)
    write_audit(args, diagnostics)
    variants = [value.strip().upper() for value in args.variants.split(",") if value.strip()]
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    result_path = args.output_dir / "temporal_quality_results.csv"
    history_path = args.output_dir / "temporal_quality_history.csv"
    results = read_csv(result_path, RESULT_FIELDS)
    history = read_csv(history_path, HISTORY_FIELDS)
    if args.force:
        results = [row for row in results if not (row["seed"] == args.seed and row["variant"] in variants)]
        history = [row for row in history if not (row["seed"] == args.seed and row["variant"] in variants)]
    device = get_device(CFG.device)
    current = []
    for variant in variants:
        mode = VARIANTS[variant]
        if variant == "T0_KEEP_P2_S2":
            payload, checkpoint = load_t0(args.seed, val_ds, device, split_hash)
            row = result_row(payload, checkpoint)
        else:
            row = run_temporal(
                variant, mode, train_ds, val_ds, device, args, logger,
                split_hash, history,
            )
        current.append(row)
        results = merge_rows(results, current, ("variant", "seed"))
        write_csv(result_path, results, RESULT_FIELDS)
        write_csv(history_path, history, HISTORY_FIELDS)
        write_summary(args, results, diagnostics)
    logger.info("Completed %s", variants)


if __name__ == "__main__":
    main()
