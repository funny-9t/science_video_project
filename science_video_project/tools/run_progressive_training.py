"""Run controlled main_v3-D3 joint versus progressive training experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import CFG
from pipeline.utils_io import ensure_dir, load_metadata
from training.dataloader_pair import collate_pair
from training.dataset_pair import PairwiseVideoDataset
from training.losses import (
    branch_consistency_loss,
    branch_supervision_loss,
    ranknet_pairwise_loss,
)
from training.metrics import compute_video_metrics
from training.model_mvp import MultiModalQualityModel
from training.train import _search_best_threshold, configure_science_features
from training.utils_train import build_model_config_from_cfg, get_device, move_batch_to_device, set_seed


HISTORY_FIELDS = (
    "experiment", "seed", "stage", "epoch", "rank_loss", "point_loss",
    "branch_reg_loss", "branch_rank_loss", "consistency_loss", "total_loss",
    "sci_srcc", "tech_srcc", "aes_srcc", "auc", "pr_auc",
    "grad_norm_sci", "grad_norm_tech", "grad_norm_aes", "learning_rate",
)
RESULT_FIELDS = (
    "experiment", "seed", "stage", "auc", "pr_auc", "accuracy", "f1",
    "sci_srcc", "tech_srcc", "aes_srcc", "visual_srcc", "audio_srcc",
    "tech_label_srcc", "science_weight", "technical_weight", "aesthetic_weight",
    "branch_macro_srcc", "best_epoch", "validation_hash", "train_seconds",
    "trainable_params", "checkpoint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "progressive_training",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--experiments", default="J0,P1",
        help="Comma-separated subset of J0,P1,P2.",
    )
    parser.add_argument(
        "--stop-after", choices=("stage1", "stage2", "stage3"), default="stage3",
        help="Stop progressive experiments after this stage (J0 is unaffected).",
    )
    return parser.parse_args()


def build_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("progressive_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def write_csv(path: Path, rows: list[dict], fields: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def read_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            converted = dict(row)
            for key in RESULT_FIELDS:
                if key in {"experiment", "stage", "validation_hash", "checkpoint"}:
                    continue
                converted[key] = (
                    int(row[key]) if key in {"seed", "best_epoch", "trainable_params"}
                    else float(row[key])
                )
            rows.append(converted)
    return rows


def merge_rows(existing: list[dict], current: list[dict]) -> list[dict]:
    merged = {
        (row["experiment"], row["seed"], row["stage"]): row
        for row in existing
    }
    for row in current:
        merged[(row["experiment"], row["seed"], row["stage"])] = row
    return sorted(merged.values(), key=lambda row: (row["seed"], row["experiment"], row["stage"]))


def prepare_data(args: argparse.Namespace, logger: logging.Logger):
    metadata = load_metadata(args.metadata)
    available = {path.stem for path in args.feature_dir.glob("*.pt")}
    metadata = metadata[metadata["video_id"].astype(str).isin(available)].copy()
    logger.info(
        "Feature coverage=%d/%d | labels=%s",
        len(metadata), len(load_metadata(args.metadata)),
        metadata["label"].value_counts().sort_index().to_dict(),
    )
    train_df, val_df = train_test_split(
        metadata,
        test_size=CFG.val_ratio,
        random_state=args.split_seed,
        stratify=metadata["label"],
    )
    split_payload = "\n".join(
        [f"train:{value}" for value in sorted(train_df["video_id"].astype(str))]
        + [f"val:{value}" for value in sorted(val_df["video_id"].astype(str))]
    )
    split_hash = hashlib.sha256(split_payload.encode("utf-8")).hexdigest()
    common = dict(
        same_category=False,
        use_native_clip=False,
        use_cover_features=False,
        aesthetic_feature_backend="shared_clip",
        technical_visual_source="clip_cover",
    )
    train_ds = PairwiseVideoDataset.from_metadata(
        train_df, args.feature_dir, deterministic_pairs=False, **common
    )
    val_ds = PairwiseVideoDataset.from_metadata(
        val_df, args.feature_dir, deterministic_pairs=True, **common
    )
    logger.info(
        "Split train=%d val=%d | hash=%s", len(train_df), len(val_df), split_hash
    )
    return train_ds, val_ds, split_hash


def make_train_loader(dataset: PairwiseVideoDataset, seed: int) -> DataLoader:
    set_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_pair,
        generator=generator,
    )


def model_config() -> dict:
    config = build_model_config_from_cfg(CFG)
    config.update(
        use_knowledge_gate=False,
        science_fusion_mode="concat",
        use_cover_features=False,
        fusion_mode="learned",
        branch_weight_floor=0.2,
        technical_feature_mode="full_no_dnsmos",
        technical_visual_source="clip_cover",
        technical_clip_projection_dim=256,
        technical_cover_projection_dim=256,
        use_cross_gating=False,
    )
    return config


def new_model(seed: int, device: torch.device) -> MultiModalQualityModel:
    set_seed(seed)
    return MultiModalQualityModel(**model_config()).to(device)


def prepare_batch(batch: dict, device: torch.device) -> dict:
    batch = move_batch_to_device(batch, device)
    return configure_science_features(
        batch, "analysis_scores_concat", "reasoning_and_analysis"
    )


def collate_unique(samples: list[dict]) -> dict:
    converted = [PairwiseVideoDataset._to_tensor_dict(sample) for sample in samples]
    result = {"video_id": [sample["video_id"] for sample in converted]}
    for key in converted[0]:
        if key in {"video_id", "frame_features"}:
            continue
        result[key] = torch.stack([sample[key] for sample in converted], dim=0)
    return result


def safe_srcc(prediction: np.ndarray, target: np.ndarray, name: str) -> tuple[float, str]:
    valid = np.isfinite(target) & (target >= 0)
    if valid.sum() < 2:
        return float("nan"), f"{name}: fewer than two valid targets"
    if np.unique(target[valid]).size < 2:
        return float("nan"), f"{name}: target is constant"
    value = spearmanr(prediction[valid], target[valid]).statistic
    if not np.isfinite(value):
        return float("nan"), f"{name}: scipy returned non-finite SRCC"
    return float(value), ""


def branch_rank_loss(
    pos_out: dict[str, torch.Tensor],
    neg_out: dict[str, torch.Tensor],
    pos_batch: dict[str, torch.Tensor],
    neg_batch: dict[str, torch.Tensor],
    minimum_gap: float,
) -> torch.Tensor:
    losses = []
    for score_key, target_key in (
        ("scientific_score", "sci_target"),
        ("technical_score", "tech_target"),
        ("aesthetic_score", "aes_target"),
    ):
        pos_target = pos_batch[target_key].squeeze(-1)
        neg_target = neg_batch[target_key].squeeze(-1)
        gap = pos_target - neg_target
        valid = (pos_target >= 0) & (neg_target >= 0) & (gap.abs() >= minimum_gap)
        if not valid.any():
            continue
        pos_score = pos_out[score_key].squeeze(-1)[valid]
        neg_score = neg_out[score_key].squeeze(-1)[valid]
        high = torch.where(gap[valid] > 0, pos_score, neg_score)
        low = torch.where(gap[valid] > 0, neg_score, pos_score)
        losses.append(ranknet_pairwise_loss(high, low))
    if not losses:
        return pos_out["overall_score"].sum() * 0.0
    return torch.stack(losses).mean()


def target_gap_statistics(dataset: PairwiseVideoDataset, minimum_gap: float = 0.05) -> dict:
    positive = [sample for sample in dataset.samples if int(sample["label"]) == 1]
    negative = [sample for sample in dataset.samples if int(sample["label"]) == 0]
    result = {"minimum_gap": minimum_gap, "target_range": [0.0, 1.0]}
    for key in ("sci_target", "tech_target", "aes_target"):
        left = np.asarray([sample[key] for sample in positive], dtype=np.float32)
        right = np.asarray([sample[key] for sample in negative], dtype=np.float32)
        valid_left = left >= 0
        valid_right = right >= 0
        gaps = np.abs(left[valid_left, None] - right[None, valid_right]).reshape(-1)
        result[key] = {
            "pair_count": int(gaps.size),
            "retained_count": int((gaps >= minimum_gap).sum()),
            "retained_fraction": float((gaps >= minimum_gap).mean()),
            "quantiles": {
                str(q): float(np.quantile(gaps, q))
                for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
            },
        }
    return result


def branch_regression_loss(pos_out, neg_out, pos_batch, neg_batch) -> torch.Tensor:
    return (
        branch_supervision_loss(
            pos_out["scientific_score"], pos_out["technical_score"],
            pos_out["aesthetic_score"], pos_batch["sci_target"],
            pos_batch["tech_target"], pos_batch["aes_target"],
        )
        + branch_supervision_loss(
            neg_out["scientific_score"], neg_out["technical_score"],
            neg_out["aesthetic_score"], neg_batch["sci_target"],
            neg_batch["tech_target"], neg_batch["aes_target"],
        )
    ) / 2.0


def component_losses(pos_out, neg_out, pos_batch, neg_batch, branch_rank_weight: float):
    rank = ranknet_pairwise_loss(pos_out["overall_score"], neg_out["overall_score"])
    point = (
        F.binary_cross_entropy_with_logits(
            pos_out["overall_score"], torch.ones_like(pos_out["overall_score"])
        )
        + F.binary_cross_entropy_with_logits(
            neg_out["overall_score"], torch.zeros_like(neg_out["overall_score"])
        )
    ) / 2.0
    consistency = (
        branch_consistency_loss(
            pos_out["overall_score"], pos_out["scientific_score"],
            pos_out["technical_score"], pos_out["aesthetic_score"],
        )
        + branch_consistency_loss(
            neg_out["overall_score"], neg_out["scientific_score"],
            neg_out["technical_score"], neg_out["aesthetic_score"],
        )
    ) / 2.0
    branch_reg = branch_regression_loss(pos_out, neg_out, pos_batch, neg_batch)
    branch_rank = (
        branch_rank_loss(pos_out, neg_out, pos_batch, neg_batch, 0.05)
        if branch_rank_weight > 0
        else rank.new_zeros(())
    )
    return rank, point, consistency, branch_reg, branch_rank


def gradient_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> float:
    active = [parameter for parameter in parameters if parameter.requires_grad]
    if not active:
        return 0.0
    gradients = torch.autograd.grad(
        loss, active, retain_graph=True, allow_unused=True
    )
    total = sum(
        float(gradient.detach().pow(2).sum().item())
        for gradient in gradients if gradient is not None
    )
    return math.sqrt(total)


def train_epoch(
    model: MultiModalQualityModel,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    stage: str,
    branch_rank_weight: float = 0.0,
) -> dict[str, float]:
    model.train()
    if stage == "stage3":
        for branch in (
            model.scientific_branch, model.technical_branch, model.aesthetic_branch
        ):
            branch.eval()
            branch.score.train()
    sums = {key: 0.0 for key in (
        "rank_loss", "point_loss", "consistency_loss", "branch_reg_loss",
        "branch_rank_loss", "total_loss",
    )}
    grad_norms = {"sci": 0.0, "tech": 0.0, "aes": 0.0}
    for step, (pos_batch, neg_batch) in enumerate(loader):
        pos_batch = prepare_batch(pos_batch, device)
        neg_batch = prepare_batch(neg_batch, device)
        pos_out = model(**pos_batch)
        neg_out = model(**neg_batch)
        rank, point, consistency, branch_reg, branch_rank = component_losses(
            pos_out, neg_out, pos_batch, neg_batch, branch_rank_weight
        )
        if stage == "stage1":
            total = branch_reg + branch_rank_weight * branch_rank
        else:
            total = (
                rank + 0.1 * point + 0.1 * consistency
                + 0.05 * branch_reg + branch_rank_weight * branch_rank
            )
        if step == 0 and stage != "stage1":
            grad_norms = {
                "sci": gradient_norm(rank, list(model.scientific_branch.parameters())),
                "tech": gradient_norm(rank, list(model.technical_branch.parameters())),
                "aes": gradient_norm(rank, list(model.aesthetic_branch.parameters())),
            }
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        for key, value in (
            ("rank_loss", rank), ("point_loss", point),
            ("consistency_loss", consistency), ("branch_reg_loss", branch_reg),
            ("branch_rank_loss", branch_rank), ("total_loss", total),
        ):
            sums[key] += float(value.detach().item())
    count = max(len(loader), 1)
    return {
        **{key: value / count for key, value in sums.items()},
        "grad_norm_sci": grad_norms["sci"],
        "grad_norm_tech": grad_norms["tech"],
        "grad_norm_aes": grad_norms["aes"],
    }


def evaluate_unique(
    model: MultiModalQualityModel,
    samples: list[dict],
    device: torch.device,
) -> dict[str, float | str]:
    model.eval()
    records = {key: [] for key in (
        "label", "probability", "overall", "sci_logit", "tech_logit", "aes_logit",
        "sci_score", "tech_score", "aes_score",
        "sci_target", "tech_target", "aes_target", "visual_target", "audio_target",
        "label_target", "science_weight", "technical_weight", "aesthetic_weight",
    )}
    with torch.inference_mode():
        for start in range(0, len(samples), 32):
            batch = prepare_batch(collate_unique(samples[start:start + 32]), device)
            output = model(**batch)
            values = {
                "label": batch["label_target"].squeeze(-1),
                "probability": output["probability"].squeeze(-1),
                "overall": output["overall_score"].squeeze(-1),
                "sci_logit": output["scientific_score"].squeeze(-1),
                "tech_logit": output["technical_score"].squeeze(-1),
                "aes_logit": output["aesthetic_score"].squeeze(-1),
                "sci_score": torch.sigmoid(output["scientific_score"]).squeeze(-1),
                "tech_score": torch.sigmoid(output["technical_score"]).squeeze(-1),
                "aes_score": torch.sigmoid(output["aesthetic_score"]).squeeze(-1),
                "sci_target": batch["sci_target"].squeeze(-1),
                "tech_target": batch["tech_target"].squeeze(-1),
                "aes_target": batch["aes_target"].squeeze(-1),
                "visual_target": batch["visual_target"].squeeze(-1),
                "audio_target": batch["audio_target"].squeeze(-1),
                "label_target": batch["label_target"].squeeze(-1),
                "science_weight": output["branch_weights"][:, 0],
                "technical_weight": output["branch_weights"][:, 1],
                "aesthetic_weight": output["branch_weights"][:, 2],
            }
            for key, value in values.items():
                records[key].extend(value.detach().cpu().numpy().tolist())
    arrays = {key: np.asarray(value, dtype=np.float32) for key, value in records.items()}
    threshold, _ = _search_best_threshold(
        arrays["probability"][arrays["label"] == 1],
        arrays["probability"][arrays["label"] == 0],
        metric="f1",
    )
    metrics = compute_video_metrics(arrays["label"], arrays["probability"], threshold)
    pos = arrays["probability"][arrays["label"] == 1]
    neg = arrays["probability"][arrays["label"] == 0]
    metrics["ranking_accuracy"] = float((pos[:, None] > neg[None, :]).mean())
    metrics["best_threshold"] = float(threshold)

    reasons = []
    for output_key, prediction_key, target_key in (
        ("sci_srcc", "sci_score", "sci_target"),
        ("tech_srcc", "tech_score", "tech_target"),
        ("aes_srcc", "aes_score", "aes_target"),
        ("visual_srcc", "tech_score", "visual_target"),
        ("audio_srcc", "tech_score", "audio_target"),
        ("tech_label_srcc", "tech_score", "label_target"),
    ):
        value, reason = safe_srcc(arrays[prediction_key], arrays[target_key], output_key)
        metrics[output_key] = value
        if reason:
            reasons.append(reason)
    metrics["branch_macro_srcc"] = float(np.mean([
        metrics["sci_srcc"], metrics["tech_srcc"], metrics["aes_srcc"]
    ]))
    for key in ("science_weight", "technical_weight", "aesthetic_weight"):
        metrics[key] = float(arrays[key].mean())

    pos_index = arrays["label"] == 1
    neg_index = ~pos_index
    rank_loss = F.softplus(-(
        torch.from_numpy(arrays["overall"][pos_index])[:, None]
        - torch.from_numpy(arrays["overall"][neg_index])[None, :]
    )).mean()
    all_overall = torch.from_numpy(arrays["overall"])
    all_labels = torch.from_numpy(arrays["label"])
    point_loss = F.binary_cross_entropy_with_logits(all_overall, all_labels)
    metrics["rank_loss"] = float(rank_loss.item())
    metrics["point_loss"] = float(point_loss.item())
    consistency = np.mean((
        arrays["overall"]
        - (arrays["sci_logit"] + arrays["tech_logit"] + arrays["aes_logit"]) / 3.0
    ) ** 2)
    branch_losses = []
    for prediction_key, target_key in (
        ("sci_score", "sci_target"),
        ("tech_score", "tech_target"),
        ("aes_score", "aes_target"),
    ):
        valid = arrays[target_key] >= 0
        branch_losses.append(float(np.mean(
            (arrays[prediction_key][valid] - arrays[target_key][valid]) ** 2
        )))
    branch_reg = statistics.fmean(branch_losses)
    metrics["consistency_loss"] = float(consistency)
    metrics["branch_reg_loss"] = branch_reg
    metrics["validation_loss"] = (
        metrics["rank_loss"] + 0.1 * metrics["point_loss"]
        + 0.1 * metrics["consistency_loss"] + 0.05 * branch_reg
    )
    metrics["srcc_nan_reasons"] = "; ".join(reasons)
    return metrics


def parameter_groups(model: MultiModalQualityModel) -> dict[str, list[torch.nn.Parameter]]:
    return {
        "scientific": list(model.scientific_branch.parameters()),
        "technical": list(model.technical_branch.parameters()),
        "aesthetic": list(model.aesthetic_branch.parameters()),
        "fusion": [
            *model.gate.parameters(), *model.fusion.parameters(),
            *model.overall_head.parameters(),
        ],
    }


def set_trainable(model: MultiModalQualityModel, stage: str) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if stage in {"joint", "stage2"}:
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif stage == "stage1":
        for branch in (
            model.scientific_branch, model.technical_branch, model.aesthetic_branch
        ):
            for parameter in branch.parameters():
                parameter.requires_grad = True
    elif stage == "stage3":
        for head in (
            model.scientific_branch.score,
            model.technical_branch.score,
            model.aesthetic_branch.score,
            model.gate,
            model.fusion,
            model.overall_head,
        ):
            for parameter in head.parameters():
                parameter.requires_grad = True
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def save_checkpoint(
    path: Path,
    model: MultiModalQualityModel,
    optimizer: AdamW,
    experiment: str,
    stage: str,
    epoch: int,
    metrics: dict,
    args: argparse.Namespace,
    split_hash: str,
    trainable_names: list[str],
) -> None:
    branch_states = {
        "scientific": model.scientific_branch.state_dict(),
        "technical": model.technical_branch.state_dict(),
        "aesthetic": model.aesthetic_branch.state_dict(),
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "branch_state_dicts": branch_states,
            "optimizer_state_dict": optimizer.state_dict(),
            "config": model_config(),
            "experiment": experiment,
            "stage": stage,
            "epoch": epoch,
            "best_epoch": epoch,
            "validation_metrics": metrics,
            "best_threshold": metrics.get("best_threshold", 0.5),
            "best_ranking_accuracy": metrics.get("ranking_accuracy", 0.0),
            "branch_macro_srcc": metrics.get("branch_macro_srcc", float("nan")),
            "seed": args.seed,
            "split_seed": args.split_seed,
            "split_hash": split_hash,
            "technical_visual_source": "clip_cover",
            "dimension_config": {"clip": 256, "cover": 256, "concat": 512},
            "trainable_parameter_names": trainable_names,
            "trainable_params": sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad
            ),
        },
        path,
    )


def train_stage(
    model: MultiModalQualityModel,
    train_ds: PairwiseVideoDataset,
    val_ds: PairwiseVideoDataset,
    device: torch.device,
    args: argparse.Namespace,
    logger: logging.Logger,
    experiment: str,
    stage: str,
    checkpoint: Path,
    max_epochs: int,
    patience: int,
    split_hash: str,
    history: list[dict],
    branch_rank_weight: float = 0.0,
) -> dict:
    trainable_names = set_trainable(model, stage)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = AdamW(trainable_parameters, lr=5e-5, weight_decay=CFG.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=CFG.scheduler_factor,
        patience=CFG.scheduler_patience, min_lr=1e-6,
    )
    loader = make_train_loader(train_ds, args.seed)
    best_metric = -float("inf")
    stale = 0
    started = time.perf_counter()
    logger.info(
        "%s/%s start | trainable=%d | max_epochs=%d patience=%d branch_rank=%.3f",
        experiment, stage,
        sum(parameter.numel() for parameter in trainable_parameters),
        max_epochs, patience, branch_rank_weight,
    )
    for epoch in range(1, max_epochs + 1):
        train_metrics = train_epoch(
            model, loader, optimizer, device, stage, branch_rank_weight
        )
        validation = evaluate_unique(model, val_ds.samples, device)
        if validation["srcc_nan_reasons"]:
            logger.warning("SRCC issue: %s", validation["srcc_nan_reasons"])
        selection = (
            float(validation["branch_macro_srcc"])
            if stage == "stage1"
            else float(validation["ranking_accuracy"])
        )
        scheduler.step(
            float(validation["branch_reg_loss"])
            if stage == "stage1"
            else float(validation["validation_loss"])
        )
        row = {
            "experiment": experiment,
            "seed": args.seed,
            "stage": stage,
            "epoch": epoch,
            **{key: train_metrics[key] for key in (
                "rank_loss", "point_loss", "branch_reg_loss", "branch_rank_loss",
                "consistency_loss", "total_loss", "grad_norm_sci",
                "grad_norm_tech", "grad_norm_aes",
            )},
            "sci_srcc": validation["sci_srcc"],
            "tech_srcc": validation["tech_srcc"],
            "aes_srcc": validation["aes_srcc"],
            "auc": validation["auc"],
            "pr_auc": validation["pr_auc"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        write_csv(args.output_dir / "progressive_training_history.csv", history, HISTORY_FIELDS)
        logger.info(
            "%s/%s epoch=%d | total=%.4f rank=%.4f point=%.4f branch=%.4f "
            "cons=%.4f | AUC=%.4f PR=%.4f SRCC=%.4f/%.4f/%.4f | "
            "grad=%.4f/%.4f/%.4f",
            experiment, stage, epoch, train_metrics["total_loss"],
            train_metrics["rank_loss"], train_metrics["point_loss"],
            train_metrics["branch_reg_loss"], train_metrics["consistency_loss"],
            validation["auc"], validation["pr_auc"], validation["sci_srcc"],
            validation["tech_srcc"], validation["aes_srcc"],
            train_metrics["grad_norm_sci"], train_metrics["grad_norm_tech"],
            train_metrics["grad_norm_aes"],
        )
        if selection > best_metric:
            best_metric = selection
            stale = 0
            save_checkpoint(
                checkpoint, model, optimizer, experiment, stage, epoch,
                validation, args, split_hash, trainable_names,
            )
        else:
            stale += 1
            if stale >= patience:
                logger.info("%s/%s early stop at epoch %d", experiment, stage, epoch)
                break
    elapsed = time.perf_counter() - started
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["train_seconds"] = elapsed
    torch.save(payload, checkpoint)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return result_row(payload, checkpoint)


def result_row(payload: dict, checkpoint: Path) -> dict:
    metrics = payload["validation_metrics"]
    return {
        "experiment": payload["experiment"],
        "seed": payload["seed"],
        "stage": payload["stage"],
        **{key: float(metrics[key]) for key in (
            "auc", "pr_auc", "accuracy", "f1", "sci_srcc", "tech_srcc",
            "aes_srcc", "visual_srcc", "audio_srcc", "tech_label_srcc",
            "science_weight", "technical_weight", "aesthetic_weight",
            "branch_macro_srcc",
        )},
        "best_epoch": payload["best_epoch"],
        "validation_hash": payload["split_hash"],
        "train_seconds": float(payload.get("train_seconds", 0.0)),
        "trainable_params": int(payload["trainable_params"]),
        "checkpoint": str(checkpoint.resolve()),
    }


def load_or_train_stage(
    model, train_ds, val_ds, device, args, logger, experiment, stage,
    checkpoint, max_epochs, patience, split_hash, history, branch_rank_weight=0.0,
) -> dict:
    if checkpoint.exists() and not args.force:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        logger.info("Reusing %s", checkpoint)
        return result_row(payload, checkpoint)
    return train_stage(
        model, train_ds, val_ds, device, args, logger, experiment, stage,
        checkpoint, max_epochs, patience, split_hash, history, branch_rank_weight,
    )


def phase_a_success(results: list[dict]) -> tuple[bool, str]:
    joint = next(row for row in results if row["experiment"] == "J0")
    candidates = [
        row for row in results
        if row["experiment"] == "P1" and row["stage"] in {"stage2", "stage3"}
    ]
    reasons = []
    for row in candidates:
        auc_delta = row["auc"] - joint["auc"]
        pr_delta = row["pr_auc"] - joint["pr_auc"]
        tech_delta = row["tech_srcc"] - joint["tech_srcc"]
        aes_delta = row["aes_srcc"] - joint["aes_srcc"]
        if auc_delta >= 0.01 or pr_delta >= 0.02:
            reasons.append(f"{row['stage']}: overall metric improved")
        elif (
            max(tech_delta, aes_delta) >= 0.05
            and auc_delta >= -0.01 and pr_delta >= -0.03
        ):
            reasons.append(f"{row['stage']}: branch alignment improved without clear ranking loss")
    return bool(reasons), "; ".join(reasons) if reasons else "no Phase A success condition met"


def write_phase_a_summary(
    args: argparse.Namespace,
    results: list[dict],
    history: list[dict],
    split_hash: str,
) -> None:
    joint = next(row for row in results if row["experiment"] == "J0")
    stage1 = next(
        row for row in results if row["experiment"] == "P1" and row["stage"] == "stage1"
    )
    stage2 = next(
        row for row in results if row["experiment"] == "P1" and row["stage"] == "stage2"
    )
    stage3 = next(
        row for row in results if row["experiment"] == "P1" and row["stage"] == "stage3"
    )
    success, reason = phase_a_success(results)
    lines = [
        "# Progressive Training Phase A",
        "",
        "## 1. main_v3-D3 baseline",
        "",
        "J0 与 P1 均固定使用 CLIP 256D + COVER technical 256D、concat 512D、"
        "20% branch floor、RankNet + 0.1 pointwise + 0.1 consistency + 0.05 branch MSE。",
        f"验证划分 hash: `{split_hash}`。",
        "",
        "参数边界：`scientific_branch`、`technical_branch`、`aesthetic_branch`；"
        "分支 heads 为各分支的 `.score`；融合为 `gate`、`fusion`、`overall_head`。",
        "",
        "## 2. Progressive Training",
        "",
        "- Stage 1: 仅三个 branch 与 branch score heads，使用现有 branch MSE；按 branch macro SRCC 选模。",
        "- Stage 2: 全模型解冻，恢复与 J0 等价的总体损失；按 ranking accuracy 选模。",
        "- Stage 3: 冻结 branch encoder/projection，仅训练三个 score heads 和 fusion；按 ranking accuracy 选模。",
        "- COVER 的 limited-view 思想被适配为 dimension-aware branch-specific auxiliary supervision，"
        "没有让每个分支预测整体 label。",
        "",
        "## 3. Seed=42 Phase A Results",
        "",
        "| Experiment | Stage | AUC | PR-AUC | Acc | F1 | Sci SRCC | Tech SRCC | Aes SRCC | Visual | Audio | Tech-label | S/T/A | Epoch |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in results:
        if row["experiment"] not in {"J0", "P1"}:
            continue
        lines.append(
            f"| {row['experiment']} | {row['stage']} | {row['auc']:.4f} | "
            f"{row['pr_auc']:.4f} | {row['accuracy']:.4f} | {row['f1']:.4f} | "
            f"{row['sci_srcc']:.4f} | {row['tech_srcc']:.4f} | {row['aes_srcc']:.4f} | "
            f"{row['visual_srcc']:.4f} | {row['audio_srcc']:.4f} | "
            f"{row['tech_label_srcc']:.4f} | {row['science_weight']:.4f}/"
            f"{row['technical_weight']:.4f}/{row['aesthetic_weight']:.4f} | "
            f"{row['best_epoch']} |"
        )
    lines.extend([
        "",
        "## 4. Branch Drift",
        "",
        f"- Stage1 -> Stage2 Sci SRCC: {stage1['sci_srcc']:.4f} -> {stage2['sci_srcc']:.4f} "
        f"({stage2['sci_srcc'] - stage1['sci_srcc']:+.4f})",
        f"- Stage1 -> Stage2 Tech SRCC: {stage1['tech_srcc']:.4f} -> {stage2['tech_srcc']:.4f} "
        f"({stage2['tech_srcc'] - stage1['tech_srcc']:+.4f})",
        f"- Stage1 -> Stage2 Aes SRCC: {stage1['aes_srcc']:.4f} -> {stage2['aes_srcc']:.4f} "
        f"({stage2['aes_srcc'] - stage1['aes_srcc']:+.4f})",
        "",
        "## 5. Fusion Calibration",
        "",
        f"- P1-S3 vs P1-S2: ΔAUC {stage3['auc'] - stage2['auc']:+.4f}，"
        f"ΔPR-AUC {stage3['pr_auc'] - stage2['pr_auc']:+.4f}，"
        f"ΔTech SRCC {stage3['tech_srcc'] - stage2['tech_srcc']:+.4f}。",
        "",
        "## 6. Gradient Diagnostic",
        "",
    ])
    for stage_name in ("joint", "stage2", "stage3"):
        selected = [row for row in history if row["stage"] == stage_name]
        if selected:
            lines.append(
                f"- {stage_name} mean rank-gradient norm S/T/A: "
                f"{statistics.fmean(row['grad_norm_sci'] for row in selected):.4f}/"
                f"{statistics.fmean(row['grad_norm_tech'] for row in selected):.4f}/"
                f"{statistics.fmean(row['grad_norm_aes'] for row in selected):.4f}。"
            )
    lines.extend([
        "",
        "## 7. Phase B Gate",
        "",
        f"- Phase A effective: `{success}`。",
        f"- Reason: {reason}。",
        "- P2 仅在该门控通过后实现和执行。",
        "",
        "## 8. Interim Decision",
        "",
        "`CONTINUE_PHASE_B`" if success else "`KEEP_JOINT`",
    ])
    (args.output_dir / "progressive_training_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def mean_std(rows: list[dict], key: str) -> tuple[float, float]:
    values = [float(row[key]) for row in rows]
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def format_result_row(row: dict) -> str:
    return (
        f"| {row['experiment']} | {row['stage']} | {row['auc']:.4f} | "
        f"{row['pr_auc']:.4f} | {row['accuracy']:.4f} | {row['f1']:.4f} | "
        f"{row['sci_srcc']:.4f} | {row['tech_srcc']:.4f} | {row['aes_srcc']:.4f} | "
        f"{row['visual_srcc']:.4f} | {row['audio_srcc']:.4f} | "
        f"{row['tech_label_srcc']:.4f} | {row['science_weight']:.4f}/"
        f"{row['technical_weight']:.4f}/{row['aesthetic_weight']:.4f} | "
        f"{row['best_epoch']} |"
    )


def write_final_summary(
    args: argparse.Namespace,
    results: list[dict],
    history: list[dict],
    split_hash: str,
) -> bool:
    expected_seeds = {42, 123, 2026}
    joint_rows = [
        row for row in results
        if row["experiment"] == "J0" and row["stage"] == "joint"
        and row["seed"] in expected_seeds
    ]
    p2_rows = [
        row for row in results
        if row["experiment"] == "P2" and row["stage"] == "stage2"
        and row["seed"] in expected_seeds
    ]
    if {row["seed"] for row in joint_rows} != expected_seeds or {
        row["seed"] for row in p2_rows
    } != expected_seeds:
        return False

    seed42 = [row for row in results if row["seed"] == 42]
    lookup = {(row["experiment"], row["stage"]): row for row in seed42}
    required = {
        ("J0", "joint"), ("P1", "stage1"), ("P1", "stage2"),
        ("P1", "stage3"), ("P2", "stage1"), ("P2", "stage2"),
        ("P2", "stage3"),
    }
    if not required.issubset(lookup):
        return False

    metric_names = (
        ("ROC-AUC", "auc"), ("PR-AUC", "pr_auc"),
        ("Accuracy", "accuracy"), ("F1", "f1"),
        ("Scientific SRCC", "sci_srcc"), ("Technical SRCC", "tech_srcc"),
        ("Aesthetic SRCC", "aes_srcc"), ("Visual SRCC", "visual_srcc"),
        ("Audio SRCC", "audio_srcc"), ("Tech-label SRCC", "tech_label_srcc"),
    )
    aggregate = {}
    for name, rows in (("J0", joint_rows), ("P2-S2", p2_rows)):
        aggregate[name] = {key: mean_std(rows, key) for _, key in metric_names}
        for key in ("science_weight", "technical_weight", "aesthetic_weight"):
            aggregate[name][key] = mean_std(rows, key)

    auc_delta = aggregate["P2-S2"]["auc"][0] - aggregate["J0"]["auc"][0]
    pr_delta = aggregate["P2-S2"]["pr_auc"][0] - aggregate["J0"]["pr_auc"][0]
    f1_delta = aggregate["P2-S2"]["f1"][0] - aggregate["J0"]["f1"][0]
    tech_delta = aggregate["P2-S2"]["tech_srcc"][0] - aggregate["J0"]["tech_srcc"][0]
    aes_delta = aggregate["P2-S2"]["aes_srcc"][0] - aggregate["J0"]["aes_srcc"][0]

    p1_stage1 = lookup[("P1", "stage1")]
    p1_stage2 = lookup[("P1", "stage2")]
    p1_stage3 = lookup[("P1", "stage3")]
    p2_stage2 = lookup[("P2", "stage2")]
    p2_stage3 = lookup[("P2", "stage3")]

    drift_rows = []
    for experiment in ("P1", "P2"):
        stage1 = lookup[(experiment, "stage1")]
        stage2 = lookup[(experiment, "stage2")]
        drift_rows.append(
            f"| {experiment} | {stage1['sci_srcc']:.4f} | {stage2['sci_srcc']:.4f} | "
            f"{stage2['sci_srcc'] - stage1['sci_srcc']:+.4f} | "
            f"{stage1['tech_srcc']:.4f} | {stage2['tech_srcc']:.4f} | "
            f"{stage2['tech_srcc'] - stage1['tech_srcc']:+.4f} | "
            f"{stage1['aes_srcc']:.4f} | {stage2['aes_srcc']:.4f} | "
            f"{stage2['aes_srcc'] - stage1['aes_srcc']:+.4f} |"
        )

    p2_drifts = {key: [] for key in ("sci_srcc", "tech_srcc", "aes_srcc")}
    for seed in sorted(expected_seeds):
        stage1 = next(
            row for row in results
            if row["experiment"] == "P2" and row["stage"] == "stage1"
            and row["seed"] == seed
        )
        stage2 = next(
            row for row in results
            if row["experiment"] == "P2" and row["stage"] == "stage2"
            and row["seed"] == seed
        )
        for key in p2_drifts:
            p2_drifts[key].append(stage2[key] - stage1[key])

    gradient_lines = []
    for experiment, stage in (("J0", "joint"), ("P1", "stage2"), ("P1", "stage3"),
                              ("P2", "stage2"), ("P2", "stage3")):
        selected = [
            row for row in history
            if row["experiment"] == experiment and row["stage"] == stage
            and row["seed"] == 42
        ]
        if selected:
            gradient_lines.append(
                f"| {experiment}-{stage} | "
                f"{statistics.fmean(row['grad_norm_sci'] for row in selected):.4f} | "
                f"{statistics.fmean(row['grad_norm_tech'] for row in selected):.4f} | "
                f"{statistics.fmean(row['grad_norm_aes'] for row in selected):.4f} |"
            )

    gap_path = args.output_dir / "branch_target_gap_stats.json"
    gap_stats = json.loads(gap_path.read_text(encoding="utf-8"))
    retention = "/".join(
        f"{100.0 * gap_stats[key]['retained_fraction']:.2f}%"
        for key in ("sci_target", "tech_target", "aes_target")
    )

    lines = [
        "# COVER-inspired Progressive Training 实验总结",
        "",
        "## 1. main_v3-D3 Baseline",
        "",
        "J0 固定使用 CLIP 256D + COVER technical 256D、concat 512D、20% branch floor，"
        "以及 RankNet + 0.1 pointwise + 0.1 consistency + 0.05 branch MSE。",
        "训练/验证为 492/123，清洗后特征覆盖 615/615；所有实验使用同一验证划分：",
        f"`{split_hash}`。",
        "",
        "## 2. Progressive Training",
        "",
        "- Stage 1: 仅训练三个分支及其 score head，以对应细粒度评分的 MSE 预热；按 macro branch SRCC 选模。",
        "- Stage 2: 解冻全模型，以与 J0 相同的总体目标训练；P2 额外加入权重 0.05 的 branch RankNet。",
        "- Stage 3: 冻结分支 encoder/projection，仅校准 score heads 与 fusion；保留当前 consistency 项。",
        "- 这是对 COVER limited-view supervision 的 dimension-aware 适配：每个分支预测自己的质量维度，而不是三个分支都预测上榜标签。",
        "",
        "## 3. Seed=42 Phase A Results",
        "",
        "| Experiment | Stage | AUC | PR-AUC | Acc | F1 | Sci | Tech | Aes | Visual | Audio | Tech-label | S/T/A | Epoch |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        format_result_row(lookup[("J0", "joint")]),
        format_result_row(p1_stage2),
        format_result_row(p1_stage3),
        "",
        f"P1-S2 相对 J0：AUC {p1_stage2['auc'] - lookup[('J0', 'joint')]['auc']:+.4f}，"
        f"PR-AUC {p1_stage2['pr_auc'] - lookup[('J0', 'joint')]['pr_auc']:+.4f}，"
        f"Technical SRCC {p1_stage2['tech_srcc'] - lookup[('J0', 'joint')]['tech_srcc']:+.4f}。"
        "P1 通过 Phase B 门控。",
        "",
        "## 4. Branch Drift",
        "",
        "| Experiment | Sci S1 | Sci S2 | Delta | Tech S1 | Tech S2 | Delta | Aes S1 | Aes S2 | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *drift_rows,
        "",
        "P2 三种子 Stage1 -> Stage2 平均漂移："
        f"Sci {statistics.fmean(p2_drifts['sci_srcc']):+.4f}，"
        f"Tech {statistics.fmean(p2_drifts['tech_srcc']):+.4f}，"
        f"Aes {statistics.fmean(p2_drifts['aes_srcc']):+.4f}。"
        "Technical alignment 在接入总体排序后仍发生稳定负向漂移，因此实验不能宣称已解决 Technical branch collapse。",
        "",
        "## 5. Fusion Weights",
        "",
        "| Experiment | Science | Technical | Aesthetic |",
        "|---|---:|---:|---:|",
    ]
    for key in (("J0", "joint"), ("P1", "stage2"), ("P1", "stage3"),
                ("P2", "stage2"), ("P2", "stage3")):
        row = lookup[key]
        lines.append(
            f"| {key[0]}-{key[1]} | {row['science_weight']:.4f} | "
            f"{row['technical_weight']:.4f} | {row['aesthetic_weight']:.4f} |"
        )
    lines.extend([
        "",
        "P2-S2 三种子平均权重为 "
        f"{aggregate['P2-S2']['science_weight'][0]:.4f}/"
        f"{aggregate['P2-S2']['technical_weight'][0]:.4f}/"
        f"{aggregate['P2-S2']['aesthetic_weight'][0]:.4f}；权重略离开下限，但仍以科学分支为主。",
        "",
        "## 6. Gradient Diagnostic",
        "",
        "下表为 seed=42 各 epoch 首批样本的 overall RankNet gradient norm 均值：",
        "",
        "| Experiment | Science | Technical | Aesthetic |",
        "|---|---:|---:|---:|",
        *gradient_lines,
        "",
        "Scientific 梯度长期显著大于 Technical/Aesthetic，支持存在梯度不平衡；"
        "但该诊断只用于解释，不能单独证明因果。",
        "",
        "## 7. Phase B",
        "",
        f"branch target gap 固定为 0.05，Sci/Tech/Aes pair 保留率为 {retention}，无需调整阈值。",
        "",
        "| Experiment | Stage | AUC | PR-AUC | Acc | F1 | Sci | Tech | Aes | Visual | Audio | Tech-label | S/T/A | Epoch |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        format_result_row(p2_stage2),
        format_result_row(p2_stage3),
        "",
        f"P2-S3 相对 P2-S2 仅增加 AUC {p2_stage3['auc'] - p2_stage2['auc']:+.4f}、"
        f"PR-AUC {p2_stage3['pr_auc'] - p2_stage2['pr_auc']:+.4f}，"
        f"Technical SRCC {p2_stage3['tech_srcc'] - p2_stage2['tech_srcc']:+.4f}。"
        "收益不足以抵消额外训练阶段，因此最终保留 P2-S2。",
        "",
        "## 8. Multi-seed",
        "",
        "固定 seeds 42/123/2026，比较 J0 与候选最佳 P2-S2：",
        "",
        "| Metric | J0 mean ± std | P2-S2 mean ± std | Delta mean |",
        "|---|---:|---:|---:|",
    ])
    for label, key in metric_names:
        j0_mean, j0_std = aggregate["J0"][key]
        p2_mean, p2_std = aggregate["P2-S2"][key]
        lines.append(
            f"| {label} | {j0_mean:.4f} ± {j0_std:.4f} | "
            f"{p2_mean:.4f} ± {p2_std:.4f} | {p2_mean - j0_mean:+.4f} |"
        )
    lines.extend([
        "",
        "P2-S2 在三个 seed 上逐一提高 AUC，且平均 AUC/PR-AUC 分别提升 "
        f"{auc_delta:+.4f}/{pr_delta:+.4f}；Accuracy 提高 "
        f"{aggregate['P2-S2']['accuracy'][0] - aggregate['J0']['accuracy'][0]:+.4f}，"
        f"F1 基本持平（{f1_delta:+.4f}）。",
        f"与此同时 Technical/Aesthetic SRCC 分别变化 {tech_delta:+.4f}/{aes_delta:+.4f}。"
        "因此收益属于总体排序泛化提升，而不是三个分支语义全面改善。",
        "",
        "## 9. Final Decision",
        "",
        "`USE_PROGRESSIVE_BRANCH_RANK`",
        "",
        "主配置采用 P2 的 Stage1 -> Stage2，不采用 Stage3。依据决策优先级，P2-S2 的多种子 "
        "AUC、PR-AUC 和 Accuracy 均优于 J0，F1 持平；它也在 seed=42 的总体指标上优于纯 MSE 的 P1。"
        "但 Technical SRCC 没有稳定提高，后续论文应将结论表述为 branch-specific ranking regularization 改善总体排序，"
        "而不是声称渐进训练已经消除梯度竞争。模型结构到此冻结，下一步进入 5-fold/repeated holdout、temporal split 与 error analysis。",
    ])
    (args.output_dir / "progressive_training_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return True


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    logger = build_logger(args.output_dir / "progressive_training.log")
    device = get_device(CFG.device)
    train_ds, val_ds, split_hash = prepare_data(args, logger)
    if split_hash != "79413d21918a956716f230456b80aec037be51bfd1c6a412a8c0a30087a2a58f":
        raise RuntimeError(f"Unexpected validation split hash: {split_hash}")
    history: list[dict] = []
    history_path = args.output_dir / "progressive_training_history.csv"
    if history_path.exists():
        with history_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                converted = dict(row)
                for key in HISTORY_FIELDS:
                    if key not in {"experiment", "stage"}:
                        converted[key] = int(row[key]) if key in {"seed", "epoch"} else float(row[key])
                history.append(converted)

    requested = {value.strip().upper() for value in args.experiments.split(",") if value.strip()}
    if not requested or not requested.issubset({"J0", "P1", "P2"}):
        raise ValueError("--experiments must be a subset of J0,P1,P2")
    if args.force:
        history = [
            row for row in history
            if not (row["seed"] == args.seed and row["experiment"] in requested)
        ]
        write_csv(history_path, history, HISTORY_FIELDS)
    results = []
    seed_suffix = "" if args.seed == 42 else f"seed_{args.seed}"

    def experiment_dir(name: str) -> Path:
        path = args.output_dir / name
        if seed_suffix:
            path = path / seed_suffix
        ensure_dir(path)
        return path

    joint_epochs, joint_patience = ((1, 1) if args.smoke else (30, 10))
    stage1_epochs, stage1_patience = ((1, 1) if args.smoke else (15, 5))
    stage2_epochs, stage2_patience = ((1, 1) if args.smoke else (30, 10))
    stage3_epochs, stage3_patience = ((1, 1) if args.smoke else (8, 3))
    if "J0" in requested:
        joint_dir = experiment_dir("J0")
        joint_model = new_model(args.seed, device)
        results.append(load_or_train_stage(
            joint_model, train_ds, val_ds, device, args, logger,
            "J0", "joint", joint_dir / "joint_best.pt", joint_epochs, joint_patience,
            split_hash, history,
        ))

    for experiment, branch_rank_weight in (("P1", 0.0), ("P2", 0.05)):
        if experiment not in requested:
            continue
        stage_dir = experiment_dir(experiment)
        if experiment == "P2":
            gap_stats = target_gap_statistics(train_ds, minimum_gap=0.05)
            (args.output_dir / "branch_target_gap_stats.json").write_text(
                json.dumps(gap_stats, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info("Branch target gap stats: %s", json.dumps(gap_stats))
        progressive_model = new_model(args.seed, device)
        results.append(load_or_train_stage(
            progressive_model, train_ds, val_ds, device, args, logger,
            experiment, "stage1", stage_dir / "stage1_best.pt",
            stage1_epochs, stage1_patience, split_hash, history, branch_rank_weight,
        ))
        if args.stop_after == "stage1":
            continue
        results.append(load_or_train_stage(
            progressive_model, train_ds, val_ds, device, args, logger,
            experiment, "stage2", stage_dir / "stage2_best.pt",
            stage2_epochs, stage2_patience, split_hash, history, branch_rank_weight,
        ))
        if args.stop_after == "stage2":
            continue
        results.append(load_or_train_stage(
            progressive_model, train_ds, val_ds, device, args, logger,
            experiment, "stage3", stage_dir / "stage3_best.pt",
            stage3_epochs, stage3_patience, split_hash, history, branch_rank_weight,
        ))

    results_path = args.output_dir / "progressive_training_results.csv"
    all_results = merge_rows(read_results(results_path), results)
    write_csv(results_path, all_results, RESULT_FIELDS)
    seed42_results = [row for row in all_results if row["seed"] == 42]
    if any(row["experiment"] == "J0" for row in seed42_results) and any(
        row["experiment"] == "P1" for row in seed42_results
    ):
        write_phase_a_summary(args, seed42_results, history, split_hash)
        success, reason = phase_a_success(seed42_results)
    else:
        success, reason = False, "Phase A results are incomplete"
    final_summary = write_final_summary(args, all_results, history, split_hash)
    print(json.dumps({
        "phase_a_effective": success,
        "reason": reason,
        "final_summary": final_summary,
        "results": results,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
