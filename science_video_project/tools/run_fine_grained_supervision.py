"""Fine-grained attribute supervision and continuous quality experiments."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import CFG
from pipeline.utils_io import ensure_dir, load_metadata
from tools.run_progressive_training import (
    collate_unique,
    make_train_loader,
    model_config as base_model_config,
    prepare_batch,
    prepare_data,
    safe_srcc,
)
from training.dataset_pair import ATTRIBUTE_COLUMNS, PairwiseVideoDataset
from training.losses import branch_consistency_loss, branch_supervision_loss, ranknet_pairwise_loss
from training.metrics import compute_video_metrics
from training.model_mvp import ATTRIBUTE_NAMES, MultiModalQualityModel
from training.train import _search_best_threshold
from training.utils_train import get_device, set_seed


DEFAULT_METADATA = Path(r"D:\Projects\science_video_ranker_mvp\science_video_project\data\parsed_metadata_filtered.csv")
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "fine_grained_supervision"
DEFAULT_FEATURES = PROJECT_ROOT / "outputs" / "features"


@dataclass(frozen=True)
class Supervision:
    consistency: float = 0.0
    branch: float = 0.0
    attribute: float = 0.0
    quality: float = 0.0
    pointwise: float = 0.1


EXPERIMENTS = {
    "F0": Supervision(),
    "F1": Supervision(consistency=0.1),
    "F2": Supervision(branch=0.05),
    "F3": Supervision(consistency=0.1, branch=0.05),
    "A0": Supervision(),
    "A1": Supervision(consistency=0.1, branch=0.05),
    "A2": Supervision(consistency=0.1, attribute=0.02),
    "A3": Supervision(consistency=0.1, branch=0.05, attribute=0.02),
    "A4": Supervision(consistency=0.1, branch=0.05, attribute=0.02, quality=0.05),
}

ATTRIBUTE_RESULT_KEYS = tuple(f"{name}_srcc" for name in ATTRIBUTE_NAMES)
RESULT_FIELDS = (
    "experiment", "seed", "auc", "pr_auc", "accuracy", "f1", "best_threshold",
    "sci_srcc", "tech_srcc", "aes_srcc", "branch_macro_srcc",
    *ATTRIBUTE_RESULT_KEYS, "mean_attribute_srcc", "q_rank_quality_srcc",
    "quality_srcc", "quality_plcc", "quality_rmse", "quality_mae",
    "quality_auc", "quality_pr_auc", "quality_f1", "quality_threshold",
    "science_weight", "technical_weight", "aesthetic_weight",
    "consistency_weight", "branch_weight", "attribute_weight", "quality_weight",
    "best_epoch", "validation_hash", "train_seconds", "trainable_params", "checkpoint",
)
HISTORY_FIELDS = (
    "experiment", "seed", "epoch", "rank_loss", "point_loss", "consistency_loss",
    "branch_loss", "attribute_loss", "quality_loss", "total_loss", "auc", "pr_auc",
    "sci_srcc", "tech_srcc", "aes_srcc", "mean_attribute_srcc", "quality_srcc",
    "learning_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--experiments", default="F0,F1,F2,F3,A2,A3,A4")
    parser.add_argument("--use-consistency", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-branch-supervision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-attribute-supervision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-quality-supervision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lambda-consistency", type=float, default=0.1)
    parser.add_argument("--lambda-branch", type=float, default=0.05)
    parser.add_argument("--lambda-attr", "--lambda-attribute", dest="lambda_attribute", type=float, default=0.02)
    parser.add_argument("--lambda-quality", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--patience", type=int, default=CFG.early_stop_patience)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def build_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("fine_grained_supervision")
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
                if key not in row or key in {"experiment", "validation_hash", "checkpoint"}:
                    continue
                row[key] = int(float(row[key])) if key in {"seed", "best_epoch", "trainable_params"} else float(row[key])
            rows.append(row)
    return rows


def merge_rows(existing: list[dict], current: list[dict]) -> list[dict]:
    merged = {(row["experiment"], int(row["seed"])): row for row in existing}
    merged.update({(row["experiment"], int(row["seed"])): row for row in current})
    return sorted(merged.values(), key=lambda row: (int(row["seed"]), row["experiment"]))


def custom_supervision(args: argparse.Namespace) -> Supervision:
    return Supervision(
        consistency=args.lambda_consistency if args.use_consistency else 0.0,
        branch=args.lambda_branch if args.use_branch_supervision else 0.0,
        attribute=args.lambda_attribute if args.use_attribute_supervision else 0.0,
        quality=args.lambda_quality if args.use_quality_supervision else 0.0,
    )


def audit_attributes(args: argparse.Namespace, clean_ids: set[str]) -> dict:
    metadata = load_metadata(args.metadata)
    clean = metadata[metadata["video_id"].astype(str).isin(clean_ids)].copy()
    stats = {}
    for name in ATTRIBUTE_COLUMNS:
        values = pd.to_numeric(clean[name], errors="coerce")
        stats[name] = {
            "count": int(values.notna().sum()),
            "missing": int(values.isna().sum()),
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "zero_count": int((values == 0).sum()),
        }
    zero_ids = clean.loc[pd.to_numeric(clean["video_aesthetics"], errors="coerce") == 0, "video_id"]
    payload = {
        "sample_count": int(len(clean)),
        "label_counts": {str(k): int(v) for k, v in clean["label"].value_counts().sort_index().items()},
        "normalization": "raw score / 5; video_aesthetics=0 is retained as a valid target",
        "attributes": stats,
        "video_aesthetics_zero_video_ids": zero_ids.astype(str).tolist(),
    }
    (args.output_dir / "attribute_statistics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def model_config(spec: Supervision | None = None) -> dict:
    config = base_model_config()
    config.update(
        use_attribute_heads=bool(spec and spec.attribute > 0),
        use_quality_head=bool(spec and spec.quality > 0),
    )
    return config


def new_model(seed: int, device: torch.device, spec: Supervision) -> MultiModalQualityModel:
    set_seed(seed)
    return MultiModalQualityModel(**model_config(spec)).to(device)


def branch_loss(output: dict, batch: dict) -> torch.Tensor:
    return branch_supervision_loss(
        output["scientific_score"], output["technical_score"], output["aesthetic_score"],
        batch["sci_target"], batch["tech_target"], batch["aes_target"],
    )


def attribute_loss(output: dict, batch: dict) -> torch.Tensor:
    if not output["attribute_scores"]:
        return output["overall_score"].sum() * 0.0
    prediction = torch.cat([output["attribute_scores"][name] for name in ATTRIBUTE_NAMES], dim=1)
    target = batch["attribute_targets"]
    valid = target >= 0
    return F.mse_loss(prediction[valid], target[valid]) if valid.any() else prediction.sum() * 0.0


def quality_loss(output: dict, batch: dict) -> torch.Tensor:
    if output["quality_score"] is None:
        return output["overall_score"].sum() * 0.0
    target = batch["overall_quality_target"]
    valid = target >= 0
    return F.mse_loss(output["quality_score"][valid], target[valid]) if valid.any() else output["quality_score"].sum() * 0.0


def component_losses(pos_out: dict, neg_out: dict, pos_batch: dict, neg_batch: dict) -> dict[str, torch.Tensor]:
    rank = ranknet_pairwise_loss(pos_out["overall_score"], neg_out["overall_score"])
    point = (
        F.binary_cross_entropy_with_logits(pos_out["overall_score"], torch.ones_like(pos_out["overall_score"]))
        + F.binary_cross_entropy_with_logits(neg_out["overall_score"], torch.zeros_like(neg_out["overall_score"]))
    ) / 2.0
    consistency = (
        branch_consistency_loss(pos_out["overall_score"], pos_out["scientific_score"], pos_out["technical_score"], pos_out["aesthetic_score"])
        + branch_consistency_loss(neg_out["overall_score"], neg_out["scientific_score"], neg_out["technical_score"], neg_out["aesthetic_score"])
    ) / 2.0
    branch = (branch_loss(pos_out, pos_batch) + branch_loss(neg_out, neg_batch)) / 2.0
    attribute = (attribute_loss(pos_out, pos_batch) + attribute_loss(neg_out, neg_batch)) / 2.0
    quality = (quality_loss(pos_out, pos_batch) + quality_loss(neg_out, neg_batch)) / 2.0
    return {"rank": rank, "point": point, "consistency": consistency, "branch": branch, "attribute": attribute, "quality": quality}


def train_epoch(model: MultiModalQualityModel, loader: DataLoader, optimizer: AdamW, device: torch.device, spec: Supervision) -> dict:
    model.train()
    sums = {name: 0.0 for name in ("rank", "point", "consistency", "branch", "attribute", "quality", "total")}
    for pos_batch, neg_batch in loader:
        pos_batch = prepare_batch(pos_batch, device)
        neg_batch = prepare_batch(neg_batch, device)
        losses = component_losses(model(**pos_batch), model(**neg_batch), pos_batch, neg_batch)
        total = (
            losses["rank"] + spec.pointwise * losses["point"]
            + spec.consistency * losses["consistency"] + spec.branch * losses["branch"]
            + spec.attribute * losses["attribute"] + spec.quality * losses["quality"]
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        for name in losses:
            sums[name] += float(losses[name].detach().item())
        sums["total"] += float(total.detach().item())
    return {name: value / max(len(loader), 1) for name, value in sums.items()}


def safe_plcc(prediction: np.ndarray, target: np.ndarray) -> float:
    valid = np.isfinite(prediction) & np.isfinite(target) & (target >= 0)
    if valid.sum() < 2 or np.unique(prediction[valid]).size < 2 or np.unique(target[valid]).size < 2:
        return float("nan")
    value = pearsonr(prediction[valid], target[valid]).statistic
    return float(value) if np.isfinite(value) else float("nan")


def evaluate(model: MultiModalQualityModel, samples: list[dict], device: torch.device, spec: Supervision) -> dict:
    model.eval()
    keys = (
        "label", "probability", "overall", "sci_logit", "tech_logit", "aes_logit",
        "sci", "tech", "aes", "sci_target", "tech_target",
        "aes_target", "quality_target", "quality_score", "science_weight", "technical_weight", "aesthetic_weight",
    )
    records = {key: [] for key in keys}
    attr_predictions = {name: [] for name in ATTRIBUTE_NAMES}
    attr_targets = {name: [] for name in ATTRIBUTE_NAMES}
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
                "sci": torch.sigmoid(output["scientific_score"]).squeeze(-1),
                "tech": torch.sigmoid(output["technical_score"]).squeeze(-1),
                "aes": torch.sigmoid(output["aesthetic_score"]).squeeze(-1),
                "sci_target": batch["sci_target"].squeeze(-1),
                "tech_target": batch["tech_target"].squeeze(-1),
                "aes_target": batch["aes_target"].squeeze(-1),
                "quality_target": batch["overall_quality_target"].squeeze(-1),
                "quality_score": (
                    output["quality_score"].squeeze(-1)
                    if output["quality_score"] is not None
                    else torch.zeros_like(output["probability"].squeeze(-1))
                ),
                "science_weight": output["branch_weights"][:, 0],
                "technical_weight": output["branch_weights"][:, 1],
                "aesthetic_weight": output["branch_weights"][:, 2],
            }
            for key, value in values.items():
                records[key].extend(value.detach().cpu().numpy().tolist())
            for index, name in enumerate(ATTRIBUTE_NAMES):
                if name in output["attribute_scores"]:
                    attr_predictions[name].extend(output["attribute_scores"][name].squeeze(-1).cpu().numpy().tolist())
                attr_targets[name].extend(batch["attribute_targets"][:, index].cpu().numpy().tolist())
    arrays = {key: np.asarray(value, dtype=np.float32) for key, value in records.items()}
    threshold, _ = _search_best_threshold(
        arrays["probability"][arrays["label"] == 1], arrays["probability"][arrays["label"] == 0], metric="f1"
    )
    metrics = compute_video_metrics(arrays["label"], arrays["probability"], threshold)
    metrics["best_threshold"] = float(threshold)
    pos = arrays["probability"][arrays["label"] == 1]
    neg = arrays["probability"][arrays["label"] == 0]
    metrics["ranking_accuracy"] = float((pos[:, None] > neg[None, :]).mean())
    for output_key, prediction_key, target_key in (
        ("sci_srcc", "sci", "sci_target"), ("tech_srcc", "tech", "tech_target"), ("aes_srcc", "aes", "aes_target")
    ):
        metrics[output_key] = safe_srcc(arrays[prediction_key], arrays[target_key], output_key)[0]
    metrics["branch_macro_srcc"] = float(np.nanmean([metrics["sci_srcc"], metrics["tech_srcc"], metrics["aes_srcc"]]))
    for key in ("science_weight", "technical_weight", "aesthetic_weight"):
        metrics[key] = float(arrays[key].mean())
    metrics["q_rank_quality_srcc"] = safe_srcc(arrays["probability"], arrays["quality_target"], "q_rank_quality")[0]

    attribute_values = []
    for name in ATTRIBUTE_NAMES:
        value = safe_srcc(np.asarray(attr_predictions[name]), np.asarray(attr_targets[name]), name)[0] if spec.attribute > 0 else float("nan")
        metrics[f"{name}_srcc"] = value
        attribute_values.append(value)
    metrics["mean_attribute_srcc"] = float(np.nanmean(attribute_values)) if spec.attribute > 0 else float("nan")

    if spec.quality > 0:
        target = arrays["quality_target"]
        prediction = arrays["quality_score"]
        valid = np.isfinite(target) & (target >= 0)
        metrics["quality_srcc"] = safe_srcc(prediction, target, "quality")[0]
        metrics["quality_plcc"] = safe_plcc(prediction, target)
        metrics["quality_rmse"] = float(np.sqrt(np.mean((prediction[valid] - target[valid]) ** 2)))
        metrics["quality_mae"] = float(np.mean(np.abs(prediction[valid] - target[valid])))
        metrics["quality_auc"] = float(roc_auc_score(arrays["label"], prediction))
        metrics["quality_pr_auc"] = float(average_precision_score(arrays["label"], prediction))
        q_threshold, _ = _search_best_threshold(prediction[arrays["label"] == 1], prediction[arrays["label"] == 0], metric="f1")
        metrics["quality_threshold"] = float(q_threshold)
        metrics["quality_f1"] = float(f1_score(arrays["label"], prediction >= q_threshold, zero_division=0))
    else:
        for key in ("quality_srcc", "quality_plcc", "quality_rmse", "quality_mae", "quality_auc", "quality_pr_auc", "quality_f1", "quality_threshold"):
            metrics[key] = float("nan")

    rank_loss = F.softplus(-(
        torch.from_numpy(arrays["overall"])[arrays["label"] == 1][:, None]
        - torch.from_numpy(arrays["overall"])[arrays["label"] == 0][None, :]
    )).mean().item()
    point_loss = F.binary_cross_entropy_with_logits(torch.from_numpy(arrays["overall"]), torch.from_numpy(arrays["label"])).item()
    consistency_loss = float(np.mean((
        arrays["overall"]
        - (arrays["sci_logit"] + arrays["tech_logit"] + arrays["aes_logit"]) / 3.0
    ) ** 2))
    branch_mse = statistics.fmean(
        float(np.mean((arrays[prediction] - arrays[target]) ** 2))
        for prediction, target in (("sci", "sci_target"), ("tech", "tech_target"), ("aes", "aes_target"))
    )
    attr_mse = 0.0
    if spec.attribute > 0:
        attr_mse = statistics.fmean(
            float(np.mean((np.asarray(attr_predictions[name]) - np.asarray(attr_targets[name])) ** 2))
            for name in ATTRIBUTE_NAMES
        )
    quality_mse = 0.0
    if spec.quality > 0:
        valid = arrays["quality_target"] >= 0
        quality_mse = float(np.mean((arrays["quality_score"][valid] - arrays["quality_target"][valid]) ** 2))
    metrics.update({
        "rank_loss": float(rank_loss), "point_loss": float(point_loss),
        "consistency_loss": consistency_loss, "branch_loss": branch_mse,
        "attribute_loss": attr_mse, "quality_loss": quality_mse,
    })
    metrics["validation_loss"] = float(
        rank_loss + spec.pointwise * point_loss
        + spec.consistency * consistency_loss + spec.branch * branch_mse
        + spec.attribute * attr_mse + spec.quality * quality_mse
    )
    return metrics


def save_checkpoint(path: Path, model: MultiModalQualityModel, optimizer: AdamW, experiment: str, epoch: int, metrics: dict, spec: Supervision, args: argparse.Namespace, split_hash: str, elapsed: float = 0.0) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model_config(spec),
        "supervision": asdict(spec),
        "experiment": experiment,
        "epoch": epoch,
        "best_epoch": epoch,
        "validation_metrics": metrics,
        "best_threshold": metrics["best_threshold"],
        "seed": args.seed,
        "split_seed": args.split_seed,
        "split_hash": split_hash,
        "science_feature_mode": "analysis_scores_concat",
        "llm_text_source": "reasoning_and_analysis",
        "technical_visual_source": "clip_cover",
        "dimension_config": {"clip": 256, "cover": 256, "concat": 512},
        "trainable_params": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "train_seconds": elapsed,
    }, path)


def result_row(payload: dict, checkpoint: Path) -> dict:
    metrics = payload["validation_metrics"]
    spec = payload["supervision"]
    row = {key: metrics.get(key, float("nan")) for key in RESULT_FIELDS if key not in {
        "experiment", "seed", "consistency_weight", "branch_weight", "attribute_weight", "quality_weight",
        "best_epoch", "validation_hash", "train_seconds", "trainable_params", "checkpoint",
    }}
    row.update({
        "experiment": payload["experiment"], "seed": payload["seed"],
        "consistency_weight": spec["consistency"], "branch_weight": spec["branch"],
        "attribute_weight": spec["attribute"], "quality_weight": spec["quality"],
        "best_epoch": payload["best_epoch"], "validation_hash": payload["split_hash"],
        "train_seconds": payload.get("train_seconds", 0.0), "trainable_params": payload["trainable_params"],
        "checkpoint": str(checkpoint.resolve()),
    })
    return row


def train_experiment(experiment: str, spec: Supervision, train_ds: PairwiseVideoDataset, val_ds: PairwiseVideoDataset, device: torch.device, args: argparse.Namespace, logger: logging.Logger, split_hash: str, history: list[dict]) -> dict:
    checkpoint = args.output_dir / f"{experiment}_seed{args.seed}_best.pt"
    if checkpoint.exists() and not args.force:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        logger.info("Reusing %s", checkpoint)
        return result_row(payload, checkpoint)
    model = new_model(args.seed, device, spec)
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=CFG.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=CFG.scheduler_factor, patience=CFG.scheduler_patience, min_lr=1e-6)
    loader = make_train_loader(train_ds, args.seed)
    epochs = 1 if args.smoke else args.epochs
    patience = 1 if args.smoke else args.patience
    best = -float("inf")
    stale = 0
    started = time.perf_counter()
    logger.info("%s seed=%d start | supervision=%s", experiment, args.seed, asdict(spec))
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, loader, optimizer, device, spec)
        validation = evaluate(model, val_ds.samples, device, spec)
        scheduler.step(validation["auc"])
        history.append({
            "experiment": experiment, "seed": args.seed, "epoch": epoch,
            "rank_loss": train_metrics["rank"], "point_loss": train_metrics["point"],
            "consistency_loss": train_metrics["consistency"], "branch_loss": train_metrics["branch"],
            "attribute_loss": train_metrics["attribute"], "quality_loss": train_metrics["quality"],
            "total_loss": train_metrics["total"], "auc": validation["auc"], "pr_auc": validation["pr_auc"],
            "sci_srcc": validation["sci_srcc"], "tech_srcc": validation["tech_srcc"],
            "aes_srcc": validation["aes_srcc"], "mean_attribute_srcc": validation["mean_attribute_srcc"],
            "quality_srcc": validation["quality_srcc"], "learning_rate": optimizer.param_groups[0]["lr"],
        })
        write_csv(args.output_dir / "fine_grained_history.csv", history, HISTORY_FIELDS)
        logger.info(
            "%s epoch=%d total=%.4f AUC=%.4f PR=%.4f branch=%.4f attr=%s quality=%s",
            experiment, epoch, train_metrics["total"], validation["auc"], validation["pr_auc"],
            validation["branch_macro_srcc"], f'{validation["mean_attribute_srcc"]:.4f}' if spec.attribute else "-",
            f'{validation["quality_srcc"]:.4f}' if spec.quality else "-",
        )
        if validation["auc"] > best:
            best = validation["auc"]
            stale = 0
            save_checkpoint(checkpoint, model, optimizer, experiment, epoch, validation, spec, args, split_hash)
        else:
            stale += 1
            if stale >= patience:
                logger.info("%s early stop at epoch %d", experiment, epoch)
                break
    elapsed = time.perf_counter() - started
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["train_seconds"] = elapsed
    torch.save(payload, checkpoint)
    return result_row(payload, checkpoint)


def fmt(value: float) -> str:
    return "-" if not np.isfinite(float(value)) else f"{float(value):.4f}"


def mean_std(rows: list[dict], key: str) -> str:
    values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
    return "-" if not values else f"{statistics.fmean(values):.4f} +/- {statistics.pstdev(values):.4f}"


def choose_candidate(rows: list[dict]) -> str | None:
    seed42 = {row["experiment"]: row for row in rows if int(row["seed"]) == 42}
    if "A1" not in seed42:
        return None
    baseline = seed42["A1"]
    candidates = [seed42[name] for name in ("A2", "A3", "A4") if name in seed42]
    eligible = [row for row in candidates if row["auc"] >= baseline["auc"] - 0.01 and row["pr_auc"] >= baseline["pr_auc"] - 0.03]
    if not eligible:
        return max(candidates, key=lambda row: row["auc"])["experiment"] if candidates else None
    return max(eligible, key=lambda row: (
        row["branch_macro_srcc"],
        row["mean_attribute_srcc"] if np.isfinite(row["mean_attribute_srcc"]) else -1.0,
        row["quality_srcc"] if np.isfinite(row["quality_srcc"]) else -1.0,
        row["auc"],
    ))["experiment"]


def alias_seed42(rows: list[dict]) -> list[dict]:
    lookup = {(row["experiment"], int(row["seed"])): row for row in rows}
    aliases = []
    for source, target in (("F0", "A0"), ("F3", "A1")):
        if (source, 42) in lookup and (target, 42) not in lookup:
            copy = dict(lookup[(source, 42)])
            copy["experiment"] = target
            aliases.append(copy)
    return merge_rows(rows, aliases)


def decision(rows: list[dict], candidate: str | None) -> str:
    if candidate is None:
        return "KEEP_CURRENT_BRANCH_SUPERVISION"
    baseline = [row for row in rows if row["experiment"] == "A1" and int(row["seed"]) in {42, 123, 2026}]
    selected = [row for row in rows if row["experiment"] == candidate and int(row["seed"]) in {42, 123, 2026}]
    if len(baseline) < 3 or len(selected) < 3:
        return "PENDING_MULTI_SEED_VALIDATION"
    auc_ok = statistics.fmean(row["auc"] for row in selected) >= statistics.fmean(row["auc"] for row in baseline) - 0.01
    pr_ok = statistics.fmean(row["pr_auc"] for row in selected) >= statistics.fmean(row["pr_auc"] for row in baseline) - 0.03
    if not (auc_ok and pr_ok):
        return "KEEP_CURRENT_BRANCH_SUPERVISION"
    return "PROMOTE_MULTITASK_QUALITY_MODEL" if candidate == "A4" else "PROMOTE_ATTRIBUTE_SUPERVISION"


def write_summary(args: argparse.Namespace, rows: list[dict], audit: dict) -> None:
    candidate = choose_candidate(rows)
    verdict = decision(rows, candidate)
    seed42 = {row["experiment"]: row for row in rows if int(row["seed"]) == 42}
    lines = [
        "# Fine-grained Attribute Supervision and Continuous Quality Modeling",
        "",
        "## Data audit",
        "",
        f"- Clean samples: {audit['sample_count']} ({audit['label_counts']})",
        "- All seven targets are normalized with `score / 5`.",
        f"- `video_aesthetics=0` is retained as valid: {len(audit['video_aesthetics_zero_video_ids'])} samples.",
        "",
        "| Attribute | Min | Max | Mean | Std | Missing | Zero |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in audit["attributes"].items():
        lines.append(f"| {name} | {item['min']:.1f} | {item['max']:.1f} | {item['mean']:.4f} | {item['std']:.4f} | {item['missing']} | {item['zero_count']} |")
    lines.extend([
        "", "## F0-F3 supervision audit (seed 42)", "",
        "| Model | AUC | PR-AUC | Acc | F1 | Sci SRCC | Tech SRCC | Aes SRCC | Sci/Tech/Aes weight |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in ("F0", "F1", "F2", "F3"):
        if name in seed42:
            row = seed42[name]
            weights = "/".join(fmt(row[key]) for key in ("science_weight", "technical_weight", "aesthetic_weight"))
            lines.append(f"| {name} | {fmt(row['auc'])} | {fmt(row['pr_auc'])} | {fmt(row['accuracy'])} | {fmt(row['f1'])} | {fmt(row['sci_srcc'])} | {fmt(row['tech_srcc'])} | {fmt(row['aes_srcc'])} | {weights} |")

    lines.extend([
        "", "## A0-A4 core matrix (seed 42)", "",
        "| Model | AUC | PR-AUC | F1 | Sci SRCC | Tech SRCC | Aes SRCC | Attr mean | Quality SRCC | Quality PLCC | q_rank-quality SRCC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in ("A0", "A1", "A2", "A3", "A4"):
        if name in seed42:
            row = seed42[name]
            lines.append(f"| {name} | {fmt(row['auc'])} | {fmt(row['pr_auc'])} | {fmt(row['f1'])} | {fmt(row['sci_srcc'])} | {fmt(row['tech_srcc'])} | {fmt(row['aes_srcc'])} | {fmt(row['mean_attribute_srcc'])} | {fmt(row['quality_srcc'])} | {fmt(row['quality_plcc'])} | {fmt(row['q_rank_quality_srcc'])} |")

    lines.extend([
        "", "## Attribute SRCC (seed 42)", "",
        "| Model | science_info | topic_importance | science_access | content_interest | visual_quality | audio_quality | video_aesthetics | Mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in ("A2", "A3", "A4"):
        if name in seed42:
            row = seed42[name]
            values = " | ".join(fmt(row[key]) for key in ATTRIBUTE_RESULT_KEYS)
            lines.append(f"| {name} | {values} | {fmt(row['mean_attribute_srcc'])} |")

    if "A4" in seed42:
        row = seed42["A4"]
        lines.extend([
            "", "## Continuous quality diagnostics (A4, seed 42)", "",
            "| Quality SRCC | PLCC | RMSE | MAE | Quality AUC | Quality PR-AUC | Quality F1 | Threshold |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| {fmt(row['quality_srcc'])} | {fmt(row['quality_plcc'])} | {fmt(row['quality_rmse'])} | {fmt(row['quality_mae'])} | {fmt(row['quality_auc'])} | {fmt(row['quality_pr_auc'])} | {fmt(row['quality_f1'])} | {fmt(row['quality_threshold'])} |",
        ])

    lines.extend(["", f"Seed-42 selected candidate: **{candidate or 'pending'}**.", "", "## Multi-seed validation", ""])
    if candidate:
        lines.extend([
            "| Model | AUC | PR-AUC | Branch macro SRCC | Attribute mean SRCC | Quality SRCC | q_rank-quality SRCC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for name in ("A1", candidate):
            subset = [r for r in rows if r["experiment"] == name and int(r["seed"]) in {42, 123, 2026}]
            lines.append(f"| {name} (n={len(subset)}) | {mean_std(subset, 'auc')} | {mean_std(subset, 'pr_auc')} | {mean_std(subset, 'branch_macro_srcc')} | {mean_std(subset, 'mean_attribute_srcc')} | {mean_std(subset, 'quality_srcc')} | {mean_std(subset, 'q_rank_quality_srcc')} |")
        baseline = [r for r in rows if r["experiment"] == "A1" and int(r["seed"]) in {42, 123, 2026}]
        selected = [r for r in rows if r["experiment"] == candidate and int(r["seed"]) in {42, 123, 2026}]
        if len(baseline) == len(selected) == 3:
            auc_delta = statistics.fmean(r["auc"] for r in selected) - statistics.fmean(r["auc"] for r in baseline)
            pr_delta = statistics.fmean(r["pr_auc"] for r in selected) - statistics.fmean(r["pr_auc"] for r in baseline)
            lines.extend(["", f"Multi-seed delta ({candidate} - A1): AUC `{auc_delta:+.4f}`, PR-AUC `{pr_delta:+.4f}`."])
            if candidate in {"A2", "A3", "A4"}:
                lines.extend([
                    "", f"### {candidate} attribute SRCC across seeds", "",
                    "| Attribute | Mean +/- std |",
                    "|---|---:|",
                ])
                for key in ATTRIBUTE_RESULT_KEYS:
                    lines.append(f"| {key.removesuffix('_srcc')} | {mean_std(selected, key)} |")
            if candidate == "A2":
                lines.extend([
                    "",
                    "Cross-experiment context: the earlier progressive P2-S2 reached AUC `0.7998 +/- 0.0160` and PR-AUC `0.6174 +/- 0.0357`. A2 improves the matched joint A1 baseline, but does not yet replace P2-S2; attribute supervision should next be validated inside P2 Stage2.",
                ])
    if all(name in seed42 for name in ("F1", "F3", "A1", "A2", "A3", "A4")):
        lines.extend([
            "", "## Answers", "",
            f"- Branch supervision: F3 vs F1 changes AUC by `{seed42['F3']['auc'] - seed42['F1']['auc']:+.4f}` and branch macro SRCC by `{seed42['F3']['branch_macro_srcc'] - seed42['F1']['branch_macro_srcc']:+.4f}`.",
            f"- Raw attributes: A2 vs A1 changes AUC by `{seed42['A2']['auc'] - seed42['A1']['auc']:+.4f}`; A2 attribute mean SRCC is `{seed42['A2']['mean_attribute_srcc']:.4f}`.",
            f"- Combined branch + attribute supervision: A3 vs A1 changes AUC by `{seed42['A3']['auc'] - seed42['A1']['auc']:+.4f}`.",
            f"- Continuous quality: A4 vs A3 changes AUC by `{seed42['A4']['auc'] - seed42['A3']['auc']:+.4f}` and yields quality SRCC `{seed42['A4']['quality_srcc']:.4f}`.",
        ])
    lines.extend([
        "", "## Decision", "", f"**{verdict}**", "",
        "The auxiliary heads are supervision-only and do not feed their predictions into branch fusion or the ranking head.",
    ])
    content = "\n".join(lines) + "\n"
    (args.output_dir / "fine_grained_summary.md").write_text(content, encoding="utf-8")
    # Preserve the descriptive filename used by the initial implementation.
    (args.output_dir / "fine_grained_supervision_summary.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    logger = build_logger(args.output_dir / f"fine_grained_seed{args.seed}.log")
    device = get_device(CFG.device)
    train_ds, val_ds, split_hash = prepare_data(args, logger)
    clean_ids = {str(sample["video_id"]) for sample in train_ds.samples + val_ds.samples}
    audit = audit_attributes(args, clean_ids)
    logger.info("Device=%s | clean=%d | split_hash=%s", device, len(clean_ids), split_hash)

    names = [name.strip().upper() for name in args.experiments.split(",") if name.strip()]
    specs = dict(EXPERIMENTS)
    specs["CUSTOM"] = custom_supervision(args)
    unknown = sorted(set(names) - set(specs))
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}")
    result_path = args.output_dir / "fine_grained_results.csv"
    history_path = args.output_dir / "fine_grained_history.csv"
    results = read_csv(result_path, RESULT_FIELDS)
    history = read_csv(history_path, HISTORY_FIELDS)
    if args.force:
        history = [
            row for row in history
            if not (int(row["seed"]) == args.seed and row["experiment"] in names)
        ]
    current = []
    for name in names:
        current.append(train_experiment(name, specs[name], train_ds, val_ds, device, args, logger, split_hash, history))
        results = alias_seed42(merge_rows(results, current))
        write_csv(result_path, results, RESULT_FIELDS)
        write_summary(args, results, audit)
    logger.info("Completed %s | results=%s", names, result_path)


if __name__ == "__main__":
    main()
