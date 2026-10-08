"""Integrate fine-grained attribute supervision into P2-S2 progressive training."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import CFG
from pipeline.utils_io import ensure_dir
from tools.run_fine_grained_supervision import (
    ATTRIBUTE_RESULT_KEYS,
    DEFAULT_FEATURES,
    DEFAULT_METADATA,
    Supervision,
    audit_attributes,
    component_losses,
    evaluate,
    model_config,
)
from tools.run_progressive_training import (
    branch_rank_loss,
    make_train_loader,
    prepare_batch,
    prepare_data,
)
from training.model_mvp import MultiModalQualityModel
from training.utils_train import get_device, set_seed


EXPECTED_HASH = "79413d21918a956716f230456b80aec037be51bfd1c6a412a8c0a30087a2a58f"
FINAL_SELECTION_OVERRIDE = "KEEP_P2_S2"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "progressive_attribute"
PROGRESSIVE_OUTPUT = PROJECT_ROOT / "outputs" / "progressive_training"


@dataclass(frozen=True)
class Plan:
    use_attribute: bool
    use_quality: bool
    use_branch: bool


PLANS = {
    "P0": Plan(False, False, True),
    "P1": Plan(True, False, True),
    "P2": Plan(True, True, True),
    "P1_SIMPLE": Plan(True, False, False),
}

RESULT_FIELDS = (
    "model", "seed", "split_seed", "validation_hash", "auc", "pr_auc", "accuracy", "f1",
    "sci_srcc", "tech_srcc", "aes_srcc", "branch_macro_srcc", "attribute_mean_srcc",
    *ATTRIBUTE_RESULT_KEYS, "quality_srcc", "quality_plcc", "quality_rmse", "quality_mae",
    "q_rank_quality_srcc", "science_weight", "technical_weight", "aesthetic_weight",
    "best_epoch", "train_seconds", "trainable_params", "checkpoint",
)
STAGE_FIELDS = (
    "model", "seed", "stage", "auc", "pr_auc", "accuracy", "f1", "sci_srcc", "tech_srcc",
    "aes_srcc", "branch_macro_srcc", "attribute_mean_srcc", "quality_srcc",
    "q_rank_quality_srcc", "attribute_loss", "quality_loss", "branch_loss", "best_epoch", "checkpoint",
)
HISTORY_FIELDS = (
    "model", "seed", "stage", "epoch", "rank_loss", "pointwise_loss", "consistency_loss",
    "branch_loss", "branch_rank_loss", "attribute_loss", "quality_loss", "total_loss",
    "auc", "pr_auc", "sci_srcc", "tech_srcc", "aes_srcc", "attribute_mean_srcc",
    "quality_srcc", "learning_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progressive-output", type=Path, default=PROGRESSIVE_OUTPUT)
    parser.add_argument("--models", default="P0,P1,P2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--lambda-attr", type=float, default=0.02)
    parser.add_argument("--lambda-quality", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def logger_for(path: Path) -> logging.Logger:
    logger = logging.getLogger("progressive_attribute")
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
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for key in fields:
                if key not in row or key in {"model", "stage", "validation_hash", "checkpoint"}:
                    continue
                row[key] = int(float(row[key])) if key in {"seed", "split_seed", "best_epoch", "trainable_params", "epoch"} else float(row[key])
            rows.append(row)
    return rows


def merge(rows: list[dict], current: list[dict], keys: tuple[str, ...]) -> list[dict]:
    merged = {tuple(row[key] for key in keys): row for row in rows}
    merged.update({tuple(row[key] for key in keys): row for row in current})
    return sorted(merged.values(), key=lambda row: tuple(row[key] for key in keys))


def supervision(plan: Plan, args: argparse.Namespace) -> Supervision:
    return Supervision(
        consistency=0.1,
        branch=0.05 if plan.use_branch else 0.0,
        attribute=args.lambda_attr if plan.use_attribute else 0.0,
        quality=args.lambda_quality if plan.use_quality else 0.0,
        pointwise=0.1,
    )


def checkpoint_name(model: str, seed: int, stage1: bool = False) -> str:
    base = f"best_{model.lower()}"
    if stage1:
        base += "_stage1"
    if seed != 42:
        base += f"_seed{seed}"
    return base + ".pt"


def new_model(seed: int, device: torch.device, spec: Supervision) -> MultiModalQualityModel:
    set_seed(seed)
    return MultiModalQualityModel(**model_config(spec)).to(device)


def set_trainable(model: MultiModalQualityModel, stage: str) -> int:
    for parameter in model.parameters():
        parameter.requires_grad = stage == "stage2"
    if stage == "stage1":
        for branch in (model.scientific_branch, model.technical_branch, model.aesthetic_branch):
            for parameter in branch.parameters():
                parameter.requires_grad = True
        if model.attribute_heads is not None:
            for parameter in model.attribute_heads.parameters():
                parameter.requires_grad = True
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def train_epoch(model, loader, optimizer, device, plan: Plan, spec: Supervision, stage: str) -> dict:
    model.train()
    if stage == "stage1" and model.quality_head is not None:
        model.quality_head.eval()
    sums = {key: 0.0 for key in (
        "rank", "point", "consistency", "branch", "branch_rank", "attribute", "quality", "total"
    )}
    for pos_batch, neg_batch in loader:
        pos_batch = prepare_batch(pos_batch, device)
        neg_batch = prepare_batch(neg_batch, device)
        pos_out = model(**pos_batch)
        neg_out = model(**neg_batch)
        losses = component_losses(pos_out, neg_out, pos_batch, neg_batch)
        branch_rank = branch_rank_loss(pos_out, neg_out, pos_batch, neg_batch, minimum_gap=0.05)
        if stage == "stage1":
            total = (
                (losses["branch"] if plan.use_branch else losses["branch"].new_zeros(()))
                + 0.05 * branch_rank + spec.attribute * losses["attribute"]
            )
        else:
            total = (
                losses["rank"] + 0.1 * losses["point"] + 0.1 * losses["consistency"]
                + spec.branch * losses["branch"] + 0.05 * branch_rank
                + spec.attribute * losses["attribute"] + spec.quality * losses["quality"]
            )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        for key in ("rank", "point", "consistency", "branch", "attribute", "quality"):
            sums[key] += float(losses[key].detach().item())
        sums["branch_rank"] += float(branch_rank.detach().item())
        sums["total"] += float(total.detach().item())
    return {key: value / max(len(loader), 1) for key, value in sums.items()}


def save_checkpoint(path, model, optimizer, model_name, stage, epoch, metrics, plan, spec, args, split_hash, train_seconds, trainable_params):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model_config(spec),
        "model": model_name,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "validation_hash": split_hash,
        "progressive_stage": stage,
        "best_epoch": epoch,
        "metrics": metrics,
        "use_attribute_supervision": plan.use_attribute,
        "use_quality_supervision": plan.use_quality,
        "use_branch_supervision": plan.use_branch,
        "lambda_attr": spec.attribute,
        "lambda_quality": spec.quality,
        "branch_rank_weight": 0.05,
        "branch_floor": 0.2,
        "train_seconds": train_seconds,
        "trainable_params": trainable_params,
        "science_feature_mode": "analysis_scores_concat",
        "llm_text_source": "reasoning_and_analysis",
    }, path)


def stage_row(payload: dict, checkpoint: Path) -> dict:
    metrics = payload["metrics"]
    return {
        "model": payload["model"], "seed": payload["seed"], "stage": payload["progressive_stage"],
        **{key: metrics.get(key, float("nan")) for key in STAGE_FIELDS if key not in {
            "model", "seed", "stage", "attribute_mean_srcc", "best_epoch", "checkpoint"
        }},
        "attribute_mean_srcc": metrics.get("mean_attribute_srcc", float("nan")),
        "best_epoch": payload["best_epoch"], "checkpoint": str(checkpoint.resolve()),
    }


def result_row(payload: dict, checkpoint: Path) -> dict:
    metrics = payload["metrics"]
    row = {
        "model": payload["model"], "seed": payload["seed"], "split_seed": payload["split_seed"],
        "validation_hash": payload["validation_hash"],
        "attribute_mean_srcc": metrics.get("mean_attribute_srcc", float("nan")),
        "best_epoch": payload["best_epoch"], "train_seconds": payload.get("train_seconds", 0.0),
        "trainable_params": payload["trainable_params"], "checkpoint": str(checkpoint.resolve()),
    }
    for key in RESULT_FIELDS:
        if key not in row and key not in {"model", "validation_hash", "checkpoint"}:
            row[key] = metrics.get(key, float("nan"))
    return row


def train_stage(model, model_name, plan, spec, stage, train_ds, val_ds, device, args, logger, split_hash, history, checkpoint):
    if checkpoint.exists() and not args.force:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        logger.info("Reusing %s", checkpoint)
        return payload
    trainable_params = set_trainable(model, stage)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=5e-5, weight_decay=CFG.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=CFG.scheduler_factor,
        patience=CFG.scheduler_patience, min_lr=1e-6,
    )
    loader = make_train_loader(train_ds, args.seed)
    max_epochs, patience = ((1, 1) if args.smoke else ((15, 5) if stage == "stage1" else (30, 10)))
    best = -float("inf")
    stale = 0
    started = time.perf_counter()
    logger.info("%s/%s start | plan=%s | trainable=%d", model_name, stage, plan, trainable_params)
    for epoch in range(1, max_epochs + 1):
        train_metrics = train_epoch(model, loader, optimizer, device, plan, spec, stage)
        validation = evaluate(model, val_ds.samples, device, spec)
        selection = validation["branch_macro_srcc"] if stage == "stage1" else validation["ranking_accuracy"]
        scheduler_metric = validation["branch_loss"] if stage == "stage1" and plan.use_branch else (
            validation["attribute_loss"] if stage == "stage1" else validation["validation_loss"]
        )
        scheduler.step(scheduler_metric)
        history.append({
            "model": model_name, "seed": args.seed, "stage": stage, "epoch": epoch,
            "rank_loss": train_metrics["rank"], "pointwise_loss": train_metrics["point"],
            "consistency_loss": train_metrics["consistency"], "branch_loss": train_metrics["branch"],
            "branch_rank_loss": train_metrics["branch_rank"], "attribute_loss": train_metrics["attribute"],
            "quality_loss": train_metrics["quality"], "total_loss": train_metrics["total"],
            "auc": validation["auc"], "pr_auc": validation["pr_auc"],
            "sci_srcc": validation["sci_srcc"], "tech_srcc": validation["tech_srcc"],
            "aes_srcc": validation["aes_srcc"], "attribute_mean_srcc": validation["mean_attribute_srcc"],
            "quality_srcc": validation["quality_srcc"], "learning_rate": optimizer.param_groups[0]["lr"],
        })
        write_csv(args.output_dir / "progressive_attribute_history.csv", history, HISTORY_FIELDS)
        logger.info(
            "%s/%s epoch=%d total=%.4f rank=%.4f branch=%.4f attr=%.4f quality=%.4f | AUC=%.4f PR=%.4f branchSRCC=%.4f attrSRCC=%s qualitySRCC=%s",
            model_name, stage, epoch, train_metrics["total"], train_metrics["rank"], train_metrics["branch"],
            train_metrics["attribute"], train_metrics["quality"], validation["auc"], validation["pr_auc"],
            validation["branch_macro_srcc"], f'{validation["mean_attribute_srcc"]:.4f}' if plan.use_attribute else "-",
            f'{validation["quality_srcc"]:.4f}' if plan.use_quality else "-",
        )
        if selection > best:
            best = selection
            stale = 0
            save_checkpoint(
                checkpoint, model, optimizer, model_name, stage, epoch, validation, plan, spec,
                args, split_hash, time.perf_counter() - started, trainable_params,
            )
        else:
            stale += 1
            if stale >= patience:
                logger.info("%s/%s early stop at epoch %d", model_name, stage, epoch)
                break
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return payload


def historical_checkpoint(root: Path, seed: int, stage: str) -> Path:
    folder = root / "P2"
    if seed != 42:
        folder = folder / f"seed_{seed}"
    return folder / f"{stage}_best.pt"


def load_p0(train_ds, val_ds, device, args, logger, split_hash) -> tuple[dict, list[dict]]:
    stages = []
    total_seconds = 0.0
    final_payload = None
    final_target = args.output_dir / checkpoint_name("P0", args.seed)
    for stage in ("stage1", "stage2"):
        source = historical_checkpoint(args.progressive_output, args.seed, stage)
        if not source.exists():
            raise FileNotFoundError(f"Historical P2-S2 checkpoint not found: {source}")
        original = torch.load(source, map_location="cpu", weights_only=False)
        model = MultiModalQualityModel(**original["config"]).to(device)
        model.load_state_dict(original["model_state_dict"], strict=True)
        spec = Supervision(consistency=0.1, branch=0.05, pointwise=0.1)
        metrics = evaluate(model, val_ds.samples, device, spec)
        target = args.output_dir / checkpoint_name("P0", args.seed, stage1=(stage == "stage1"))
        payload = dict(original)
        payload.update({
            "model": "P0", "seed": args.seed, "split_seed": args.split_seed,
            "validation_hash": split_hash, "progressive_stage": stage,
            "best_epoch": original.get("best_epoch", original.get("epoch", 0)), "metrics": metrics,
            "use_attribute_supervision": False, "use_quality_supervision": False,
            "use_branch_supervision": True, "lambda_attr": 0.0, "lambda_quality": 0.0,
            "branch_rank_weight": 0.05, "branch_floor": 0.2,
            "trainable_params": original.get("trainable_params", sum(p.numel() for p in model.parameters())),
        })
        total_seconds += float(original.get("train_seconds", 0.0))
        payload["train_seconds"] = total_seconds
        if stage == "stage2":
            target = final_target
            final_payload = payload
        torch.save(payload, target)
        stages.append(stage_row(payload, target))
        logger.info("P0/%s restored from %s | AUC=%.4f PR=%.4f", stage, source, metrics["auc"], metrics["pr_auc"])
    return result_row(final_payload, final_target), stages


def train_plan(model_name, plan, train_ds, val_ds, device, args, logger, split_hash, history):
    spec = supervision(plan, args)
    model = new_model(args.seed, device, spec)
    stage1_path = args.output_dir / checkpoint_name(model_name, args.seed, stage1=True)
    stage1 = train_stage(
        model, model_name, plan, spec, "stage1", train_ds, val_ds, device, args,
        logger, split_hash, history, stage1_path,
    )
    stage2_path = args.output_dir / checkpoint_name(model_name, args.seed)
    stage2 = train_stage(
        model, model_name, plan, spec, "stage2", train_ds, val_ds, device, args,
        logger, split_hash, history, stage2_path,
    )
    stage2["train_seconds"] = float(stage1.get("train_seconds", 0.0)) + float(stage2.get("train_seconds", 0.0))
    torch.save(stage2, stage2_path)
    return result_row(stage2, stage2_path), [stage_row(stage1, stage1_path), stage_row(stage2, stage2_path)]


def trigger_simple(rows: list[dict]) -> tuple[bool, str]:
    lookup = {(row["model"], row["seed"]): row for row in rows}
    if ("P0", 42) not in lookup or ("P1", 42) not in lookup:
        return False, "seed-42 P0/P1 results are incomplete"
    p0, p1 = lookup[("P0", 42)], lookup[("P1", 42)]
    ranking_gain = p1["auc"] - p0["auc"] >= 0.005 or p1["pr_auc"] - p0["pr_auc"] >= 0.01
    alignment_ok = p1["branch_macro_srcc"] >= p0["branch_macro_srcc"] - 0.01 and p1["attribute_mean_srcc"] > 0
    return ranking_gain and alignment_ok, f"delta_auc={p1['auc'] - p0['auc']:+.4f}, delta_pr={p1['pr_auc'] - p0['pr_auc']:+.4f}, alignment_ok={alignment_ok}"


def mean_std(rows: list[dict], key: str) -> str:
    values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
    if not values:
        return "-"
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.fmean(values):.4f} +/- {std:.4f}"


def select_seed42(rows: list[dict]) -> str | None:
    lookup = {row["model"]: row for row in rows if row["seed"] == 42}
    if "P0" not in lookup or "P1" not in lookup:
        return None
    candidates = [lookup[name] for name in ("P1", "P2", "P1_SIMPLE") if name in lookup]
    viable = [row for row in candidates if row["auc"] >= lookup["P0"]["auc"] - 0.01 and row["pr_auc"] >= lookup["P0"]["pr_auc"] - 0.03]
    return max(viable, key=lambda row: (row["pr_auc"], row["auc"], row["branch_macro_srcc"]))["model"] if viable else None


def select_multi_seed(rows: list[dict]) -> str | None:
    baseline = [r for r in rows if r["model"] == "P0" and r["seed"] in {42, 123, 2026}]
    if len(baseline) < 3:
        return None
    baseline_auc = statistics.fmean(r["auc"] for r in baseline)
    baseline_pr = statistics.fmean(r["pr_auc"] for r in baseline)
    viable = []
    for name in ("P1", "P2", "P1_SIMPLE"):
        selected = [r for r in rows if r["model"] == name and r["seed"] in {42, 123, 2026}]
        if len(selected) != 3:
            continue
        auc_ok = statistics.fmean(r["auc"] for r in selected) >= baseline_auc
        pr_ok = statistics.fmean(r["pr_auc"] for r in selected) > baseline_pr
        attr_ok = statistics.fmean(r["attribute_mean_srcc"] for r in selected) > 0
        if auc_ok and pr_ok and attr_ok:
            viable.append(selected)
    if not viable:
        return None
    selected = max(
        viable,
        key=lambda group: (
            statistics.fmean(r["pr_auc"] for r in group),
            statistics.fmean(r["auc"] for r in group),
            -statistics.stdev(r["auc"] for r in group),
        ),
    )
    return selected[0]["model"]


def final_decision(rows: list[dict], candidate: str | None) -> str:
    multi_seed_candidate = select_multi_seed(rows)
    baseline = [r for r in rows if r["model"] == "P0" and r["seed"] in {42, 123, 2026}]
    complete_candidates = [
        name for name in ("P1", "P2", "P1_SIMPLE")
        if len([r for r in rows if r["model"] == name and r["seed"] in {42, 123, 2026}]) == 3
    ]
    if len(baseline) < 3 or (candidate is not None and not complete_candidates):
        return "PENDING_MULTI_SEED"
    if FINAL_SELECTION_OVERRIDE:
        return FINAL_SELECTION_OVERRIDE
    if multi_seed_candidate is None:
        return "KEEP_P2_S2"
    return {
        "P1": "PROMOTE_P2_S2_ATTRIBUTE",
        "P2": "PROMOTE_P2_S2_ATTRIBUTE_QUALITY",
        "P1_SIMPLE": "PROMOTE_P2_S2_ATTRIBUTE_SIMPLE",
    }[multi_seed_candidate]


def write_summary(args, rows, stages, history, audit):
    candidate = select_seed42(rows)
    multi_seed_candidate = select_multi_seed(rows)
    decision = final_decision(rows, candidate)
    seed42 = {row["model"]: row for row in rows if row["seed"] == 42}
    simple, reason = trigger_simple(rows)
    lines = [
        "# Progressive Training x Fine-grained Attribute Supervision",
        "", "## Configuration", "",
        f"- Clean data: {audit['sample_count']} videos; train/validation 492/123; split hash `{EXPECTED_HASH}`.",
        "- main_v3-D3, branch floor 0.2, cached features only.",
        "- Stage1: branch modules plus attribute heads; branch MSE + 0.05 branch RankNet + optional 0.02 attribute MSE; 15 epochs, patience 5.",
        "- Stage2: all modules; RankNet + 0.1 pointwise + 0.1 consistency + 0.05 branch MSE + 0.05 branch RankNet + optional auxiliary losses; 30 epochs, patience 10.",
        "- Both stages use AdamW, lr 5e-5. Stage2 inherits the best Stage1 checkpoint. Stage1 selects branch macro SRCC; Stage2 selects ranking accuracy.",
        "- P2 quality head is frozen in Stage1 and quality loss is not applied until Stage2; its Stage1 quality value is diagnostic only.",
        "", "## Seed 42", "",
        "| Model | AUC | PR-AUC | Acc | F1 | Sci | Tech | Aes | Branch macro | Attr mean | Quality SRCC | q_rank-quality | S/T/A weight | Epoch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for name in ("P0", "P1", "P2", "P1_SIMPLE"):
        if name in seed42:
            row = seed42[name]
            weights = f"{row['science_weight']:.4f}/{row['technical_weight']:.4f}/{row['aesthetic_weight']:.4f}"
            lines.append(f"| {name} | {row['auc']:.4f} | {row['pr_auc']:.4f} | {row['accuracy']:.4f} | {row['f1']:.4f} | {row['sci_srcc']:.4f} | {row['tech_srcc']:.4f} | {row['aes_srcc']:.4f} | {row['branch_macro_srcc']:.4f} | {row['attribute_mean_srcc']:.4f} | {row['quality_srcc']:.4f} | {row['q_rank_quality_srcc']:.4f} | {weights} | {row['best_epoch']} |")
    lines.extend(["", f"P1-Simple triggered: `{simple}` ({reason}).", f"Seed-42 candidate: **{candidate or 'none'}**.", "", "## Stage diagnostics", "",
                  "| Model | Stage | AUC | PR-AUC | Branch macro | Attr mean | Attr loss | Quality loss |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in sorted((r for r in stages if r["seed"] == 42), key=lambda r: (r["model"], r["stage"])):
        lines.append(f"| {row['model']} | {row['stage']} | {row['auc']:.4f} | {row['pr_auc']:.4f} | {row['branch_macro_srcc']:.4f} | {row['attribute_mean_srcc']:.4f} | {row['attribute_loss']:.4f} | {row['quality_loss']:.4f} |")
    lines.extend([
        "", "### Training loss scale (seed 42 epoch mean)", "",
        "| Model-stage | Rank | Point | Branch | Branch rank | Attribute | 0.02 Attr | Quality | 0.05 Quality | Total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, stage in (("P1", "stage1"), ("P1", "stage2"), ("P2", "stage1"), ("P2", "stage2"), ("P1_SIMPLE", "stage2")):
        selected_history = [r for r in history if r["seed"] == 42 and r["model"] == name and r["stage"] == stage]
        if not selected_history:
            continue
        mean = lambda key: statistics.fmean(r[key] for r in selected_history)
        quality_weight = 0.05 if name == "P2" and stage == "stage2" else 0.0
        lines.append(
            f"| {name}-{stage} | {mean('rank_loss'):.4f} | {mean('pointwise_loss'):.4f} | "
            f"{mean('branch_loss'):.4f} | {mean('branch_rank_loss'):.4f} | {mean('attribute_loss'):.4f} | "
            f"{0.02 * mean('attribute_loss'):.4f} | {mean('quality_loss'):.4f} | "
            f"{quality_weight * mean('quality_loss'):.4f} | {mean('total_loss'):.4f} |"
        )
    if all(name in seed42 for name in ("P0", "P1", "P2")):
        p0, p1, p2 = seed42["P0"], seed42["P1"], seed42["P2"]
        lines.extend([
            "", "## Answers", "",
            f"- Q1 P1-P0: AUC `{p1['auc']-p0['auc']:+.4f}`, PR-AUC `{p1['pr_auc']-p0['pr_auc']:+.4f}`, F1 `{p1['f1']-p0['f1']:+.4f}`, branch macro `{p1['branch_macro_srcc']-p0['branch_macro_srcc']:+.4f}`, q_rank-quality `{p1['q_rank_quality_srcc']-p0['q_rank_quality_srcc']:+.4f}`; attribute mean `{p1['attribute_mean_srcc']:.4f}`.",
            f"- Q2 P2-P1: AUC `{p2['auc']-p1['auc']:+.4f}`, PR-AUC `{p2['pr_auc']-p1['pr_auc']:+.4f}`; quality SRCC/PLCC/RMSE/MAE `{p2['quality_srcc']:.4f}/{p2['quality_plcc']:.4f}/{p2['quality_rmse']:.4f}/{p2['quality_mae']:.4f}`.",
        ])
        if "P1_SIMPLE" in seed42:
            simple_row = seed42["P1_SIMPLE"]
            lines.append(
                f"- Q3 P1-Simple-P1: AUC `{simple_row['auc']-p1['auc']:+.4f}`, "
                f"PR-AUC `{simple_row['pr_auc']-p1['pr_auc']:+.4f}`, branch macro "
                f"`{simple_row['branch_macro_srcc']-p1['branch_macro_srcc']:+.4f}`."
            )
    lines.extend(["", "## Multi-seed", ""])
    complete_names = [
        name for name in ("P0", "P1", "P2", "P1_SIMPLE")
        if len([r for r in rows if r["model"] == name and r["seed"] in {42, 123, 2026}]) == 3
    ]
    if complete_names:
        lines.extend(["| Model | AUC | PR-AUC | F1 | Branch macro | Attr mean | q_rank-quality | S/T/A weights |",
                      "|---|---:|---:|---:|---:|---:|---:|---|"])
        for name in complete_names:
            selected = [r for r in rows if r["model"] == name and r["seed"] in {42, 123, 2026}]
            weights = "/".join(mean_std(selected, key) for key in ("science_weight", "technical_weight", "aesthetic_weight"))
            lines.append(f"| {name} (n={len(selected)}) | {mean_std(selected,'auc')} | {mean_std(selected,'pr_auc')} | {mean_std(selected,'f1')} | {mean_std(selected,'branch_macro_srcc')} | {mean_std(selected,'attribute_mean_srcc')} | {mean_std(selected,'q_rank_quality_srcc')} | {weights} |")
        baseline = [r for r in rows if r["model"] == "P0" and r["seed"] in {42, 123, 2026}]
        if len(baseline) == 3:
            for name in (n for n in complete_names if n != "P0"):
                selected = [r for r in rows if r["model"] == name and r["seed"] in {42, 123, 2026}]
                lines.extend([
                    "",
                    f"Multi-seed delta ({name}-P0): AUC `{statistics.fmean(r['auc'] for r in selected)-statistics.fmean(r['auc'] for r in baseline):+.4f}`, "
                    f"PR-AUC `{statistics.fmean(r['pr_auc'] for r in selected)-statistics.fmean(r['pr_auc'] for r in baseline):+.4f}`, "
                    f"F1 `{statistics.fmean(r['f1'] for r in selected)-statistics.fmean(r['f1'] for r in baseline):+.4f}`, "
                    f"branch macro `{statistics.fmean(r['branch_macro_srcc'] for r in selected)-statistics.fmean(r['branch_macro_srcc'] for r in baseline):+.4f}`, "
                    f"q_rank-quality `{statistics.fmean(r['q_rank_quality_srcc'] for r in selected)-statistics.fmean(r['q_rank_quality_srcc'] for r in baseline):+.4f}`.",
                ])
        for name in (n for n in complete_names if n != "P0"):
            selected = [r for r in rows if r["model"] == name and r["seed"] in {42, 123, 2026}]
            lines.extend([
                "", f"### {name} attribute SRCC", "", "| Attribute | Mean +/- std |", "|---|---:|",
            ])
            for key in ATTRIBUTE_RESULT_KEYS:
                lines.append(f"| {key.removesuffix('_srcc')} | {mean_std(selected,key)} |")
            if name == "P2":
                lines.extend([
                    "", "### P2 quality modeling", "",
                    "| Metric | Mean +/- std |", "|---|---:|",
                    f"| Quality SRCC | {mean_std(selected, 'quality_srcc')} |",
                    f"| Quality PLCC | {mean_std(selected, 'quality_plcc')} |",
                    f"| Quality RMSE | {mean_std(selected, 'quality_rmse')} |",
                    f"| Quality MAE | {mean_std(selected, 'quality_mae')} |",
                ])
    branch_answer = "REMOVE_BRANCH_SUPERVISION" if decision == "PROMOTE_P2_S2_ATTRIBUTE_SIMPLE" else "KEEP_BRANCH_SUPERVISION"
    lines.extend(["", "## Final", "", f"- Branch supervision: **{branch_answer}**"])
    if decision != "PROMOTE_P2_S2_ATTRIBUTE_QUALITY" and "P2" in seed42:
        lines.append("- Quality head is useful for diagnostic continuous quality modeling, but is not promoted into the final ranking model.")
    lines.append(f"- Protocol-gate candidate: **{multi_seed_candidate or 'P0'}**")
    if FINAL_SELECTION_OVERRIDE:
        lines.append(
            "- Final selection override: retain P0 because P1's ranking gain is marginal and "
            "F1, branch macro SRCC, and q_rank-quality SRCC all decline."
        )
    lines.append(f"- Decision: **{decision}**")
    (args.output_dir / "progressive_attribute_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    logger = logger_for(args.output_dir / f"progressive_attribute_seed{args.seed}.log")
    device = get_device(CFG.device)
    train_ds, val_ds, split_hash = prepare_data(args, logger)
    if args.split_seed != 42 or split_hash != EXPECTED_HASH:
        raise RuntimeError(f"Unexpected split: seed={args.split_seed}, hash={split_hash}")
    clean_ids = {str(sample["video_id"]) for sample in train_ds.samples + val_ds.samples}
    audit = audit_attributes(args, clean_ids)
    names = [name.strip().upper().replace("-", "_") for name in args.models.split(",") if name.strip()]
    unknown = sorted(set(names) - set(PLANS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")

    result_path = args.output_dir / "progressive_attribute_results.csv"
    stage_path = args.output_dir / "progressive_attribute_stage_results.csv"
    history_path = args.output_dir / "progressive_attribute_history.csv"
    results = read_csv(result_path, RESULT_FIELDS)
    stages = read_csv(stage_path, STAGE_FIELDS)
    history = read_csv(history_path, HISTORY_FIELDS)
    if args.force:
        history = [row for row in history if not (row["seed"] == args.seed and row["model"] in names)]
    current_results, current_stages = [], []
    for name in names:
        if name == "P1_SIMPLE":
            allowed, reason = trigger_simple(results + current_results)
            if not allowed:
                raise RuntimeError(f"P1-Simple trigger not met: {reason}")
        if name == "P0":
            result, stage_rows = load_p0(train_ds, val_ds, device, args, logger, split_hash)
        else:
            result, stage_rows = train_plan(name, PLANS[name], train_ds, val_ds, device, args, logger, split_hash, history)
        current_results.append(result)
        current_stages.extend(stage_rows)
        results = merge(results, current_results, ("model", "seed"))
        stages = merge(stages, current_stages, ("model", "seed", "stage"))
        write_csv(result_path, results, RESULT_FIELDS)
        write_csv(stage_path, stages, STAGE_FIELDS)
        write_csv(history_path, history, HISTORY_FIELDS)
        write_summary(args, results, stages, history, audit)
    logger.info("Completed %s | results=%s", names, result_path)


if __name__ == "__main__":
    main()
