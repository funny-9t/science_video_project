"""Run multi-seed stability validation after the A2 seed-42 gate passes."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import CFG
from training.utils_train import get_device
from tools import run_aesthetic_prompt_a2 as experiment


SEEDS = (42, 123, 2026)
METRICS = (
    "aes_srcc", "aes_plcc", "aes_mse", "aes_mae", "auc", "pr_auc",
    "accuracy", "f1", "overall_srcc", "science_weight",
    "technical_weight", "aesthetic_weight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata", type=Path,
        default=Path(r"D:\Projects\science_video_ranker_mvp\science_video_project\data\parsed_metadata_filtered.csv"),
    )
    parser.add_argument(
        "--feature-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "features",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "experiments" / "aesthetic_prompt_a2",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--force-training", action="store_true")
    return parser.parse_args()


def a1_checkpoint(seed: int, stage: str) -> Path:
    root = PROJECT_ROOT / "outputs" / "progressive_training" / "P2"
    if seed != 42:
        root = root / f"seed_{seed}"
    return root / f"{stage}_best.pt"


def mean_std(rows: list[dict], variant: str, metric: str) -> tuple[float, float]:
    values = [float(row[metric]) for row in rows if row["variant"] == variant]
    return statistics.fmean(values), statistics.stdev(values)


def update_metric_json(path: Path, rows: list[dict], variant: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["multi_seed_stage2"] = {
        "seeds": list(SEEDS),
        "metrics": {
            metric: {
                "mean": mean_std(rows, variant, metric)[0],
                "std": mean_std(rows, variant, metric)[1],
            }
            for metric in METRICS
        },
    }
    experiment.write_json(path, payload)


def append_summary(output_dir: Path, rows: list[dict]) -> None:
    summary_path = output_dir / "aesthetic_prompt_experiment_summary.md"
    text = summary_path.read_text(encoding="utf-8")
    marker = "\n## 13. Multi-seed 稳定性验证与最终决策\n"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n"
    aggregate = {
        variant: {
            metric: mean_std(rows, variant, metric) for metric in METRICS
        }
        for variant in experiment.PROMPT_VERSIONS
    }
    a1 = aggregate["a1_original"]
    a2 = aggregate["a2_domain_semantic"]
    branch_improved = (
        a2["aes_srcc"][0] > a1["aes_srcc"][0]
        and a2["aes_plcc"][0] > a1["aes_plcc"][0]
    )
    overall_stable = (
        a2["pr_auc"][0] >= a1["pr_auc"][0] - 0.02
        and a2["auc"][0] >= a1["auc"][0] - 0.01
        and a2["overall_srcc"][0] >= a1["overall_srcc"][0] - 0.02
    )
    final_decision = (
        "多种子均值支持 A2：建议升级为主模型美学 Prompt，并保留 A1 作为消融基线。"
        if branch_improved and overall_stable else
        "多种子均值未同时支持美学分支改善与整体稳定：主模型继续保留 A1，A2 作为负结果。"
    )
    lines = [
        "",
        "## 13. Multi-seed 稳定性验证与最终决策",
        "",
        "固定 seeds 42/123/2026，所有 seed 使用同一 split_seed=42 和同一 validation hash。",
        "",
        "| Metric | A1 mean ± std | A2 mean ± std | Delta mean |",
        "|---|---:|---:|---:|",
    ]
    labels = (
        ("Aesthetic SRCC", "aes_srcc"), ("Aesthetic PLCC", "aes_plcc"),
        ("Aesthetic MSE", "aes_mse"), ("Aesthetic MAE", "aes_mae"),
        ("ROC-AUC", "auc"), ("PR-AUC", "pr_auc"), ("F1", "f1"),
        ("Overall SRCC", "overall_srcc"),
    )
    for label, metric in labels:
        a1_mean, a1_std = a1[metric]
        a2_mean, a2_std = a2[metric]
        lines.append(
            f"| {label} | {a1_mean:.4f} ± {a1_std:.4f} | "
            f"{a2_mean:.4f} ± {a2_std:.4f} | {a2_mean - a1_mean:+.4f} |"
        )
    lines.extend([
        "",
        f"多种子 branch improvement=`{branch_improved}`，overall stability=`{overall_stable}`。",
        "",
        final_decision,
    ])
    summary_path.write_text(text.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    metadata, train_df, val_df, split_hash = experiment.split_metadata(
        args.metadata, args.feature_dir, args.split_seed
    )
    if split_hash != experiment.EXPECTED_SPLIT_HASH:
        raise ValueError(f"Unexpected validation hash: {split_hash}")
    a2_cache_dir = experiment.feature_cache_dir(args.output_dir, "a2_domain_semantic")
    if not all(
        experiment.valid_a2_cache(a2_cache_dir / f"{video_id}.pt")
        for video_id in metadata["video_id"].astype(str)
    ):
        raise ValueError("A2 cache is incomplete or stale; run the seed-42 experiment first")

    _, a1_val = experiment.build_datasets(
        train_df, val_df, args.feature_dir, "a1_original", a2_cache_dir
    )
    a2_train, a2_val = experiment.build_datasets(
        train_df, val_df, args.feature_dir, "a2_domain_semantic", a2_cache_dir
    )
    device = get_device(CFG.device)
    rows = []
    for seed in SEEDS:
        seed_args = argparse.Namespace(
            output_dir=args.output_dir,
            seed=seed,
            split_seed=args.split_seed,
            force_training=args.force_training,
        )
        _, a2_stage2 = experiment.train_a2(
            a2_train, a2_val, device, seed_args, split_hash
        )
        for variant, dataset, checkpoint in (
            ("a1_original", a1_val, a1_checkpoint(seed, "stage2")),
            ("a2_domain_semantic", a2_val, a2_stage2),
        ):
            metrics, _ = experiment.load_checkpoint_metrics(
                checkpoint, dataset, device, split_hash
            )
            rows.append({
                "seed": seed,
                "variant": variant,
                "validation_hash": split_hash,
                **{metric: metrics[metric] for metric in METRICS},
                "best_epoch": metrics["best_epoch"],
                "checkpoint": metrics["checkpoint"],
            })
    experiment.write_csv(args.output_dir / "multiseed_metrics.csv", rows)
    update_metric_json(args.output_dir / "a1_metrics.json", rows, "a1_original")
    update_metric_json(args.output_dir / "a2_metrics.json", rows, "a2_domain_semantic")
    append_summary(args.output_dir, rows)
    print(f"Multi-seed experiment complete: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
