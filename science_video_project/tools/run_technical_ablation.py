"""Run controlled technical-feature ablations on a fixed data split."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODES = ("visual_only", "visual_dnsmos", "dnsmos_only", "full_no_dnsmos", "full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--seeds", default="42,123,2026")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--force", action="store_true", help="Retrain even when a checkpoint exists.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "technical_ablation",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "logs" / "technical_ablation.json",
    )
    return parser.parse_args()


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for mode in MODES:
        group = [row for row in rows if row["mode"] == mode]
        if not group:
            continue

        def metric(name: str) -> list[float]:
            return [float(row[name]) for row in group]

        summary.append({
            "mode": mode,
            "runs": len(group),
            "ranking_accuracy_mean": statistics.mean(metric("ranking_accuracy")),
            "ranking_accuracy_std": statistics.pstdev(metric("ranking_accuracy")) if len(group) > 1 else 0.0,
            "video_auc_mean": statistics.mean(metric("video_auc")),
            "video_auc_std": statistics.pstdev(metric("video_auc")) if len(group) > 1 else 0.0,
            "video_pr_auc_mean": statistics.mean(metric("video_pr_auc")),
            "technical_srcc_mean": statistics.mean(metric("technical_srcc")),
            "technical_srcc_std": statistics.pstdev(metric("technical_srcc")) if len(group) > 1 else 0.0,
        })
    return sorted(summary, key=lambda item: item["video_auc_mean"], reverse=True)


def write_report(path: Path, rows: list[dict]) -> None:
    payload = {"runs": rows, "summary": summarize(rows)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown = set(modes) - set(MODES)
    if unknown:
        raise ValueError(f"Unknown technical feature modes: {sorted(unknown)}")
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for mode in modes:
        for seed in seeds:
            checkpoint = args.output_dir / f"{mode}_seed{seed}.pt"
            command = [
                sys.executable,
                "training/train.py",
                "--epochs", str(args.epochs),
                "--early_stop_patience", str(args.patience),
                "--lr", str(args.lr),
                "--seed", str(seed),
                "--split_seed", str(args.split_seed),
                "--pair_scope", "global",
                "--loss_type", "ranknet",
                "--lambda_pointwise", "0.1",
                "--branch_weight_floor", "0.2",
                "--disable_cross_gating",
                "--science_feature_mode", "analysis_scores_concat",
                "--llm_text_source", "reasoning_and_analysis",
                "--technical_feature_mode", mode,
                "--aesthetic_feature_backend", "legacy",
                "--checkpoint", str(checkpoint),
            ]
            print(f"\n=== technical={mode} seed={seed} ===", flush=True)
            if checkpoint.exists() and not args.force:
                print(f"Reusing checkpoint: {checkpoint}", flush=True)
            else:
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            metrics = payload.get("validation_metrics", {})
            rows.append({
                "mode": mode,
                "seed": seed,
                "ranking_accuracy": float(payload.get("best_ranking_accuracy", 0.0)),
                "video_auc": float(metrics.get("auc", 0.0)),
                "video_pr_auc": float(metrics.get("pr_auc", 0.0)),
                "technical_srcc": float(metrics.get("technical_srcc", 0.0)),
                "scientific_srcc": float(metrics.get("scientific_srcc", 0.0)),
                "aesthetic_srcc": float(metrics.get("aesthetic_srcc", 0.0)),
                "best_epoch": int(payload.get("best_epoch", 0)),
                "checkpoint": str(checkpoint),
            })
            write_report(args.report, rows)

    report = {"runs": rows, "summary": summarize(rows)}
    print("\n=== technical ablation summary ===")
    for row in report["summary"]:
        print(
            f"{row['mode']}: auc={row['video_auc_mean']:.4f}+-{row['video_auc_std']:.4f} "
            f"pr_auc={row['video_pr_auc_mean']:.4f} "
            f"tech_srcc={row['technical_srcc_mean']:.4f}+-{row['technical_srcc_std']:.4f}"
        )


if __name__ == "__main__":
    main()
