"""Run COVER-inspired ablations while keeping the scientific branch fixed."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = {
    "baseline_legacy": [],
    "ranknet_calibrated": ["--loss_type", "ranknet", "--lambda_pointwise", "0.1"],
    "ranknet_no_llm": [
        "--loss_type", "ranknet", "--lambda_pointwise", "0.1",
        "--science_feature_mode", "none",
    ],
    "ranknet_scores_only": [
        "--loss_type", "ranknet", "--lambda_pointwise", "0.1",
        "--science_feature_mode", "scores",
    ],
    "ranknet_analysis_only": [
        "--loss_type", "ranknet", "--lambda_pointwise", "0.1",
        "--science_feature_mode", "analysis_concat", "--llm_text_source", "analysis",
    ],
    "ranknet_floor_005": [
        "--loss_type", "ranknet", "--lambda_pointwise", "0.1",
        "--branch_weight_floor", "0.05",
    ],
    "ranknet_floor_010": [
        "--loss_type", "ranknet", "--lambda_pointwise", "0.1",
        "--branch_weight_floor", "0.10",
    ],
    "ranknet_floor_015": [
        "--loss_type", "ranknet", "--lambda_pointwise", "0.1",
        "--branch_weight_floor", "0.15",
    ],
    "ranknet_floor_020": [
        "--loss_type", "ranknet", "--lambda_pointwise", "0.1",
        "--branch_weight_floor", "0.20",
    ],
    "ranknet_floor_025": [
        "--loss_type", "ranknet", "--lambda_pointwise", "0.1",
        "--branch_weight_floor", "0.25",
    ],
    "native_clip": [
        "--loss_type", "ranknet", "--lambda_pointwise", "0.1", "--use_native_clip",
    ],
    "average_no_cover": [
        "--loss_type", "ranknet", "--lambda_pointwise", "0.1",
        "--fusion_mode", "average", "--lambda_consistency", "0",
    ],
    "average_no_cover_staged": [
        "--loss_type", "ranknet", "--lambda_pointwise", "0.1",
        "--fusion_mode", "average", "--lambda_consistency", "0",
        "--branch_pretrain_epochs", "5", "--branch_pretrain_supervision", "1.0",
    ],
    "cover_learned": [
        "--loss_type", "ranknet", "--lambda_pointwise", "0.1",
        "--use_cover_features",
    ],
    "cover_average": [
        "--loss_type", "ranknet", "--lambda_pointwise", "0.1",
        "--use_cover_features", "--fusion_mode", "average", "--lambda_consistency", "0",
    ],
    "cover_average_no_cross_gate": [
        "--loss_type", "ranknet", "--lambda_pointwise", "0.1",
        "--use_cover_features", "--fusion_mode", "average", "--lambda_consistency", "0",
        "--disable_cross_gating",
    ],
    "cover_native_average": [
        "--loss_type", "ranknet", "--lambda_pointwise", "0.1",
        "--use_native_clip", "--use_cover_features", "--fusion_mode", "average", "--lambda_consistency", "0",
    ],
    "cover_average_no_llm": [
        "--loss_type", "ranknet", "--lambda_pointwise", "0.1",
        "--use_cover_features", "--fusion_mode", "average", "--lambda_consistency", "0",
        "--science_feature_mode", "none",
    ],
    "cover_average_scores_concat": [
        "--loss_type", "ranknet", "--lambda_pointwise", "0.1",
        "--use_cover_features", "--fusion_mode", "average", "--lambda_consistency", "0",
        "--science_feature_mode", "analysis_scores_concat",
    ],
    "cover_average_scores_ifg": [
        "--loss_type", "ranknet", "--lambda_pointwise", "0.1",
        "--use_cover_features", "--fusion_mode", "average", "--lambda_consistency", "0",
        "--science_feature_mode", "analysis_scores_ifg",
    ],
    "cover_average_full_ifg": [
        "--loss_type", "ranknet", "--lambda_pointwise", "0.1",
        "--use_cover_features", "--fusion_mode", "average", "--lambda_consistency", "0",
        "--science_feature_mode", "full_ifg",
        "--llm_text_source", "reasoning_and_analysis",
    ],
    "cover_average_staged": [
        "--loss_type", "ranknet", "--lambda_pointwise", "0.1",
        "--use_cover_features", "--fusion_mode", "average", "--lambda_consistency", "0",
        "--branch_pretrain_epochs", "5", "--branch_pretrain_supervision", "1.0",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", default=",".join(EXPERIMENTS))
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--science-feature-mode",
        choices=["none", "scores", "analysis_concat", "analysis_scores_concat", "analysis_scores_ifg", "full_ifg"],
        default="analysis_scores_concat",
    )
    parser.add_argument(
        "--llm-text-source",
        choices=["analysis", "reasoning_and_analysis"],
        default="reasoning_and_analysis",
    )
    parser.add_argument("--disable-audio", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "cover_ablation",
    )
    parser.add_argument(
        "--report", type=Path,
        default=PROJECT_ROOT / "outputs" / "logs" / "cover_ablation.json",
    )
    return parser.parse_args()


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for name in sorted({row["name"] for row in rows}):
        group = [row for row in rows if row["name"] == name]
        ranks = [row["ranking_accuracy"] for row in group]
        aucs = [row["video_auc"] for row in group]
        prs = [row["video_pr_auc"] for row in group]
        science = [row["scientific_srcc"] for row in group]
        technical = [row["technical_srcc"] for row in group]
        aesthetic = [row["aesthetic_srcc"] for row in group]
        science_weights = [row["scientific_fusion_weight"] for row in group]
        summary.append({
            "name": name,
            "runs": len(group),
            "ranking_mean": statistics.mean(ranks),
            "ranking_std": statistics.pstdev(ranks) if len(ranks) > 1 else 0.0,
            "video_auc_mean": statistics.mean(aucs),
            "video_pr_auc_mean": statistics.mean(prs),
            "scientific_srcc_mean": statistics.mean(science),
            "technical_srcc_mean": statistics.mean(technical),
            "aesthetic_srcc_mean": statistics.mean(aesthetic),
            "scientific_fusion_weight_mean": statistics.mean(science_weights),
        })
    return sorted(summary, key=lambda item: item["ranking_mean"], reverse=True)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.report = args.report.resolve()
    names = [name.strip() for name in args.experiments.split(",") if name.strip()]
    unknown = set(names) - set(EXPERIMENTS)
    if unknown:
        raise ValueError(f"Unknown experiments: {sorted(unknown)}")
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for name in names:
        for seed in seeds:
            checkpoint = args.output_dir / f"{name}_seed{seed}.pt"
            command = [
                sys.executable, "training/train.py",
                "--epochs", str(args.epochs),
                "--early_stop_patience", str(args.patience),
                "--lr", str(args.lr),
                "--seed", str(seed),
                "--split_seed", str(args.split_seed),
                "--pair_scope", "global",
                "--science_feature_mode", args.science_feature_mode,
                "--llm_text_source", args.llm_text_source,
                "--checkpoint", str(checkpoint),
            ]
            if args.disable_audio:
                command.append("--disable_audio")
            if name == "baseline_legacy":
                command += ["--loss_type", "focal"]
            command += EXPERIMENTS[name]
            print(f"\n=== {name} seed={seed} ===", flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            metrics = payload.get("validation_metrics", {})
            rows.append({
                "name": name,
                "seed": seed,
                "ranking_accuracy": float(payload.get("best_ranking_accuracy", 0.0)),
                "video_auc": float(metrics.get("auc", 0.0)),
                "video_pr_auc": float(metrics.get("pr_auc", 0.0)),
                "f1": float(metrics.get("f1", 0.0)),
                "scientific_srcc": float(metrics.get("scientific_srcc", 0.0)),
                "technical_srcc": float(metrics.get("technical_srcc", 0.0)),
                "aesthetic_srcc": float(metrics.get("aesthetic_srcc", 0.0)),
                "scientific_fusion_weight": float(
                    metrics.get("scientific_fusion_weight", 0.0)
                ),
                "best_epoch": int(payload.get("best_epoch", 0)),
                "best_threshold": float(payload.get("best_threshold", 0.5)),
                "checkpoint": str(checkpoint),
            })
            args.report.write_text(
                json.dumps({"runs": rows, "summary": summarize(rows)}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    report = {"runs": rows, "summary": summarize(rows)}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== summary ===")
    for row in report["summary"]:
        print(
            f"{row['name']}: rank={row['ranking_mean']:.4f}+-{row['ranking_std']:.4f} "
            f"auc={row['video_auc_mean']:.4f} pr_auc={row['video_pr_auc_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
