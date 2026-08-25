"""Run the scientific-branch ablation matrix and summarize checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    ("no_llm", "none", "analysis"),
    ("scores_only", "scores", "analysis"),
    ("analysis_only_concat", "analysis_concat", "analysis"),
    ("analysis_scores_concat", "analysis_scores_concat", "analysis"),
    ("reasoning_scores_concat", "analysis_scores_concat", "reasoning_and_analysis"),
    ("analysis_scores_ifg", "analysis_scores_ifg", "analysis"),
    ("reasoning_scores_ifg", "analysis_scores_ifg", "reasoning_and_analysis"),
    ("full_ifg", "full_ifg", "reasoning_and_analysis"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "checkpoints" / "science_ablation")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "outputs" / "logs" / "science_ablation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for name, mode, text_source in EXPERIMENTS:
        checkpoint = args.output_dir / f"{name}.pt"
        command = [
            sys.executable,
            "training/train.py",
            "--epochs",
            str(args.epochs),
            "--early_stop_patience",
            str(args.patience),
            "--lr",
            str(args.lr),
            "--pair_scope",
            "global",
            "--technical_feature_mode",
            "full",
            "--aesthetic_feature_backend",
            "legacy",
            "--loss_type",
            "focal",
            "--science_feature_mode",
            mode,
            "--llm_text_source",
            text_source,
            "--checkpoint",
            str(checkpoint),
        ]
        print(f"\n=== {name} ===", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        results.append(
            {
                "name": name,
                "science_feature_mode": mode,
                "llm_text_source": text_source,
                "ranking_accuracy": float(payload.get("best_ranking_accuracy", 0.0)),
                "best_epoch": int(payload.get("best_epoch", 0)),
                "best_threshold": float(payload.get("best_threshold", 0.5)),
                "validation_metrics": payload.get("validation_metrics", {}),
                "checkpoint": str(checkpoint),
            }
        )
        args.report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    results.sort(key=lambda item: item["ranking_accuracy"], reverse=True)
    args.report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== ranking ===")
    for result in results:
        print(f"{result['name']}: {result['ranking_accuracy']:.4f}")


if __name__ == "__main__":
    main()
