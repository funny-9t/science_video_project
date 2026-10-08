"""Compare legacy dual-CLIP aesthetics with shared ViT-L/14 prompt features."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("legacy", "shared_clip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--seeds", default="42,123,2026")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "aesthetic_ablation",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "logs" / "aesthetic_ablation.json",
    )
    parser.add_argument(
        "--legacy-checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "technical_ablation",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "features",
    )
    return parser.parse_args()


def feature_statistics(feature_dir: Path) -> dict:
    legacy = []
    shared = []
    for path in feature_dir.glob("*.pt"):
        sample = torch.load(path, map_location="cpu", weights_only=False)
        old = np.asarray(sample.get("aes_feat", []), dtype=np.float32)
        new = np.asarray(sample.get("aes_shared_clip_feat", []), dtype=np.float32)
        if old.shape == (7,) and new.shape == (7,):
            legacy.append(old)
            shared.append(new)
    if not legacy:
        return {"count": 0}
    old = np.stack(legacy)
    new = np.stack(shared)
    correlations = []
    for index in range(old.shape[1]):
        value = np.corrcoef(old[:, index], new[:, index])[0, 1]
        correlations.append(float(value) if np.isfinite(value) else 0.0)
    return {
        "count": int(old.shape[0]),
        "dimension_pearson": correlations,
        "legacy_mean": old.mean(axis=0).tolist(),
        "legacy_std": old.std(axis=0).tolist(),
        "shared_mean": new.mean(axis=0).tolist(),
        "shared_std": new.std(axis=0).tolist(),
    }


def summarize(rows: list[dict]) -> list[dict]:
    result = []
    for backend in BACKENDS:
        group = [row for row in rows if row["backend"] == backend]
        if not group:
            continue
        def values(key: str) -> list[float]:
            return [float(row[key]) for row in group]
        result.append({
            "backend": backend,
            "runs": len(group),
            "video_auc_mean": statistics.mean(values("video_auc")),
            "video_auc_std": statistics.pstdev(values("video_auc")) if len(group) > 1 else 0.0,
            "video_pr_auc_mean": statistics.mean(values("video_pr_auc")),
            "aesthetic_srcc_mean": statistics.mean(values("aesthetic_srcc")),
            "aesthetic_srcc_std": statistics.pstdev(values("aesthetic_srcc")) if len(group) > 1 else 0.0,
        })
    return sorted(result, key=lambda item: item["video_auc_mean"], reverse=True)


def main() -> None:
    args = parse_args()
    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    unknown = set(backends) - set(BACKENDS)
    if unknown:
        raise ValueError(f"Unknown aesthetic backends: {sorted(unknown)}")
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for backend in backends:
        for seed in seeds:
            checkpoint = args.output_dir / f"{backend}_seed{seed}.pt"
            legacy_checkpoint = args.legacy_checkpoint_dir / f"full_no_dnsmos_seed{seed}.pt"
            source = checkpoint
            if backend == "legacy" and legacy_checkpoint.exists() and not args.force:
                source = legacy_checkpoint
                print(f"Reusing technical-ablation checkpoint: {source}")
            elif checkpoint.exists() and not args.force:
                print(f"Reusing checkpoint: {checkpoint}")
            else:
                command = [
                    sys.executable, "training/train.py",
                    "--profile", "main_v2",
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
                    "--technical_feature_mode", "full_no_dnsmos",
                    "--aesthetic_feature_backend", backend,
                    "--checkpoint", str(checkpoint),
                ]
                print(f"\n=== aesthetic={backend} seed={seed} ===", flush=True)
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            payload = torch.load(source, map_location="cpu", weights_only=False)
            metrics = payload.get("validation_metrics", {})
            rows.append({
                "backend": backend,
                "seed": seed,
                "ranking_accuracy": float(payload.get("best_ranking_accuracy", 0.0)),
                "video_auc": float(metrics.get("auc", 0.0)),
                "video_pr_auc": float(metrics.get("pr_auc", 0.0)),
                "aesthetic_srcc": float(metrics.get("aesthetic_srcc", 0.0)),
                "technical_srcc": float(metrics.get("technical_srcc", 0.0)),
                "checkpoint": str(source),
            })
            report = {
                "feature_statistics": feature_statistics(args.feature_dir),
                "runs": rows,
                "summary": summarize(rows),
            }
            args.report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    print("\n=== aesthetic ablation summary ===")
    for row in summarize(rows):
        print(
            f"{row['backend']}: auc={row['video_auc_mean']:.4f}+-{row['video_auc_std']:.4f} "
            f"pr_auc={row['video_pr_auc_mean']:.4f} "
            f"aes_srcc={row['aesthetic_srcc_mean']:.4f}+-{row['aesthetic_srcc_std']:.4f}"
        )


if __name__ == "__main__":
    main()
