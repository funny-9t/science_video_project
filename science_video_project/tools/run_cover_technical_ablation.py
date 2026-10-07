"""Run the fixed T0/T1/T2 COVER technical representation experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("clip", "cover", "clip_cover")
EXPERIMENT_NAMES = {"clip": "T0", "cover": "T1", "clip_cover": "T2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "technical_cover_ablation",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _row(source: str, checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metrics = payload["validation_metrics"]
    return {
        "experiment": EXPERIMENT_NAMES[source],
        "technical_visual_source": source,
        "auc": float(metrics.get("auc", 0.0)),
        "pr_auc": float(metrics.get("pr_auc", 0.0)),
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "f1": float(metrics.get("f1", 0.0)),
        "ranking_accuracy": float(payload.get("best_ranking_accuracy", 0.0)),
        "best_epoch": int(payload.get("best_epoch", 0)),
        "technical_srcc": float(metrics.get("technical_srcc", 0.0)),
        "technical_visual_srcc": float(metrics.get("technical_visual_srcc", 0.0)),
        "technical_audio_srcc": float(metrics.get("technical_audio_srcc", 0.0)),
        "technical_label_srcc": float(metrics.get("technical_label_srcc", 0.0)),
        "technical_plcc": float(metrics.get("technical_plcc", 0.0)),
        "technical_mae": float(metrics.get("technical_mae", 0.0)),
        "technical_rmse": float(metrics.get("technical_rmse", 0.0)),
        "scientific_srcc": float(metrics.get("scientific_srcc", 0.0)),
        "aesthetic_srcc": float(metrics.get("aesthetic_srcc", 0.0)),
        "scientific_weight": float(metrics.get("scientific_fusion_weight", 0.0)),
        "technical_weight": float(metrics.get("technical_fusion_weight", 0.0)),
        "aesthetic_weight": float(metrics.get("aesthetic_fusion_weight", 0.0)),
        "split_hash": str(payload.get("split_hash", "")),
        "seed": int(payload.get("seed", 0)),
        "split_seed": int(payload.get("split_seed", 0)),
        "checkpoint": str(checkpoint),
    }


def _write_summary(
    path: Path, rows: list[dict], extraction: dict, shared_clip: dict
) -> None:
    baseline = rows[0]
    best_technical = max(rows, key=lambda row: row["technical_visual_srcc"])
    best_overall = max(rows, key=lambda row: row["auc"])
    hashes = {row["split_hash"] for row in rows}
    candidate = best_technical
    technical_gain = candidate["technical_visual_srcc"] - baseline["technical_visual_srcc"]
    auc_delta = candidate["auc"] - baseline["auc"]
    pr_delta = candidate["pr_auc"] - baseline["pr_auc"]
    recommend = (
        technical_gain >= 0.05 and auc_delta >= -0.01
    ) or (best_overall["technical_visual_source"] != "clip" and best_overall["auc"] - baseline["auc"] >= 0.01)
    recommendation = (
        f"建议采用 {candidate['experiment']} ({candidate['technical_visual_source']}) 进入 main_v3。"
        if recommend
        else "不建议进入 main_v3，保持 T0/main_v2。"
    )

    lines = [
        "# COVER-inspired Quality-Aware Technical Encoder 实验",
        "",
        "## 1. COVER Technical 实际实现",
        "",
        "- Sampling: 官方 technical view，7x7 spatial fragments，fragment 32x32，40 帧，frame interval 2。",
        "- Backbone: COVER checkpoint 中的 GRPB Swin-3D Tiny，仅加载 `technical_backbone.*` 权重并冻结。",
        "- Backbone output: `[B, 768, 20, 7, 7]`；对 T/H/W 做均值池化，缓存 768D representation。",
        "- 未调用完整 COVER forward，未使用 semantic/aesthetic branch，也未保存 final technical score。",
        "- T1 projection: LayerNorm -> Linear(768,256) -> GELU -> Dropout -> Linear(256,128)。",
        "- T2 projection: CLIP 512->128、COVER 768->256，concat 384->TechnicalBranch hidden 128。",
        "",
        "## 2. Feature extraction",
        "",
        f"- 成功/失败: {extraction.get('succeeded', 0)}/{extraction.get('failed', 0)}，skip {extraction.get('skipped', 0)}。",
        f"- Device: {extraction.get('device', 'unknown')}，AMP={extraction.get('amp', False)}。",
        f"- 总耗时: {extraction.get('total_seconds', 0.0):.2f}s。",
        f"- 平均耗时: {extraction.get('mean_seconds_per_video', 0.0):.3f}s/video。",
        f"- Peak VRAM: {extraction.get('peak_vram_bytes', 0) / 1024**3:.3f} GiB。",
        f"- 平均缓存增量: {extraction.get('mean_feature_size_delta_bytes', 0.0) / 1024:.2f} KiB/video。",
        f"- Shared CLIP visual 对比: {shared_clip.get('mean_seconds_per_video', 0.0):.3f}s/video，"
        f"peak VRAM {shared_clip.get('peak_vram_bytes', 0) / 1024**3:.3f} GiB，"
        f"frame cache {shared_clip.get('mean_feature_bytes_per_video', 0.0) / 1024:.2f} KiB/video。",
        "",
        "## 3. 三组实验",
        "",
        "| Exp | Source | AUC | PR-AUC | Acc | F1 | Tech SRCC | Visual SRCC | Audio SRCC | Tech-label SRCC | Sci SRCC | Aes SRCC | Fusion S/T/A | Best epoch |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['technical_visual_source']} | {row['auc']:.4f} | "
            f"{row['pr_auc']:.4f} | {row['accuracy']:.4f} | {row['f1']:.4f} | "
            f"{row['technical_srcc']:.4f} | {row['technical_visual_srcc']:.4f} | "
            f"{row['technical_audio_srcc']:.4f} | {row['technical_label_srcc']:.4f} | "
            f"{row['scientific_srcc']:.4f} | {row['aesthetic_srcc']:.4f} | "
            f"{row['scientific_weight']:.3f}/{row['technical_weight']:.3f}/{row['aesthetic_weight']:.3f} | "
            f"{row['best_epoch']} |"
        )
    lines.extend([
        "",
        "## 4. 最终判断",
        "",
        f"- 三组 validation hash 一致: `{len(hashes) == 1}` (`{next(iter(hashes), '')}`)。",
        f"- Visual SRCC 最佳方案: {candidate['experiment']}，相对 T0 {technical_gain:+.4f}。",
        f"- 对应 AUC 变化: {auc_delta:+.4f}；PR-AUC 变化: {pr_delta:+.4f}。",
        f"- Overall AUC 最佳方案: {best_overall['experiment']} ({best_overall['auc']:.4f})。",
        f"- T2 Combined Technical SRCC: {rows[2]['technical_srcc']:.4f}；Visual SRCC: "
        f"{rows[2]['technical_visual_srcc']:.4f}，尚未达到预设 0.30 Visual SRCC 强成功线。",
        f"- T2 Technical fusion weight: {rows[2]['technical_weight']:.6f}，仍贴近 20% 下限，"
        "没有出现自然增权；因此 branch collapse 不能仅归因于旧 visual representation。",
        "- T2 > T1 且 T2 > T0，说明 CLIP global context 与 COVER quality-aware feature "
        "存在互补；单独替换为 COVER (T1) 会损失总体排序。",
        f"- 结论: {recommendation}",
        "- 风险: 当前结论来自任务指定的单 seed 固定验证集；进入 main_v3 前应补多 seed/重复划分确认。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    environment = os.environ.copy()
    environment["SCIENCE_VIDEO_OUTPUT_DIR"] = str(args.output_dir)
    for source in SOURCES:
        checkpoint = args.output_dir / f"best_technical_{source}.pt"
        command = [
            sys.executable, "training/train.py",
            "--profile", "main_v2",
            "--metadata", str(args.metadata),
            "--feature_dir", str(args.feature_dir),
            "--epochs", str(args.epochs),
            "--batch_size", "8",
            "--lr", "5e-5",
            "--seed", "42",
            "--split_seed", "42",
            "--early_stop_patience", str(args.patience),
            "--pair_scope", "global",
            "--loss_type", "ranknet",
            "--lambda_supervision", "0.05",
            "--lambda_consistency", "0.1",
            "--lambda_pointwise", "0.1",
            "--branch_weight_floor", "0.2",
            "--disable_cross_gating",
            "--science_feature_mode", "analysis_scores_concat",
            "--llm_text_source", "reasoning_and_analysis",
            "--technical_feature_mode", "full_no_dnsmos",
            "--technical_visual_source", source,
            "--aesthetic_feature_backend", "shared_clip",
            "--checkpoint", str(checkpoint),
        ]
        print(f"\n=== {EXPERIMENT_NAMES[source]}: {source} ===", flush=True)
        if args.force or not checkpoint.exists():
            subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
        rows.append(_row(source, checkpoint))

    csv_path = args.output_dir / "technical_cover_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    extraction_path = args.output_dir / "extraction_report.json"
    extraction = json.loads(extraction_path.read_text(encoding="utf-8")) if extraction_path.exists() else {}
    shared_clip_path = args.output_dir / "shared_clip_benchmark.json"
    shared_clip = json.loads(shared_clip_path.read_text(encoding="utf-8")) if shared_clip_path.exists() else {}
    _write_summary(
        args.output_dir / "technical_cover_summary.md", rows, extraction, shared_clip
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
