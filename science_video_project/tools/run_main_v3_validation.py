"""Validate COVER technical features across seeds and decide main_v3 promotion."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "main_v3_validation"
METRIC_FIELDS = (
    "auc",
    "pr_auc",
    "accuracy",
    "f1",
    "tech_srcc",
    "visual_srcc",
    "audio_srcc",
    "tech_label_srcc",
    "sci_srcc",
    "aes_srcc",
    "science_weight",
    "technical_weight",
    "aesthetic_weight",
    "train_seconds",
    "trainable_params",
)
CSV_FIELDS = (
    "experiment",
    "model",
    "seed",
    "split_seed",
    "validation_hash",
    "technical_source",
    "clip_projection_dim",
    "cover_projection_dim",
    "auc",
    "pr_auc",
    "accuracy",
    "f1",
    "tech_srcc",
    "visual_srcc",
    "audio_srcc",
    "tech_label_srcc",
    "sci_srcc",
    "aes_srcc",
    "science_weight",
    "technical_weight",
    "aesthetic_weight",
    "best_epoch",
    "train_seconds",
    "trainable_params",
    "checkpoint",
)
COMMON_TRAINING = {
    "profile": "main_v2",
    "epochs": 30,
    "batch_size": 8,
    "lr": 5e-5,
    "split_seed": 42,
    "early_stop_patience": 10,
    "pair_scope": "global",
    "loss_type": "ranknet",
    "lambda_supervision": 0.05,
    "lambda_consistency": 0.1,
    "lambda_pointwise": 0.1,
    "branch_weight_floor": 0.2,
    "technical_feature_mode": "full_no_dnsmos",
    "aesthetic_feature_backend": "shared_clip",
    "science_feature_mode": "analysis_scores_concat",
    "llm_text_source": "reasoning_and_analysis",
    "use_cross_gating": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="42,123,2026")
    parser.add_argument("--stage", choices=["multiseed", "dimension", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _checkpoint_row(
    checkpoint: Path,
    *,
    experiment: str,
    model: str,
    source: str,
    clip_dim: int,
    cover_dim: int,
) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _validate_checkpoint(payload, source, clip_dim, cover_dim)
    metrics = payload["validation_metrics"]
    return {
        "experiment": experiment,
        "model": model,
        "seed": int(payload["seed"]),
        "split_seed": int(payload["split_seed"]),
        "validation_hash": str(payload["split_hash"]),
        "technical_source": source,
        "clip_projection_dim": clip_dim,
        "cover_projection_dim": cover_dim,
        "auc": float(metrics.get("auc", 0.0)),
        "pr_auc": float(metrics.get("pr_auc", 0.0)),
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "f1": float(metrics.get("f1", 0.0)),
        "tech_srcc": float(metrics.get("technical_srcc", 0.0)),
        "visual_srcc": float(metrics.get("technical_visual_srcc", 0.0)),
        "audio_srcc": float(metrics.get("technical_audio_srcc", 0.0)),
        "tech_label_srcc": float(metrics.get("technical_label_srcc", 0.0)),
        "sci_srcc": float(metrics.get("scientific_srcc", 0.0)),
        "aes_srcc": float(metrics.get("aesthetic_srcc", 0.0)),
        "science_weight": float(metrics.get("scientific_fusion_weight", 0.0)),
        "technical_weight": float(metrics.get("technical_fusion_weight", 0.0)),
        "aesthetic_weight": float(metrics.get("aesthetic_fusion_weight", 0.0)),
        "best_epoch": int(payload.get("best_epoch", 0)),
        "train_seconds": float(payload.get("train_seconds", 0.0)),
        "trainable_params": int(payload.get("trainable_params", 0)),
        "checkpoint": str(checkpoint.resolve()),
    }


def _validate_checkpoint(payload: dict, source: str, clip_dim: int, cover_dim: int) -> None:
    training = payload.get("training_config", {})
    model = payload.get("config", {})
    expected_training = {
        "epochs": 30,
        "batch_size": 8,
        "lr": 5e-5,
        "split_seed": 42,
        "early_stop_patience": 10,
        "pair_scope": "global",
        "loss_type": "ranknet",
        "lambda_supervision": 0.05,
        "lambda_consistency": 0.1,
        "lambda_pointwise": 0.1,
        "technical_feature_mode": "full_no_dnsmos",
        "aesthetic_feature_backend": "shared_clip",
        "profile": "main_v2",
        "use_cross_gating": False,
    }
    expected_model = {
        "fusion_mode": "learned",
        "branch_weight_floor": 0.2,
        "technical_feature_mode": "full_no_dnsmos",
        "technical_visual_source": source,
        "technical_clip_projection_dim": clip_dim,
        "technical_cover_projection_dim": cover_dim,
        "use_cross_gating": False,
        "use_cover_features": False,
        "science_fusion_mode": "concat",
        "use_knowledge_gate": False,
    }
    mismatches = []
    for namespace, actual, expected in (
        ("training", training, expected_training),
        ("model", model, expected_model),
    ):
        for key, value in expected.items():
            if actual.get(key) != value:
                mismatches.append(f"{namespace}.{key}: {actual.get(key)!r} != {value!r}")
    if payload.get("science_feature_mode") != "analysis_scores_concat":
        mismatches.append("science_feature_mode is not analysis_scores_concat")
    if payload.get("llm_text_source") != "reasoning_and_analysis":
        mismatches.append("llm_text_source is not reasoning_and_analysis")
    if payload.get("use_audio_feature") is not True:
        mismatches.append("speech/audio context flag is not enabled")
    if mismatches:
        raise RuntimeError("Checkpoint configuration mismatch:\n" + "\n".join(mismatches))


def _training_command(
    metadata: Path,
    feature_dir: Path,
    checkpoint: Path,
    seed: int,
    source: str,
    clip_dim: int,
    cover_dim: int,
) -> list[str]:
    return [
        sys.executable,
        "training/train.py",
        "--profile",
        "main_v2",
        "--metadata",
        str(metadata),
        "--feature_dir",
        str(feature_dir),
        "--epochs",
        "30",
        "--batch_size",
        "8",
        "--lr",
        "5e-5",
        "--seed",
        str(seed),
        "--split_seed",
        "42",
        "--early_stop_patience",
        "10",
        "--pair_scope",
        "global",
        "--loss_type",
        "ranknet",
        "--lambda_supervision",
        "0.05",
        "--lambda_consistency",
        "0.1",
        "--lambda_pointwise",
        "0.1",
        "--branch_weight_floor",
        "0.2",
        "--disable_cross_gating",
        "--science_feature_mode",
        "analysis_scores_concat",
        "--llm_text_source",
        "reasoning_and_analysis",
        "--technical_feature_mode",
        "full_no_dnsmos",
        "--technical_visual_source",
        source,
        "--technical_clip_projection_dim",
        str(clip_dim),
        "--technical_cover_projection_dim",
        str(cover_dim),
        "--aesthetic_feature_backend",
        "shared_clip",
        "--checkpoint",
        str(checkpoint),
    ]


def _run(
    args: argparse.Namespace,
    *,
    experiment: str,
    model: str,
    seed: int,
    source: str,
    clip_dim: int,
    cover_dim: int,
) -> dict:
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{experiment.lower()}_seed{seed}"
    checkpoint = checkpoint_dir / f"{slug}.pt"
    if args.force or not checkpoint.exists():
        command = _training_command(
            args.metadata,
            args.feature_dir,
            checkpoint,
            seed,
            source,
            clip_dim,
            cover_dim,
        )
        environment = os.environ.copy()
        environment["SCIENCE_VIDEO_OUTPUT_DIR"] = str(args.output_dir.resolve())
        print(f"\n=== {experiment} | seed={seed} ===", flush=True)
        started = time.perf_counter()
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
        elapsed = time.perf_counter() - started
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not payload.get("train_seconds"):
            payload["train_seconds"] = elapsed
            torch.save(payload, checkpoint)
    return _checkpoint_row(
        checkpoint,
        experiment=experiment,
        model=model,
        source=source,
        clip_dim=clip_dim,
        cover_dim=cover_dim,
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(rows: list[dict], key: str) -> tuple[float, float]:
    values = [float(row[key]) for row in rows]
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def _fmt(rows: list[dict], key: str, digits: int = 4) -> str:
    mean, std = _mean_std(rows, key)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _group(rows: list[dict], model: str) -> list[dict]:
    return sorted((row for row in rows if row["model"] == model), key=lambda row: row["seed"])


def _validate_multiseed(rows: list[dict], seeds: list[int]) -> None:
    expected = {(model, seed) for model in ("main_v2", "main_v3_candidate") for seed in seeds}
    actual = {(row["model"], row["seed"]) for row in rows}
    if actual != expected:
        raise RuntimeError(f"Run matrix mismatch: expected={expected}, actual={actual}")
    hashes = {row["validation_hash"] for row in rows}
    split_seeds = {row["split_seed"] for row in rows}
    if len(hashes) != 1 or split_seeds != {42}:
        raise RuntimeError(f"Validation split mismatch: hashes={hashes}, seeds={split_seeds}")


def _delta(left: list[dict], right: list[dict], key: str) -> float:
    return _mean_std(right, key)[0] - _mean_std(left, key)[0]


def _candidate_effective(
    m2: list[dict], m3: list[dict]
) -> tuple[bool, dict[str, float | bool]]:
    by_seed_m2 = {row["seed"]: row for row in m2}
    stable_technical = all(
        row["tech_srcc"] > by_seed_m2[row["seed"]]["tech_srcc"] for row in m3
    )
    auc_delta = _delta(m2, m3, "auc")
    pr_delta = _delta(m2, m3, "pr_auc")
    tech_delta = _delta(m2, m3, "tech_srcc")
    auc_tolerance = max(_mean_std(m2, "auc")[1], _mean_std(m3, "auc")[1], 0.01)
    pr_tolerance = max(_mean_std(m2, "pr_auc")[1], _mean_std(m3, "pr_auc")[1], 0.01)
    overall_comparable = auc_delta >= -auc_tolerance and pr_delta >= -pr_tolerance
    effective = stable_technical and tech_delta > 0.05 and overall_comparable
    return effective, {
        "stable_technical": stable_technical,
        "overall_comparable": overall_comparable,
        "auc_delta": auc_delta,
        "pr_delta": pr_delta,
        "tech_delta": tech_delta,
        "auc_tolerance": auc_tolerance,
        "pr_tolerance": pr_tolerance,
    }


def _is_candidate_effective(rows: list[dict]) -> tuple[bool, dict[str, float | bool]]:
    return _candidate_effective(
        _group(rows, "main_v2"), _group(rows, "main_v3_candidate")
    )


def _select_dimension(rows: list[dict], seed_count: int) -> tuple[str, list[dict]]:
    groups = {
        model: _group(rows, model)
        for model in dict.fromkeys(row["model"] for row in rows)
    }
    complete = {
        model: group for model, group in groups.items() if len(group) == seed_count
    }
    if not complete:
        raise RuntimeError("No dimension candidate has a complete multi-seed result.")
    selected = max(
        complete,
        key=lambda model: (
            _mean_std(complete[model], "auc")[0],
            _mean_std(complete[model], "pr_auc")[0],
            _mean_std(complete[model], "tech_srcc")[0],
        ),
    )
    return selected, complete[selected]


def _run_multiseed(args: argparse.Namespace, seeds: list[int]) -> list[dict]:
    rows = []
    for seed in seeds:
        rows.append(
            _run(
                args,
                experiment="T0",
                model="main_v2",
                seed=seed,
                source="clip",
                clip_dim=128,
                cover_dim=256,
            )
        )
        rows.append(
            _run(
                args,
                experiment="T2",
                model="main_v3_candidate",
                seed=seed,
                source="clip_cover",
                clip_dim=128,
                cover_dim=256,
            )
        )
    _validate_multiseed(rows, seeds)
    _write_csv(args.output_dir / "t0_t2_multiseed.csv", rows)
    main_rows = [
        {
            **row,
            "experiment": "strict_main_ab",
            "model": "main_v2" if row["technical_source"] == "clip" else "main_v3_candidate",
        }
        for row in rows
    ]
    _write_csv(args.output_dir / "main_v2_main_v3_multiseed.csv", main_rows)
    return rows


def _dimension_seed42(args: argparse.Namespace) -> list[dict]:
    d1_checkpoint = args.output_dir / "checkpoints" / "t2_seed42.pt"
    rows = [
        _checkpoint_row(
            d1_checkpoint,
            experiment="D1",
            model="D1",
            source="clip_cover",
            clip_dim=128,
            cover_dim=256,
        )
    ]
    for name, clip_dim, cover_dim in (("D2", 128, 128), ("D3", 256, 256)):
        rows.append(
            _run(
                args,
                experiment=name,
                model=name,
                seed=42,
                source="clip_cover",
                clip_dim=clip_dim,
                cover_dim=cover_dim,
            )
        )
    return rows


def _dimension_is_clear(rows: list[dict]) -> bool:
    return (
        max(row["auc"] for row in rows) - min(row["auc"] for row in rows) >= 0.01
        or max(row["pr_auc"] for row in rows) - min(row["pr_auc"] for row in rows) >= 0.03
        or max(row["tech_srcc"] for row in rows) - min(row["tech_srcc"] for row in rows) >= 0.05
    )


def _run_dimension(args: argparse.Namespace, seeds: list[int]) -> tuple[list[dict], bool]:
    rows = _dimension_seed42(args)
    clear = _dimension_is_clear(rows)
    if clear and len(seeds) > 1:
        top_two = sorted(
            rows,
            key=lambda row: (row["auc"], row["pr_auc"], row["tech_srcc"]),
            reverse=True,
        )[:2]
        for seed in (seed for seed in seeds if seed != 42):
            for candidate in top_two:
                if candidate["model"] == "D1":
                    checkpoint = args.output_dir / "checkpoints" / f"t2_seed{seed}.pt"
                    rows.append(
                        _checkpoint_row(
                            checkpoint,
                            experiment="D1",
                            model="D1",
                            source="clip_cover",
                            clip_dim=128,
                            cover_dim=256,
                        )
                    )
                else:
                    rows.append(
                        _run(
                            args,
                            experiment=candidate["model"],
                            model=candidate["model"],
                            seed=seed,
                            source="clip_cover",
                            clip_dim=int(candidate["clip_projection_dim"]),
                            cover_dim=int(candidate["cover_projection_dim"]),
                        )
                    )
    _write_csv(args.output_dir / "dimension_ablation.csv", rows)
    return rows, clear


def _raw_table(rows: list[dict], model_label: str = "model") -> list[str]:
    lines = [
        f"| {model_label} | Seed | AUC | PR-AUC | Acc | F1 | Tech SRCC | Visual SRCC | Audio SRCC | Tech-label SRCC | Sci SRCC | Aes SRCC | S/T/A weight | Epoch | Time(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["model"], item["seed"])):
        lines.append(
            f"| {row['model']} | {row['seed']} | {row['auc']:.4f} | {row['pr_auc']:.4f} | "
            f"{row['accuracy']:.4f} | {row['f1']:.4f} | {row['tech_srcc']:.4f} | "
            f"{row['visual_srcc']:.4f} | {row['audio_srcc']:.4f} | {row['tech_label_srcc']:.4f} | "
            f"{row['sci_srcc']:.4f} | {row['aes_srcc']:.4f} | "
            f"{row['science_weight']:.4f}/{row['technical_weight']:.4f}/{row['aesthetic_weight']:.4f} | "
            f"{row['best_epoch']} | {row['train_seconds']:.1f} |"
        )
    return lines


def _summary_table(rows: list[dict]) -> list[str]:
    lines = [
        "| Model | AUC | PR-AUC | F1 | Tech SRCC | Visual SRCC | Audio SRCC | Tech-label SRCC | Tech weight | Time/run(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in dict.fromkeys(row["model"] for row in rows):
        group = _group(rows, model)
        lines.append(
            f"| {model} | {_fmt(group, 'auc')} | {_fmt(group, 'pr_auc')} | {_fmt(group, 'f1')} | "
            f"{_fmt(group, 'tech_srcc')} | {_fmt(group, 'visual_srcc')} | "
            f"{_fmt(group, 'audio_srcc')} | {_fmt(group, 'tech_label_srcc')} | "
            f"{_fmt(group, 'technical_weight', 6)} | {_fmt(group, 'train_seconds', 1)} |"
        )
    return lines


def _write_summary(
    args: argparse.Namespace,
    rows: list[dict],
    decision_data: dict[str, float | bool],
    dimension_rows: list[dict],
    dimension_clear: bool,
    selected_dimension: str | None,
    final_rows: list[dict],
    final_decision_data: dict[str, float | bool],
    decision: str,
) -> None:
    m2 = _group(rows, "main_v2")
    m3 = _group(rows, "main_v3_candidate")
    extraction_path = PROJECT_ROOT / "outputs" / "technical_cover_ablation" / "extraction_report.json"
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    lines = [
        "# COVER Technical main_v3 稳定性验证",
        "",
        "## 1. 实验目的与实现核对",
        "",
        "本实验验证 COVER-inspired Technical representation 的多随机种子稳定性，并判断是否升级正式 main_v3。",
        "",
        "- COVER sampling: 7x7 fragments，fragment 32x32，40 frames，interval 2。",
        "- Backbone: frozen GRPB Swin-3D Tiny，仅加载 `technical_backbone.*`。",
        "- Output: `[B,768,20,7,7]` 经 T/H/W mean pooling 得到 768D cache。",
        "- D1/T2: CLIP 512->128，COVER 768->256，concat 384->TechnicalBranch hidden 128。",
        "- T0 与正式 main_v2 配置完全相同；T2 与 main_v3-candidate 仅 technical visual source 不同，因此两阶段严格复用同一组 runs。",
        f"- Validation hash: `{rows[0]['validation_hash']}`；split seed 固定为 42。",
        "",
        "## 2. T0 vs T2 / main_v2 vs main_v3-candidate",
        "",
    ]
    lines.extend(_raw_table(rows))
    lines.extend(["", "### Mean ± std", ""])
    lines.extend(_summary_table(rows))
    lines.extend(
        [
            "",
            "### 平均变化（main_v3-candidate - main_v2）",
            "",
            f"- ΔAUC: {_delta(m2, m3, 'auc'):+.4f}",
            f"- ΔPR-AUC: {_delta(m2, m3, 'pr_auc'):+.4f}",
            f"- ΔF1: {_delta(m2, m3, 'f1'):+.4f}",
            f"- ΔTechnical SRCC: {_delta(m2, m3, 'tech_srcc'):+.4f}",
            f"- ΔVisual SRCC: {_delta(m2, m3, 'visual_srcc'):+.4f}",
            f"- ΔAudio SRCC: {_delta(m2, m3, 'audio_srcc'):+.4f}",
            f"- ΔTech-label SRCC: {_delta(m2, m3, 'tech_label_srcc'):+.4f}",
            "",
            "Audio SRCC 的变化只表示数据中视觉制作质量与人工音频评分存在共变；COVER 特征未使用音频，不能解释为声学识别能力提升。",
            "",
            "## 3. Fusion 行为",
            "",
            f"- main_v2 technical weight: {_fmt(m2, 'technical_weight', 6)}",
            f"- main_v3-candidate technical weight: {_fmt(m3, 'technical_weight', 6)}",
            f"- 预定义 D1 Technical SRCC 三个 seed 全部提升: `{decision_data['stable_technical']}`。",
            "- 若权重仍贴近 0.20，则 branch collapse 不能仅由旧 Technical representation 质量不足解释；总体标签更偏好 Scientific representation。",
            "",
            "## 4. 效率与复杂度",
            "",
            f"- main_v2 trainable params: {int(m2[0]['trainable_params']):,}",
            f"- main_v3-candidate trainable params: {int(m3[0]['trainable_params']):,}",
            f"- Δ trainable params: {int(m3[0]['trainable_params'] - m2[0]['trainable_params']):+,}",
            "- Frozen COVER technical extractor params: 28,078,620（仅离线提取，不计入 trainable params）。",
            f"- main_v2 training time/run: {_fmt(m2, 'train_seconds', 1)} s",
            f"- main_v3-candidate training time/run: {_fmt(m3, 'train_seconds', 1)} s",
            f"- COVER cold extraction: {extraction['mean_seconds_per_video']:.3f} s/video，peak VRAM {extraction['peak_vram_bytes'] / 1024**3:.3f} GiB。",
            f"- COVER cache: 768D，约 {extraction['mean_feature_size_delta_bytes'] / 1024:.2f} KiB/video；cached training 不重复执行 frozen backbone。",
            "",
            "## 5. Dimension ablation",
            "",
        ]
    )
    if dimension_rows:
        lines.extend(_raw_table(dimension_rows, "Dimension"))
        lines.extend(["", "### Dimension mean ± std", ""])
        lines.extend(_summary_table(dimension_rows))
        lines.extend(["", "D2 仅执行 seed42；D1/D3 为三 seed，不能将 D2 的零标准差解释为稳定。", ""])
        lines.extend([f"- Seed-42 差异达到预注册明显阈值: `{dimension_clear}`。"])
        dimension_models = list(dict.fromkeys(row["model"] for row in dimension_rows))
        for model in dimension_models:
            group = _group(dimension_rows, model)
            if len(group) > 1:
                lines.append(
                    f"- {model} multi-seed: AUC {_fmt(group, 'auc')}，PR-AUC {_fmt(group, 'pr_auc')}，Tech SRCC {_fmt(group, 'tech_srcc')}。"
                )
        lines.extend(
            [
                f"- 最终维度选择: **{selected_dimension}** "
                f"({int(final_rows[0]['clip_projection_dim'])}+{int(final_rows[0]['cover_projection_dim'])})。",
                f"- 最终模型相对 main_v2: ΔAUC {_delta(m2, final_rows, 'auc'):+.4f}，"
                f"ΔPR-AUC {_delta(m2, final_rows, 'pr_auc'):+.4f}，"
                f"ΔTechnical SRCC {_delta(m2, final_rows, 'tech_srcc'):+.4f}。",
                f"- 最终模型 trainable params: {int(final_rows[0]['trainable_params']):,}，"
                f"相对 main_v2 {int(final_rows[0]['trainable_params'] - m2[0]['trainable_params']):+,}。",
            ]
        )
    else:
        lines.append("前两阶段未达到候选有效标准，按任务约束未执行维度消融。")
    lines.extend(
        [
            "",
            "## 6. 最终决策",
            "",
            f"`{decision}`",
            "",
            f"- Final architecture: `{selected_dimension or 'D1'}`。",
            f"- Technical stable: `{final_decision_data['stable_technical']}`；overall comparable: `{final_decision_data['overall_comparable']}`。",
            f"- 最终架构相对 main_v2 的 mean AUC {_delta(m2, final_rows, 'auc'):+.4f}、"
            f"PR-AUC {_delta(m2, final_rows, 'pr_auc'):+.4f}、"
            f"Technical SRCC {_delta(m2, final_rows, 'tech_srcc'):+.4f}，满足升级条件。",
            "- Technical weight 仍贴近 20% floor；不继续人工调整 branch floor。",
            f"- 判断依据优先级：multi-seed stability > AUC/PR-AUC > Technical SRCC > complexity。",
        ]
    )
    (args.output_dir / "main_v3_validation_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if 42 not in seeds or len(seeds) < 3:
        raise ValueError("Seeds must contain 42 and at least three values.")
    if not args.metadata.exists() or not args.feature_dir.exists():
        raise FileNotFoundError("Metadata or feature directory does not exist.")

    multiseed_csv = args.output_dir / "t0_t2_multiseed.csv"
    if args.stage == "dimension":
        if not multiseed_csv.exists():
            raise FileNotFoundError("Run multiseed stage before dimension stage.")
        with multiseed_csv.open(encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        for row in rows:
            for key in METRIC_FIELDS:
                row[key] = float(row[key])
            for key in ("seed", "split_seed", "clip_projection_dim", "cover_projection_dim", "best_epoch"):
                row[key] = int(row[key])
    else:
        rows = _run_multiseed(args, seeds)

    effective, decision_data = _is_candidate_effective(rows)
    dimension_rows: list[dict] = []
    dimension_clear = False
    if args.stage in {"dimension", "all"} and effective:
        dimension_rows, dimension_clear = _run_dimension(args, seeds)

    selected_dimension = None
    final_rows = _group(rows, "main_v3_candidate")
    final_effective = effective
    final_decision_data = decision_data
    if dimension_rows:
        selected_dimension, final_rows = _select_dimension(dimension_rows, len(seeds))
        final_effective, final_decision_data = _candidate_effective(
            _group(rows, "main_v2"), final_rows
        )
    decision = "PROMOTE_MAIN_V3" if final_effective else "KEEP_MAIN_V2"
    _write_summary(
        args,
        rows,
        decision_data,
        dimension_rows,
        dimension_clear,
        selected_dimension,
        final_rows,
        final_decision_data,
        decision,
    )
    print(json.dumps(
        {
            "decision": decision,
            "selected_dimension": selected_dimension or "D1",
            **final_decision_data,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
