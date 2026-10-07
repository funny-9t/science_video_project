"""Run the controlled A1 versus A2 aesthetic prompt experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import CFG
from pipeline.step_aesthetic_clip import (
    AESTHETIC_FEATURE_VERSIONS,
    AESTHETIC_PROMPT_SETS,
    SharedCLIPAestheticScorer,
)
from pipeline.utils_io import load_metadata, load_pt
from training.dataset_pair import PairwiseVideoDataset
from training.metrics import compute_video_metrics
from training.train import _search_best_threshold
from training.utils_train import get_device
from tools import run_progressive_training as progressive


EXPECTED_SPLIT_HASH = "79413d21918a956716f230456b80aec037be51bfd1c6a412a8c0a30087a2a58f"
PROMPT_VERSIONS = ("a1_original", "a2_domain_semantic")


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
    parser.add_argument(
        "--aesthetic-prompt-version", choices=(*PROMPT_VERSIONS, "all"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--sanity-size", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force-training", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def correlation(prediction: np.ndarray, target: np.ndarray, method: str) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(prediction) & np.isfinite(target)
    if valid.sum() < 2:
        return float("nan")
    if np.unique(prediction[valid]).size < 2 or np.unique(target[valid]).size < 2:
        return float("nan")
    result = (
        spearmanr(prediction[valid], target[valid]).statistic
        if method == "spearman"
        else pearsonr(prediction[valid], target[valid]).statistic
    )
    return float(result)


def split_metadata(metadata_path: Path, feature_dir: Path, split_seed: int):
    metadata = load_metadata(metadata_path)
    available = {path.stem for path in feature_dir.glob("*.pt")}
    metadata = metadata[metadata["video_id"].astype(str).isin(available)].copy()
    train_df, val_df = train_test_split(
        metadata,
        test_size=CFG.val_ratio,
        random_state=split_seed,
        stratify=metadata["label"],
    )
    split_payload = "\n".join(
        [f"train:{value}" for value in sorted(train_df["video_id"].astype(str))]
        + [f"val:{value}" for value in sorted(val_df["video_id"].astype(str))]
    )
    split_hash = hashlib.sha256(split_payload.encode("utf-8")).hexdigest()
    return metadata, train_df, val_df, split_hash


def feature_cache_dir(output_dir: Path, prompt_version: str) -> Path:
    return output_dir / "cache" / prompt_version


def save_prompt_configs(output_dir: Path) -> None:
    for prompt_version in PROMPT_VERSIONS:
        suffix = "a1" if prompt_version == "a1_original" else "a2"
        write_json(
            output_dir / f"prompt_config_{suffix}.json",
            {
                "prompt_version": prompt_version,
                "feature_version": AESTHETIC_FEATURE_VERSIONS[prompt_version],
                "feature_dim": len(AESTHETIC_PROMPT_SETS[prompt_version]),
                "prompts": AESTHETIC_PROMPT_SETS[prompt_version],
            },
        )


def valid_a2_cache(path: Path) -> bool:
    if not path.exists():
        return False
    payload = load_pt(path)
    feature = np.asarray(payload.get("feature", []), dtype=np.float32)
    metadata = payload.get("metadata", {})
    return (
        feature.shape == (7,)
        and np.isfinite(feature).all()
        and metadata.get("prompt_version") == "a2_domain_semantic"
        and metadata.get("feature_version")
        == AESTHETIC_FEATURE_VERSIONS["a2_domain_semantic"]
    )


def extract_a2_features(
    metadata: pd.DataFrame,
    feature_dir: Path,
    output_dir: Path,
    force: bool,
) -> tuple[Path, SharedCLIPAestheticScorer]:
    cache_dir = feature_cache_dir(output_dir, "a2_domain_semantic")
    cache_dir.mkdir(parents=True, exist_ok=True)
    scorer = SharedCLIPAestheticScorer(
        CFG.clip_model_name,
        device=CFG.device,
        prompt_version="a2_domain_semantic",
    )
    prompt_names = [item["name"] for item in scorer.prompts]
    for row in metadata.itertuples(index=False):
        video_id = str(row.video_id)
        cache_path = cache_dir / f"{video_id}.pt"
        if not force and valid_a2_cache(cache_path):
            continue
        sample = load_pt(feature_dir / f"{video_id}.pt")
        frame_features = np.asarray(sample.get("frame_features", []), dtype=np.float32)
        if frame_features.ndim != 2 or frame_features.shape[1] != CFG.clip_video_dim:
            raise ValueError(f"Invalid cached frame features for {video_id}: {frame_features.shape}")
        result = scorer.score_frame_features(frame_features, aggregation="mean")
        feature = np.asarray(
            [result["dimensions"][name] for name in prompt_names], dtype=np.float32
        )
        torch.save(
            {
                "video_id": video_id,
                "feature": feature,
                "metadata": {
                    "prompt_version": result["prompt_version"],
                    "feature_version": result["feature_version"],
                    "clip_model": str(CFG.clip_model_name),
                    "aggregation": "mean",
                    "feature_dim": 7,
                    "prompt_names": prompt_names,
                    "frame_sampling_version": sample.get("clip_sampling_version", ""),
                },
            },
            cache_path,
        )
    return cache_dir, scorer


def load_prompt_feature(
    video_id: str, prompt_version: str, feature_dir: Path, a2_cache_dir: Path
) -> np.ndarray:
    if prompt_version == "a1_original":
        sample = load_pt(feature_dir / f"{video_id}.pt")
        return np.asarray(sample["aes_shared_clip_feat"], dtype=np.float32)
    return np.asarray(load_pt(a2_cache_dir / f"{video_id}.pt")["feature"], dtype=np.float32)


def run_feature_sanity(
    metadata: pd.DataFrame,
    feature_dir: Path,
    a2_cache_dir: Path,
    scorer: SharedCLIPAestheticScorer,
    output_dir: Path,
    sample_size: int,
    seed: int,
) -> None:
    metadata = metadata.copy()
    metadata["video_id"] = metadata["video_id"].astype(str)
    metadata = metadata.sort_values("video_id")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(metadata), size=min(sample_size, len(metadata)), replace=False)
    selected = metadata.iloc[np.sort(indices)]
    rows = []
    for row in selected.itertuples(index=False):
        video_id = str(row.video_id)
        a1 = load_prompt_feature(video_id, "a1_original", feature_dir, a2_cache_dir)
        a2 = load_prompt_feature(video_id, "a2_domain_semantic", feature_dir, a2_cache_dir)
        rows.append({
            "video_id": video_id,
            "a1_7d": json.dumps(a1.tolist()),
            "a2_7d": json.dumps(a2.tolist()),
            "aes_target": float(row.aes_target),
        })
    write_csv(output_dir / "feature_sanity.csv", rows)

    all_a1 = np.stack([
        load_prompt_feature(str(video_id), "a1_original", feature_dir, a2_cache_dir)
        for video_id in metadata["video_id"]
    ])
    all_a2 = np.stack([
        load_prompt_feature(str(video_id), "a2_domain_semantic", feature_dir, a2_cache_dir)
        for video_id in metadata["video_id"]
    ])
    first_id = str(selected.iloc[0]["video_id"])
    first_frames = load_pt(feature_dir / f"{first_id}.pt")["frame_features"]
    repeat_1 = scorer.score_frame_features(first_frames)["dimensions"]
    repeat_2 = scorer.score_frame_features(first_frames)["dimensions"]
    deterministic_delta = max(abs(repeat_1[key] - repeat_2[key]) for key in repeat_1)
    cache_valid = sum(
        valid_a2_cache(a2_cache_dir / f"{video_id}.pt")
        for video_id in metadata["video_id"]
    )

    def max_off_diagonal(features: np.ndarray) -> float:
        matrix = np.corrcoef(features, rowvar=False)
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        return float(np.nanmax(np.abs(matrix[mask])))

    a1_std = all_a1.std(axis=0)
    a2_std = all_a2.std(axis=0)
    lines = [
        "# Aesthetic Prompt Feature Sanity Check",
        "",
        f"- Videos checked in the random table: {len(rows)} (seed={seed}).",
        f"- Full feature population: {len(metadata)}.",
        f"- A2 cache metadata valid: {cache_valid}/{len(metadata)}.",
        f"- NaN/Inf: A1={int((~np.isfinite(all_a1)).sum())}, A2={int((~np.isfinite(all_a2)).sum())}.",
        f"- Per-dimension standard deviation A1: `{a1_std.tolist()}`.",
        f"- Per-dimension standard deviation A2: `{a2_std.tolist()}`.",
        f"- Near-constant dimensions (std < 1e-6): A1={int((a1_std < 1e-6).sum())}, A2={int((a2_std < 1e-6).sum())}.",
        f"- Maximum absolute within-set off-diagonal Pearson correlation: A1={max_off_diagonal(all_a1):.6f}, A2={max_off_diagonal(all_a2):.6f}.",
        f"- Mean absolute value: A1={np.abs(all_a1).mean():.6f}, A2={np.abs(all_a2).mean():.6f}.",
        f"- Global standard deviation: A1={all_a1.std():.6f}, A2={all_a2.std():.6f}.",
        f"- Deterministic repeat max absolute delta for `{first_id}`: {deterministic_delta:.10f}.",
        "- Aggregation remains frame-wise mean; no sigmoid, softmax, or normalization was added after the CLIP similarity difference.",
    ]
    (output_dir / "feature_sanity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_dir: Path,
    prompt_version: str,
    a2_cache_dir: Path,
):
    common = dict(
        same_category=False,
        use_native_clip=False,
        use_cover_features=False,
        aesthetic_feature_backend="shared_clip",
        technical_visual_source="clip_cover",
    )
    if prompt_version == "a2_domain_semantic":
        common.update(
            aesthetic_prompt_feature_dir=a2_cache_dir,
            aesthetic_prompt_version=prompt_version,
        )
    train_ds = PairwiseVideoDataset.from_metadata(
        train_df, feature_dir, deterministic_pairs=False, **common
    )
    val_ds = PairwiseVideoDataset.from_metadata(
        val_df, feature_dir, deterministic_pairs=True, **common
    )
    return train_ds, val_ds


def collect_predictions(model, samples: list[dict], device: torch.device) -> dict[str, np.ndarray]:
    keys = (
        "label", "probability", "overall", "sci_score", "tech_score", "aes_score",
        "sci_target", "tech_target", "aes_target", "overall_target",
        "science_weight", "technical_weight", "aesthetic_weight",
    )
    records = {key: [] for key in keys}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(samples), 32):
            batch = progressive.prepare_batch(
                progressive.collate_unique(samples[start:start + 32]), device
            )
            output = model(**batch)
            values = {
                "label": batch["label_target"].squeeze(-1),
                "probability": output["probability"].squeeze(-1),
                "overall": output["overall_score"].squeeze(-1),
                "sci_score": torch.sigmoid(output["scientific_score"]).squeeze(-1),
                "tech_score": torch.sigmoid(output["technical_score"]).squeeze(-1),
                "aes_score": torch.sigmoid(output["aesthetic_score"]).squeeze(-1),
                "sci_target": batch["sci_target"].squeeze(-1),
                "tech_target": batch["tech_target"].squeeze(-1),
                "aes_target": batch["aes_target"].squeeze(-1),
                "overall_target": batch["overall_quality_target"].squeeze(-1),
                "science_weight": output["branch_weights"][:, 0],
                "technical_weight": output["branch_weights"][:, 1],
                "aesthetic_weight": output["branch_weights"][:, 2],
            }
            for key, value in values.items():
                records[key].extend(value.detach().cpu().numpy().tolist())
    return {key: np.asarray(value, dtype=np.float32) for key, value in records.items()}


def evaluate_model(model, samples: list[dict], device: torch.device) -> tuple[dict, dict]:
    arrays = collect_predictions(model, samples, device)
    threshold, _ = _search_best_threshold(
        arrays["probability"][arrays["label"] == 1],
        arrays["probability"][arrays["label"] == 0],
        metric="f1",
    )
    metrics = compute_video_metrics(arrays["label"], arrays["probability"], threshold)
    positive = arrays["probability"][arrays["label"] == 1]
    negative = arrays["probability"][arrays["label"] == 0]
    metrics["ranking_accuracy"] = float((positive[:, None] > negative[None, :]).mean())
    metrics["best_threshold"] = float(threshold)
    metrics["overall_srcc"] = correlation(
        arrays["overall"], arrays["overall_target"], "spearman"
    )
    for prefix in ("sci", "tech", "aes"):
        metrics[f"{prefix}_srcc"] = correlation(
            arrays[f"{prefix}_score"], arrays[f"{prefix}_target"], "spearman"
        )
    metrics["aes_plcc"] = correlation(
        arrays["aes_score"], arrays["aes_target"], "pearson"
    )
    error = arrays["aes_score"] - arrays["aes_target"]
    metrics["aes_mse"] = float(np.mean(error ** 2))
    metrics["aes_mae"] = float(np.mean(np.abs(error)))
    metrics["branch_macro_srcc"] = float(np.mean([
        metrics["sci_srcc"], metrics["tech_srcc"], metrics["aes_srcc"]
    ]))
    for key in ("science_weight", "technical_weight", "aesthetic_weight"):
        metrics[key] = float(arrays[key].mean())
    return metrics, arrays


def load_checkpoint_metrics(
    checkpoint: Path,
    val_ds: PairwiseVideoDataset,
    device: torch.device,
    expected_hash: str,
) -> tuple[dict, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actual_hash = payload.get("split_hash", payload.get("validation_hash", ""))
    if actual_hash != expected_hash:
        raise ValueError(f"Checkpoint split hash mismatch: {actual_hash} != {expected_hash}")
    model = progressive.new_model(int(payload["seed"]), device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    metrics, arrays = evaluate_model(model, val_ds.samples, device)
    metrics.update({
        "best_epoch": int(payload["best_epoch"]),
        "checkpoint": str(checkpoint.resolve()),
    })
    return metrics, arrays


def train_a2(
    train_ds: PairwiseVideoDataset,
    val_ds: PairwiseVideoDataset,
    device: torch.device,
    args: argparse.Namespace,
    split_hash: str,
) -> tuple[Path, Path]:
    training_dir = args.output_dir / "training" / "a2_domain_semantic" / f"seed_{args.seed}"
    training_dir.mkdir(parents=True, exist_ok=True)
    log_path = training_dir / "training.log"
    logger = progressive.build_logger(log_path)
    stage_args = SimpleNamespace(
        seed=args.seed,
        split_seed=args.split_seed,
        output_dir=training_dir,
        force=args.force_training,
    )
    history: list[dict] = []
    model = progressive.new_model(args.seed, device)
    stage1 = training_dir / "stage1_best.pt"
    progressive.load_or_train_stage(
        model, train_ds, val_ds, device, stage_args, logger,
        "A2_P2", "stage1", stage1, 15, 5, split_hash, history,
        branch_rank_weight=0.05,
    )
    stage2 = training_dir / "stage2_best.pt"
    progressive.load_or_train_stage(
        model, train_ds, val_ds, device, stage_args, logger,
        "A2_P2", "stage2", stage2, 30, 10, split_hash, history,
        branch_rank_weight=0.05,
    )
    return stage1, stage2


def branch_rows(
    variant: str,
    stage: str,
    metrics: dict,
    val_df: pd.DataFrame,
    feature_dir: Path,
    a2_cache_dir: Path,
) -> list[dict]:
    rows = [{
        "variant": variant,
        "stage": stage,
        "scope": "aesthetic_branch",
        "dimension": "model_output",
        "srcc": metrics["aes_srcc"],
        "plcc": metrics["aes_plcc"],
        "mse": metrics["aes_mse"],
        "mae": metrics["aes_mae"],
    }]
    if stage != "stage2":
        return rows
    names = [item["name"] for item in AESTHETIC_PROMPT_SETS[variant]]
    features = np.stack([
        load_prompt_feature(str(video_id), variant, feature_dir, a2_cache_dir)
        for video_id in val_df["video_id"].astype(str)
    ])
    target = val_df["aes_target"].to_numpy(dtype=np.float32)
    for index, name in enumerate(names):
        rows.append({
            "variant": variant,
            "stage": "raw_feature",
            "scope": "prompt_dimension",
            "dimension": name,
            "srcc": correlation(features[:, index], target, "spearman"),
            "plcc": correlation(features[:, index], target, "pearson"),
            "mse": "",
            "mae": "",
        })
    return rows


def dimension_correlation_rows(
    val_df: pd.DataFrame,
    feature_dir: Path,
    a2_cache_dir: Path,
    prediction_arrays: dict[str, dict],
) -> list[dict]:
    features = {}
    labels = []
    for variant in PROMPT_VERSIONS:
        names = [item["name"] for item in AESTHETIC_PROMPT_SETS[variant]]
        values = np.stack([
            load_prompt_feature(str(video_id), variant, feature_dir, a2_cache_dir)
            for video_id in val_df["video_id"].astype(str)
        ])
        for index, name in enumerate(names):
            key = f"{variant}:{name}"
            features[key] = values[:, index]
            labels.append(key)
    rows = []
    for left in labels:
        for right in labels:
            rows.append({
                "analysis": "a1_a2_matrix",
                "left": left,
                "right": right,
                "metric": "plcc",
                "value": correlation(features[left], features[right], "pearson"),
            })
    target = val_df["aes_target"].to_numpy(dtype=np.float32)
    for variant in PROMPT_VERSIONS:
        for item in AESTHETIC_PROMPT_SETS[variant]:
            key = f"{variant}:{item['name']}"
            for metric in ("spearman", "pearson"):
                rows.append({
                    "analysis": "dimension_vs_aes_target",
                    "left": key,
                    "right": "aes_target",
                    "metric": "srcc" if metric == "spearman" else "plcc",
                    "value": correlation(features[key], target, metric),
                })
            rows.append({
                "analysis": "dimension_vs_technical_score",
                "left": key,
                "right": f"{variant}:s_tech",
                "metric": "srcc",
                "value": correlation(
                    features[key], prediction_arrays[variant]["tech_score"], "spearman"
                ),
            })
    return rows


def write_reports(
    args: argparse.Namespace,
    split_hash: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_dir: Path,
    a2_cache_dir: Path,
    stage_metrics: dict[str, dict[str, dict]],
    stage2_arrays: dict[str, dict],
) -> None:
    branch_diagnostics = []
    for variant in PROMPT_VERSIONS:
        for stage in ("stage1", "stage2"):
            branch_diagnostics.extend(branch_rows(
                variant, stage, stage_metrics[variant][stage], val_df,
                feature_dir, a2_cache_dir,
            ))
    write_csv(args.output_dir / "branch_diagnostics.csv", branch_diagnostics)

    dimension_rows = dimension_correlation_rows(
        val_df, feature_dir, a2_cache_dir, stage2_arrays
    )
    write_csv(args.output_dir / "dimension_correlations.csv", dimension_rows)

    for variant, suffix in (("a1_original", "a1"), ("a2_domain_semantic", "a2")):
        write_json(args.output_dir / f"{suffix}_metrics.json", {
            "variant": variant,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "validation_hash": split_hash,
            "train_size": len(train_df),
            "validation_size": len(val_df),
            "stage1": stage_metrics[variant]["stage1"],
            "stage2": stage_metrics[variant]["stage2"],
        })

    comparison_rows = []
    for stage in ("stage1", "stage2"):
        for metric in (
            "aes_srcc", "aes_plcc", "aes_mse", "aes_mae", "auc", "pr_auc",
            "accuracy", "f1", "overall_srcc", "science_weight",
            "technical_weight", "aesthetic_weight",
        ):
            a1 = float(stage_metrics["a1_original"][stage][metric])
            a2 = float(stage_metrics["a2_domain_semantic"][stage][metric])
            comparison_rows.append({
                "stage": stage,
                "metric": metric,
                "a1_original": a1,
                "a2_domain_semantic": a2,
                "delta_a2_minus_a1": a2 - a1,
            })
    write_csv(args.output_dir / "comparison.csv", comparison_rows)

    a1 = stage_metrics["a1_original"]["stage2"]
    a2 = stage_metrics["a2_domain_semantic"]["stage2"]
    branch_improved = a2["aes_srcc"] > a1["aes_srcc"] and a2["aes_plcc"] > a1["aes_plcc"]
    overall_stable = (
        a2["pr_auc"] >= a1["pr_auc"] - 0.02
        and a2["auc"] >= a1["auc"] - 0.01
        and a2["overall_srcc"] >= a1["overall_srcc"] - 0.02
    )
    recommend = branch_improved and overall_stable
    decision = (
        "A2 improves both primary aesthetic correlations without a clear overall regression; "
        "it is a candidate for multi-seed validation."
        if recommend else
        "A2 does not satisfy the branch-improvement plus overall-stability condition; keep A1 "
        "as the main-model prompt set and retain A2 as a controlled negative/diagnostic result."
    )
    lines = [
        "# Semantic-Guided Aesthetic Prompt Experiment",
        "",
        "## 1. 实验目的",
        "",
        "在 KEEP_P2_S2 中只替换 7 维美学 Prompt 属性空间，比较原始 A1 与领域解耦 A2。",
        "",
        "## 2. 当前代码审计结论",
        "",
        "主路径使用缓存的 ViT-L/14 逐帧特征和同一模型的文本塔；训练读取 7D PT 缓存，"
        "并非在线运行 legacy ViT-B/32。完整审计见 `code_audit.md`。",
        "",
        "## 3. A1 / A2 Prompt 设计",
        "",
        "| A1 | 潜在问题 | A2 对应处理 |",
        "|---|---|---|",
        "| clarity | 与技术 sharpness/blur 重叠 | 删除 |",
        "| cleanliness | 混入 clutter/noise | 拆除，不直接保留 |",
        "| composition | 合理 | 保留并规范化 |",
        "| appeal | 措辞较泛 | 改为 visual_appeal |",
        "| lighting | 与 color 混合 | 拆为 lighting + color_harmony |",
        "| text_readability | 更接近技术表达 | 删除 |",
        "| professional | 多概念混杂 | 改为 visual_refinement / visual_presentation |",
        "",
        "## 4. 控制变量",
        "",
        f"训练/验证={len(train_df)}/{len(val_df)}，seed={args.seed}，split_seed={args.split_seed}，"
        f"validation hash=`{split_hash}`。骨干、帧特征、网络、P2-S2 两阶段训练、损失、优化器、"
        "scheduler、batch size 和选模规则均保持一致。A1 历史缓存与 checkpoint 未覆盖。",
        "",
        "## 5. 特征 sanity check",
        "",
        "A2 已写入独立版本缓存；完整方差、共线性、量级和确定性检查见 `feature_sanity.md`。",
        "",
        "## 6. 美学分支结果",
        "",
        "| Variant | Stage | SRCC_aes | PLCC_aes | MSE_aes | MAE_aes |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for variant in PROMPT_VERSIONS:
        for stage in ("stage1", "stage2"):
            metric = stage_metrics[variant][stage]
            lines.append(
                f"| {variant} | {stage} | {metric['aes_srcc']:.4f} | "
                f"{metric['aes_plcc']:.4f} | {metric['aes_mse']:.4f} | {metric['aes_mae']:.4f} |"
            )
    lines.extend([
        "",
        "Stage 1 只训练三个分支，且每个分支由自身标签损失驱动，因此作为 branch-only 诊断；"
        "Stage 2 是完整 KEEP_P2_S2 结果。",
        "",
        "## 7. 整体模型结果",
        "",
        "项目没有名为 AUCF1 的历史指标；为保持可比性，这里报告 ROC-AUC、PR-AUC、F1 与 overall SRCC。",
        "",
        "| Variant | ROC-AUC | PR-AUC | F1 | Overall SRCC | S/T/A weight |",
        "|---|---:|---:|---:|---:|---|",
        f"| a1_original | {a1['auc']:.4f} | {a1['pr_auc']:.4f} | {a1['f1']:.4f} | "
        f"{a1['overall_srcc']:.4f} | {a1['science_weight']:.4f}/{a1['technical_weight']:.4f}/{a1['aesthetic_weight']:.4f} |",
        f"| a2_domain_semantic | {a2['auc']:.4f} | {a2['pr_auc']:.4f} | {a2['f1']:.4f} | "
        f"{a2['overall_srcc']:.4f} | {a2['science_weight']:.4f}/{a2['technical_weight']:.4f}/{a2['aesthetic_weight']:.4f} |",
        "",
        "## 8. 维度级相关性",
        "",
        "`dimension_correlations.csv` 包含 A1/A2 14 维完整 Pearson 矩阵、每维与 aes_target 的"
        " SRCC/PLCC，以及每维与对应模型 s_tech 的 SRCC。",
        "",
        "## 9. A1 vs A2 结果解释",
        "",
        f"A2-A1: aes SRCC {a2['aes_srcc'] - a1['aes_srcc']:+.4f}，"
        f"aes PLCC {a2['aes_plcc'] - a1['aes_plcc']:+.4f}，"
        f"PR-AUC {a2['pr_auc'] - a1['pr_auc']:+.4f}，"
        f"ROC-AUC {a2['auc'] - a1['auc']:+.4f}，"
        f"overall SRCC {a2['overall_srcc'] - a1['overall_srcc']:+.4f}。",
        "",
        "## 10. 是否支持实验假设",
        "",
        f"Primary branch improvement: `{branch_improved}`；overall stability: `{overall_stable}`。",
        "",
        "## 11. 是否建议将 A2 纳入主模型",
        "",
        decision,
        "",
        "## 12. 下一步建议",
        "",
        "仅当 seed=42 同时满足美学分支改善与整体稳定时，再运行 seeds 123/2026；否则停止对 A2 调参，"
        "避免在固定验证集上搜索 Prompt。",
    ])
    (args.output_dir / "aesthetic_prompt_experiment_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_smoke_test(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_dir: Path,
    a2_cache_dir: Path,
    device: torch.device,
) -> None:
    a1_train, _ = build_datasets(
        train_df, val_df, feature_dir, "a1_original", a2_cache_dir
    )
    a2_train, _ = build_datasets(
        train_df, val_df, feature_dir, "a2_domain_semantic", a2_cache_dir
    )
    if not np.allclose(a1_train.samples[0]["text_feat"], a2_train.samples[0]["text_feat"]):
        raise AssertionError("Non-aesthetic features changed between A1 and A2")
    if np.allclose(a1_train.samples[0]["aes_feat"], a2_train.samples[0]["aes_feat"]):
        raise AssertionError("A1 and A2 aesthetic features unexpectedly match")
    model = progressive.new_model(42, device)
    batch = progressive.prepare_batch(
        progressive.collate_unique(a2_train.samples[:2]), device
    )
    with torch.inference_mode():
        output = model(**batch)
    if output["aesthetic_score"].shape != (2, 1):
        raise AssertionError(f"Unexpected aesthetic output shape: {output['aesthetic_score'].shape}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_prompt_configs(args.output_dir)
    metadata, train_df, val_df, split_hash = split_metadata(
        args.metadata, args.feature_dir, args.split_seed
    )
    if args.seed == 42 and args.split_seed == 42 and split_hash != EXPECTED_SPLIT_HASH:
        raise ValueError(f"Unexpected validation hash: {split_hash}")
    a2_cache_dir, scorer = extract_a2_features(
        metadata, args.feature_dir, args.output_dir, args.force_features
    )
    run_feature_sanity(
        metadata, args.feature_dir, a2_cache_dir, scorer, args.output_dir,
        args.sanity_size, args.seed,
    )
    device = get_device(CFG.device)
    if args.smoke:
        run_smoke_test(train_df, val_df, args.feature_dir, a2_cache_dir, device)
        print(f"Smoke test passed for {len(metadata)} videos; split hash={split_hash}")
        return

    a1_train, a1_val = build_datasets(
        train_df, val_df, args.feature_dir, "a1_original", a2_cache_dir
    )
    a2_train, a2_val = build_datasets(
        train_df, val_df, args.feature_dir, "a2_domain_semantic", a2_cache_dir
    )
    del a1_train
    a1_stage1 = PROJECT_ROOT / "outputs" / "progressive_training" / "P2" / "stage1_best.pt"
    a1_stage2 = PROJECT_ROOT / "outputs" / "progressive_training" / "P2" / "stage2_best.pt"
    if not a1_stage1.exists() or not a1_stage2.exists():
        raise FileNotFoundError("Historical A1 P2-S2 checkpoints are required")
    a2_stage1, a2_stage2 = train_a2(a2_train, a2_val, device, args, split_hash)

    stage_metrics: dict[str, dict[str, dict]] = {
        "a1_original": {}, "a2_domain_semantic": {}
    }
    stage2_arrays = {}
    for variant, dataset, checkpoints in (
        ("a1_original", a1_val, (a1_stage1, a1_stage2)),
        ("a2_domain_semantic", a2_val, (a2_stage1, a2_stage2)),
    ):
        for stage, checkpoint in zip(("stage1", "stage2"), checkpoints):
            metrics, arrays = load_checkpoint_metrics(
                checkpoint, dataset, device, split_hash
            )
            stage_metrics[variant][stage] = metrics
            if stage == "stage2":
                stage2_arrays[variant] = arrays

    write_reports(
        args, split_hash, train_df, val_df, args.feature_dir, a2_cache_dir,
        stage_metrics, stage2_arrays,
    )
    print(f"Experiment complete: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
