"""Run the AADB-only phase of E3 v2 multi-attribute aesthetic validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import CFG
from pipeline.step_aesthetic_clip import AESTHETIC_PROMPT_SETS
from training.model_mvp import MultiModalQualityModel
from training.utils_train import set_seed


SEEDS = (42, 123, 2026)
EXPECTED_SPLIT_HASH = "79413d21918a956716f230456b80aec037be51bfd1c6a412a8c0a30087a2a58f"
ATTRIBUTES = (
    "BalacingElements", "ColorHarmony", "Content", "DoF", "Light", "MotionBlur",
    "Object", "Repetition", "RuleOfThirds", "Symmetry", "VividColor",
)
BRANCH_CACHE_VERSION = "e3v2_aadb_keep_p2_s2_aesthetic_embedding_v1"
PROBE_VERSION = "e3v2_aadb_frozen_linear_probe_v1"

# Defined from the current A1 prompt wording before any E3-v2 correlations run.
MAPPINGS = (
    ("clarity", "MotionBlur", "Partial", "Generic sharpness overlaps motion blur, but also covers defocus and resolution."),
    ("cleanliness", None, "None", "AADB does not provide a clutter or visual-cleanliness attribute."),
    ("composition", "BalacingElements", "Partial", "Element balance is one component of broad composition and framing."),
    ("composition", "RuleOfThirds", "Partial", "Rule of thirds is one possible composition rule, not the full prompt meaning."),
    ("composition", "Symmetry", "Partial", "Symmetry may support composition but is not required by the prompt."),
    ("appeal", "Content", "Partial", "Content preference can affect appeal, but generic appeal is broader."),
    ("lighting", "Light", "Strong", "Both explicitly evaluate good versus poor illumination."),
    ("lighting", "ColorHarmony", "Partial", "The prompt mentions balanced colors in addition to lighting."),
    ("lighting", "VividColor", "Partial", "Bright colors overlap vividness, but illumination remains part of the prompt."),
    ("text_readability", None, "None", "AADB has no text or scientific-diagram readability attribute."),
    ("professional", None, "None", "AADB has no scientific-visualization professionalism attribute."),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir", type=Path,
        default=Path(r"F:\test_data\AADB_datasetImages_warp256\datasetImages_warp256"),
    )
    parser.add_argument(
        "--annotation-dir", type=Path,
        default=PROJECT_ROOT / "data" / "external" / "aadb_official"
        / "imgListFiles_label" / "imgListFiles_label",
    )
    parser.add_argument(
        "--clip-cache", type=Path,
        default=PROJECT_ROOT / "outputs" / "aadb_prompt_alignment" / "cache"
        / "external_validation" / "aadb" / "aadb_clip_prompt_cache.pt",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "outputs" / "progressive_training" / "P2" / "stage2_best.pt",
    )
    parser.add_argument(
        "--previous-divide-results", type=Path,
        default=PROJECT_ROOT / "outputs" / "external_aesthetic_validation"
        / "external_aesthetic_results.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "aesthetic_external_validation_v2",
    )
    parser.add_argument("--force-branch-cache", action="store_true")
    return parser.parse_args()


def read_label_file(path: Path) -> dict[str, float]:
    values = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            image_id, value = line.strip().rsplit(maxsplit=1)
            values[image_id] = float(value)
    return values


def load_split(annotation_dir: Path, phase: str, split: str) -> pd.DataFrame:
    overall = read_label_file(annotation_dir / f"imgList{phase}Regression_score.txt")
    frame = pd.DataFrame({"image_id": list(overall), "overall_score": list(overall.values())})
    frame["split"] = split
    for attribute in ATTRIBUTES:
        values = read_label_file(annotation_dir / f"imgList{phase}Regression_{attribute}.txt")
        if set(values) != set(overall):
            raise ValueError(f"{phase}/{attribute} IDs do not match the overall-score file")
        frame[attribute] = frame["image_id"].map(values)
    return frame


def load_annotations(annotation_dir: Path) -> tuple[pd.DataFrame, dict]:
    train = load_split(annotation_dir, "Train", "train")
    validation_raw = load_split(annotation_dir, "Validation", "validation")
    test = load_split(annotation_dir, "TestNew", "test")
    overlap = sorted(set(validation_raw["image_id"]) & set(test["image_id"]))
    validation = validation_raw.loc[~validation_raw["image_id"].isin(overlap)].copy()
    frame = pd.concat([train, validation, test], ignore_index=True)
    if frame["image_id"].duplicated().any():
        raise ValueError("AADB protocol contains duplicate image IDs after de-overlap")
    split_text = "\n".join(f"{row.split}:{row.image_id}" for row in frame.itertuples(index=False))
    audit = {
        "raw_train": len(train), "raw_validation": len(validation_raw), "raw_testnew": len(test),
        "validation_test_overlap": len(overlap), "validation_after_deoverlap": len(validation),
        "protocol_samples": len(frame),
        "split_hash": hashlib.sha256(split_text.encode("utf-8")).hexdigest(),
        "overlap_examples": overlap[:10],
    }
    return frame, audit


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def correlation(prediction: np.ndarray, target: np.ndarray, kind: str) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(prediction) & np.isfinite(target)
    prediction, target = prediction[valid], target[valid]
    if prediction.size < 2 or np.unique(prediction).size < 2 or np.unique(target).size < 2:
        return float("nan")
    if kind == "spearman":
        prediction, target = rankdata(prediction), rankdata(target)
    x = torch.from_numpy(prediction.copy())
    y = torch.from_numpy(target.copy())
    x, y = x - x.mean(), y - y.mean()
    denominator = torch.sqrt(torch.sum(x * x) * torch.sum(y * y)).clamp_min(1e-15)
    return float((torch.sum(x * y) / denominator).item())


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = prediction - target
    return {
        "SRCC": correlation(prediction, target, "spearman"),
        "PLCC": correlation(prediction, target, "pearson"),
        "MAE": float(np.mean(np.abs(error))),
        "MSE": float(np.mean(error ** 2)),
    }


def write_branch_audits(args: argparse.Namespace, frame: pd.DataFrame, audit: dict, model: MultiModalQualityModel) -> None:
    prompts = AESTHETIC_PROMPT_SETS["a1_original"]
    branch_params = sum(parameter.numel() for parameter in model.aesthetic_branch.parameters())
    branch_audit = f"""# KEEP_P2_S2 Aesthetic Branch Audit

- Checkpoint: `{args.checkpoint.resolve()}`
- CLIP backbone: `{CFG.clip_model_name}` (ViT-L/14)
- Native CLIP feature: 768 dimensions; legacy branch visual input: 512 dimensions.
- Prompt set: `a1_original`, {len(prompts)} positive/negative prompt pairs.
- Prompt score: cosine(image, positive text) - cosine(image, negative text), after L2 normalization.
- Prompt feature dimension: {len(prompts)}.
- Branch inputs: visual 512D, text 768D, audio 384D, prompt 7D.
- Aesthetic embedding: 128D from the frozen branch fusion MLP.
- Score head: frozen `Linear(128, 1)` followed by sigmoid outside the branch.
- Frozen aesthetic-branch parameters: {branch_params:,}; trainable encoder parameters in this experiment: 0.
- AADB policy: cached image CLIP vectors provide the visual input; text/audio context is unavailable and zero-filled.
"""
    (args.output_dir / "aesthetic_branch_audit.md").write_text(branch_audit, encoding="utf-8")

    intentions = {
        "clarity": "visual sharpness, focus, and fine detail",
        "cleanliness": "low clutter, noise, and distracting elements",
        "composition": "composition and professional framing",
        "appeal": "generic visual attractiveness",
        "lighting": "lighting quality with color balance and brightness",
        "text_readability": "readability of text and labels in scientific diagrams",
        "professional": "professional quality of scientific visualization",
    }
    lines = [
        "# Prompt Definition Audit", "",
        "Meanings below come from the current A1 wording and project comments, before AADB correlations.", "",
        "| Prompt ID | Name | Positive prompt | Negative prompt | Intended aesthetic meaning |",
        "|---|---|---|---|---|",
    ]
    for index, prompt in enumerate(prompts, start=1):
        lines.append(
            f"| P{index} | {prompt['name']} | {prompt['positive']} | {prompt['negative']} | {intentions[prompt['name']]} |"
        )
    (args.output_dir / "prompt_definition_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    stats = []
    for column in ("overall_score", *ATTRIBUTES):
        values = frame[column]
        stats.append({
            "attribute": column, "sample_count": int(values.notna().sum()),
            "missing_count": int(values.isna().sum()), "mean": float(values.mean()),
            "std": float(values.std(ddof=0)), "min": float(values.min()), "max": float(values.max()),
        })
    pd.DataFrame(stats).to_csv(args.output_dir / "aadb_attribute_statistics.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# AADB Dataset Audit", "",
        f"- Local image directory: `{args.image_dir}`",
        f"- Annotation directory: `{args.annotation_dir}`",
        f"- Raw official phases: Train={audit['raw_train']}, Validation={audit['raw_validation']}, TestNew={audit['raw_testnew']}.",
        f"- Validation/TestNew overlap: {audit['validation_test_overlap']} images.",
        f"- Leakage-safe protocol: Train={int(frame['split'].eq('train').sum())}, Validation={int(frame['split'].eq('validation').sum())}, TestNew={int(frame['split'].eq('test').sum())}.",
        f"- Protocol samples: {len(frame)}; split hash: `{audit['split_hash']}`.",
        "- TestNew has priority; overlapping validation rows are removed before early stopping.", "",
        "| Attribute | Samples | Missing | Mean | Std | Min | Max |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in stats:
        lines.append(
            f"| {item['attribute']} | {item['sample_count']} | {item['missing_count']} | {item['mean']:.4f} | {item['std']:.4f} | {item['min']:.4f} | {item['max']:.4f} |"
        )
    (args.output_dir / "aadb_dataset_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_preregistered_mapping(output_dir: Path) -> None:
    lines = [
        "# Preregistered Prompt-AADB Attribute Mapping", "",
        "This table is fixed before E3-v2 correlations are computed. Only Strong matches are primary evidence.", "",
        "| Prompt | AADB Attribute | Match Level | Rationale |", "|---|---|---|---|",
    ]
    for prompt, attribute, level, rationale in MAPPINGS:
        lines.append(f"| {prompt} | {attribute or '-'} | {level} | {rationale} |")
    (output_dir / "prompt_attribute_mapping_preregistered.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )


def load_model(checkpoint: Path, device: torch.device) -> MultiModalQualityModel:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("split_hash") != EXPECTED_SPLIT_HASH:
        raise ValueError(f"Unexpected KEEP_P2_S2 split hash: {payload.get('split_hash')}")
    model = MultiModalQualityModel(**payload["config"]).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def load_clip_cache(path: Path, frame: pd.DataFrame) -> dict:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache.get("feature_version") != "e4_aadb_vitl14_a1_prompts_v1":
        raise ValueError(f"Unexpected AADB CLIP cache version: {cache.get('feature_version')}")
    if cache.get("image_ids") != frame["image_id"].tolist():
        raise ValueError("AADB CLIP cache order does not match the audited protocol")
    if tuple(cache["clip_embeddings"].shape) != (len(frame), 768):
        raise ValueError("Unexpected AADB CLIP embedding shape")
    if tuple(cache["prompt_scores"].shape) != (len(frame), 7):
        raise ValueError("Unexpected AADB prompt-score shape")
    return cache


def build_branch_cache(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    clip_cache: dict,
    model: MultiModalQualityModel,
    device: torch.device,
) -> dict:
    target = args.output_dir / "cache" / "external_validation" / "aadb" / "aadb_aesthetic_embedding_cache.pt"
    if target.exists() and not args.force_branch_cache:
        payload = torch.load(target, map_location="cpu", weights_only=False)
        if payload.get("feature_version") == BRANCH_CACHE_VERSION and payload.get("image_ids") == frame["image_id"].tolist():
            return payload
    target.parent.mkdir(parents=True, exist_ok=True)
    native = clip_cache["clip_embeddings"].float()
    legacy = F.adaptive_avg_pool1d(native.unsqueeze(1), 512).squeeze(1)
    prompts = clip_cache["prompt_scores"].float()
    embeddings = []
    scores = []
    with torch.inference_mode():
        for start in range(0, len(frame), 512):
            stop = min(start + 512, len(frame))
            batch = stop - start
            embedding, logit = model.aesthetic_branch(
                legacy[start:stop].to(device),
                torch.zeros(batch, 768, device=device),
                torch.zeros(batch, 384, device=device),
                prompts[start:stop].to(device),
            )
            embeddings.append(embedding.cpu().float())
            scores.append(torch.sigmoid(logit).squeeze(-1).cpu().float())
    payload = {
        "feature_version": BRANCH_CACHE_VERSION,
        "image_ids": frame["image_id"].tolist(), "splits": frame["split"].tolist(),
        "aesthetic_embeddings": torch.cat(embeddings),
        "aesthetic_scores": torch.cat(scores),
        "metadata": {
            "source_clip_cache": str(args.clip_cache.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "missing_context_policy": "zero_text_and_audio",
            "visual_projection": "adaptive_avg_pool1d_768_to_512_for_legacy_checkpoint",
            "encoder_trainable_params": 0,
        },
    }
    torch.save(payload, target)
    return payload


def train_probe(
    features: np.ndarray,
    target: np.ndarray,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    test_index: np.ndarray,
    seed: int,
    name: str,
    output_dir: Path,
) -> tuple[np.ndarray, dict]:
    set_seed(seed)
    probe = nn.Linear(features.shape[1], 1)
    optimizer = AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features[train_index]).float(), torch.from_numpy(target[train_index]).float()),
        batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    best_mse = float("inf")
    stale = 0
    history = []
    checkpoint = output_dir / f"{name}_seed_{seed}.pt"
    for epoch in range(1, 101):
        probe.train()
        losses = []
        for x, y in loader:
            prediction = probe(x).squeeze(-1)
            loss = F.mse_loss(prediction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        probe.eval()
        with torch.inference_mode():
            validation_prediction = probe(torch.from_numpy(features[validation_index]).float()).squeeze(-1).numpy()
        result = regression_metrics(validation_prediction, target[validation_index])
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), **result})
        if result["MSE"] < best_mse:
            best_mse = result["MSE"]
            stale = 0
            torch.save({
                "model_state_dict": probe.state_dict(), "probe_version": PROBE_VERSION,
                "representation": name, "seed": seed, "best_epoch": epoch,
                "validation_metrics": result, "feature_dim": features.shape[1],
                "trainable_params": features.shape[1] + 1, "encoder_trainable_params": 0,
                "early_stopping_metric": "validation_MSE",
            }, checkpoint)
        else:
            stale += 1
            if stale >= 10:
                break
    pd.DataFrame(history).to_csv(output_dir / f"{name}_history_seed_{seed}.csv", index=False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    probe.load_state_dict(payload["model_state_dict"])
    probe.eval()
    with torch.inference_mode():
        prediction = probe(torch.from_numpy(features[test_index]).float()).squeeze(-1).numpy()
    return prediction, payload


def summarize_probe(rows: list[dict], representation: str) -> dict:
    selected = [row for row in rows if row["representation"] == representation and row["seed"] != "3 seeds"]
    summary = {
        "representation": representation, "seed": "3 seeds",
        "feature_dim": selected[0]["feature_dim"],
        "trainable_params": selected[0]["trainable_params"], "encoder_trainable_params": 0,
    }
    for key in ("SRCC", "PLCC", "MAE", "MSE"):
        values = np.asarray([row[key] for row in selected], dtype=np.float64)
        summary[key] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=0))
    rows.append(summary)
    return summary


def run_evaluation(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    clip_cache: dict,
    branch_cache: dict,
    audit: dict,
) -> None:
    output = args.output_dir
    prompt_names = list(clip_cache["prompt_names"])
    prompt_scores = clip_cache["prompt_scores"].numpy()
    embeddings = branch_cache["aesthetic_embeddings"].numpy()
    branch_scores = branch_cache["aesthetic_scores"].numpy()
    overall = frame["overall_score"].to_numpy(np.float32)
    train_index = np.flatnonzero(frame["split"].eq("train").to_numpy())
    validation_index = np.flatnonzero(frame["split"].eq("validation").to_numpy())
    test_index = np.flatnonzero(frame["split"].eq("test").to_numpy())
    prompt_index = {name: index for index, name in enumerate(prompt_names)}

    attribute_rows = []
    for prompt in prompt_names:
        for attribute in ATTRIBUTES:
            prediction = prompt_scores[test_index, prompt_index[prompt]]
            target = frame[attribute].to_numpy(np.float32)[test_index]
            level = "None"
            rationale = "Unmatched control"
            for mapped_prompt, mapped_attribute, mapped_level, mapped_rationale in MAPPINGS:
                if prompt == mapped_prompt and attribute == mapped_attribute:
                    level, rationale = mapped_level, mapped_rationale
                    break
            attribute_rows.append({
                "prompt": prompt, "attribute": attribute, "match_level": level,
                "is_strong_match": level == "Strong", "SRCC": correlation(prediction, target, "spearman"),
                "PLCC": correlation(prediction, target, "pearson"), "rationale": rationale,
            })
    attribute_frame = pd.DataFrame(attribute_rows)
    attribute_frame.to_csv(output / "aadb_prompt_attribute_results.csv", index=False, encoding="utf-8-sig")

    specificity_rows = []
    for prompt, attribute, level, _ in MAPPINGS:
        if level != "Strong" or attribute is None:
            continue
        prompt_rows = attribute_frame.loc[attribute_frame["prompt"].eq(prompt)]
        matched = float(prompt_rows.loc[prompt_rows["attribute"].eq(attribute), "SRCC"].iloc[0])
        unmatched = prompt_rows.loc[~prompt_rows["attribute"].eq(attribute), "SRCC"].abs()
        mean_unmatched = float(unmatched.mean())
        specificity_rows.append({
            "prompt": prompt, "matched_attribute": attribute, "matched_SRCC": matched,
            "mean_unmatched_abs_SRCC": mean_unmatched,
            "specificity": abs(matched) - mean_unmatched,
            "matched_rank_by_abs_SRCC": int(
                prompt_rows["SRCC"].abs().rank(method="min", ascending=False).loc[
                    prompt_rows["attribute"].eq(attribute)
                ].iloc[0]
            ),
            "num_attributes": len(ATTRIBUTES),
        })
    pd.DataFrame(specificity_rows).to_csv(output / "aadb_prompt_specificity.csv", index=False, encoding="utf-8-sig")

    matrix_values = np.eye(len(prompt_names), dtype=np.float64)
    redundancy_rows = []
    for left_index, left in enumerate(prompt_names):
        for right_index in range(left_index + 1, len(prompt_names)):
            right = prompt_names[right_index]
            value = correlation(prompt_scores[test_index, left_index], prompt_scores[test_index, right_index], "spearman")
            matrix_values[left_index, right_index] = value
            matrix_values[right_index, left_index] = value
            redundancy_rows.append({
                "prompt_pair": f"{left} <-> {right}", "SRCC": value,
                "absolute_SRCC": abs(value), "above_0_8": abs(value) > 0.8,
            })
    pd.DataFrame(matrix_values, index=prompt_names, columns=prompt_names).to_csv(
        output / "prompt_correlation_matrix.csv", encoding="utf-8-sig",
    )
    redundancy_rows.sort(key=lambda item: item["absolute_SRCC"], reverse=True)
    pd.DataFrame(redundancy_rows).to_csv(output / "prompt_redundancy.csv", index=False, encoding="utf-8-sig")

    prompt_overall_rows = []
    validation_prompt_metrics = {}
    for name in prompt_names:
        index = prompt_index[name]
        validation_prompt_metrics[name] = correlation(
            prompt_scores[validation_index, index], overall[validation_index], "spearman",
        )
        result = regression_metrics(prompt_scores[test_index, index], overall[test_index])
        prompt_overall_rows.append({
            "prompt": name, "selection_split": "not selected on test", **result,
            "validation_SRCC": validation_prompt_metrics[name],
        })
    pd.DataFrame(prompt_overall_rows).to_csv(
        output / "aadb_single_prompt_overall_results.csv", index=False, encoding="utf-8-sig",
    )
    selected_prompt = max(validation_prompt_metrics, key=validation_prompt_metrics.get)
    selected_index = prompt_index[selected_prompt]
    selected_metrics = regression_metrics(prompt_scores[test_index, selected_index], overall[test_index])
    mean_prompt_prediction = prompt_scores[test_index].mean(axis=1)
    mean_prompt_metrics = regression_metrics(mean_prompt_prediction, overall[test_index])
    branch_score_metrics = regression_metrics(branch_scores[test_index], overall[test_index])

    probe_dir = output / "linear_probes"
    probe_dir.mkdir(exist_ok=True)
    predictions_dir = output / "aesthetic_predictions"
    predictions_dir.mkdir(exist_ok=True)
    probe_rows = []
    prompt_probe_predictions = []
    embedding_probe_predictions = []
    for representation, features, collector in (
        ("multi_prompt_linear_probe", prompt_scores.astype(np.float32), prompt_probe_predictions),
        ("aesthetic_embedding_linear_probe", embeddings.astype(np.float32), embedding_probe_predictions),
    ):
        for seed in SEEDS:
            prediction, payload = train_probe(
                features, overall, train_index, validation_index, test_index,
                seed, representation, probe_dir,
            )
            result = regression_metrics(prediction, overall[test_index])
            probe_rows.append({
                "representation": representation, "seed": seed, **result,
                "feature_dim": features.shape[1], "trainable_params": payload["trainable_params"],
                "encoder_trainable_params": 0, "best_epoch": payload["best_epoch"],
            })
            collector.append(prediction)
    prompt_summary = summarize_probe(probe_rows, "multi_prompt_linear_probe")
    embedding_summary = summarize_probe(probe_rows, "aesthetic_embedding_linear_probe")
    pd.DataFrame(probe_rows).to_csv(output / "probe_multiseed_results.csv", index=False, encoding="utf-8-sig")

    overall_rows = [
        {"Representation": f"Best Single Prompt (validation-selected: {selected_prompt})", **selected_metrics,
         "SRCC_std": "", "PLCC_std": "", "selection_policy": "highest validation SRCC"},
        {"Representation": "Mean of Prompts", **mean_prompt_metrics,
         "SRCC_std": "", "PLCC_std": "", "selection_policy": "fixed unweighted mean"},
        {"Representation": "Multi-Prompt Linear Probe", **{key: prompt_summary[key] for key in ("SRCC", "PLCC", "MAE", "MSE")},
         "SRCC_std": prompt_summary["SRCC_std"], "PLCC_std": prompt_summary["PLCC_std"],
         "selection_policy": "3 seeds; official validation MSE early stopping"},
        {"Representation": "Aesthetic Embedding Linear Probe", **{key: embedding_summary[key] for key in ("SRCC", "PLCC", "MAE", "MSE")},
         "SRCC_std": embedding_summary["SRCC_std"], "PLCC_std": embedding_summary["PLCC_std"],
         "selection_policy": "3 seeds; official validation MSE early stopping"},
        {"Representation": "Frozen Branch Score (diagnostic)", **branch_score_metrics,
         "SRCC_std": "", "PLCC_std": "", "selection_policy": "no training"},
    ]
    pd.DataFrame(overall_rows).to_csv(output / "aadb_overall_results.csv", index=False, encoding="utf-8-sig")

    predictions = frame.iloc[test_index][["image_id", "overall_score"]].copy()
    predictions[f"single_prompt_{selected_prompt}"] = prompt_scores[test_index, selected_index]
    predictions["mean_prompt"] = mean_prompt_prediction
    predictions["frozen_branch_score"] = branch_scores[test_index]
    for seed, prediction in zip(SEEDS, prompt_probe_predictions):
        predictions[f"multi_prompt_probe_seed_{seed}"] = prediction
    for seed, prediction in zip(SEEDS, embedding_probe_predictions):
        predictions[f"embedding_probe_seed_{seed}"] = prediction
    predictions["multi_prompt_probe_mean"] = np.mean(np.stack(prompt_probe_predictions), axis=0)
    predictions["embedding_probe_mean"] = np.mean(np.stack(embedding_probe_predictions), axis=0)
    predictions.to_csv(predictions_dir / "aadb_test_predictions.csv", index=False, encoding="utf-8-sig")

    strong = specificity_rows[0] if specificity_rows else None
    mean_abs_corr = float(np.mean([row["absolute_SRCC"] for row in redundancy_rows]))
    max_pair = redundancy_rows[0]
    high_pair_count = sum(row["above_0_8"] for row in redundancy_rows)
    divide_text = "Previous DIVIDE metrics were not found."
    if args.previous_divide_results.is_file():
        divide = pd.read_csv(args.previous_divide_results)
        zero = divide.loc[divide["protocol"].eq("Zero-shot s_aes")].iloc[0]
        probe = divide.loc[divide["protocol"].eq("Frozen Linear Probe mean+-std")].iloc[0]
        divide_text = (
            f"The earlier supplementary DIVIDE video test obtained zero-shot SRCC={zero['SRCC']:.4f} "
            f"and embedding-probe SRCC={probe['SRCC']:.4f} +/- {probe['SRCC_std']:.4f}. "
            "It tests a different video/domain definition and is not the primary validation here."
        )
    specificity_text = (
        f"`{strong['prompt']} -> {strong['matched_attribute']}`: matched SRCC={strong['matched_SRCC']:.4f}, "
        f"mean unmatched |SRCC|={strong['mean_unmatched_abs_SRCC']:.4f}, "
        f"specificity={strong['specificity']:.4f}, rank={strong['matched_rank_by_abs_SRCC']}/{strong['num_attributes']}."
        if strong else "No preregistered Strong match was available."
    )
    complement = prompt_summary["SRCC"] > selected_metrics["SRCC"]
    embedding_retains = embedding_summary["SRCC"] >= prompt_summary["SRCC"]
    summary = f"""# Aesthetic External Validation v2: AADB-Only Summary

## 1. Research Question

Does the current multi-attribute aesthetic design have external semantic validity and overall-aesthetic transfer on locally available AADB?

## 2. Branch Definition

KEEP_P2_S2 uses seven fixed A1 prompt contrasts over frozen ViT-L/14 features. Their 7D scores and a 512D legacy visual feature enter the frozen Aesthetic Branch, which outputs a 128D embedding and a sigmoid score. No prompt, CLIP weight, or branch parameter is changed.

## 3. Dataset-Definition Matching

AADB directly provides overall aesthetics plus eleven human aesthetic attributes, so it matches the project's multi-attribute definition more closely than a single video-aesthetic MOS. This phase uses only the local F-drive AADB images. AVA is intentionally deferred by user scope.

## 4. AADB Attribute Alignment

The sole conservative Strong mapping is `lighting -> Light`, with SRCC={strong['matched_SRCC']:.4f} and PLCC={float(attribute_frame.loc[(attribute_frame['prompt'].eq('lighting')) & (attribute_frame['attribute'].eq('Light')), 'PLCC'].iloc[0]):.4f}.

## 5. Matched-vs-Unmatched Analysis

{specificity_text}

## 6. Prompt Redundancy

Across 21 pairs, mean absolute SRCC={mean_abs_corr:.4f}, maximum absolute SRCC={max_pair['absolute_SRCC']:.4f} (`{max_pair['prompt_pair']}`), and {high_pair_count} pairs exceed |SRCC|>0.8.

## 7. AADB Overall Aesthetics

The best single prompt is selected only on official validation (`{selected_prompt}`, validation SRCC={validation_prompt_metrics[selected_prompt]:.4f}), then evaluated once on TestNew: SRCC={selected_metrics['SRCC']:.4f}, PLCC={selected_metrics['PLCC']:.4f}. Mean prompt reaches SRCC={mean_prompt_metrics['SRCC']:.4f}. Multi-prompt probe reaches SRCC={prompt_summary['SRCC']:.4f} +/- {prompt_summary['SRCC_std']:.4f}, PLCC={prompt_summary['PLCC']:.4f} +/- {prompt_summary['PLCC_std']:.4f}. Aesthetic embedding probe reaches SRCC={embedding_summary['SRCC']:.4f} +/- {embedding_summary['SRCC_std']:.4f}, PLCC={embedding_summary['PLCC']:.4f} +/- {embedding_summary['PLCC_std']:.4f}.

## 8. AVA Overall Generalization

Not run in this AADB-only phase.

## 9. AVA Style Semantic Validation

Not run in this AADB-only phase.

## 10. Cross-Dataset Comparison

AADB-versus-AVA consistency cannot yet be assessed because AVA was explicitly excluded from this run.

## 11. Relation to Previous DIVIDE Result

{divide_text}

## 12. Conclusion

1. **Low redundancy:** {'yes' if high_pair_count == 0 else 'not fully'}; no automatic prompt deletion is performed.
2. **Attribute semantics:** the preregistered Strong match has specificity {strong['specificity']:.4f}; this is {'positive' if strong['specificity'] > 0 else 'non-positive'} but limited to one exact match.
3. **Multi-prompt complementarity:** {'supported' if complement else 'not supported'} because the multi-prompt probe {'exceeds' if complement else 'does not exceed'} the validation-selected single prompt on TestNew.
4. **Overall aesthetics:** the multi-prompt probe {'provides' if prompt_summary['SRCC'] > 0 else 'does not provide'} positive external prediction.
5. **Embedding retention:** the 128D branch embedding {'retains or improves on' if embedding_retains else 'underperforms'} the raw 7D prompt probe on AADB.
6. **AADB/AVA consistency:** not answerable until the separately scoped AVA phase is run.
"""
    (output / "aesthetic_external_validation_summary.md").write_text(summary, encoding="utf-8")
    (output / "supplementary_divide_cross_domain_video_aesthetic_test.md").write_text(
        "# Supplementary Cross-Domain Video Aesthetic Test\n\n" + divide_text + "\n",
        encoding="utf-8",
    )
    (output / "experiment_manifest.json").write_text(json.dumps({
        "experiment": "E3_v2_AADB_only", "seeds": list(SEEDS),
        "aadb_split_hash": audit["split_hash"], "num_samples": len(frame),
        "clip_cache": str(args.clip_cache.resolve()), "branch_cache_version": BRANCH_CACHE_VERSION,
        "ava_executed": False, "prompt_version": "a1_original",
        "selected_prompt_policy": "highest official-validation SRCC; test never used for selection",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, audit = load_annotations(args.annotation_dir)
    missing_images = [image_id for image_id in frame["image_id"] if not (args.image_dir / image_id).is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} local AADB images; examples: {missing_images[:3]}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    # These artifacts freeze definitions and mappings before any correlation is computed.
    write_branch_audits(args, frame, audit, model)
    write_preregistered_mapping(args.output_dir)

    clip_cache = load_clip_cache(args.clip_cache, frame)
    branch_cache = build_branch_cache(args, frame, clip_cache, model, device)
    run_evaluation(args, frame, clip_cache, branch_cache, audit)
    print(f"E3 v2 AADB-only complete: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
