"""Run E4: validate KEEP_P2_S2 A1 aesthetic prompts on AADB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import CFG
from pipeline.step_aesthetic_clip import AESTHETIC_PROMPT_SETS, SharedCLIPAestheticScorer
from pipeline.step_video import VideoEncoder
from training.utils_train import set_seed


SEEDS = (42, 123, 2026)
ATTRIBUTES = (
    "BalacingElements", "ColorHarmony", "Content", "DoF", "Light", "MotionBlur",
    "Object", "Repetition", "RuleOfThirds", "Symmetry", "VividColor",
)
FEATURE_VERSION = "e4_aadb_vitl14_a1_prompts_v1"
PROBE_VERSION = "e4_aadb_prompt_linear_probe_v1"

MAPPINGS = (
    ("clarity", "MotionBlur", "Partial", "Sharpness overlaps with blur, but motion blur is narrower than generic blur/focus."),
    ("cleanliness", None, "None", "AADB has no clutter or visual-cleanliness attribute."),
    ("composition", "BalacingElements", "Partial", "Balanced elements contribute to composition but do not cover framing as a whole."),
    ("composition", "RuleOfThirds", "Partial", "Rule of thirds is one composition rule; the prompt is intentionally broader."),
    ("composition", "Symmetry", "Partial", "Symmetry can affect composition but is not required by the prompt."),
    ("appeal", "Content", "Partial", "Generic visual appeal can reflect content preference but is not content-specific."),
    ("lighting", "Light", "Strong", "Both directly contrast good versus poor lighting."),
    ("lighting", "ColorHarmony", "Partial", "The prompt also mentions balanced colors, but color harmony is not its sole meaning."),
    ("lighting", "VividColor", "Partial", "Bright colors overlap with vividness, while the prompt also includes illumination."),
    ("text_readability", None, "None", "AADB has no text or diagram-readability attribute."),
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
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "aadb_prompt_alignment",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--force", action="store_true")
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
            raise ValueError(f"{phase}/{attribute} image IDs do not match overall score file")
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
        raise ValueError("AADB de-overlapped protocol still contains duplicate image IDs")
    audit = {
        "train_raw": len(train), "validation_raw": len(validation_raw), "test_raw": len(test),
        "validation_test_overlap": len(overlap), "validation_after_deoverlap": len(validation),
        "unique_protocol_images": len(frame), "overlap_examples": overlap[:10],
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


def torch_pearson(left: np.ndarray, right: np.ndarray) -> float:
    x = torch.from_numpy(np.asarray(left, dtype=np.float64).copy())
    y = torch.from_numpy(np.asarray(right, dtype=np.float64).copy())
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.sqrt(torch.sum(x * x) * torch.sum(y * y)).clamp_min(1e-15)
    return float((torch.sum(x * y) / denominator).item())


def correlation(prediction: np.ndarray, target: np.ndarray, kind: str) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(prediction) & np.isfinite(target)
    prediction, target = prediction[valid], target[valid]
    if prediction.size < 2 or np.unique(prediction).size < 2 or np.unique(target).size < 2:
        return float("nan")
    if kind == "spearman":
        prediction = rankdata(prediction)
        target = rankdata(target)
    return torch_pearson(prediction, target)


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    error = np.asarray(prediction, np.float64) - np.asarray(target, np.float64)
    return {
        "SRCC": correlation(prediction, target, "spearman"),
        "PLCC": correlation(prediction, target, "pearson"),
        "MSE": float(np.mean(error ** 2)),
        "MAE": float(np.mean(np.abs(error))),
    }


def encode_shard(
    encoder: VideoEncoder,
    paths: list[Path],
    batch_size: int,
) -> torch.Tensor:
    output = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            images = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = encoder.processor(images=images, return_tensors="pt")
            pixels = inputs["pixel_values"].to(encoder.device, dtype=encoder.model.dtype)
            embedding = encoder.model(pixel_values=pixels).image_embeds
            output.append(F.normalize(embedding, dim=-1).cpu().half())
    return torch.cat(output)


def extract_cache(args: argparse.Namespace, frame: pd.DataFrame) -> dict:
    cache_dir = args.output_dir / "cache" / "external_validation" / "aadb"
    shard_dir = cache_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [args.image_dir / image_id for image_id in frame["image_id"]]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} AADB images; examples: {missing[:3]}")

    encoder = VideoEncoder(CFG.clip_model_name, CFG.device, batch_size=args.batch_size)
    shard_payloads = []
    for start in range(0, len(frame), args.shard_size):
        stop = min(start + args.shard_size, len(frame))
        target = shard_dir / f"shard_{start:05d}_{stop:05d}.pt"
        expected_ids = frame["image_id"].iloc[start:stop].tolist()
        payload = None
        if target.exists() and not args.force:
            candidate = torch.load(target, map_location="cpu", weights_only=False)
            if (
                candidate.get("feature_version") == FEATURE_VERSION
                and candidate.get("image_ids") == expected_ids
                and tuple(candidate.get("clip_embeddings", torch.empty(0)).shape) == (stop - start, 768)
            ):
                payload = candidate
        if payload is None:
            embeddings = encode_shard(encoder, image_paths[start:stop], args.batch_size)
            payload = {
                "feature_version": FEATURE_VERSION,
                "image_ids": expected_ids,
                "clip_embeddings": embeddings,
            }
            torch.save(payload, target)
        shard_payloads.append(payload)
        print(f"AADB CLIP cache {stop}/{len(frame)}")
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    clip_embeddings = torch.cat([item["clip_embeddings"] for item in shard_payloads]).float()
    scorer = SharedCLIPAestheticScorer(
        CFG.clip_model_name, device=CFG.device, prompt_version="a1_original",
    )
    with torch.inference_mode():
        image_features = F.normalize(clip_embeddings.to(scorer.device, dtype=scorer.dtype), dim=-1)
        prompt_scores = (
            image_features @ scorer.positive_features.T
            - image_features @ scorer.negative_features.T
        ).cpu().float()
    payload = {
        "feature_version": FEATURE_VERSION,
        "image_ids": frame["image_id"].tolist(),
        "splits": frame["split"].tolist(),
        "clip_embeddings": clip_embeddings.half(),
        "prompt_names": scorer.prompt_names,
        "prompt_scores": prompt_scores,
        "overall_score": torch.tensor(frame["overall_score"].to_numpy(), dtype=torch.float32),
        "attribute_scores": {
            attribute: torch.tensor(frame[attribute].to_numpy(), dtype=torch.float32)
            for attribute in ATTRIBUTES
        },
        "metadata": {
            "clip_model": str(CFG.clip_model_name), "prompt_version": "a1_original",
            "score_formula": "cosine(image, positive) - cosine(image, negative)",
            "validation_test_overlap_removed": True,
        },
    }
    combined = cache_dir / "aadb_clip_prompt_cache.pt"
    torch.save(payload, combined)
    return payload


def load_or_extract_cache(args: argparse.Namespace, frame: pd.DataFrame) -> dict:
    path = args.output_dir / "cache" / "external_validation" / "aadb" / "aadb_clip_prompt_cache.pt"
    if path.exists() and not args.force:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("feature_version") == FEATURE_VERSION and payload.get("image_ids") == frame["image_id"].tolist():
            return payload
    return extract_cache(args, frame)


def train_probe(
    features: np.ndarray,
    target: np.ndarray,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    test_index: np.ndarray,
    seed: int,
    output_dir: Path,
) -> tuple[np.ndarray, dict]:
    set_seed(seed)
    probe = nn.Linear(features.shape[1], 1)
    optimizer = AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features[train_index]).float(), torch.from_numpy(target[train_index]).float()),
        batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    history = []
    best_srcc = -float("inf")
    stale = 0
    checkpoint = output_dir / f"aadb_prompt_probe_seed_{seed}.pt"
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
            prediction = probe(torch.from_numpy(features[validation_index]).float()).squeeze(-1).numpy()
        result = regression_metrics(prediction, target[validation_index])
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), **result})
        if result["SRCC"] > best_srcc:
            best_srcc = result["SRCC"]
            stale = 0
            torch.save({
                "model_state_dict": probe.state_dict(), "probe_version": PROBE_VERSION,
                "seed": seed, "best_epoch": epoch, "validation_metrics": result,
                "feature_dim": features.shape[1], "trainable_params": features.shape[1] + 1,
            }, checkpoint)
        else:
            stale += 1
            if stale >= 10:
                break
    pd.DataFrame(history).to_csv(output_dir / f"aadb_prompt_probe_history_seed_{seed}.csv", index=False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    probe.load_state_dict(payload["model_state_dict"])
    probe.eval()
    with torch.inference_mode():
        prediction = probe(torch.from_numpy(features[test_index]).float()).squeeze(-1).numpy()
    return prediction, payload


def create_reports(args: argparse.Namespace, frame: pd.DataFrame, audit: dict, cache: dict) -> None:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    prompt_names = list(cache["prompt_names"])
    prompt_scores = cache["prompt_scores"].numpy()
    overall = frame["overall_score"].to_numpy(np.float32)
    test_index = np.flatnonzero(frame["split"].eq("test").to_numpy())
    train_index = np.flatnonzero(frame["split"].eq("train").to_numpy())
    validation_index = np.flatnonzero(frame["split"].eq("validation").to_numpy())
    prompt_to_index = {name: index for index, name in enumerate(prompt_names)}

    attribute_stats = []
    for attribute in ("overall_score", *ATTRIBUTES):
        values = frame[attribute]
        attribute_stats.append({
            "attribute": attribute, "sample_count": int(values.notna().sum()),
            "missing_count": int(values.isna().sum()), "mean": float(values.mean()),
            "std": float(values.std(ddof=0)), "min": float(values.min()), "max": float(values.max()),
        })
    pd.DataFrame(attribute_stats).to_csv(output / "aadb_attribute_statistics.csv", index=False, encoding="utf-8-sig")
    stats_lines = [
        "# AADB Dataset Audit", "",
        f"- Image directory: `{args.image_dir}`",
        f"- Official annotation directory: `{args.annotation_dir}`",
        f"- Raw split sizes: Train={audit['train_raw']}, TestNew={audit['test_raw']}, Validation={audit['validation_raw']}.",
        f"- Validation/TestNew overlap: {audit['validation_test_overlap']} images.",
        f"- Leakage-safe protocol: Train={len(train_index)}, Validation={len(validation_index)}, TestNew={len(test_index)}; TestNew has priority and overlapping validation rows are excluded.",
        f"- Unique images encoded: {len(frame)}.", "",
        "| Attribute | Samples | Missing | Mean | Std | Min | Max |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in attribute_stats:
        stats_lines.append(
            f"| {item['attribute']} | {item['sample_count']} | {item['missing_count']} | {item['mean']:.4f} | {item['std']:.4f} | {item['min']:.4f} | {item['max']:.4f} |"
        )
    (output / "aadb_dataset_audit.md").write_text("\n".join(stats_lines) + "\n", encoding="utf-8")

    prompt_rows = []
    intentions = {
        "clarity": "visual sharpness and detail", "cleanliness": "low clutter and distraction",
        "composition": "composition and framing", "appeal": "generic visual appeal",
        "lighting": "lighting and color balance", "text_readability": "scientific text readability",
        "professional": "professional scientific presentation",
    }
    for index, prompt in enumerate(AESTHETIC_PROMPT_SETS["a1_original"]):
        prompt_rows.append((index, prompt["name"], prompt["positive"], prompt["negative"], intentions[prompt["name"]]))
    prompt_lines = [
        "# Current Aesthetic Prompts", "",
        f"- CLIP backbone: `{CFG.clip_model_name}` (ViT-L/14).",
        "- Text embedding: frozen `CLIPTextModelWithProjection`, L2 normalized.",
        "- Image embedding: frozen `CLIPVisionModelWithProjection`, L2 normalized.",
        "- Score: cosine(image, positive) - cosine(image, negative).", "",
        "| ID | Prompt | Positive | Negative | Semantic intention |", "|---:|---|---|---|---|",
    ]
    for row in prompt_rows:
        prompt_lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |")
    (output / "current_aesthetic_prompts.md").write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")

    mapping_lines = [
        "# Prompt-Attribute Mapping", "",
        "Only Strong matches enter the primary attribute-alignment conclusion. Partial matches are supplementary.", "",
        "| Prompt | AADB Attribute | Match Level | Rationale |", "|---|---|---|---|",
    ]
    for prompt, attribute, level, rationale in MAPPINGS:
        mapping_lines.append(f"| {prompt} | {attribute or '-'} | {level} | {rationale} |")
    (output / "prompt_attribute_mapping.md").write_text("\n".join(mapping_lines) + "\n", encoding="utf-8")

    overall_metrics = {}
    for prompt in prompt_names:
        index = prompt_to_index[prompt]
        overall_metrics[prompt] = {
            "SRCC": correlation(prompt_scores[test_index, index], overall[test_index], "spearman"),
            "PLCC": correlation(prompt_scores[test_index, index], overall[test_index], "pearson"),
        }
    result_rows = []
    for prompt, attribute, level, rationale in MAPPINGS:
        index = prompt_to_index[prompt]
        if attribute:
            target = frame[attribute].to_numpy(np.float32)
            attribute_srcc = correlation(prompt_scores[test_index, index], target[test_index], "spearman")
            attribute_plcc = correlation(prompt_scores[test_index, index], target[test_index], "pearson")
        else:
            attribute_srcc = float("nan")
            attribute_plcc = float("nan")
        result_rows.append({
            "prompt": prompt, "attribute": attribute or "", "match_level": level,
            "attribute_SRCC": attribute_srcc, "attribute_PLCC": attribute_plcc,
            "overall_SRCC": overall_metrics[prompt]["SRCC"],
            "overall_PLCC": overall_metrics[prompt]["PLCC"], "rationale": rationale,
        })
    pd.DataFrame(result_rows).to_csv(output / "aadb_prompt_attribute_results.csv", index=False, encoding="utf-8-sig")

    matrix_values = np.eye(len(prompt_names), dtype=np.float64)
    for left_index in range(len(prompt_names)):
        for right_index in range(left_index + 1, len(prompt_names)):
            value = correlation(
                prompt_scores[test_index, left_index],
                prompt_scores[test_index, right_index],
                "spearman",
            )
            matrix_values[left_index, right_index] = value
            matrix_values[right_index, left_index] = value
    matrix = pd.DataFrame(matrix_values, index=prompt_names, columns=prompt_names)
    matrix.to_csv(output / "prompt_correlation_matrix.csv", encoding="utf-8-sig")
    redundancy = []
    for left_index, left in enumerate(prompt_names):
        for right in prompt_names[left_index + 1:]:
            value = float(matrix.loc[left, right])
            redundancy.append({
                "prompt_pair": f"{left} <-> {right}", "SRCC": value,
                "absolute_SRCC": abs(value), "highly_correlated": abs(value) > 0.8,
            })
    redundancy.sort(key=lambda item: item["absolute_SRCC"], reverse=True)
    pd.DataFrame(redundancy).to_csv(output / "prompt_redundancy.csv", index=False, encoding="utf-8-sig")

    probe_dir = output / "linear_probes"
    probe_dir.mkdir(exist_ok=True)
    probe_results = []
    probe_predictions = []
    prediction_frame = frame.iloc[test_index][["image_id", "overall_score"]].copy()
    for prompt in prompt_names:
        prediction_frame[f"prompt_{prompt}"] = prompt_scores[test_index, prompt_to_index[prompt]]
    for seed in SEEDS:
        prediction, payload = train_probe(
            prompt_scores, overall, train_index, validation_index, test_index, seed, probe_dir,
        )
        result = regression_metrics(prediction, overall[test_index])
        probe_results.append({
            "protocol": "All Prompt Linear Probe", "seed": seed, **result,
            "num_train": len(train_index), "num_validation": len(validation_index),
            "num_test": len(test_index), "feature_dim": len(prompt_names),
            "trainable_params": len(prompt_names) + 1, "best_epoch": payload["best_epoch"],
        })
        probe_predictions.append(prediction)
        prediction_frame[f"probe_seed_{seed}"] = prediction
    prediction_frame["probe_mean"] = np.mean(np.stack(probe_predictions), axis=0)
    prediction_frame.to_csv(output / "aadb_predictions.csv", index=False, encoding="utf-8-sig")
    summary_row = {
        "protocol": "All Prompt Linear Probe mean+-std", "seed": "3 seeds",
        "num_train": len(train_index), "num_validation": len(validation_index),
        "num_test": len(test_index), "feature_dim": len(prompt_names),
        "trainable_params": len(prompt_names) + 1,
    }
    for key in ("SRCC", "PLCC", "MSE", "MAE"):
        values = np.array([row[key] for row in probe_results])
        summary_row[key] = float(values.mean())
        summary_row[f"{key}_std"] = float(values.std(ddof=0))
    probe_results.append(summary_row)
    pd.DataFrame(probe_results).to_csv(output / "aadb_probe_results.csv", index=False, encoding="utf-8-sig")

    strong = [row for row in result_rows if row["match_level"] == "Strong"]
    best_overall = max(overall_metrics.items(), key=lambda item: item[1]["SRCC"])
    high_pairs = [row for row in redundancy if row["highly_correlated"]]
    strongest_pair = redundancy[0]
    strong_text = ", ".join(
        f"`{row['prompt']}->{row['attribute']}` SRCC={row['attribute_SRCC']:.4f}"
        for row in strong
    ) or "No Strong mappings"
    summary = f"""# E4 AADB Prompt Alignment Summary

## Main Findings

1. **External attribute semantics:** {strong_text}. Partial mappings are reported separately and are not promoted to primary evidence.
2. **Best aligned prompt:** among Strong matches, `{strong[0]['prompt'] if strong else 'N/A'}` is the only semantically exact candidate under the conservative mapping policy.
3. **Redundancy:** {len(high_pairs)} of {len(redundancy)} prompt pairs have |SRCC| > 0.8. The strongest pair is `{strongest_pair['prompt_pair']}` (SRCC={strongest_pair['SRCC']:.4f}).
4. **Overall transfer:** the best individual prompt is `{best_overall[0]}` (SRCC={best_overall[1]['SRCC']:.4f}, PLCC={best_overall[1]['PLCC']:.4f}); the 7-prompt linear probe reaches SRCC={summary_row['SRCC']:.4f} +/- {summary_row['SRCC_std']:.4f}, PLCC={summary_row['PLCC']:.4f} +/- {summary_row['PLCC_std']:.4f}.
5. **Disentanglement conclusion:** prompt disentanglement is supported only if the Strong attribute correlation is meaningful and redundancy remains limited; the measured values above are retained without prompt redesign or post-hoc selection.

## Protocol Note

The official annotation package labels phases as Train/TestNew/Validation, but Validation overlaps TestNew by {audit['validation_test_overlap']} images. Those rows were excluded from validation so the final test set never influenced early stopping.
"""
    (output / "aadb_prompt_alignment_summary.md").write_text(summary, encoding="utf-8")
    (output / "experiment_manifest.json").write_text(json.dumps({
        "experiment": "E4", "feature_version": FEATURE_VERSION, "seeds": list(SEEDS),
        "clip_model": str(CFG.clip_model_name), "prompt_version": "a1_original", "audit": audit,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, audit = load_annotations(args.annotation_dir)
    cache = load_or_extract_cache(args, frame)
    create_reports(args, frame, audit, cache)
    print(f"E4 complete: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
