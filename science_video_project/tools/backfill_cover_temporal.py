"""Cache spatially pooled COVER technical sequences without changing sampling."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import CFG
from pipeline.step_cover_technical import COVERTechnicalFeatureExtractor
from pipeline.utils_io import get_video_id, list_videos, load_metadata, load_pt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=CFG.metadata_csv)
    parser.add_argument("--video-dir", type=Path, default=CFG.video_dir)
    parser.add_argument("--feature-dir", type=Path, default=CFG.feature_dir)
    parser.add_argument("--cover-root", type=Path, default=CFG.cover_root)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "temporal_quality_statistics" / "cache",
    )
    parser.add_argument("--device", default=CFG.device)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--report", type=Path,
        default=PROJECT_ROOT / "outputs" / "temporal_quality_statistics" / "extraction_report.json",
    )
    return parser.parse_args()


def valid_cache(path: Path) -> bool:
    if not path.exists():
        return False
    payload = load_pt(path)
    feature = np.asarray(payload.get("temporal_feature", []), dtype=np.float32)
    return (
        payload.get("feature_version")
        == COVERTechnicalFeatureExtractor.TEMPORAL_FEATURE_VERSION
        and feature.ndim == 2
        and feature.shape[0] >= 1
        and feature.shape[1] == COVERTechnicalFeatureExtractor.FEATURE_DIM
        and np.isfinite(feature).all()
    )


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.metadata)
    available_features = {path.stem for path in args.feature_dir.glob("*.pt")}
    metadata = metadata[metadata["video_id"].astype(str).isin(available_features)].copy()
    videos = {}
    for path in list_videos(args.video_dir):
        videos.setdefault(get_video_id(path), path)

    jobs = []
    skipped = 0
    missing_video = []
    for video_id in metadata["video_id"].astype(str):
        output = args.output_dir / f"{video_id}.pt"
        if valid_cache(output) and not args.force:
            skipped += 1
            continue
        if video_id not in videos:
            missing_video.append(video_id)
            continue
        jobs.append((video_id, videos[video_id], output))
    if args.limit > 0:
        jobs = jobs[:args.limit]

    extractor = COVERTechnicalFeatureExtractor(
        args.cover_root, device=args.device, use_amp=args.amp
    )
    failures = []
    times = []
    shapes = {}
    pooled_max_abs_errors = []
    short_sequences = 0
    started = time.perf_counter()
    for video_id, video_path, output in tqdm(jobs, desc="COVER temporal"):
        try:
            item_started = time.perf_counter()
            temporal = extractor.extract_temporal(video_path)
            if temporal.shape[0] < 2:
                short_sequences += 1
            source = load_pt(args.feature_dir / f"{video_id}.pt")
            pooled = np.asarray(source["cover_technical_feat"], dtype=np.float32)
            error = float(np.max(np.abs(temporal.mean(dim=0).numpy() - pooled)))
            pooled_max_abs_errors.append(error)
            shape_key = "x".join(str(value) for value in temporal.shape)
            shapes[shape_key] = shapes.get(shape_key, 0) + 1
            atomic_save(
                {
                    "video_id": video_id,
                    "temporal_feature": temporal.numpy(),
                    "feature_version": extractor.TEMPORAL_FEATURE_VERSION,
                    "source_feature_version": extractor.FEATURE_VERSION,
                    "sampling": extractor.sampling_config,
                    "checkpoint": str(extractor.checkpoint_path),
                    "pooling": "spatial mean of [B,768,T,H,W], stored as [T,768]",
                    "pooled_max_abs_error": error,
                },
                output,
            )
            times.append(time.perf_counter() - item_started)
        except Exception as exc:
            failures.append({"video_id": video_id, "error": repr(exc)})

    report = {
        "feature_version": extractor.TEMPORAL_FEATURE_VERSION,
        "source_feature_version": extractor.FEATURE_VERSION,
        "checkpoint": str(extractor.checkpoint_path),
        "sampling": extractor.sampling_config,
        "metadata_count": int(len(metadata)),
        "scheduled": len(jobs),
        "succeeded": len(times),
        "skipped": skipped,
        "failed": len(failures),
        "missing_video": missing_video,
        "sequence_shapes": shapes,
        "num_short_temporal_sequences": short_sequences,
        "mean_seconds_per_video": float(np.mean(times)) if times else 0.0,
        "total_seconds": time.perf_counter() - started,
        "pooled_max_abs_error_max": (
            float(np.max(pooled_max_abs_errors)) if pooled_max_abs_errors else 0.0
        ),
        "failures": failures,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures or missing_video:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
