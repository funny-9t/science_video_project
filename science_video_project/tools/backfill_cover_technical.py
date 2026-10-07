"""Backfill only the frozen COVER technical representation into feature files."""

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
    parser.add_argument("--cover-root", type=Path, default=CFG.cover_root)
    parser.add_argument("--source-feature-dir", type=Path, default=CFG.feature_dir)
    parser.add_argument("--output-feature-dir", type=Path, default=CFG.feature_dir)
    parser.add_argument("--device", default=CFG.device)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--benchmark-count", type=int, default=20)
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "outputs" / "technical_cover_ablation" / "extraction_report.json")
    return parser.parse_args()


def _valid(sample: dict) -> bool:
    value = np.asarray(sample.get("cover_technical_feat", []), dtype=np.float32)
    return (
        value.shape == (COVERTechnicalFeatureExtractor.FEATURE_DIM,)
        and np.isfinite(value).all()
        and sample.get("cover_technical_feature_version")
        == COVERTechnicalFeatureExtractor.FEATURE_VERSION
    )


def _atomic_save(sample: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(sample, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    args.output_feature_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.metadata)
    videos = {}
    for path in list_videos(args.video_dir):
        videos.setdefault(get_video_id(path), path)

    jobs = []
    skipped = 0
    missing_source = []
    missing_video = []
    for video_id in metadata["video_id"].astype(str):
        source = args.source_feature_dir / f"{video_id}.pt"
        output = args.output_feature_dir / f"{video_id}.pt"
        if not source.exists() and not output.exists():
            missing_source.append(video_id)
            continue
        if video_id not in videos:
            missing_video.append(video_id)
            continue
        existing = load_pt(output if output.exists() else source)
        if _valid(existing) and not args.force:
            skipped += 1
            continue
        jobs.append((video_id, videos[video_id], source, output))
    if args.limit > 0:
        jobs = jobs[: args.limit]

    extractor = COVERTechnicalFeatureExtractor(
        args.cover_root, device=args.device, use_amp=args.amp
    )
    extraction_times = []
    size_deltas = []
    failures = []
    if extractor.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(extractor.device)
    started = time.perf_counter()
    for video_id, video_path, source, output in tqdm(jobs, desc="COVER technical"):
        try:
            if output.exists():
                sample = load_pt(output)
            else:
                sample = load_pt(source)
            before = source.stat().st_size if source.exists() else output.stat().st_size
            item_started = time.perf_counter()
            feature = extractor.extract(video_path)
            sample["cover_technical_feat"] = np.asarray(feature.tolist(), dtype=np.float32)
            sample["cover_technical_feature_version"] = extractor.FEATURE_VERSION
            sample["cover_technical_feature_dim"] = extractor.FEATURE_DIM
            _atomic_save(sample, output)
            extraction_times.append(time.perf_counter() - item_started)
            size_deltas.append(output.stat().st_size - before)
        except Exception as exc:
            failures.append({"video_id": video_id, "error": repr(exc)})

    elapsed = time.perf_counter() - started
    peak_vram = (
        int(torch.cuda.max_memory_allocated(extractor.device))
        if extractor.device.type == "cuda"
        else 0
    )
    benchmark = extraction_times[: max(args.benchmark_count, 0)]
    payload = {
        "feature_version": extractor.FEATURE_VERSION,
        "feature_dim": extractor.FEATURE_DIM,
        "backbone": "COVER swin_tiny_grpb (SwinTransformer3D)",
        "checkpoint": str(extractor.checkpoint_path),
        "sampling": extractor.sampling_config,
        "pooling": "mean over temporal and spatial axes of [B,768,T,H,W]",
        "device": str(extractor.device),
        "amp": extractor.use_amp,
        "metadata_count": int(len(metadata)),
        "scheduled": len(jobs),
        "succeeded": len(extraction_times),
        "failed": len(failures),
        "skipped": skipped,
        "missing_source": missing_source,
        "missing_video": missing_video,
        "total_seconds": elapsed,
        "mean_seconds_per_video": float(np.mean(extraction_times)) if extraction_times else 0.0,
        "benchmark_count": len(benchmark),
        "benchmark_mean_seconds_per_video": float(np.mean(benchmark)) if benchmark else 0.0,
        "peak_vram_bytes": peak_vram,
        "mean_feature_size_delta_bytes": float(np.mean(size_deltas)) if size_deltas else 0.0,
        "failures": failures,
    }
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if failures or missing_source or missing_video:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
