"""Benchmark the current shared CLIP visual encoder on cached video frames."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import CFG
from pipeline.step_video import VideoEncoder
from pipeline.utils_io import load_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=CFG.metadata_csv)
    parser.add_argument("--frame-dir", type=Path, default=CFG.frame_dir)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--device", default=CFG.device)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata)
    video_ids = [
        video_id for video_id in metadata["video_id"].astype(str)
        if (args.frame_dir / video_id).exists()
    ][: args.count]
    if len(video_ids) != args.count:
        raise RuntimeError(f"Expected {args.count} frame directories, found {len(video_ids)}")

    encoder = VideoEncoder(CFG.clip_model_name, args.device)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    bytes_per_video = []
    for video_id in video_ids:
        started = time.perf_counter()
        features = encoder.encode_frame_features(
            args.frame_dir / video_id, max_frames=CFG.clip_max_frames
        )
        if features.ndim != 2 or features.shape[-1] != CFG.clip_video_dim:
            raise RuntimeError(f"Invalid CLIP features for {video_id}: {tuple(features.shape)}")
        times.append(time.perf_counter() - started)
        bytes_per_video.append(features.numel() * features.element_size())

    payload = {
        "encoder": "Shared CLIP ViT-L/14 visual encoder",
        "device": str(device),
        "count": len(video_ids),
        "max_frames": CFG.clip_max_frames,
        "total_seconds": float(sum(times)),
        "mean_seconds_per_video": float(np.mean(times)),
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "mean_feature_bytes_per_video": float(np.mean(bytes_per_video)),
        "video_ids": video_ids,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
