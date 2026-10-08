"""Filter metadata against local videos and report video-level statistics."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".webm"}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=script_dir / "parsed_metadata.csv")
    parser.add_argument("--output", type=Path, default=script_dir / "parsed_metadata_filtered.csv")
    parser.add_argument("--video-dir", type=Path, default=Path(r"F:\Data\172.16.29.65"))
    return parser.parse_args()


def get_video_id(path: Path) -> str:
    """Use the same trailing-number rule as the feature pipeline."""
    match = re.search(r"(\d{6,})$", path.stem)
    return match.group(1) if match else path.stem


def label_counts(df: pd.DataFrame) -> dict[int, int]:
    labels = pd.to_numeric(df["label"], errors="coerce")
    return {int(key): int(value) for key, value in labels.value_counts().sort_index().items()}


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.input, dtype={"video_id": str})
    metadata["video_id"] = metadata["video_id"].astype(str).str.strip()
    metadata["label"] = pd.to_numeric(metadata["label"], errors="coerce").fillna(0).astype(int)

    video_paths = [
        path
        for path in args.video_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    available_ids = {get_video_id(path) for path in video_paths}
    filtered = metadata[metadata["video_id"].isin(available_ids)].copy()

    label_nunique = filtered.groupby("video_id")["label"].nunique()
    conflicting_ids = set(label_nunique[label_nunique > 1].index.astype(str))
    non_conflicting = filtered[~filtered["video_id"].isin(conflicting_ids)]
    train_candidates = non_conflicting.drop_duplicates("video_id", keep="last")

    print(f"Original rows: {len(metadata)}")
    print(f"Video files: {len(video_paths)}")
    print(f"Available unique video IDs: {len(available_ids)}")
    print(f"Filtered rows: {len(filtered)} | labels: {label_counts(filtered)}")
    print(f"Filtered unique video IDs: {filtered['video_id'].nunique()}")
    print(f"Conflicting-label video IDs: {len(conflicting_ids)}")
    print(
        "Train candidates after conflict removal and deduplication: "
        f"{len(train_candidates)} | labels: {label_counts(train_candidates)}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
