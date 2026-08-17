import json
import random
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_videos(video_dir: str | Path) -> List[Path]:
    base = Path(video_dir)
    exts = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".webm"}
    if not base.exists():
        return []
    return sorted([p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def get_video_id(video_path: str | Path) -> str:
    stem = Path(video_path).stem
    match = re.search(r"(\d{6,})$", stem)
    return match.group(1) if match else stem


def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"metadata file not found: {path}")
    df = pd.read_csv(path)
    required = {"video_id", "label", "category", "title", "tags", "duration"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"parsed_metadata.csv missing columns: {sorted(missing)}")

    df["video_id"] = df["video_id"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df["category"] = df["category"].fillna("unknown").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["tags"] = df["tags"].fillna("").astype(str)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)
    if "verified" not in df.columns:
        df["verified"] = 0
    df["verified"] = pd.to_numeric(df["verified"], errors="coerce").fillna(0).astype(int)
    if "publish_time" not in df.columns:
        df["publish_time"] = ""
    df["publish_time"] = df["publish_time"].fillna("").astype(str)

    label_counts = df.groupby("video_id")["label"].nunique()
    conflicting_ids = set(label_counts[label_counts > 1].index.astype(str))
    if conflicting_ids:
        warnings.warn(
            f"Dropping {len(conflicting_ids)} video IDs with conflicting labels.",
            RuntimeWarning,
            stacklevel=2,
        )
        df = df[~df["video_id"].isin(conflicting_ids)]

    duplicate_count = int(df.duplicated("video_id", keep="last").sum())
    if duplicate_count:
        warnings.warn(
            f"Deduplicating {duplicate_count} repeated metadata rows by video_id.",
            RuntimeWarning,
            stacklevel=2,
        )
        df = df.drop_duplicates("video_id", keep="last")
    return df.reset_index(drop=True)


def save_pt(obj: Any, path: str | Path) -> None:
    torch.save(obj, Path(path))


def load_pt(path: str | Path) -> Any:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
