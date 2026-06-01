from datetime import datetime
from typing import Dict, List

import numpy as np


class MetaFeatureBuilder:
    def __init__(self, categories: List[str], out_dim: int = 16, cat_bins: int = 8):
        self.out_dim = int(out_dim)
        self.cat_bins = int(cat_bins)
        uniq = sorted(set([str(c) for c in categories if str(c).strip()]))
        self.cat2idx = {c: i % self.cat_bins for i, c in enumerate(uniq)}

    @staticmethod
    def _safe_float(v, default: float = 0.0) -> float:
        try:
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    @staticmethod
    def _parse_publish_time(value: str) -> tuple[float, float, float, float]:
        if not value:
            return 0.0, 0.0, 0.0, 0.0
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", ""))
            hour = dt.hour / 24.0
            weekday = dt.weekday() / 7.0
            hour_sin = float(np.sin(2 * np.pi * hour))
            hour_cos = float(np.cos(2 * np.pi * hour))
            day_sin = float(np.sin(2 * np.pi * weekday))
            day_cos = float(np.cos(2 * np.pi * weekday))
            return hour_sin, hour_cos, day_sin, day_cos
        except Exception:
            return 0.0, 0.0, 0.0, 0.0

    def build(self, row: Dict) -> np.ndarray:
        duration = self._safe_float(row.get("duration", 0.0), 0.0)
        title = str(row.get("title", "") or "")
        tags = str(row.get("tags", "") or "")
        verified = self._safe_float(row.get("verified", 0), 0.0)

        tag_count = float(len([t for t in tags.split("|") if t.strip()]))
        hour_sin, hour_cos, day_sin, day_cos = self._parse_publish_time(str(row.get("publish_time", "") or ""))

        # 8 dense features
        dense = np.array(
            [
                duration / 300.0,
                min(len(title), 120) / 120.0,
                min(tag_count, 20.0) / 20.0,
                1.0 if verified > 0 else 0.0,
                hour_sin,
                hour_cos,
                day_sin,
                day_cos,
            ],
            dtype=np.float32,
        )

        # 8-dim category encoding
        cat_vec = np.zeros(self.cat_bins, dtype=np.float32)
        category = str(row.get("category", "unknown") or "unknown")
        idx = self.cat2idx.get(category)
        if idx is not None:
            cat_vec[idx] = 1.0

        feat = np.concatenate([dense, cat_vec], axis=0).astype(np.float32)
        if feat.shape[0] != self.out_dim:
            raise RuntimeError(f"meta feature dimension must be {self.out_dim}, got {feat.shape[0]}")
        return feat
