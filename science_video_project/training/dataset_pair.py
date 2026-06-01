import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pipeline.utils_io import load_pt


class PairwiseVideoDataset(Dataset):
    def __init__(self, samples: List[Dict], same_category: bool = True):
        self.samples = samples
        self.same_category = same_category

        self.pos = [s for s in samples if int(s["label"]) == 1]
        self.neg = [s for s in samples if int(s["label"]) == 0]
        if not self.pos or not self.neg:
            raise ValueError("PairwiseVideoDataset needs both positive and negative samples.")

        self.cat_map: Dict[str, Dict[str, List[Dict]]] = {}
        for s in samples:
            cat = str(s.get("category", "unknown"))
            if cat not in self.cat_map:
                self.cat_map[cat] = {"pos": [], "neg": []}
            key = "pos" if int(s["label"]) == 1 else "neg"
            self.cat_map[cat][key].append(s)

        self.valid_cats = [c for c, v in self.cat_map.items() if v["pos"] and v["neg"]]
        self.length = max(len(self.pos), len(self.neg))

    @classmethod
    def from_metadata(
        cls,
        metadata_df: pd.DataFrame,
        feature_dir: str | Path,
        same_category: bool = True,
    ) -> "PairwiseVideoDataset":
        feature_dir = Path(feature_dir)
        samples = []
        for _, row in metadata_df.iterrows():
            video_id = str(row["video_id"])
            pt_path = feature_dir / f"{video_id}.pt"
            if not pt_path.exists():
                continue
            sample = load_pt(pt_path)
            sample["label"] = int(row["label"])
            sample["category"] = str(row["category"])
            samples.append(sample)
        return cls(samples, same_category=same_category)

    def __len__(self) -> int:
        return self.length

    def _sample_pair(self) -> tuple[Dict, Dict]:
        if self.same_category and self.valid_cats:
            cat = random.choice(self.valid_cats)
            pos = random.choice(self.cat_map[cat]["pos"])
            neg = random.choice(self.cat_map[cat]["neg"])
            return pos, neg
        return random.choice(self.pos), random.choice(self.neg)

    @staticmethod
    def _to_tensor_dict(sample: Dict) -> Dict[str, torch.Tensor]:
        return {
            "text_feat": torch.tensor(sample["text_feat"], dtype=torch.float32),
            "video_feat": torch.tensor(sample["video_feat"], dtype=torch.float32),
            "audio_feat": torch.tensor(sample["audio_feat"], dtype=torch.float32),
            "meta_feat": torch.tensor(sample["meta_feat"], dtype=torch.float32),
            "aes_feat": torch.tensor(sample.get("aes_feat", np.zeros(7, dtype=np.float32)), dtype=torch.float32),
            "sci_hand_feat": torch.tensor(sample.get("sci_hand_feat", np.zeros(5, dtype=np.float32)), dtype=torch.float32),
            "frame_features": torch.tensor(sample.get("frame_features", np.zeros((0, 512), dtype=np.float32)), dtype=torch.float32),
            "engagement_target": torch.tensor([sample.get("engagement_target", 0.0)], dtype=torch.float32),
        }

    def __getitem__(self, idx: int) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        pos, neg = self._sample_pair()
        return self._to_tensor_dict(pos), self._to_tensor_dict(neg)
