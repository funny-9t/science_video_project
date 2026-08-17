import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pipeline.utils_io import load_pt


class PairwiseVideoDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        same_category: bool = True,
        pos_aug_noise: float = 0.05,
        deterministic_pairs: bool = False,
        use_native_clip: bool = False,
    ):
        self.samples = samples
        self.same_category = same_category
        self.pos_aug_noise = pos_aug_noise
        self.deterministic_pairs = deterministic_pairs
        self.use_native_clip = use_native_clip

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
        self.pos_oversample = max(1, len(self.neg) // max(len(self.pos), 1))
        self.fixed_pairs = self._build_fixed_pairs() if deterministic_pairs else []
        self.length = (
            len(self.fixed_pairs)
            if self.fixed_pairs
            else max(len(self.pos) * self.pos_oversample, len(self.neg))
        )

    @classmethod
    def from_metadata(
        cls,
        metadata_df: pd.DataFrame,
        feature_dir: str | Path,
        same_category: bool = True,
        pos_aug_noise: float = 0.05,
        deterministic_pairs: bool = False,
        use_native_clip: bool = False,
        use_cover_features: bool = False,
    ) -> "PairwiseVideoDataset":
        feature_dir = Path(feature_dir)
        samples = []
        for _, row in metadata_df.iterrows():
            video_id = str(row["video_id"])
            pt_path = feature_dir / f"{video_id}.pt"
            if not pt_path.exists():
                continue
            sample = load_pt(pt_path)
            if use_native_clip:
                native = np.asarray(sample.get("clip_video_feat", []), dtype=np.float32)
                if native.shape != (768,):
                    raise ValueError(f"Missing native CLIP feature for {video_id}: {native.shape}")
                sample["video_feat"] = native
            if use_cover_features:
                from pipeline.step_cover import COVERFeatureExtractor

                cover = np.asarray(sample.get("cover_feat", []), dtype=np.float32)
                version = sample.get("cover_feature_version", "")
                if cover.shape != (3,) or not np.isfinite(cover).all():
                    raise ValueError(f"Missing COVER feature for {video_id}: {cover.shape}")
                if version != COVERFeatureExtractor.FEATURE_VERSION:
                    raise ValueError(
                        f"Stale COVER feature for {video_id}: {version!r}; "
                        f"expected {COVERFeatureExtractor.FEATURE_VERSION!r}"
                    )
            sample["label"] = int(row["label"])
            sample["category"] = str(row["category"])
            for key in ["sci_target", "tech_target", "aes_target"]:
                val = row.get(key, -1)
                sample[key] = float(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else -1.0
            samples.append(sample)
        return cls(
            samples,
            same_category=same_category,
            pos_aug_noise=pos_aug_noise,
            deterministic_pairs=deterministic_pairs,
            use_native_clip=use_native_clip,
        )

    def __len__(self) -> int:
        return self.length

    def _build_fixed_pairs(self) -> List[tuple[Dict, Dict]]:
        pairs: List[tuple[Dict, Dict]] = []
        if self.same_category and self.valid_cats:
            for cat in sorted(self.valid_cats):
                for pos in self.cat_map[cat]["pos"]:
                    for neg in self.cat_map[cat]["neg"]:
                        pairs.append((pos, neg))
        else:
            for pos in self.pos:
                for neg in self.neg:
                    pairs.append((pos, neg))
        return pairs

    def _sample_pair(self) -> tuple[Dict, Dict]:
        if self.same_category and self.valid_cats:
            cat = random.choice(self.valid_cats)
            pos = random.choice(self.cat_map[cat]["pos"])
            neg = random.choice(self.cat_map[cat]["neg"])
            return pos, neg
        return random.choice(self.pos), random.choice(self.neg)

    @staticmethod
    def _fixed_vector(sample: Dict, key: str, dim: int) -> np.ndarray:
        value = np.asarray(sample.get(key, np.zeros(dim)), dtype=np.float32).reshape(-1)
        result = np.zeros(dim, dtype=np.float32)
        result[: min(dim, value.size)] = value[:dim]
        return result

    @staticmethod
    def _to_tensor_dict(sample: Dict) -> Dict:
        return {
            "video_id": str(sample.get("video_id", "")),
            "text_feat": torch.tensor(sample["text_feat"], dtype=torch.float32),
            "video_feat": torch.tensor(sample["video_feat"], dtype=torch.float32),
            "clip_video_feat": torch.tensor(
                PairwiseVideoDataset._fixed_vector(sample, "clip_video_feat", 768), dtype=torch.float32
            ),
            "audio_feat": torch.tensor(sample["audio_feat"], dtype=torch.float32),
            "meta_feat": torch.tensor(sample["meta_feat"], dtype=torch.float32),
            "aes_feat": torch.tensor(sample.get("aes_feat", np.zeros(7, dtype=np.float32)), dtype=torch.float32),
            "dnsmos_feat": torch.tensor(sample.get("dnsmos_feat", np.zeros(3, dtype=np.float32)), dtype=torch.float32),
            "wpm": torch.tensor([sample.get("wpm", 0.0)], dtype=torch.float32),
            "speech_rhythm_feat": torch.tensor(
                sample.get("speech_rhythm_feat", np.zeros(6, dtype=np.float32)),
                dtype=torch.float32,
            ),
            "sci_hand_feat": torch.tensor(sample.get("sci_hand_feat", np.zeros(5, dtype=np.float32)), dtype=torch.float32),
            "llm_knowledge_feat": torch.tensor(
                PairwiseVideoDataset._fixed_vector(sample, "llm_knowledge_feat", 4), dtype=torch.float32
            ),
            "llm_analysis_feat": torch.tensor(
                PairwiseVideoDataset._fixed_vector(sample, "llm_analysis_feat", 768), dtype=torch.float32
            ),
            "llm_analysis_only_feat": torch.tensor(
                PairwiseVideoDataset._fixed_vector(sample, "llm_analysis_only_feat", 768), dtype=torch.float32
            ),
            "llm_reasoning_analysis_feat": torch.tensor(
                PairwiseVideoDataset._fixed_vector(sample, "llm_reasoning_analysis_feat", 768), dtype=torch.float32
            ),
            "cover_feat": torch.tensor(
                PairwiseVideoDataset._fixed_vector(sample, "cover_feat", 3), dtype=torch.float32
            ),
            "frame_features": torch.tensor(
                sample.get("frame_features", np.zeros((0, 512), dtype=np.float32)),
                dtype=torch.float32,
            ),
            "engagement_target": torch.tensor([sample.get("engagement_target", 0.0)], dtype=torch.float32),
            "sci_target": torch.tensor([sample.get("sci_target", -1.0)], dtype=torch.float32),
            "tech_target": torch.tensor([sample.get("tech_target", -1.0)], dtype=torch.float32),
            "aes_target": torch.tensor([sample.get("aes_target", -1.0)], dtype=torch.float32),
        }

    def __getitem__(self, idx: int) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if self.fixed_pairs:
            pos, neg = self.fixed_pairs[idx % len(self.fixed_pairs)]
        else:
            pos, neg = self._sample_pair()
        return self._to_tensor_dict(pos), self._to_tensor_dict(neg)
