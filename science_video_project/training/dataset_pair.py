import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pipeline.utils_io import load_pt


ATTRIBUTE_COLUMNS = (
    "science_info",
    "topic_importance",
    "science_access",
    "content_interest",
    "visual_quality",
    "audio_quality",
    "video_aesthetics",
)


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
        aesthetic_feature_backend: str = "legacy",
        aesthetic_prompt_feature_dir: str | Path | None = None,
        aesthetic_prompt_version: str | None = None,
        technical_visual_source: str = "clip",
        cover_temporal_feature_dir: str | Path | None = None,
    ) -> "PairwiseVideoDataset":
        feature_dir = Path(feature_dir)
        aesthetic_prompt_dir = (
            Path(aesthetic_prompt_feature_dir) if aesthetic_prompt_feature_dir else None
        )
        temporal_dir = Path(cover_temporal_feature_dir) if cover_temporal_feature_dir else None
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
            if aesthetic_feature_backend == "shared_clip":
                from pipeline.step_aesthetic_clip import SharedCLIPAestheticScorer

                shared_aesthetic = np.asarray(
                    sample.get("aes_shared_clip_feat", []), dtype=np.float32
                )
                version = sample.get("aes_shared_clip_version", "")
                if shared_aesthetic.shape != (7,) or not np.isfinite(shared_aesthetic).all():
                    raise ValueError(
                        f"Missing shared CLIP aesthetic feature for {video_id}: "
                        f"{shared_aesthetic.shape}"
                    )
                if version != SharedCLIPAestheticScorer.FEATURE_VERSION:
                    raise ValueError(
                        f"Stale shared aesthetic feature for {video_id}: {version!r}; "
                        f"expected {SharedCLIPAestheticScorer.FEATURE_VERSION!r}"
                    )
                sample["aes_feat"] = shared_aesthetic
            if aesthetic_prompt_dir is not None:
                if not aesthetic_prompt_version:
                    raise ValueError(
                        "aesthetic_prompt_version is required with "
                        "aesthetic_prompt_feature_dir"
                    )
                aesthetic_path = aesthetic_prompt_dir / f"{video_id}.pt"
                if not aesthetic_path.exists():
                    raise ValueError(f"Missing aesthetic prompt feature for {video_id}")
                aesthetic_payload = load_pt(aesthetic_path)
                aesthetic_feature = np.asarray(
                    aesthetic_payload.get("feature", []), dtype=np.float32
                )
                metadata = aesthetic_payload.get("metadata", {})
                if aesthetic_feature.shape != (7,) or not np.isfinite(aesthetic_feature).all():
                    raise ValueError(
                        f"Invalid aesthetic prompt feature for {video_id}: "
                        f"{aesthetic_feature.shape}"
                    )
                if metadata.get("prompt_version") != aesthetic_prompt_version:
                    raise ValueError(
                        f"Stale aesthetic prompt feature for {video_id}: "
                        f"{metadata.get('prompt_version')!r}; expected "
                        f"{aesthetic_prompt_version!r}"
                    )
                sample["aes_feat"] = aesthetic_feature
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
            if technical_visual_source in {"cover", "clip_cover"}:
                from pipeline.step_cover_technical import COVERTechnicalFeatureExtractor

                technical = np.asarray(
                    sample.get("cover_technical_feat", []), dtype=np.float32
                )
                version = sample.get("cover_technical_feature_version", "")
                if technical.shape != (COVERTechnicalFeatureExtractor.FEATURE_DIM,):
                    raise ValueError(
                        f"Missing COVER technical feature for {video_id}: {technical.shape}"
                    )
                if not np.isfinite(technical).all():
                    raise ValueError(f"Non-finite COVER technical feature for {video_id}")
                if version != COVERTechnicalFeatureExtractor.FEATURE_VERSION:
                    raise ValueError(
                        f"Stale COVER technical feature for {video_id}: {version!r}; "
                        f"expected {COVERTechnicalFeatureExtractor.FEATURE_VERSION!r}"
                    )
                if temporal_dir is not None:
                    temporal_path = temporal_dir / f"{video_id}.pt"
                    if not temporal_path.exists():
                        raise ValueError(f"Missing COVER temporal feature for {video_id}")
                    temporal_payload = load_pt(temporal_path)
                    temporal = np.asarray(
                        temporal_payload.get("temporal_feature", []), dtype=np.float32
                    )
                    temporal_version = temporal_payload.get("feature_version", "")
                    if temporal.ndim != 2 or temporal.shape[0] < 1 or temporal.shape[1] != 768:
                        raise ValueError(
                            f"Invalid COVER temporal feature for {video_id}: {temporal.shape}"
                        )
                    if not np.isfinite(temporal).all():
                        raise ValueError(f"Non-finite COVER temporal feature for {video_id}")
                    if temporal_version != COVERTechnicalFeatureExtractor.TEMPORAL_FEATURE_VERSION:
                        raise ValueError(
                            f"Stale COVER temporal feature for {video_id}: {temporal_version!r}; "
                            f"expected {COVERTechnicalFeatureExtractor.TEMPORAL_FEATURE_VERSION!r}"
                        )
                    sample["cover_temporal_feat"] = temporal
            sample["label"] = int(row["label"])
            sample["category"] = str(row["category"])
            for key in ["sci_target", "tech_target", "aes_target"]:
                val = row.get(key, -1)
                sample[key] = float(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else -1.0
            attribute_targets = []
            for key in ATTRIBUTE_COLUMNS:
                value = pd.to_numeric(row.get(key, np.nan), errors="coerce")
                attribute_targets.append(float(value) / 5.0 if np.isfinite(value) else -1.0)
            sample["attribute_targets"] = np.asarray(attribute_targets, dtype=np.float32)
            sample["overall_quality_target"] = (
                float(np.mean(attribute_targets))
                if all(value >= 0 for value in attribute_targets)
                else -1.0
            )
            for source_key, target_key in (
                ("visual_quality", "visual_target"),
                ("audio_quality", "audio_target"),
            ):
                value = pd.to_numeric(row.get(source_key, np.nan), errors="coerce")
                sample[target_key] = float(value) / 5.0 if np.isfinite(value) else -1.0
            sample["label_target"] = float(row["label"])
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
            "cover_technical_feat": torch.tensor(
                PairwiseVideoDataset._fixed_vector(sample, "cover_technical_feat", 768),
                dtype=torch.float32,
            ),
            "cover_temporal_feat": torch.tensor(
                sample.get("cover_temporal_feat", np.zeros((0, 768), dtype=np.float32)),
                dtype=torch.float32,
            ),
            "frame_features": torch.tensor(
                sample.get("frame_features", np.zeros((0, 512), dtype=np.float32)),
                dtype=torch.float32,
            ),
            "engagement_target": torch.tensor([sample.get("engagement_target", 0.0)], dtype=torch.float32),
            "sci_target": torch.tensor([sample.get("sci_target", -1.0)], dtype=torch.float32),
            "tech_target": torch.tensor([sample.get("tech_target", -1.0)], dtype=torch.float32),
            "aes_target": torch.tensor([sample.get("aes_target", -1.0)], dtype=torch.float32),
            "visual_target": torch.tensor([sample.get("visual_target", -1.0)], dtype=torch.float32),
            "audio_target": torch.tensor([sample.get("audio_target", -1.0)], dtype=torch.float32),
            "label_target": torch.tensor([sample.get("label_target", -1.0)], dtype=torch.float32),
            "attribute_targets": torch.tensor(
                sample.get("attribute_targets", np.full(7, -1.0, dtype=np.float32)),
                dtype=torch.float32,
            ),
            "overall_quality_target": torch.tensor(
                [sample.get("overall_quality_target", -1.0)], dtype=torch.float32
            ),
        }

    def __getitem__(self, idx: int) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if self.fixed_pairs:
            pos, neg = self.fixed_pairs[idx % len(self.fixed_pairs)]
        else:
            pos, neg = self._sample_pair()
        return self._to_tensor_dict(pos), self._to_tensor_dict(neg)
