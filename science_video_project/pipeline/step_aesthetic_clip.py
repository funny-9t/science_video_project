"""
CLIP-based prompt scoring for aesthetic quality evaluation.

Uses CLIP's multimodal embedding space to compare video frames against
carefully designed text prompts that capture different aesthetic dimensions.

Usage (standalone):
    scorer = CLIPAestheticScorer(device="cuda")
    scores = scorer.score_frames(frame_dir="outputs/frames/demo")
    print(scores)

Usage (integrated with pipeline):
    from pipeline.step_aesthetic_clip import CLIPAestheticScorer, DEFAULT_PROMPTS
    scorer = CLIPAestheticScorer(prompts=DEFAULT_PROMPTS)
    aes_scores = scorer.score_frames(frame_path)
    # aes_scores["aesthetic_score"]  →  aggregated score
    # aes_scores["dimensions"]       →  per-dimension scores
"""

from __future__ import annotations

from pathlib import Path

import clip
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
#  Default prompt sets — designed for science short video aesthetics
# ---------------------------------------------------------------------------

# Each entry is a dict: {"name": ..., "positive": ..., "negative": ...}
# Score = similarity(pos) - similarity(neg)  →  [-1, 1], higher = better

DEFAULT_PROMPTS = [
    # --- Visual clarity & resolution ---
    {
        "name": "clarity",
        "positive": "a sharp clear image with fine details",
        "negative": "a blurry fuzzy out-of-focus image",
    },
    {
        "name": "cleanliness",
        "positive": "a clean visual design with no distracting elements",
        "negative": "a cluttered messy noisy image",
    },
    # --- Composition ---
    {
        "name": "composition",
        "positive": "a well-composed professionally framed shot",
        "negative": "a poorly composed awkwardly framed shot",
    },
    # --- Visual appeal ---
    {
        "name": "appeal",
        "positive": "a visually appealing attractive beautiful image",
        "negative": "an ugly unpleasant难看 image",
    },
    # --- Lighting & color ---
    {
        "name": "lighting",
        "positive": "good lighting with balanced bright colors",
        "negative": "poor lighting with dark dull flat colors",
    },
    # --- Science-specific: diagram / text readability ---
    {
        "name": "text_readability",
        "positive": "clear readable text and labels in a scientific diagram",
        "negative": "unreadable messy text and labels in a scientific diagram",
    },
    {
        "name": "professional",
        "positive": "a professional high-quality scientific visualization",
        "negative": "an amateur low-quality screenshot",
    },
]


# Shorter subset for fast prototyping
QUICK_PROMPTS = [
    {
        "name": "clean_design",
        "positive": "a clean visual design with no distracting elements",
        "negative": "a cluttered messy noisy image",
    },
    {
        "name": "visual_quality",
        "positive": "a high-quality visually appealing image",
        "negative": "a low-quality blurry image",
    },
]


# ---------------------------------------------------------------------------
#  Scorer
# ---------------------------------------------------------------------------


class CLIPAestheticScorer:
    """Score video frames using CLIP text-prompt similarity.

    Args:
        prompts: List of prompt dicts, each with "name", "positive", "negative".
        clip_model_name: OpenAI CLIP model name or local path.
        device: "cuda" or "cpu".
        batch_size: Frames per batch.
    """

    def __init__(
        self,
        prompts: list[dict] | None = None,
        clip_model_name: str = "ViT-B/32",
        device: str = "cuda",
        batch_size: int = 32,
    ):
        self.prompts = prompts or DEFAULT_PROMPTS
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        print(f"[CLIPAestheticScorer] Loading CLIP model: {clip_model_name}")
        self.model, self.preprocess = clip.load(clip_model_name, device=self.device)
        self.model.eval()

        # Precompute text features for all prompts
        self._prompt_names: list[str] = []
        self._pos_text_feats: list[torch.Tensor] = []
        self._neg_text_feats: list[torch.Tensor] = []

        with torch.no_grad():
            for p in self.prompts:
                self._prompt_names.append(p["name"])
                pos = self._encode_text(p["positive"])
                neg = self._encode_text(p["negative"])
                self._pos_text_feats.append(pos)
                self._neg_text_feats.append(neg)

        print(f"[CLIPAestheticScorer] Loaded {len(self.prompts)} prompt dimensions")
        for n in self._prompt_names:
            print(f"  - {n}")

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _encode_text(self, text: str) -> torch.Tensor:
        tokens = clip.tokenize([text]).to(self.device)
        feat = self.model.encode_text(tokens)
        feat = F.normalize(feat, dim=-1)
        return feat  # (1, dim)

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(x)
        return F.normalize(feat, dim=-1)  # (1, dim)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def score_single_frame(self, image: Image.Image) -> dict[str, float]:
        """Score a single PIL image. Returns per-dimension scores."""
        img_feat = self._encode_image(image)  # (1, dim)

        scores = {}
        for name, pos, neg in zip(
            self._prompt_names, self._pos_text_feats, self._neg_text_feats
        ):
            sim_pos = (img_feat @ pos.T).item()
            sim_neg = (img_feat @ neg.T).item()
            # Contrastive score: how much more "positive" than "negative"
            scores[name] = float(sim_pos - sim_neg)  # range ≈ [-2, 2]

        return scores

    def score_frames(
        self,
        frame_dir: str | Path,
        aggregation: str = "mean",
    ) -> dict:
        """Score all frames in a directory.

        Args:
            frame_dir: Path containing *.jpg frame images.
            aggregation: "mean" | "median" | "min" | "max".

        Returns:
            dict with:
                - "aesthetic_score": scalar, aggregated across frames & dims
                - "dimensions": dict of per-dimension scores
                - "per_frame": list of per-frame scores (if verbose)
        """
        frame_paths = sorted(Path(frame_dir).glob("*.jpg"))
        if not frame_paths:
            raise FileNotFoundError(f"No .jpg frames found in {frame_dir}")

        # Collect per-frame dimension scores
        all_dims: list[dict[str, float]] = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            all_dims.append(self.score_single_frame(img))

        # Aggregate across frames for each dimension
        dim_scores: dict[str, float] = {}
        for name in self._prompt_names:
            values = [d[name] for d in all_dims]
            if aggregation == "mean":
                dim_scores[name] = float(torch.tensor(values).mean())
            elif aggregation == "median":
                dim_scores[name] = float(torch.tensor(values).median())
            elif aggregation == "min":
                dim_scores[name] = float(torch.tensor(values).min())
            elif aggregation == "max":
                dim_scores[name] = float(torch.tensor(values).max())
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

        # Overall aesthetic score = mean of all dimensions
        aesthetic_score = float(torch.tensor(list(dim_scores.values())).mean())

        return {
            "aesthetic_score": aesthetic_score,
            "dimensions": dim_scores,
            "num_frames": len(frame_paths),
        }

    def score_frames_batched(
        self,
        frame_dir: str | Path,
        aggregation: str = "mean",
    ) -> dict:
        """Batched version — loads frames in batches for GPU efficiency."""
        frame_paths = sorted(Path(frame_dir).glob("*.jpg"))
        if not frame_paths:
            raise FileNotFoundError(f"No .jpg frames found in {frame_dir}")

        # Preprocess all images
        all_imgs = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            all_imgs.append(self.preprocess(img))

        # Batched encoding
        all_feats: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(all_imgs), self.batch_size):
                batch = torch.stack(all_imgs[i: i + self.batch_size]).to(self.device)
                feats = self.model.encode_image(batch)
                all_feats.append(F.normalize(feats, dim=-1))

        img_feats = torch.cat(all_feats, dim=0)  # (N, dim)

        # Compute dimension scores
        pos_all = torch.cat(self._pos_text_feats, dim=0)  # (D, dim)
        neg_all = torch.cat(self._neg_text_feats, dim=0)  # (D, dim)

        sim_pos = img_feats @ pos_all.T  # (N, D)
        sim_neg = img_feats @ neg_all.T  # (N, D)
        delta = sim_pos - sim_neg  # (N, D)

        # Aggregate
        if aggregation == "mean":
            agg = delta.mean(dim=0)  # (D,)
        elif aggregation == "median":
            agg = delta.median(dim=0).values
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")

        dim_scores = {name: float(agg[i]) for i, name in enumerate(self._prompt_names)}
        aesthetic_score = float(agg.mean())

        return {
            "aesthetic_score": aesthetic_score,
            "dimensions": dim_scores,
            "num_frames": len(frame_paths),
        }


# ---------------------------------------------------------------------------
#  Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    scorer = CLIPAestheticScorer(prompts=QUICK_PROMPTS, device="cuda")

    frame_dir = sys.argv[1] if len(sys.argv) > 1 else r"D:\Projects\science_video_ranker_mvp\science_video_project\outputs\frames"
    result = scorer.score_frames_batched(frame_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
