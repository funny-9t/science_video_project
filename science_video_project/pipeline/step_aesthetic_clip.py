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

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

SHARED_AESTHETIC_VERSION = "shared_clip_vitl14_prompts_v1"
A2_SHARED_AESTHETIC_VERSION = "shared_clip_vitl14_a2_domain_semantic_v1"


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


AESTHETIC_PROMPTS_A2 = [
    {
        "name": "composition",
        "positive": "a video frame with refined and well-balanced composition",
        "negative": "a video frame with poor and unbalanced composition",
    },
    {
        "name": "color_harmony",
        "positive": "a video frame with harmonious and visually pleasing colors",
        "negative": "a video frame with disharmonious and visually unpleasant colors",
    },
    {
        "name": "lighting",
        "positive": "a video frame with balanced and aesthetically pleasing lighting",
        "negative": "a video frame with unbalanced and aesthetically poor lighting",
    },
    {
        "name": "visual_hierarchy",
        "positive": "a video frame with a clear and well-organized visual hierarchy",
        "negative": "a video frame with a confusing and disorganized visual hierarchy",
    },
    {
        "name": "visual_presentation",
        "positive": "a video frame with well-arranged and coherent visual presentation",
        "negative": "a video frame with poorly arranged and confusing visual presentation",
    },
    {
        "name": "visual_refinement",
        "positive": "a polished video frame with refined visual details",
        "negative": "a rough video frame with poorly handled visual details",
    },
    {
        "name": "visual_appeal",
        "positive": "a visually appealing and aesthetically pleasing video frame",
        "negative": "a visually unappealing and aesthetically poor video frame",
    },
]


AESTHETIC_PROMPT_SETS = {
    "a1_original": DEFAULT_PROMPTS,
    "a2_domain_semantic": AESTHETIC_PROMPTS_A2,
}

AESTHETIC_FEATURE_VERSIONS = {
    "a1_original": SHARED_AESTHETIC_VERSION,
    "a2_domain_semantic": A2_SHARED_AESTHETIC_VERSION,
}


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

        try:
            import clip as openai_clip
        except ImportError as exc:
            raise RuntimeError(
                "The legacy aesthetic scorer requires the openai-clip package. "
                "Use SharedCLIPAestheticScorer with the local ViT-L/14 model."
            ) from exc
        self._clip = openai_clip
        print(f"[CLIPAestheticScorer] Loading CLIP model: {clip_model_name}")
        self.model, self.preprocess = self._clip.load(clip_model_name, device=self.device)
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
        tokens = self._clip.tokenize([text]).to(self.device)
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
        """Encode frame images in batches with the legacy aesthetic CLIP."""
        frame_paths = sorted(Path(frame_dir).glob("*.jpg"))
        if not frame_paths:
            raise FileNotFoundError(f"No .jpg frames found in {frame_dir}")

        all_imgs = []
        for fp in frame_paths:
            with Image.open(fp) as image:
                all_imgs.append(self.preprocess(image.convert("RGB")))

        all_feats: list[torch.Tensor] = []
        with torch.inference_mode():
            for i in range(0, len(all_imgs), self.batch_size):
                batch = torch.stack(all_imgs[i: i + self.batch_size]).to(self.device)
                feats = self.model.encode_image(batch)
                all_feats.append(F.normalize(feats, dim=-1))

        img_feats = torch.cat(all_feats, dim=0)
        pos_all = torch.cat(self._pos_text_feats, dim=0)
        neg_all = torch.cat(self._neg_text_feats, dim=0)
        delta = img_feats @ pos_all.T - img_feats @ neg_all.T
        if aggregation == "mean":
            aggregated = delta.mean(dim=0)
        elif aggregation == "median":
            aggregated = delta.median(dim=0).values
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")

        dimensions = {
            name: float(aggregated[index])
            for index, name in enumerate(self._prompt_names)
        }
        return {
            "aesthetic_score": float(aggregated.mean()),
            "dimensions": dimensions,
            "num_frames": len(frame_paths),
        }


class SharedCLIPAestheticScorer:
    """Prompt scorer operating directly on cached ViT-L/14 frame embeddings."""

    FEATURE_VERSION = SHARED_AESTHETIC_VERSION

    def __init__(
        self,
        model_path: str | Path,
        prompts: list[dict] | None = None,
        device: str = "cuda",
        prompt_version: str = "a1_original",
    ):
        if prompt_version not in AESTHETIC_PROMPT_SETS:
            raise ValueError(
                f"Unknown aesthetic prompt version: {prompt_version!r}; "
                f"expected one of {sorted(AESTHETIC_PROMPT_SETS)}"
            )
        self.prompt_version = prompt_version
        self.prompts = prompts or AESTHETIC_PROMPT_SETS[prompt_version]
        self.feature_version = AESTHETIC_FEATURE_VERSIONS[prompt_version]
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model_path = str(model_path)
        tokenizer = CLIPTokenizerFast.from_pretrained(model_path, local_files_only=True)
        text_model = CLIPTextModelWithProjection.from_pretrained(
            model_path,
            local_files_only=True,
        ).to(self.device)
        text_model.eval()
        if self.device.type == "cuda":
            text_model = text_model.to(dtype=self.dtype)

        positive = [item["positive"] for item in self.prompts]
        negative = [item["negative"] for item in self.prompts]
        tokens = tokenizer(
            positive + negative,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        with torch.inference_mode():
            text_features = text_model(**tokens).text_embeds
            text_features = F.normalize(text_features, dim=-1)
        split = len(self.prompts)
        self.positive_features = text_features[:split].detach()
        self.negative_features = text_features[split:].detach()
        self.prompt_names = [item["name"] for item in self.prompts]
        del text_model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def score_frame_features(
        self,
        frame_features: torch.Tensor,
        aggregation: str = "mean",
    ) -> dict:
        features = torch.as_tensor(frame_features, dtype=self.dtype, device=self.device)
        if features.ndim != 2 or features.shape[-1] != self.positive_features.shape[-1]:
            raise ValueError(
                f"Expected frame features shaped (N, {self.positive_features.shape[-1]}), "
                f"got {tuple(features.shape)}"
            )
        features = F.normalize(features, dim=-1)
        delta = features @ self.positive_features.T - features @ self.negative_features.T
        if aggregation == "mean":
            aggregated = delta.mean(dim=0)
        elif aggregation == "median":
            aggregated = delta.median(dim=0).values
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        aggregated = aggregated.detach().cpu().float()
        dimensions = {
            name: float(aggregated[index])
            for index, name in enumerate(self.prompt_names)
        }
        return {
            "aesthetic_score": float(aggregated.mean()),
            "dimensions": dimensions,
            "num_frames": int(features.shape[0]),
            "feature_version": self.feature_version,
            "prompt_version": self.prompt_version,
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
