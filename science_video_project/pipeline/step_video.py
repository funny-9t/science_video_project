from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection

try:
    import clip
except ImportError:
    clip = None


class VideoEncoder:
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda", batch_size: int = 32):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.backend = "openai"
        self.preprocess = None
        self.processor = None
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.batch_size = max(1, int(batch_size))

        resolved = self._resolve_clip_model(model_name)
        if resolved["backend"] == "hf":
            self.backend = "hf"
            self.model, self.processor = self._load_hf_vision_model(resolved["path"])
        else:
            if clip is None:
                raise RuntimeError(
                    "The openai-clip package is required for an OpenAI CLIP checkpoint. "
                    "Use the local Hugging Face CLIP path for main_v2."
                )
            self.model, self.preprocess = clip.load(resolved["path"], device=self.device)
        self.model.eval()
        self.model = self.model.to(self.device, dtype=self.dtype if self.device.type == "cuda" else torch.float32)

    @staticmethod
    def _resolve_clip_model(model_name: str) -> dict:
        path = Path(model_name)
        if path.exists():
            if path.is_file():
                return {"backend": "openai", "path": str(path)}
            if list(path.glob("*.safetensors")) or (path / "model.safetensors").exists():
                return {"backend": "hf", "path": str(path)}
            candidates = list(path.glob("*.pt"))
            if candidates:
                return {"backend": "openai", "path": str(candidates[0])}
            raise RuntimeError(f"No CLIP checkpoint found under {path}")
        return {"backend": "openai", "path": model_name}

    @staticmethod
    def _load_hf_vision_model(model_path: str) -> tuple[torch.nn.Module, CLIPImageProcessor]:
        path = Path(model_path)
        config = CLIPVisionConfig.from_pretrained(path, local_files_only=True)
        model = CLIPVisionModelWithProjection(config)

        state_dict: dict[str, torch.Tensor] = {}
        safetensor_path = path / "model.safetensors"
        if not safetensor_path.exists():
            candidates = list(path.glob("*.safetensors"))
            if candidates:
                safetensor_path = candidates[0]
        if not safetensor_path.exists():
            raise FileNotFoundError(f"No safetensors file found under {path}")

        with safe_open(str(safetensor_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("vision_model.") or key.startswith("visual_projection."):
                    # Skip non-persistent buffers (e.g. position_ids) that the model
                    # creates internally and are not part of its own state_dict.
                    if key.endswith(".position_ids"):
                        continue
                    state_dict[key] = handle.get_tensor(key)

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            missing_preview = ", ".join(missing_keys[:5])
            raise RuntimeError(f"Missing CLIP vision weights: {missing_preview}")
        if unexpected_keys:
            unexpected_preview = ", ".join(unexpected_keys[:5])
            raise RuntimeError(f"Unexpected CLIP vision weights: {unexpected_preview}")

        processor = CLIPImageProcessor.from_pretrained(path, local_files_only=True)
        return model, processor

    @staticmethod
    def _global_uniform_paths(frame_paths: list[Path], max_frames: int | None) -> list[Path]:
        if max_frames is None or max_frames <= 0 or len(frame_paths) <= max_frames:
            return frame_paths
        indices = torch.linspace(0, len(frame_paths) - 1, steps=max_frames).round().long().tolist()
        return [frame_paths[index] for index in indices]

    def encode_frame_features(
        self,
        frame_dir: str | Path,
        max_frames: int | None = None,
    ) -> torch.Tensor:
        """Return one native CLIP embedding per sampled frame.

        CLIP embedding channels are not spatially ordered, so adaptive pooling
        across the channel axis is not a valid dimensionality projection.  The
        native sequence is kept for temporal modelling and COVER-style fusion.
        """
        frame_paths = sorted(Path(frame_dir).glob("*.jpg"))
        if not frame_paths:
            raise RuntimeError(f"No frame images in {frame_dir}")
        frame_paths = self._global_uniform_paths(frame_paths, max_frames)

        feats = []
        with torch.no_grad():
            for start in range(0, len(frame_paths), self.batch_size):
                images = []
                for path in frame_paths[start : start + self.batch_size]:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                if self.backend == "hf":
                    inputs = self.processor(images=images, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(self.device, dtype=self.model.dtype)
                    outputs = self.model(pixel_values=pixel_values)
                    feat = outputs.image_embeds
                else:
                    batch = torch.stack([self.preprocess(image) for image in images]).to(self.device)
                    feat = self.model.encode_image(batch)

                feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                feats.append(feat.detach().cpu())

        return torch.cat(feats, dim=0).float()

    def encode_native_video(
        self,
        frame_dir: str | Path,
        max_frames: int | None = None,
    ) -> torch.Tensor:
        """Mean-pool frames while preserving CLIP's native embedding space."""
        return self.encode_frame_features(frame_dir, max_frames=max_frames).mean(dim=0)

    @staticmethod
    def legacy_project(video_feat: torch.Tensor, output_dim: int = 512) -> torch.Tensor:
        """Reproduce the historical 512-d feature for old checkpoints only."""
        if video_feat.numel() == output_dim:
            return video_feat.float()
        return F.adaptive_avg_pool1d(video_feat.view(1, 1, -1), output_dim).view(-1).float()

    def encode_frames(self, frame_dir: str | Path) -> torch.Tensor:
        """Backward-compatible legacy feature used by existing checkpoints."""
        return self.legacy_project(self.encode_native_video(frame_dir), output_dim=512)
