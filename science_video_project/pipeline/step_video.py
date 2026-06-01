from pathlib import Path

import clip
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection


class VideoEncoder:
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.backend = "openai"
        self.preprocess = None
        self.processor = None
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        resolved = self._resolve_clip_model(model_name)
        if resolved["backend"] == "hf":
            self.backend = "hf"
            self.model, self.processor = self._load_hf_vision_model(resolved["path"])
        else:
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

    def encode_frames(self, frame_dir: str | Path) -> torch.Tensor:
        frame_paths = sorted(Path(frame_dir).glob("*.jpg"))
        if not frame_paths:
            raise RuntimeError(f"No frame images in {frame_dir}")

        feats = []
        with torch.no_grad():
            for p in frame_paths:
                img = Image.open(p).convert("RGB")
                if self.backend == "hf":
                    inputs = self.processor(images=img, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(self.device, dtype=self.model.dtype)
                    outputs = self.model(pixel_values=pixel_values)
                    feat = outputs.image_embeds
                else:
                    image = self.preprocess(img).unsqueeze(0).to(self.device)
                    feat = self.model.encode_image(image)

                feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                feats.append(feat.squeeze(0).detach().cpu())

        video_feat = torch.stack(feats, dim=0).mean(dim=0).float()
        if video_feat.numel() != 512:
            video_feat = F.adaptive_avg_pool1d(video_feat.view(1, 1, -1), 512).view(-1)
        if video_feat.numel() != 512:
            raise RuntimeError(f"video embedding dimension must be 512, got {video_feat.numel()}")
        return video_feat
