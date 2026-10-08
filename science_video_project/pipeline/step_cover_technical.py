from __future__ import annotations

import hashlib
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml


class COVERTechnicalFeatureExtractor:
    """Frozen COVER technical view/backbone without the other COVER branches."""

    FEATURE_VERSION = "cover_technical_swin3d_tiny_grpb_ytugc_v1"
    TEMPORAL_FEATURE_VERSION = "cover_technical_swin3d_tiny_grpb_ytugc_temporal_v1"
    FEATURE_DIM = 768

    def __init__(
        self,
        cover_root: str | Path,
        device: str = "cuda",
        config_name: str = "cover.yml",
        use_amp: bool = False,
    ) -> None:
        self.cover_root = Path(cover_root).resolve()
        if not self.cover_root.exists():
            raise FileNotFoundError(f"COVER repository not found: {self.cover_root}")
        if str(self.cover_root) not in sys.path:
            sys.path.insert(0, str(self.cover_root))

        from cover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
        from cover.models.swin_backbone import SwinTransformer3D

        with (self.cover_root / config_name).open("r", encoding="utf-8") as handle:
            options = yaml.safe_load(handle)
        technical_options = options["data"]["val-ytugc"]["args"]["sample_types"]["technical"]
        backbone_options = options["model"]["args"]["backbone"]["technical"]
        if backbone_options.get("type") != "swin_tiny_grpb":
            raise ValueError(
                "This extractor expects COVER's swin_tiny_grpb technical backbone, "
                f"got {backbone_options.get('type')!r}."
            )

        self.device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.sample_options = {"technical": dict(technical_options)}
        self.sampler = UnifiedFrameSampler(
            technical_options["clip_len"] // technical_options["t_frag"],
            technical_options["t_frag"],
            technical_options["frame_interval"],
            technical_options["num_clips"],
        )
        self._decompose = spatial_temporal_view_decomposition

        # COVER maps swin_tiny_grpb to the default GRPB-enabled SwinTransformer3D.
        self.backbone = SwinTransformer3D().to(self.device)
        checkpoint_path = Path(options["test_load_path"])
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.cover_root / checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        prefix = "technical_backbone."
        technical_state = {
            key[len(prefix):]: value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith(prefix)
        }
        if not technical_state:
            raise RuntimeError(f"No technical backbone weights in {checkpoint_path}")
        self.backbone.load_state_dict(technical_state, strict=True)
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        self.checkpoint_path = checkpoint_path
        self.mean = torch.tensor([123.675, 116.28, 103.53])
        self.std = torch.tensor([58.395, 57.12, 57.375])

    @property
    def sampling_config(self) -> dict[str, int]:
        fields = (
            "fragments_h", "fragments_w", "fsize_h", "fsize_w", "aligned",
            "clip_len", "t_frag", "frame_interval", "num_clips",
        )
        return {name: int(self.sample_options["technical"][name]) for name in fields}

    def _prepare_view(self, video_path: str | Path) -> torch.Tensor:
        # COVER's test sampler still draws fragment offsets, so make them stable per video.
        seed = int(hashlib.sha256(Path(video_path).stem.encode("utf-8")).hexdigest()[:8], 16)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            views, _ = self._decompose(
                str(video_path), self.sample_options, {"technical": self.sampler}
            )
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

        value = views["technical"]
        normalized = (
            ((value.permute(1, 2, 3, 0) - self.mean) / self.std)
            .permute(3, 0, 1, 2)
            .unsqueeze(0)
        )
        return normalized.to(self.device)

    def extract(self, video_path: str | Path) -> torch.Tensor:
        view = self._prepare_view(video_path)
        amp = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), amp:
            feature_map = self.backbone(view)
            feature = feature_map.mean(dim=(2, 3, 4)).squeeze(0).float().cpu()
        if feature.shape != (self.FEATURE_DIM,) or not torch.isfinite(feature).all():
            raise RuntimeError(
                f"Invalid COVER technical feature for {video_path}: {tuple(feature.shape)}"
            )
        return feature

    def extract_temporal(self, video_path: str | Path) -> torch.Tensor:
        """Return the spatially pooled Swin feature sequence as [T, D]."""
        view = self._prepare_view(video_path)
        amp = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), amp:
            feature_map = self.backbone(view)
            feature = feature_map.mean(dim=(3, 4)).squeeze(0).transpose(0, 1)
            feature = feature.float().cpu()
        if (
            feature.ndim != 2
            or feature.shape[0] < 1
            or feature.shape[1] != self.FEATURE_DIM
            or not torch.isfinite(feature).all()
        ):
            raise RuntimeError(
                f"Invalid COVER temporal feature for {video_path}: {tuple(feature.shape)}"
            )
        return feature
