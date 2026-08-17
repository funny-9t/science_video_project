from __future__ import annotations

import sys
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
import yaml


class COVERFeatureExtractor:
    """Frozen adapter around the official COVER implementation."""

    FEATURE_VERSION = "cover_official_ytugc_deterministic_v2"
    BRANCH_ORDER = ("semantic", "technical", "aesthetic")

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
        from cover.models import COVER

        self._decompose = spatial_temporal_view_decomposition
        config_path = self.cover_root / config_name
        with config_path.open("r", encoding="utf-8") as handle:
            options = yaml.safe_load(handle)

        self.device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.sample_options = options["data"]["val-ytugc"]["args"]["sample_types"]
        self.samplers = {
            name: UnifiedFrameSampler(
                sample["clip_len"] // sample["t_frag"],
                sample["t_frag"],
                sample["frame_interval"],
                sample["num_clips"],
            )
            for name, sample in self.sample_options.items()
        }

        self.model = COVER(**options["model"]["args"]).to(self.device)
        checkpoint_path = Path(options["test_load_path"])
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.cover_root / checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        self.model.eval()

        self.mean = torch.tensor([123.675, 116.28, 103.53])
        self.std = torch.tensor([58.395, 57.12, 57.375])
        self.mean_clip = torch.tensor([122.77, 116.75, 104.09])
        self.std_clip = torch.tensor([68.50, 66.63, 70.32])

    def _prepare_views(self, video_path: str | Path) -> dict[str, torch.Tensor]:
        seed = int(
            hashlib.sha256(Path(video_path).stem.encode("utf-8")).hexdigest()[:8], 16
        )
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            views, _ = self._decompose(
                str(video_path), self.sample_options, self.samplers
            )
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        prepared = {}
        for name in self.BRANCH_ORDER:
            value = views[name]
            num_clips = self.sample_options[name].get("num_clips", 1)
            mean, std = (
                (self.mean_clip, self.std_clip)
                if name == "semantic"
                else (self.mean, self.std)
            )
            prepared[name] = (
                ((value.permute(1, 2, 3, 0) - mean) / std)
                .permute(3, 0, 1, 2)
                .reshape(value.shape[0], num_clips, -1, *value.shape[2:])
                .transpose(0, 1)
                .to(self.device)
            )
        return prepared

    def score(self, video_path: str | Path) -> torch.Tensor:
        """Return [semantic, technical, aesthetic] branch scores."""
        views = self._prepare_views(video_path)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16 if self.device.type == "cuda" else torch.bfloat16,
            enabled=self.use_amp,
        ):
            outputs = self.model(views, reduce_scores=False)
        scores = torch.tensor(
            [float(output.mean().item()) for output in outputs], dtype=torch.float32
        )
        if scores.numel() != 3 or not torch.isfinite(scores).all():
            raise RuntimeError(f"Invalid COVER scores for {video_path}: {scores.tolist()}")
        return scores
