import math

import ctranslate2
import numpy as np
import torch
import torch.nn.functional as F
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

from pipeline.step_speech_features import configure_nvidia_dll_path


class AudioEncoder:
    def __init__(
        self,
        model_path: str = "base",
        device: str = "cuda",
        out_dim: int = 384,
        shared_model: WhisperModel | None = None,
        max_segments: int = 8,
    ):
        requested_device = "cuda" if str(device).lower().startswith("cuda") else "cpu"
        if requested_device == "cuda":
            configure_nvidia_dll_path()
        if shared_model is not None:
            self.model = shared_model
            self.backend_device = str(self.model.model.device)
        else:
            try:
                compute_type = "float16" if requested_device == "cuda" else "int8"
                self.model = WhisperModel(
                    model_path,
                    device=requested_device,
                    compute_type=compute_type,
                )
                self.backend_device = requested_device
            except RuntimeError:
                self.model = WhisperModel(model_path, device="cpu", compute_type="int8")
                self.backend_device = "cpu"
        self.out_dim = int(out_dim)
        self.max_segments = max(1, int(max_segments))

    def _audio_windows(self, audio: np.ndarray) -> list[np.ndarray]:
        chunk_samples = int(self.model.feature_extractor.n_samples)
        if audio.size <= chunk_samples:
            padded = np.pad(audio, (0, chunk_samples - audio.size))
            return [padded.astype(np.float32, copy=False)]

        total_chunks = int(math.ceil(audio.size / chunk_samples))
        count = min(self.max_segments, total_chunks)
        starts = np.linspace(0, audio.size - chunk_samples, num=count, dtype=np.int64)
        return [
            audio[int(start) : int(start) + chunk_samples].astype(np.float32, copy=False)
            for start in starts
        ]

    def encode(self, wav_path: str) -> torch.Tensor:
        audio = decode_audio(
            str(wav_path),
            sampling_rate=self.model.feature_extractor.sampling_rate,
        )
        if audio.size == 0:
            raise ValueError(f"Audio is empty: {wav_path}")

        windows = self._audio_windows(audio)
        features = np.stack(
            [self.model.feature_extractor(window)[..., :-1] for window in windows],
            axis=0,
        )
        encoded = self.model.encode(features)
        encoded = encoded.to_device(ctranslate2.Device.cpu)
        encoded_np = np.array(encoded).astype(np.float32, copy=False)
        if encoded_np.ndim != 3 or encoded_np.shape[0] != len(windows):
            raise RuntimeError(
                f"Unexpected Whisper encoder output shape {encoded_np.shape} for {wav_path}"
            )
        pooled = torch.from_numpy(encoded_np.mean(axis=(0, 1))).float()

        mapped = F.adaptive_avg_pool1d(pooled.view(1, 1, -1), self.out_dim).view(-1)
        mapped = F.normalize(mapped, p=2, dim=0).detach().cpu().float()
        if mapped.numel() != self.out_dim:
            raise RuntimeError(f"audio embedding dimension must be {self.out_dim}, got {mapped.numel()}")
        if not torch.isfinite(mapped).all() or float(mapped.norm()) == 0.0:
            raise RuntimeError(f"Invalid audio embedding generated for {wav_path}")
        return mapped
