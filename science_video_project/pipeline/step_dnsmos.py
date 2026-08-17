"""Efficient DNSMOS P.835 audio-quality features."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import torch


class DNSMOSScorer:
    """Return normalized [overall, signal, background] DNSMOS scores."""

    SAMPLE_RATE = 16000
    INPUT_SECONDS = 9.01

    def __init__(
        self,
        device: str = "cpu",
        max_segments: int = 20,
        batch_size: int = 32,
        num_threads: int = 1,
    ) -> None:
        self.device = device
        self.max_segments = max(1, int(max_segments))
        self.batch_size = max(1, int(batch_size))
        self.num_threads = max(1, int(num_threads))
        self._session = None

    def _lazy_init(self) -> None:
        if self._session is not None:
            return

        spec = importlib.util.find_spec("speechmos")
        if spec is None or not spec.submodule_search_locations:
            raise ImportError("speechmos not installed. Run: pip install speechmos")

        import onnxruntime as ort

        package_dir = Path(next(iter(spec.submodule_search_locations)))
        model_path = package_dir / "dnsmos_models" / "sig_bak_ovr.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"DNSMOS model not found: {model_path}")

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.num_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    @staticmethod
    def _polyfit(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sig_raw, bak_raw, ovr_raw = raw[:, 0], raw[:, 1], raw[:, 2]
        sig = np.polyval([-0.08397278, 1.22083953, 0.0052439], sig_raw)
        bak = np.polyval([-0.13166888, 1.60915514, -0.39604546], bak_raw)
        ovr = np.polyval([-0.06766283, 1.11546468, 0.04602535], ovr_raw)
        return sig, bak, ovr

    def _build_segments(self, audio: np.ndarray) -> np.ndarray:
        segment_samples = int(self.INPUT_SECONDS * self.SAMPLE_RATE)
        if len(audio) < segment_samples:
            repeats = math.ceil(segment_samples / max(len(audio), 1))
            audio = np.tile(audio, repeats)
            return audio[:segment_samples][None, :].astype(np.float32)

        max_start = len(audio) - segment_samples
        window_count = max_start // self.SAMPLE_RATE + 1
        if window_count <= self.max_segments:
            starts = np.arange(window_count, dtype=np.int64) * self.SAMPLE_RATE
        else:
            starts = np.linspace(0, max_start, self.max_segments, dtype=np.int64)
        return np.stack(
            [audio[start : start + segment_samples] for start in np.unique(starts)],
            axis=0,
        ).astype(np.float32)

    def score(self, wav_path: str | Path) -> torch.Tensor:
        self._lazy_init()
        try:
            import soundfile as sf

            audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=-1)
            if sample_rate != self.SAMPLE_RATE:
                from scipy.signal import resample_poly

                divisor = math.gcd(int(sample_rate), self.SAMPLE_RATE)
                audio = resample_poly(
                    audio,
                    self.SAMPLE_RATE // divisor,
                    int(sample_rate) // divisor,
                ).astype(np.float32)
            audio = np.nan_to_num(audio, copy=False)
            if audio.size == 0:
                return torch.zeros(3, dtype=torch.float32)

            segments = self._build_segments(audio)
            raw_outputs = []
            for start in range(0, len(segments), self.batch_size):
                batch = segments[start : start + self.batch_size]
                raw_outputs.append(self._session.run(None, {"input_1": batch})[0])
            raw = np.concatenate(raw_outputs, axis=0)
            sig, bak, ovr = self._polyfit(raw)
            mos = np.asarray([ovr.mean(), sig.mean(), bak.mean()], dtype=np.float32)

            # DNSMOS is on [1, 5]; map it to a true [0, 1] interval.
            normalized = np.clip((mos - 1.0) / 4.0, 0.0, 1.0)
            return torch.from_numpy(normalized)
        except Exception:
            return torch.zeros(3, dtype=torch.float32)
