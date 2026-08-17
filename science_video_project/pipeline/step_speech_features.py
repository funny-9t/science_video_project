from __future__ import annotations

import json
import os
import re
import site
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


SPEECH_RATE_SCALE = 600.0
_DLL_HANDLES: list[object] = []


def configure_nvidia_dll_path() -> list[Path]:
    """Expose pip-installed CUDA DLLs to CTranslate2 on Windows."""
    if os.name != "nt":
        return []

    roots = {Path(sysconfig.get_paths()["purelib"])}
    roots.update(Path(p) for p in site.getsitepackages())
    candidates = []
    for root in roots:
        candidates.extend(
            [
                root / "nvidia" / "cublas" / "bin",
                root / "nvidia" / "cudnn" / "bin",
            ]
        )

    found = [path for path in candidates if path.exists()]
    if found:
        current = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join([*(str(p) for p in found), current])
        for path in found:
            try:
                _DLL_HANDLES.append(os.add_dll_directory(str(path)))
            except (AttributeError, FileNotFoundError, OSError):
                pass
    return found


def count_spoken_units(text: str) -> int:
    """Count Chinese characters plus Latin words/numbers as speech units."""
    content = text or ""
    chinese = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", content)
    latin_or_number = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?", content)
    return len(chinese) + len(latin_or_number)


def compute_speech_features(
    segments: Iterable[tuple[float, float, str]],
    total_duration: float,
) -> tuple[float, np.ndarray]:
    """Return raw whole-video WPM and six raw rhythm features."""
    segment_list = list(segments)
    if total_duration <= 0:
        return 0.0, np.zeros(6, dtype=np.float32)

    total_units = sum(count_spoken_units(text) for _, _, text in segment_list)
    wpm = total_units / (total_duration / 60.0)

    segment_wpms: list[float] = []
    total_speech_time = 0.0
    total_pause_time = 0.0
    previous_end: float | None = None
    for start, end, text in segment_list:
        duration = max(float(end) - float(start), 0.0)
        if duration <= 0:
            continue
        segment_wpms.append(count_spoken_units(text) / (duration / 60.0))
        total_speech_time += duration
        if previous_end is not None:
            gap = float(start) - previous_end
            if gap > 0.5:
                total_pause_time += gap
        previous_end = float(end)

    if not segment_wpms:
        return float(wpm), np.zeros(6, dtype=np.float32)

    rates = np.asarray(segment_wpms, dtype=np.float32)
    rhythm = np.asarray(
        [
            rates.mean(),
            rates.std(),
            rates.min(),
            rates.max(),
            total_pause_time / total_duration,
            total_speech_time / total_duration,
        ],
        dtype=np.float32,
    )
    return float(wpm), rhythm


def normalize_speech_features(wpm: float, rhythm: np.ndarray) -> tuple[float, np.ndarray]:
    """Scale speech rates for stable MLP training while preserving ratios."""
    rhythm_norm = np.asarray(rhythm, dtype=np.float32).copy()
    rhythm_norm[:4] = np.clip(rhythm_norm[:4] / SPEECH_RATE_SCALE, 0.0, 2.0)
    rhythm_norm[4:] = np.clip(rhythm_norm[4:], 0.0, 1.0)
    return float(np.clip(float(wpm) / SPEECH_RATE_SCALE, 0.0, 2.0)), rhythm_norm


@dataclass
class TranscriptResult:
    text: str
    segments: list[tuple[float, float, str]]
    language: str = "zh"


class SpeechTranscriber:
    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        compute_type: str | None = None,
        batch_size: int = 8,
        language: str = "zh",
        cpu_fallback: bool = True,
    ) -> None:
        self.model_path = str(model_path)
        self.requested_device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.language = language
        self.cpu_fallback = cpu_fallback
        self.device = device
        self.model = None
        self.pipeline = None
        try:
            self._initialize(device)
        except RuntimeError:
            if device != "cuda" or not cpu_fallback:
                raise
            self._initialize("cpu")

    def _initialize(self, device: str) -> None:
        configure_nvidia_dll_path()
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        compute_type = self.compute_type or ("float16" if device == "cuda" else "int8")
        self.model = WhisperModel(self.model_path, device=device, compute_type=compute_type)
        self.pipeline = BatchedInferencePipeline(model=self.model) if device == "cuda" else None
        self.device = device

    def _transcribe_once(self, wav_path: str | Path) -> TranscriptResult:
        kwargs = {
            "beam_size": 1,
            "language": self.language,
            "vad_filter": True,
            "condition_on_previous_text": False,
        }
        if self.pipeline is not None:
            segments, info = self.pipeline.transcribe(
                str(wav_path), batch_size=self.batch_size, **kwargs
            )
        else:
            segments, info = self.model.transcribe(str(wav_path), **kwargs)

        result_segments = [
            (float(seg.start), float(seg.end), seg.text.strip())
            for seg in segments
            if seg.text and seg.text.strip()
        ]
        text = " ".join(text for _, _, text in result_segments).strip()
        language = getattr(info, "language", None) or self.language
        return TranscriptResult(text=text, segments=result_segments, language=str(language))

    def transcribe(self, wav_path: str | Path) -> TranscriptResult:
        try:
            return self._transcribe_once(wav_path)
        except RuntimeError as exc:
            if self.device == "cuda" and "out of memory" in str(exc).lower() and self.batch_size > 1:
                self.batch_size = max(1, self.batch_size // 2)
                return self._transcribe_once(wav_path)
            if self.device != "cuda" or not self.cpu_fallback:
                raise
            self.model = None
            self.pipeline = None
            self._initialize("cpu")
            return self._transcribe_once(wav_path)


def load_transcript(path: str | Path) -> TranscriptResult | None:
    transcript_path = Path(path)
    if not transcript_path.exists():
        return None
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments = [
        (float(item["start"]), float(item["end"]), str(item["text"]))
        for item in payload.get("segments", [])
    ]
    return TranscriptResult(
        text=str(payload.get("text", "")),
        segments=segments,
        language=str(payload.get("language", "zh")),
    )


def save_transcript(result: TranscriptResult, path: str | Path) -> None:
    transcript_path = Path(path)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "language": result.language,
        "text": result.text,
        "segments": [
            {"start": start, "end": end, "text": text}
            for start, end, text in result.segments
        ],
    }
    temp_path = transcript_path.with_suffix(transcript_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(transcript_path)
