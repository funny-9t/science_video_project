import os
import shutil
import subprocess
from pathlib import Path

import cv2

from pipeline.utils_io import ensure_dir


class VideoExtractor:
    def __init__(self, audio_sr: int = 16000):
        self.audio_sr = int(audio_sr)
        self.ffmpeg_bin = self._resolve_ffmpeg()

    def _resolve_ffmpeg(self) -> str:
        override = os.environ.get("FFMPEG_PATH", "").strip()
        if override:
            path = Path(override)
            if path.is_dir():
                path = path / "ffmpeg.exe"
            if path.exists():
                return str(path)

        detected = shutil.which("ffmpeg")
        if detected:
            return detected

        raise RuntimeError("ffmpeg not found. Set FFMPEG_PATH or add ffmpeg to PATH.")

    def extract_audio(self, video_path: str | Path, wav_path: str | Path) -> None:
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-ac",
            "1",
            "-ar",
            str(self.audio_sr),
            str(wav_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "ffmpeg failed to extract audio. "
                f"returncode={result.returncode}, stderr={result.stderr.strip()}"
            )

    def extract_frames(self, video_path: str | Path, out_dir: str | Path, fps: int = 1) -> int:
        out = Path(out_dir)
        ensure_dir(out)

        cap = cv2.VideoCapture(str(video_path))
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if src_fps <= 0:
            src_fps = 25.0
        interval = max(1, int(round(src_fps / max(1, int(fps)))))

        count = 0
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if count % interval == 0:
                frame_path = out / f"{saved:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                saved += 1
            count += 1
        cap.release()

        if saved == 0:
            raise RuntimeError(f"No frame extracted from {video_path}")
        return saved
