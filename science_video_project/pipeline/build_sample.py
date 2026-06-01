from typing import Dict

import numpy as np
import torch


def _as_float32_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def build_sample(
    video_id: str,
    category: str,
    label: int,
    text_feat: torch.Tensor,
    video_feat: torch.Tensor,
    audio_feat: torch.Tensor,
    meta_feat: np.ndarray,
    aes_feat: torch.Tensor | np.ndarray | None = None,
) -> Dict:
    text_np = _as_float32_numpy(text_feat)
    video_np = _as_float32_numpy(video_feat)
    audio_np = _as_float32_numpy(audio_feat)
    meta_np = _as_float32_numpy(meta_feat)
    aes_np = _as_float32_numpy(aes_feat) if aes_feat is not None else np.zeros(7, dtype=np.float32)

    if text_np.shape[-1] != 768:
        raise RuntimeError(f"text feature dim must be 768, got {text_np.shape[-1]}")
    if video_np.shape[-1] != 512:
        raise RuntimeError(f"video feature dim must be 512, got {video_np.shape[-1]}")
    if audio_np.shape[-1] != 384:
        raise RuntimeError(f"audio feature dim must be 384, got {audio_np.shape[-1]}")
    if meta_np.shape[-1] != 16:
        raise RuntimeError(f"meta feature dim must be 16, got {meta_np.shape[-1]}")

    return {
        "video_id": str(video_id),
        "category": str(category),
        "label": int(label),
        "text_feat": text_np,
        "video_feat": video_np,
        "audio_feat": audio_np,
        "meta_feat": meta_np,
        "aes_feat": aes_np,
    }
