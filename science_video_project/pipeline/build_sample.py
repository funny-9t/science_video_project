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
    clip_video_feat: torch.Tensor | np.ndarray | None = None,
    aes_feat: torch.Tensor | np.ndarray | None = None,
    dnsmos_feat: torch.Tensor | np.ndarray | None = None,
    wpm: float | None = None,
    speech_rhythm_feat: torch.Tensor | np.ndarray | None = None,
    sci_hand_feat: torch.Tensor | np.ndarray | None = None,
    llm_knowledge_feat: torch.Tensor | np.ndarray | None = None,
    llm_analysis_feat: torch.Tensor | np.ndarray | None = None,
    llm_analysis_only_feat: torch.Tensor | np.ndarray | None = None,
    llm_reasoning_analysis_feat: torch.Tensor | np.ndarray | None = None,
    cover_feat: torch.Tensor | np.ndarray | None = None,
    wpm_raw: float | None = None,
    speech_rhythm_raw_feat: torch.Tensor | np.ndarray | None = None,
    frame_features: torch.Tensor | np.ndarray | None = None,
    engagement_target: float | None = None,
    # ── 细粒度分支监督目标 (归一化到 [0,1]) ──
    sci_target: float | None = None,
    tech_target: float | None = None,
    aes_target: float | None = None,
) -> Dict:
    text_np = _as_float32_numpy(text_feat)
    video_np = _as_float32_numpy(video_feat)
    clip_video_np = (
        _as_float32_numpy(clip_video_feat)
        if clip_video_feat is not None
        else np.zeros(768, dtype=np.float32)
    )
    audio_np = _as_float32_numpy(audio_feat)
    meta_np = _as_float32_numpy(meta_feat)
    aes_np = _as_float32_numpy(aes_feat) if aes_feat is not None else np.zeros(7, dtype=np.float32)
    dnsmos_np = _as_float32_numpy(dnsmos_feat) if dnsmos_feat is not None else np.zeros(3, dtype=np.float32)
    wpm_val = float(wpm) if wpm is not None else 0.0
    rhythm_np = _as_float32_numpy(speech_rhythm_feat) if speech_rhythm_feat is not None else np.zeros(6, dtype=np.float32)
    sci_np = _as_float32_numpy(sci_hand_feat) if sci_hand_feat is not None else np.zeros(5, dtype=np.float32)
    llm_np = _as_float32_numpy(llm_knowledge_feat) if llm_knowledge_feat is not None else np.zeros(4, dtype=np.float32)
    llm_analysis_np = (
        _as_float32_numpy(llm_analysis_feat)
        if llm_analysis_feat is not None
        else np.zeros(768, dtype=np.float32)
    )
    llm_analysis_only_np = (
        _as_float32_numpy(llm_analysis_only_feat)
        if llm_analysis_only_feat is not None
        else llm_analysis_np.copy()
    )
    llm_reasoning_analysis_np = (
        _as_float32_numpy(llm_reasoning_analysis_feat)
        if llm_reasoning_analysis_feat is not None
        else llm_analysis_np.copy()
    )
    cover_np = (
        _as_float32_numpy(cover_feat)
        if cover_feat is not None
        else np.zeros(3, dtype=np.float32)
    )
    rhythm_raw_np = (
        _as_float32_numpy(speech_rhythm_raw_feat)
        if speech_rhythm_raw_feat is not None
        else np.zeros(6, dtype=np.float32)
    )
    frame_np = _as_float32_numpy(frame_features) if frame_features is not None else np.zeros(0, dtype=np.float32)
    eng_target = float(engagement_target) if engagement_target is not None else 0.0
    st = float(sci_target) if sci_target is not None else -1.0
    tt = float(tech_target) if tech_target is not None else -1.0
    at = float(aes_target) if aes_target is not None else -1.0

    if text_np.shape[-1] != 768:
        raise RuntimeError(f"text feature dim must be 768, got {text_np.shape[-1]}")
    if video_np.shape[-1] not in {512, 768}:
        raise RuntimeError(f"video feature dim must be 512 or 768, got {video_np.shape[-1]}")
    if clip_video_np.shape[-1] != 768:
        raise RuntimeError(f"native CLIP feature dim must be 768, got {clip_video_np.shape[-1]}")
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
        "clip_video_feat": clip_video_np,
        "audio_feat": audio_np,
        "meta_feat": meta_np,
        "aes_feat": aes_np,
        "dnsmos_feat": dnsmos_np,
        "wpm": wpm_val,
        "wpm_raw": float(wpm_raw) if wpm_raw is not None else 0.0,
        "speech_rhythm_feat": rhythm_np,
        "speech_rhythm_raw_feat": rhythm_raw_np,
        "sci_hand_feat": sci_np,
        "llm_knowledge_feat": llm_np,
        "llm_analysis_feat": llm_analysis_np,
        "llm_analysis_only_feat": llm_analysis_only_np,
        "llm_reasoning_analysis_feat": llm_reasoning_analysis_np,
        "cover_feat": cover_np,
        "frame_features": frame_np,
        "engagement_target": eng_target,
        # 细粒度分支监督目标 (-1 表示缺失)
        "sci_target": st,
        "tech_target": tt,
        "aes_target": at,
    }
