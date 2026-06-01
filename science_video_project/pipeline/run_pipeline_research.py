"""
扩展流水线 — 提取研究版所需全部特征

在原有 pipeline 基础上新增:
  - sci_hand_feat: 手工科学性特征 (§7)
  - frame_features: 逐帧 CLIP 特征用于时序编码器 (§9)
  - engagement_target: 传播力标签用于辅助分支 (§6)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pipeline.build_sample import build_sample
from pipeline.config import CFG
from pipeline.step_aesthetic_clip import CLIPAestheticScorer, DEFAULT_PROMPTS
from pipeline.step_audio import AudioEncoder
from pipeline.step_extract import VideoExtractor
from pipeline.step_meta import MetaFeatureBuilder
from pipeline.step_science_features import ScienceFeatureExtractor
from pipeline.step_text import TextEncoder
from pipeline.step_video import VideoEncoder
from pipeline.utils_io import ensure_dir, get_video_id, list_videos, load_metadata, save_pt, set_seed


def _build_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("pipeline_research")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def extract_per_frame_features(
    video_encoder: VideoEncoder, frame_dir: str | Path
) -> torch.Tensor:
    """提取逐帧 CLIP 特征 (不池化) — 用于时序编码器。

    Returns:
        (N_frames, 512) float32 tensor
    """
    import numpy as np
    from PIL import Image

    frame_paths = sorted(Path(frame_dir).glob("*.jpg"))
    if not frame_paths:
        return torch.zeros(0, 512, dtype=torch.float32)

    feats = []
    with torch.no_grad():
        for p in frame_paths:
            img = Image.open(p).convert("RGB")
            if video_encoder.backend == "hf":
                inputs = video_encoder.processor(images=img, return_tensors="pt")
                pixel = inputs["pixel_values"].to(video_encoder.device, dtype=video_encoder.model.dtype)
                out = video_encoder.model(pixel_values=pixel)
                feat = out.image_embeds
            else:
                image = video_encoder.preprocess(img).unsqueeze(0).to(video_encoder.device)
                feat = video_encoder.model.encode_image(image)
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            feats.append(feat.squeeze(0).detach().cpu())

    return torch.stack(feats, dim=0).float()  # (N, 512)


def compute_engagement_target(row: dict) -> float:
    """根据元数据计算传播力标签 (用于 EngagementBranch 辅助训练)。

    标签: log(likes+1) + log(shares+1) + log(comments+1) 归一化后二值化。
    """
    likes = float(row.get("likes", row.get("like_count", 0)) or 0)
    shares = float(row.get("shares", row.get("share_count", 0)) or 0)
    comments = float(row.get("comments", row.get("comment_count", 0)) or 0)

    score = np.log1p(likes) + np.log1p(shares) + np.log1p(comments)
    # 简单阈值: 高于均值 → 高传播 (1)
    # 实际使用时建议基于 all videos 的分布设定阈值
    return 1.0 if score > 5.0 else 0.0  # 粗略阈值


def main() -> None:
    try:
        set_seed(CFG.seed)
        ensure_dir(CFG.output_dir)
        for d in [CFG.frame_dir, CFG.audio_dir, CFG.feature_dir, CFG.log_dir]:
            ensure_dir(d)

        logger = _build_logger(CFG.log_dir / "pipeline_research.log")
        metadata = load_metadata(CFG.metadata_csv)
        row_map = {str(row.video_id): row.to_dict() for _, row in metadata.iterrows()}

        videos = list_videos(CFG.video_dir)
        if not videos:
            raise FileNotFoundError(f"No video files found in: {CFG.video_dir}")

        # 编码器初始化
        meta_builder = MetaFeatureBuilder(categories=metadata["category"].tolist(), out_dim=CFG.meta_dim)
        sci_extractor = ScienceFeatureExtractor()  # §7 新增

        if CFG.ffmpeg_path:
            os.environ["FFMPEG_PATH"] = str(CFG.ffmpeg_path)

        asr_path = Path(CFG.asr_model_path).resolve()
        audio_path = Path(CFG.audio_model_path).resolve()
        if not asr_path.exists():
            raise FileNotFoundError(f"ASR model path not found: {asr_path}")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio model path not found: {audio_path}")

        extractor = VideoExtractor(audio_sr=CFG.audio_sr)
        text_encoder = TextEncoder(CFG.text_model_name, str(asr_path), CFG.device, CFG.local_files_only)
        video_encoder = VideoEncoder(CFG.clip_model_name, CFG.device)
        audio_encoder = AudioEncoder(str(audio_path), CFG.device, out_dim=CFG.audio_dim, shared_model=text_encoder.asr)
        aes_scorer = CLIPAestheticScorer(prompts=DEFAULT_PROMPTS, device=CFG.device)

        processed = 0
        skipped = 0

        for video_path in tqdm(videos, desc="Extracting research features"):
            video_id = get_video_id(video_path)
            if video_id not in row_map:
                skipped += 1
                logger.warning("Skip %s: not found in parsed_metadata.csv", video_id)
                continue

            out_pt = CFG.feature_dir / f"{video_id}.pt"
            if CFG.skip_existing and out_pt.exists():
                skipped += 1
                logger.info("Skip %s: feature exists", video_id)
                continue

            row = row_map[video_id]
            wav_path = CFG.audio_dir / f"{video_id}.wav"
            frame_path = CFG.frame_dir / video_id
            ensure_dir(frame_path)

            # 基础提取
            extractor.extract_audio(video_path, wav_path)
            extractor.extract_frames(video_path, frame_path, fps=CFG.frame_fps)

            # 特征编码
            text_result = text_encoder.encode_fields(
                wav_path=str(wav_path),
                title=str(row.get("title", "") or ""),
                tags=str(row.get("tags", "") or ""),
            )
            video_feat = video_encoder.encode_frames(frame_path)
            audio_feat = audio_encoder.encode(str(wav_path))
            meta_feat = meta_builder.build(row)

            # 美学
            aes_result = aes_scorer.score_frames_batched(frame_path)
            dim_names = list(aes_result["dimensions"].keys())
            aes_feat = torch.tensor([aes_result["dimensions"][n] for n in dim_names], dtype=torch.float32)

            # §7 手工科学性特征
            sci_hand = sci_extractor.extract_vector(text_result.subtitle)

            # §9 逐帧特征 (用于时序编码器)
            frame_feats = extract_per_frame_features(video_encoder, frame_path)

            # §6 传播力标签
            eng_target = compute_engagement_target(row)

            sample = build_sample(
                video_id=video_id,
                category=str(row.get("category", "unknown")),
                label=int(row.get("label", 0)),
                text_feat=text_result.embedding,
                video_feat=video_feat,
                audio_feat=audio_feat,
                meta_feat=meta_feat,
                aes_feat=aes_feat,
                sci_hand_feat=sci_hand,
                frame_features=frame_feats,
                engagement_target=eng_target,
            )
            save_pt(sample, out_pt)
            processed += 1

        logger.info("Research pipeline completed: processed=%d, skipped=%d", processed, skipped)
        logger.info("Features saved in %s", CFG.feature_dir)
    except Exception as exc:
        print(f"Research pipeline failed: {exc}")
        raise


if __name__ == "__main__":
    main()
