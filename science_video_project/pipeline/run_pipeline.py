import logging
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
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
from pipeline.step_dnsmos import DNSMOSScorer
from pipeline.step_extract import VideoExtractor
from pipeline.step_meta import MetaFeatureBuilder
from pipeline.step_science_features import ScienceFeatureExtractor
from pipeline.step_llm_knowledge import LLMKnowledgeExtractor, DummyLLMKnowledgeExtractor
from pipeline.step_speech_features import (
    TranscriptResult,
    compute_speech_features,
    normalize_speech_features,
    save_transcript,
)
from pipeline.step_text import TextEncoder
from pipeline.step_video import VideoEncoder
from pipeline.utils_io import ensure_dir, get_video_id, list_videos, load_metadata, save_pt, set_seed


def _compute_speech_rhythm(segments: list, total_duration: float) -> np.ndarray:
    """从 Whisper 段级时间戳计算语速节奏特征 (6维)。

    返回:
        np.ndarray shape=(6,):
          [0] seg_wpm_mean   — 段级 WPM 均值
          [1] seg_wpm_std    — 段级 WPM 标准差
          [2] seg_wpm_min    — 最慢段 WPM
          [3] seg_wpm_max    — 最快段 WPM
          [4] pause_ratio    — 停顿占比 (段间间隔>0.5s / 总时长)
          [5] speech_density — 说话占比 (段内时间 / 总时长)
    """
    if not segments or total_duration <= 0:
        return np.zeros(6, dtype=np.float32)

    seg_wpms = []
    total_speech_time = 0.0
    total_pause_time = 0.0

    for i, (start, end, text) in enumerate(segments):
        seg_dur = end - start
        if seg_dur <= 0:
            continue
        char_count = len(text.replace(" ", ""))
        wpm = char_count / (seg_dur / 60.0)
        seg_wpms.append(wpm)
        total_speech_time += seg_dur

        # 计算与前一段的间隔（停顿）
        if i > 0:
            gap = start - segments[i - 1][1]
            if gap > 0.5:  # 超过 0.5s 算有效停顿
                total_pause_time += gap

    if not seg_wpms:
        return np.zeros(6, dtype=np.float32)

    seg_wpms = np.array(seg_wpms, dtype=np.float32)

    return np.array([
        float(np.mean(seg_wpms)),           # seg_wpm_mean
        float(np.std(seg_wpms)),            # seg_wpm_std
        float(np.min(seg_wpms)),            # seg_wpm_min
        float(np.max(seg_wpms)),            # seg_wpm_max
        total_pause_time / total_duration,   # pause_ratio
        total_speech_time / total_duration,  # speech_density
    ], dtype=np.float32)


def _build_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("pipeline")
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


def main() -> None:
    try:
        set_seed(CFG.seed)
        ensure_dir(CFG.output_dir)
        ensure_dir(CFG.frame_dir)
        ensure_dir(CFG.audio_dir)
        ensure_dir(CFG.transcript_dir)
        ensure_dir(CFG.feature_dir)
        ensure_dir(CFG.log_dir)

        logger = _build_logger(CFG.log_dir / "pipeline.log")
        metadata = load_metadata(CFG.metadata_csv)
        row_map = {str(row.video_id): row.to_dict() for _, row in metadata.iterrows()}

        videos = list_videos(CFG.video_dir)
        if not videos:
            raise FileNotFoundError(f"No video files found in: {CFG.video_dir}")

        meta_builder = MetaFeatureBuilder(categories=metadata["category"].tolist(), out_dim=CFG.meta_dim)
        if CFG.ffmpeg_path:
            os.environ["FFMPEG_PATH"] = str(CFG.ffmpeg_path)

        asr_path = CFG.asr_model_path  # 可能是模型名如 "tiny" 或本地路径
        audio_path = CFG.audio_model_path
        # 仅本地路径才需要 exists 检查，模型名由 faster-whisper 自动下载
        asr_path_obj = Path(asr_path)
        if asr_path_obj.exists() or not asr_path_obj.is_absolute():
            pass  # 模型名或存在的本地路径，OK
        else:
            raise FileNotFoundError(f"ASR model path not found: {asr_path_obj}")
        audio_path_obj = Path(audio_path)
        if not audio_path_obj.exists() and audio_path_obj.is_absolute():
            raise FileNotFoundError(f"Audio model path not found: {audio_path_obj}")

        extractor = VideoExtractor(audio_sr=CFG.audio_sr)
        text_encoder = TextEncoder(CFG.text_model_name, asr_path, CFG.device, CFG.local_files_only)
        video_encoder = VideoEncoder(CFG.clip_model_name, CFG.device)
        audio_encoder = AudioEncoder(audio_path, CFG.device, out_dim=CFG.audio_dim, shared_model=text_encoder.asr)
        aes_scorer = CLIPAestheticScorer(prompts=DEFAULT_PROMPTS, device=CFG.device)
        dnsmos_scorer = DNSMOSScorer(device=CFG.device)

        # ── §7+ LLM 知识科学性特征提取器 ──
        sci_hand_extractor = ScienceFeatureExtractor()
        api_key = CFG.llm_api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if CFG.use_llm_knowledge and api_key:
            llm_extractor = LLMKnowledgeExtractor(
                api_key=api_key,
                model=CFG.llm_model_name,
                api_base=CFG.llm_api_base,
                cache_dir=CFG.llm_cache_dir,
                temperature=CFG.llm_temperature,
            )
        else:
            llm_extractor = DummyLLMKnowledgeExtractor()
            if CFG.use_llm_knowledge and not api_key:
                logger.warning("LLM knowledge enabled but no API key — using DummyLLM fallback")

        processed = 0
        skipped = 0

        for video_path in tqdm(videos, desc="Extracting features"):
            video_id = get_video_id(video_path)
            if video_id not in row_map:
                skipped += 1
                logger.warning("Skip %s: not found in metadata", video_id)
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

            if not wav_path.exists():
                extractor.extract_audio(video_path, wav_path)
            if not frame_path.exists() or not any(frame_path.iterdir()):
                extractor.extract_frames(video_path, frame_path, fps=CFG.frame_fps)

            text_result = text_encoder.encode_fields(
                wav_path=str(wav_path),
                title=str(row.get("title", "") or ""),
                tags=str(row.get("tags", "") or ""),
            )
            frame_features = video_encoder.encode_frame_features(
                frame_path, max_frames=CFG.clip_max_frames
            )
            clip_video_feat = frame_features.mean(dim=0)
            video_feat = video_encoder.legacy_project(clip_video_feat, output_dim=CFG.video_dim)
            audio_feat = audio_encoder.encode(str(wav_path))
            meta_feat = meta_builder.build(row)

            # CLIP prompt-based aesthetic scoring → (D,)-dim feature vector
            aes_result = aes_scorer.score_frames_batched(frame_path)
            dim_names = list(aes_result["dimensions"].keys())
            aes_feat = torch.tensor(
                [aes_result["dimensions"][n] for n in dim_names],
                dtype=torch.float32,
            )  # shape: (D,)

            # DNSMOS 客观音频质量评分 → (3,) [ovrl_mos, sig_mos, bak_mos]
            dnsmos_feat = dnsmos_scorer.score(str(wav_path))

            subtitle_text = text_result.subtitle or ""
            save_transcript(
                TranscriptResult(subtitle_text, text_result.segments),
                CFG.transcript_dir / f"{video_id}.json",
            )
            try:
                audio_duration = sf.info(str(wav_path)).duration  # 秒
            except Exception:
                audio_duration = 0.0
            wpm_raw, speech_rhythm_raw_feat = compute_speech_features(
                text_result.segments, audio_duration
            )
            wpm, speech_rhythm_feat = normalize_speech_features(
                wpm_raw, speech_rhythm_raw_feat
            )

            # ── §7 手工科学性特征 ──
            sci_hand_feat = sci_hand_extractor.extract_vector(subtitle_text)

            # ── §7+ LLM 知识科学性特征 ──
            science_text = "\n".join(
                part for part in [
                    str(row.get("title", "") or "").strip(),
                    str(row.get("tags", "") or "").strip(),
                    subtitle_text,
                ] if part
            )
            llm_details = llm_extractor.extract_details(science_text, verbose=(processed == 0))
            llm_knowledge_feat = llm_details.scores
            llm_analysis_only_feat = text_encoder.text_embedding(
                llm_details.feature_text(include_reasoning=False)
            )
            llm_reasoning_analysis_feat = text_encoder.text_embedding(
                llm_details.feature_text(include_reasoning=True)
            )
            llm_analysis_feat = (
                llm_reasoning_analysis_feat
                if CFG.llm_include_reasoning
                else llm_analysis_only_feat
            )

            sample = build_sample(
                video_id=video_id,
                category=str(row.get("category", "unknown")),
                label=int(row.get("label", 0)),
                text_feat=text_result.embedding,
                video_feat=video_feat,
                clip_video_feat=clip_video_feat,
                audio_feat=audio_feat,
                meta_feat=meta_feat,
                aes_feat=aes_feat,
                dnsmos_feat=dnsmos_feat,
                wpm=wpm,
                wpm_raw=wpm_raw,
                speech_rhythm_feat=speech_rhythm_feat,
                speech_rhythm_raw_feat=speech_rhythm_raw_feat,
                sci_hand_feat=sci_hand_feat,
                llm_knowledge_feat=llm_knowledge_feat,
                llm_analysis_feat=llm_analysis_feat,
                llm_analysis_only_feat=llm_analysis_only_feat,
                llm_reasoning_analysis_feat=llm_reasoning_analysis_feat,
                frame_features=frame_features,
            )
            sample["clip_sampling_version"] = f"global_uniform_{CFG.clip_max_frames}_v1"
            save_pt(sample, out_pt)
            processed += 1

        logger.info("Pipeline completed: processed=%d, skipped=%d", processed, skipped)
        logger.info("Features saved in %s", CFG.feature_dir)
    except Exception as exc:
        print(f"Pipeline failed: {exc}")
        raise


if __name__ == "__main__":
    main()
