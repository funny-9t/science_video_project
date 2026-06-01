import logging
import os
import sys
from pathlib import Path

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
from pipeline.step_text import TextEncoder
from pipeline.step_video import VideoEncoder
from pipeline.utils_io import ensure_dir, get_video_id, list_videos, load_metadata, save_pt, set_seed


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

        for video_path in tqdm(videos, desc="Extracting features"):
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

            extractor.extract_audio(video_path, wav_path)
            extractor.extract_frames(video_path, frame_path, fps=CFG.frame_fps)

            text_result = text_encoder.encode_fields(
                wav_path=str(wav_path),
                title=str(row.get("title", "") or ""),
                tags=str(row.get("tags", "") or ""),
            )
            video_feat = video_encoder.encode_frames(frame_path)
            audio_feat = audio_encoder.encode(str(wav_path))
            meta_feat = meta_builder.build(row)

            # CLIP prompt-based aesthetic scoring → (D,)-dim feature vector
            aes_result = aes_scorer.score_frames_batched(frame_path)
            dim_names = list(aes_result["dimensions"].keys())
            aes_feat = torch.tensor(
                [aes_result["dimensions"][n] for n in dim_names],
                dtype=torch.float32,
            )  # shape: (D,)

            sample = build_sample(
                video_id=video_id,
                category=str(row.get("category", "unknown")),
                label=int(row.get("label", 0)),
                text_feat=text_result.embedding,
                video_feat=video_feat,
                audio_feat=audio_feat,
                meta_feat=meta_feat,
                aes_feat=aes_feat,
            )
            save_pt(sample, out_pt)
            processed += 1

        logger.info("Pipeline completed: processed=%d, skipped=%d", processed, skipped)
        logger.info("Features saved in %s", CFG.feature_dir)
    except Exception as exc:
        print(f"Pipeline failed: {exc}")
        raise


if __name__ == "__main__":
    main()
