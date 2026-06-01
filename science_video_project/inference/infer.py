import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.build_sample import build_sample
from pipeline.config import CFG
from pipeline.step_aesthetic_clip import CLIPAestheticScorer, DEFAULT_PROMPTS
from pipeline.step_audio import AudioEncoder
from pipeline.step_extract import VideoExtractor
from pipeline.step_meta import MetaFeatureBuilder
from pipeline.step_text import TextEncoder
from pipeline.step_video import VideoEncoder
from pipeline.utils_io import ensure_dir, get_video_id, load_metadata, save_json
from training.model_mvp import MultiModalQualityModel
from training.utils_train import adapt_checkpoint_state_dict, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for a single science short video")
    parser.add_argument("--video", required=True, help="Path to one video file")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint, e.g. outputs/checkpoints/best.pt")
    parser.add_argument("--metadata", default=str(CFG.metadata_csv), help="Metadata csv path")
    parser.add_argument("--output", default="", help="Optional output json path")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> MultiModalQualityModel:
    model = MultiModalQualityModel(
        text_dim=CFG.text_dim,
        video_dim=CFG.video_dim,
        audio_dim=CFG.audio_dim,
        meta_dim=CFG.meta_dim,
        aes_dim=CFG.aes_dim,
        hidden_dim=CFG.hidden_dim,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(adapt_checkpoint_state_dict(state_dict))
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    ensure_dir(CFG.audio_dir)
    ensure_dir(CFG.frame_dir)

    video_id = get_video_id(video_path)
    metadata = load_metadata(args.metadata)
    categories = metadata["category"].tolist()
    row_df = metadata[metadata["video_id"].astype(str) == video_id]
    if len(row_df) > 0:
        row = row_df.iloc[0].to_dict()
    else:
        row = {
            "video_id": video_id,
            "label": 0,
            "category": "unknown",
            "title": "",
            "tags": "",
            "duration": 0,
            "verified": 0,
            "publish_time": "",
        }

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
    meta_builder = MetaFeatureBuilder(categories=categories + [row["category"]], out_dim=CFG.meta_dim)
    aes_scorer = CLIPAestheticScorer(prompts=DEFAULT_PROMPTS, device=CFG.device)

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
    )

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

    device = get_device(CFG.device)
    model = load_model(args.checkpoint, device)
    inputs = {
        "text_feat": torch.tensor(sample["text_feat"], dtype=torch.float32).unsqueeze(0).to(device),
        "video_feat": torch.tensor(sample["video_feat"], dtype=torch.float32).unsqueeze(0).to(device),
        "audio_feat": torch.tensor(sample["audio_feat"], dtype=torch.float32).unsqueeze(0).to(device),
        "meta_feat": torch.tensor(sample["meta_feat"], dtype=torch.float32).unsqueeze(0).to(device),
        "aes_feat": torch.tensor(sample["aes_feat"], dtype=torch.float32).unsqueeze(0).to(device),
    }

    with torch.no_grad():
        out = model(**inputs)

    probability = float(out["probability"].item())
    result = {
        "scientific_score": float(out["scientific_score"].item()),
        "technical_score": float(out["technical_score"].item()),
        "aesthetic_score": float(out["aesthetic_score"].item()),
        "overall_score": float(out["overall_score"].item()),
        "probability": probability,
        "prediction": "上榜" if probability >= CFG.threshold else "未上榜",
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        save_json(result, args.output)


if __name__ == "__main__":
    main()
