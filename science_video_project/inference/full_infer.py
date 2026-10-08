import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.infer import (
    _build_llm_extractor,
    _validate_model_path,
    apply_science_feature_mode,
    load_model,
)
from pipeline.config import CFG
from pipeline.step_aesthetic_clip import (
    CLIPAestheticScorer,
    DEFAULT_PROMPTS,
    SharedCLIPAestheticScorer,
)
from pipeline.step_audio import AudioEncoder
from pipeline.step_dnsmos import DNSMOSScorer
from pipeline.step_extract import VideoExtractor
from pipeline.step_meta import MetaFeatureBuilder
from pipeline.step_science_features import ScienceFeatureExtractor
from pipeline.step_speech_features import compute_speech_features, normalize_speech_features
from pipeline.step_text import TextEncoder
from pipeline.step_video import VideoEncoder
from pipeline.utils_io import ensure_dir, get_video_id, save_json
from training.utils_train import get_device


class InferenceEngine:
    """Inference engine that mirrors the training feature pipeline."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        allow_llm_online_enroll: bool = False,
    ):
        self.device = get_device(device)
        print(f"Using device: {self.device}")

        self.model, self.threshold, self.runtime_config = load_model(checkpoint_path, self.device)
        print(f"Loaded checkpoint: {checkpoint_path}")

        asr_path = _validate_model_path(CFG.asr_model_path, "ASR")
        audio_path = _validate_model_path(CFG.audio_model_path, "Audio")

        self.video_encoder = VideoEncoder(CFG.clip_model_name, device)
        if self.runtime_config["aesthetic_feature_backend"] == "shared_clip":
            self.aes_scorer = SharedCLIPAestheticScorer(
                CFG.clip_model_name, prompts=DEFAULT_PROMPTS, device=device
            )
        else:
            self.aes_scorer = CLIPAestheticScorer(prompts=DEFAULT_PROMPTS, device=device)
        self.text_encoder = TextEncoder(CFG.text_model_name, asr_path, device, CFG.local_files_only)
        self.audio_encoder = AudioEncoder(audio_path, device, out_dim=CFG.audio_dim, shared_model=self.text_encoder.asr)
        self.video_extractor = VideoExtractor(audio_sr=CFG.audio_sr)
        self.use_dnsmos = self.runtime_config["technical_feature_mode"] in {
            "full", "visual_dnsmos", "dnsmos_only"
        }
        self.dnsmos_scorer = DNSMOSScorer(device=device) if self.use_dnsmos else None
        self.sci_hand_extractor = ScienceFeatureExtractor()
        self.llm_extractor = _build_llm_extractor(allow_llm_online_enroll)

        print("All components initialized")

    def infer_from_video(self, video_path: str, row_data: dict) -> dict:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        video_id = get_video_id(video_path)
        ensure_dir(CFG.audio_dir)
        ensure_dir(CFG.frame_dir)

        wav_path = CFG.audio_dir / f"{video_id}.wav"
        frame_path = CFG.frame_dir / video_id
        ensure_dir(frame_path)

        print("Extracting audio and frames...")
        self.video_extractor.extract_audio(video_path, wav_path)
        self.video_extractor.extract_frames(video_path, frame_path, fps=CFG.frame_fps)

        print("Encoding features...")
        text_result = self.text_encoder.encode_fields(
            wav_path=str(wav_path),
            title=str(row_data.get("title", "") or ""),
            tags=str(row_data.get("tags", "") or ""),
        )
        frame_features = self.video_encoder.encode_frame_features(
            frame_path, max_frames=CFG.clip_max_frames
        )
        clip_video_feat = frame_features.mean(dim=0)
        video_feat = (
            clip_video_feat
            if self.runtime_config["video_dim"] == CFG.clip_video_dim
            else self.video_encoder.legacy_project(
                clip_video_feat, output_dim=self.runtime_config["video_dim"]
            )
        )
        audio_feat = self.audio_encoder.encode(str(wav_path))

        aes_result = (
            self.aes_scorer.score_frame_features(frame_features)
            if self.runtime_config["aesthetic_feature_backend"] == "shared_clip"
            else self.aes_scorer.score_frames_batched(frame_path)
        )
        aes_feat = torch.tensor(
            [aes_result["dimensions"][n] for n in aes_result["dimensions"].keys()],
            dtype=torch.float32,
        )
        dnsmos_feat = (
            self.dnsmos_scorer.score(str(wav_path))
            if self.dnsmos_scorer is not None
            else torch.zeros(CFG.dnsmos_dim, dtype=torch.float32)
        )

        subtitle_text = text_result.subtitle or ""
        audio_duration = 0.0
        try:
            import soundfile as sf

            audio_duration = sf.info(str(wav_path)).duration
        except Exception:
            pass
        wpm_raw, rhythm_raw = compute_speech_features(text_result.segments, audio_duration)
        wpm, rhythm = normalize_speech_features(wpm_raw, rhythm_raw)
        science_text = "\n".join(
            part for part in [
                str(row_data.get("title", "") or "").strip(),
                str(row_data.get("tags", "") or "").strip(),
                subtitle_text,
            ] if part
        )
        llm_details = self.llm_extractor.extract_details(science_text, verbose=False)
        llm_analysis_feat = self.text_encoder.text_embedding(
            llm_details.feature_text(
                include_reasoning=self.runtime_config["llm_text_source"] == "reasoning_and_analysis"
            )
        )

        return {
            "text_feat": text_result.embedding,
            "video_feat": video_feat,
            "audio_feat": audio_feat,
            "aes_feat": aes_feat,
            "dnsmos_feat": dnsmos_feat,
            "wpm": wpm,
            "speech_rhythm_feat": rhythm,
            "sci_hand_feat": self.sci_hand_extractor.extract_vector(subtitle_text),
            "llm_knowledge_feat": llm_details.scores,
            "llm_analysis_feat": llm_analysis_feat,
        }

    def predict(self, features: dict, row_data: dict) -> dict:
        meta_builder = MetaFeatureBuilder(
            categories=row_data.get("categories", [row_data.get("category", "unknown")]),
            out_dim=CFG.meta_dim,
        )
        meta_feat = meta_builder.build(row_data)

        inputs = {
            key: torch.tensor(features[key], dtype=torch.float32).unsqueeze(0).to(self.device)
            for key in [
                "text_feat",
                "video_feat",
                "audio_feat",
                "aes_feat",
                "dnsmos_feat",
                "speech_rhythm_feat",
                "sci_hand_feat",
                "llm_knowledge_feat",
                "llm_analysis_feat",
            ]
        }
        inputs["meta_feat"] = torch.tensor(meta_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
        inputs["wpm"] = torch.tensor([features["wpm"]], dtype=torch.float32).unsqueeze(0).to(self.device)
        inputs = apply_science_feature_mode(
            inputs,
            str(self.runtime_config["science_feature_mode"]),
            bool(self.runtime_config["use_audio_feature"]),
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probability = float(outputs["probability"].item())
        return {
            "scientific_score": float(outputs["scientific_score"].item()),
            "technical_score": float(outputs["technical_score"].item()),
            "aesthetic_score": float(outputs["aesthetic_score"].item()),
            "overall_score": float(outputs["overall_score"].item()),
            "probability": probability,
            "threshold": self.threshold,
            "prediction": "top" if probability >= self.threshold else "not_top",
        }

    def full_inference(self, video_path: str, row_data: dict) -> dict:
        features = self.infer_from_video(video_path, row_data)
        result = self.predict(features, row_data)
        result["video_id"] = row_data.get("video_id", "unknown")
        result["timestamp"] = datetime.now().isoformat()
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Science video quality inference")
    parser.add_argument("--mode", choices=["video"], default="video", help="Inference mode")
    parser.add_argument("--video", help="Video file path")
    parser.add_argument("--checkpoint", default=str(CFG.checkpoint_dir / "best.pt"), help="Checkpoint path")
    parser.add_argument("--video_id", help="Video ID")
    parser.add_argument("--title", default="", help="Video title")
    parser.add_argument("--tags", default="", help="Tags separated by |")
    parser.add_argument("--category", default="unknown", help="Video category")
    parser.add_argument("--duration", type=float, default=0.0, help="Video duration")
    parser.add_argument("--verified", type=int, default=0, help="Verified flag")
    parser.add_argument("--publish_time", default="", help="Publish time")
    parser.add_argument("--output", default="", help="Output JSON path")
    parser.add_argument("--llm-online-enroll", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.video or not args.checkpoint:
        print("Error: --video and --checkpoint are required")
        return

    engine = InferenceEngine(
        args.checkpoint,
        allow_llm_online_enroll=args.llm_online_enroll,
    )
    row_data = {
        "video_id": args.video_id or get_video_id(args.video),
        "title": args.title,
        "tags": args.tags,
        "category": args.category,
        "duration": args.duration,
        "verified": args.verified,
        "publish_time": args.publish_time,
        "label": 0,
        "categories": [args.category],
    }

    print("Starting inference...")
    result = engine.full_inference(args.video, row_data)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        save_json(result, args.output)
        print(f"Result saved to: {args.output}")


if __name__ == "__main__":
    main()
