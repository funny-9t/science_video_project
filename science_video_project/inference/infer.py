import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import CFG
from pipeline.utils_io import get_video_id
from training.model_mvp import MultiModalQualityModel
from training.utils_train import (
    adapt_checkpoint_state_dict,
    build_model_config_from_cfg,
    get_device,
    merge_checkpoint_model_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for a single science short video")
    parser.add_argument("--video", required=True, help="Path to one video file")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint, e.g. outputs/checkpoints/best.pt")
    parser.add_argument("--metadata", default=str(CFG.metadata_csv), help="Metadata csv path")
    parser.add_argument("--output", default="", help="Optional output json path")
    parser.add_argument(
        "--llm-online-enroll",
        action="store_true",
        help="Allow one DeepSeek request when this video's cache entry is missing.",
    )
    return parser.parse_args()


def _compute_speech_rhythm(segments: list, total_duration: float) -> np.ndarray:
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
        seg_wpms.append(char_count / (seg_dur / 60.0))
        total_speech_time += seg_dur
        if i > 0:
            gap = start - segments[i - 1][1]
            if gap > 0.5:
                total_pause_time += gap

    if not seg_wpms:
        return np.zeros(6, dtype=np.float32)

    seg_wpms = np.array(seg_wpms, dtype=np.float32)
    return np.array(
        [
            float(np.mean(seg_wpms)),
            float(np.std(seg_wpms)),
            float(np.min(seg_wpms)),
            float(np.max(seg_wpms)),
            total_pause_time / total_duration,
            total_speech_time / total_duration,
        ],
        dtype=np.float32,
    )


def _validate_model_path(model_path: str, label: str) -> str:
    path = Path(model_path)
    if path.exists() or not path.is_absolute():
        return model_path
    raise FileNotFoundError(f"{label} model path not found: {path}")


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[MultiModalQualityModel, float, dict[str, object]]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = merge_checkpoint_model_config(ckpt, build_model_config_from_cfg(CFG))
    model = MultiModalQualityModel(**model_config).to(device)

    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    adapted = adapt_checkpoint_state_dict(state_dict, model.state_dict())
    missing_keys, unexpected_keys = model.load_state_dict(adapted, strict=False)
    if unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected_keys[:5]}")
    if missing_keys:
        missing_preview = ", ".join(missing_keys[:5])
        print(f"[load_model] Missing keys initialized from model defaults: {missing_preview}")

    model.eval()
    threshold = float(ckpt.get("best_threshold", CFG.threshold)) if isinstance(ckpt, dict) else float(CFG.threshold)
    runtime_config = {
        "science_feature_mode": str(ckpt.get("science_feature_mode", "full_ifg")),
        "llm_text_source": str(
            ckpt.get(
                "llm_text_source",
                "reasoning_and_analysis" if CFG.llm_include_reasoning else "analysis",
            )
        ),
        "use_audio_feature": bool(ckpt.get("use_audio_feature", True)),
        "use_cover_features": bool(model_config.get("use_cover_features", False)),
        "technical_feature_mode": str(model_config.get("technical_feature_mode", "full")),
        "aesthetic_feature_backend": str(
            ckpt.get("training_config", {}).get("aesthetic_feature_backend", "legacy")
        ),
        "video_dim": int(model_config.get("video_dim", CFG.video_dim)),
    } if isinstance(ckpt, dict) else {
        "science_feature_mode": "full_ifg",
        "llm_text_source": "reasoning_and_analysis" if CFG.llm_include_reasoning else "analysis",
        "use_audio_feature": True,
        "use_cover_features": False,
        "technical_feature_mode": "full",
        "aesthetic_feature_backend": "legacy",
        "video_dim": CFG.video_dim,
    }
    return model, threshold, runtime_config


def apply_science_feature_mode(
    inputs: dict[str, torch.Tensor],
    mode: str,
    use_audio_feature: bool = True,
) -> dict[str, torch.Tensor]:
    if mode in {"none", "scores"}:
        inputs["llm_analysis_feat"] = torch.zeros_like(inputs["llm_analysis_feat"])
    if mode in {"none", "analysis_concat"}:
        inputs["llm_knowledge_feat"] = torch.zeros_like(inputs["llm_knowledge_feat"])
    if mode != "full_ifg":
        inputs["sci_hand_feat"] = torch.zeros_like(inputs["sci_hand_feat"])
    if not use_audio_feature:
        inputs["audio_feat"] = torch.zeros_like(inputs["audio_feat"])
    return inputs


def _build_llm_extractor(allow_online_enroll: bool = False):
    from pipeline.step_llm_knowledge import LLMKnowledgeExtractor

    api_key = CFG.llm_api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if allow_online_enroll and not api_key:
        raise RuntimeError("--llm-online-enroll requires DEEPSEEK_API_KEY.")
    return LLMKnowledgeExtractor(
        api_key=api_key if allow_online_enroll else None,
        model=CFG.llm_model_name,
        api_base=CFG.llm_api_base,
        cache_dir=CFG.llm_cache_dir,
        temperature=CFG.llm_temperature,
        cache_policy="prefer" if allow_online_enroll else "require",
    )


def main() -> None:
    from pipeline.build_sample import build_sample
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
    from pipeline.utils_io import ensure_dir, load_metadata, save_json

    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    device = get_device(CFG.device)
    model, threshold, runtime_config = load_model(args.checkpoint, device)

    ensure_dir(CFG.audio_dir)
    ensure_dir(CFG.frame_dir)

    video_id = get_video_id(video_path)
    metadata = load_metadata(args.metadata)
    categories = metadata["category"].tolist()
    row_df = metadata[metadata["video_id"].astype(str) == video_id]
    row = row_df.iloc[0].to_dict() if len(row_df) > 0 else {
        "video_id": video_id,
        "label": 0,
        "category": "unknown",
        "title": "",
        "tags": "",
        "duration": 0,
        "verified": 0,
        "publish_time": "",
    }

    cover_feat = torch.zeros(CFG.cover_dim, dtype=torch.float32)
    if runtime_config["use_cover_features"]:
        from pipeline.step_cover import COVERFeatureExtractor

        cover_extractor = COVERFeatureExtractor(CFG.cover_root, device=CFG.device)
        cover_feat = cover_extractor.score(video_path)
        del cover_extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    asr_path = _validate_model_path(CFG.asr_model_path, "ASR")
    audio_path = _validate_model_path(CFG.audio_model_path, "Audio")

    extractor = VideoExtractor(audio_sr=CFG.audio_sr)
    video_encoder = VideoEncoder(CFG.clip_model_name, CFG.device)
    meta_builder = MetaFeatureBuilder(categories=categories + [row["category"]], out_dim=CFG.meta_dim)
    if runtime_config["aesthetic_feature_backend"] == "shared_clip":
        aes_scorer = SharedCLIPAestheticScorer(
            CFG.clip_model_name, prompts=DEFAULT_PROMPTS, device=CFG.device
        )
    else:
        aes_scorer = CLIPAestheticScorer(prompts=DEFAULT_PROMPTS, device=CFG.device)
    text_encoder = TextEncoder(CFG.text_model_name, asr_path, CFG.device, CFG.local_files_only)
    audio_encoder = AudioEncoder(audio_path, CFG.device, out_dim=CFG.audio_dim, shared_model=text_encoder.asr)
    use_dnsmos = runtime_config["technical_feature_mode"] in {
        "full", "visual_dnsmos", "dnsmos_only"
    }
    dnsmos_scorer = DNSMOSScorer(device=CFG.device) if use_dnsmos else None
    sci_hand_extractor = ScienceFeatureExtractor()
    llm_extractor = _build_llm_extractor(args.llm_online_enroll)

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
    frame_features = video_encoder.encode_frame_features(
        frame_path, max_frames=CFG.clip_max_frames
    )
    clip_video_feat = frame_features.mean(dim=0)
    video_feat = (
        clip_video_feat
        if runtime_config["video_dim"] == CFG.clip_video_dim
        else video_encoder.legacy_project(clip_video_feat, output_dim=runtime_config["video_dim"])
    )
    audio_feat = audio_encoder.encode(str(wav_path))
    meta_feat = meta_builder.build(row)

    aes_result = (
        aes_scorer.score_frame_features(frame_features)
        if runtime_config["aesthetic_feature_backend"] == "shared_clip"
        else aes_scorer.score_frames_batched(frame_path)
    )
    aes_feat = torch.tensor(
        [aes_result["dimensions"][n] for n in aes_result["dimensions"].keys()],
        dtype=torch.float32,
    )
    dnsmos_feat = (
        dnsmos_scorer.score(str(wav_path))
        if dnsmos_scorer is not None
        else torch.zeros(CFG.dnsmos_dim, dtype=torch.float32)
    )

    subtitle_text = text_result.subtitle or ""
    audio_duration = 0.0
    try:
        import soundfile as sf

        audio_duration = sf.info(str(wav_path)).duration
    except Exception:
        pass
    wpm_raw, speech_rhythm_raw_feat = compute_speech_features(text_result.segments, audio_duration)
    wpm, speech_rhythm_feat = normalize_speech_features(wpm_raw, speech_rhythm_raw_feat)
    sci_hand_feat = sci_hand_extractor.extract_vector(subtitle_text)
    science_text = "\n".join(
        part for part in [
            str(row.get("title", "") or "").strip(),
            str(row.get("tags", "") or "").strip(),
            subtitle_text,
        ] if part
    )
    llm_details = llm_extractor.extract_details(science_text, verbose=False)
    llm_knowledge_feat = llm_details.scores
    llm_analysis_feat = text_encoder.text_embedding(
        llm_details.feature_text(
            include_reasoning=runtime_config["llm_text_source"] == "reasoning_and_analysis"
        )
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
        cover_feat=cover_feat,
        frame_features=frame_features,
    )

    inputs = {
        key: torch.tensor(sample[key], dtype=torch.float32).unsqueeze(0).to(device)
        for key in [
            "text_feat",
            "video_feat",
            "audio_feat",
            "meta_feat",
            "aes_feat",
            "dnsmos_feat",
            "speech_rhythm_feat",
            "sci_hand_feat",
            "llm_knowledge_feat",
            "llm_analysis_feat",
            "cover_feat",
        ]
    }
    inputs["wpm"] = torch.tensor([sample["wpm"]], dtype=torch.float32).unsqueeze(0).to(device)
    inputs = apply_science_feature_mode(
        inputs,
        str(runtime_config["science_feature_mode"]),
        bool(runtime_config["use_audio_feature"]),
    )

    with torch.no_grad():
        out = model(**inputs)

    probability = float(out["probability"].item())
    result = {
        "scientific_score": float(out["scientific_score"].item()),
        "technical_score": float(out["technical_score"].item()),
        "aesthetic_score": float(out["aesthetic_score"].item()),
        "overall_score": float(out["overall_score"].item()),
        "probability": probability,
        "threshold": threshold,
        "prediction": "top" if probability >= threshold else "not_top",
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        save_json(result, args.output)


if __name__ == "__main__":
    main()
