"""Incrementally add audio, speech, DNSMOS, and DeepSeek features to existing .pt files."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import CFG
from pipeline.step_audio import AudioEncoder
from pipeline.step_dnsmos import DNSMOSScorer
from pipeline.step_llm_knowledge import (
    LLMKnowledgeExtractor,
    PROMPT_VERSION,
    RobertaAnalysisEncoder,
)
from pipeline.step_science_features import ScienceFeatureExtractor
from pipeline.step_speech_features import (
    SpeechTranscriber,
    compute_speech_features,
    load_transcript,
    normalize_speech_features,
    save_transcript,
)
from pipeline.step_video import VideoEncoder
from pipeline.utils_io import get_video_id, list_videos, load_metadata


VALID_FEATURES = {"audio", "speech", "dnsmos", "llm", "clip", "cover"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", default="speech,dnsmos",
        help="Comma-separated: audio,speech,dnsmos,llm,clip,cover",
    )
    parser.add_argument("--feature-dir", type=Path, default=CFG.feature_dir)
    parser.add_argument("--audio-dir", type=Path, default=CFG.audio_dir)
    parser.add_argument("--transcript-dir", type=Path, default=CFG.transcript_dir)
    parser.add_argument("--frame-dir", type=Path, default=CFG.frame_dir)
    parser.add_argument("--video-dir", type=Path, default=CFG.video_dir)
    parser.add_argument("--cover-root", type=Path, default=CFG.cover_root)
    parser.add_argument("--metadata", type=Path, default=CFG.metadata_csv)
    parser.add_argument("--asr-model", type=Path, default=Path(CFG.asr_model_path))
    parser.add_argument("--asr-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--asr-batch-size", type=int, default=8)
    parser.add_argument("--audio-max-segments", type=int, default=8)
    parser.add_argument("--dnsmos-max-segments", type=int, default=8)
    parser.add_argument("--llm-model", default=CFG.llm_model_name)
    parser.add_argument("--llm-text-source", choices=["analysis", "reasoning_and_analysis"], default="reasoning_and_analysis")
    parser.add_argument("--llm-cache-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--video-id",
        action="append",
        dest="video_ids",
        help="Process an exact video ID; repeat to process multiple samples.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute selected feature fields")
    parser.add_argument("--force-transcripts", action="store_true")
    parser.add_argument("--report", type=Path, default=CFG.log_dir / "feature_backfill_report.json")
    return parser.parse_args()


def parse_features(raw: str) -> set[str]:
    features = {item.strip().lower() for item in raw.split(",") if item.strip()}
    unknown = features - VALID_FEATURES
    if unknown or not features:
        raise ValueError(f"Invalid features {sorted(unknown)}; choose from {sorted(VALID_FEATURES)}")
    return features


def save_feature_file(sample: dict, path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(sample, temp_path)
    temp_path.replace(path)


def needs_field(sample: dict, key: str, force: bool) -> bool:
    return force or key not in sample


def build_science_text(row: dict, transcript: str) -> str:
    parts = [
        str(row.get("title", "") or "").strip(),
        str(row.get("tags", "") or "").strip(),
        (transcript or "").strip(),
    ]
    return "\n".join(part for part in parts if part)


def main() -> None:
    args = parse_args()
    selected = parse_features(args.features)
    args.transcript_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata, dtype={"video_id": str}).fillna("")
    row_map = {str(row["video_id"]): row.to_dict() for _, row in metadata.iterrows()}
    def audio_size(path: Path) -> float:
        audio_path = args.audio_dir / f"{path.stem}.wav"
        size = audio_path.stat().st_size if audio_path.exists() else 0
        return float(size) if size > 0 else float("inf")

    feature_paths = sorted(args.feature_dir.glob("*.pt"), key=audio_size)
    if "cover" in selected:
        effective_ids = set(load_metadata(args.metadata)["video_id"].astype(str))
        feature_paths = [path for path in feature_paths if path.stem in effective_ids]
    if args.video_ids:
        requested_ids = {str(video_id).strip() for video_id in args.video_ids}
        available_ids = {path.stem for path in feature_paths}
        missing_ids = sorted(requested_ids - available_ids)
        if missing_ids:
            raise FileNotFoundError(f"Feature files not found for video IDs: {missing_ids}")
        feature_paths = [path for path in feature_paths if path.stem in requested_ids]
    if args.limit > 0:
        feature_paths = feature_paths[: args.limit]

    transcriber = None

    dnsmos = None
    if "dnsmos" in selected:
        dnsmos = DNSMOSScorer(max_segments=args.dnsmos_max_segments)

    audio_encoder = None
    if "audio" in selected:
        audio_encoder = AudioEncoder(
            str(args.asr_model),
            device=args.asr_device,
            out_dim=CFG.audio_dim,
            max_segments=args.audio_max_segments,
        )

    llm = None
    analysis_encoder = None
    if "llm" in selected:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key and not args.llm_cache_only:
            raise RuntimeError("DEEPSEEK_API_KEY must be set for final LLM feature generation.")
        llm = LLMKnowledgeExtractor(
            api_key=api_key or None,
            model=args.llm_model,
            api_base=CFG.llm_api_base,
            cache_dir=CFG.llm_cache_dir,
        )
        analysis_encoder = RobertaAnalysisEncoder(CFG.text_model_name, device=CFG.device)

    video_encoder = None
    if "clip" in selected:
        video_encoder = VideoEncoder(CFG.clip_model_name, CFG.device)

    cover_extractor = None
    video_map = {}
    if "cover" in selected:
        from pipeline.step_cover import COVERFeatureExtractor

        cover_extractor = COVERFeatureExtractor(args.cover_root, device=CFG.device)
        video_map = {get_video_id(path): path for path in list_videos(args.video_dir)}

    science_extractor = ScienceFeatureExtractor()
    counters = {"updated": 0, "unchanged": 0, "missing_audio": 0, "errors": 0}
    errors = []
    started = time.time()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) if dnsmos is not None else None
    progress = tqdm(feature_paths, desc="Backfilling features")
    for feature_path in progress:
        video_id = feature_path.stem
        wav_path = args.audio_dir / f"{video_id}.wav"
        needs_audio = bool(selected & {"audio", "speech", "dnsmos", "llm"})
        if needs_audio and (not wav_path.exists() or wav_path.stat().st_size == 0):
            counters["missing_audio"] += 1
            errors.append({"video_id": video_id, "error": "audio missing or empty"})
            continue

        sample = torch.load(feature_path, map_location="cpu", weights_only=False)
        row = row_map.get(video_id, {})
        changed = False
        try:
            if "cover" in selected and (
                args.force
                or "cover_feat" not in sample
                or sample.get("cover_feature_version") != cover_extractor.FEATURE_VERSION
            ):
                video_path = video_map.get(video_id)
                if video_path is None:
                    counters.setdefault("missing_video", 0)
                    counters["missing_video"] += 1
                    raise FileNotFoundError(f"source video missing for id: {video_id}")
                sample["cover_feat"] = cover_extractor.score(video_path).numpy()
                sample["cover_feature_version"] = cover_extractor.FEATURE_VERSION
                changed = True

            if "clip" in selected:
                frame_dir = args.frame_dir / video_id
                if not frame_dir.exists() or not any(frame_dir.glob("*.jpg")):
                    counters.setdefault("missing_frames", 0)
                    counters["missing_frames"] += 1
                    raise FileNotFoundError(f"frame images missing: {frame_dir}")
                existing_frames = np.asarray(sample.get("frame_features", []))
                clip_missing = (
                    args.force
                    or "clip_video_feat" not in sample
                    or existing_frames.ndim != 2
                    or existing_frames.shape[0] == 0
                    or sample.get("clip_sampling_version")
                    != f"global_uniform_{CFG.clip_max_frames}_v1"
                )
                if clip_missing:
                    frame_features = video_encoder.encode_frame_features(
                        frame_dir, max_frames=CFG.clip_max_frames
                    )
                    sample["frame_features"] = frame_features.numpy()
                    sample["clip_video_feat"] = frame_features.mean(dim=0).numpy()
                    sample["clip_sampling_version"] = (
                        f"global_uniform_{CFG.clip_max_frames}_v1"
                    )
                    changed = True

            if "audio" in selected and needs_field(sample, "audio_feat", args.force):
                sample["audio_feat"] = audio_encoder.encode(str(wav_path)).numpy()
                changed = True

            dnsmos_future = None
            if "dnsmos" in selected and needs_field(sample, "dnsmos_feat", args.force):
                dnsmos_future = executor.submit(dnsmos.score, wav_path)

            transcript_path = args.transcript_dir / f"{video_id}.json"
            transcript = None if args.force_transcripts else load_transcript(transcript_path)
            needs_transcript = (
                ("speech" in selected and needs_field(sample, "speech_rhythm_feat", args.force))
                or ("llm" in selected and needs_field(sample, "llm_analysis_feat", args.force))
            )
            if transcript is None and needs_transcript:
                if transcriber is None:
                    transcriber = SpeechTranscriber(
                        args.asr_model,
                        device=args.asr_device,
                        batch_size=args.asr_batch_size,
                    )
                transcript = transcriber.transcribe(wav_path)
                save_transcript(transcript, transcript_path)

            if "speech" in selected and needs_field(sample, "speech_rhythm_feat", args.force):
                duration = float(sf.info(wav_path).duration)
                wpm_raw, rhythm_raw = compute_speech_features(transcript.segments, duration)
                wpm, rhythm = normalize_speech_features(wpm_raw, rhythm_raw)
                sample["wpm"] = wpm
                sample["wpm_raw"] = wpm_raw
                sample["speech_rhythm_feat"] = rhythm
                sample["speech_rhythm_raw_feat"] = rhythm_raw
                sample["sci_hand_feat"] = science_extractor.extract_vector(transcript.text).numpy()
                changed = True

            if dnsmos_future is not None:
                sample["dnsmos_feat"] = dnsmos_future.result().numpy()
                changed = True

            if "llm" in selected and needs_field(sample, "llm_analysis_feat", args.force):
                science_text = build_science_text(row, transcript.text)
                details = llm.extract_details(science_text)
                if not details.is_valid:
                    raise ValueError("DeepSeek returned an invalid structured response")
                analysis_only_feat = analysis_encoder.encode(
                    details.feature_text(include_reasoning=False)
                )
                reasoning_analysis_feat = analysis_encoder.encode(
                    details.feature_text(include_reasoning=True)
                )
                include_reasoning = args.llm_text_source == "reasoning_and_analysis"
                sample["llm_knowledge_feat"] = details.scores
                sample["llm_analysis_only_feat"] = analysis_only_feat
                sample["llm_reasoning_analysis_feat"] = reasoning_analysis_feat
                sample["llm_analysis_feat"] = (
                    reasoning_analysis_feat if include_reasoning else analysis_only_feat
                )
                sample["llm_model"] = details.model
                sample["llm_prompt_version"] = PROMPT_VERSION
                sample["llm_text_source"] = args.llm_text_source
                changed = True

            if changed:
                save_feature_file(sample, feature_path)
                counters["updated"] += 1
            else:
                counters["unchanged"] += 1
        except Exception as exc:
            counters["errors"] += 1
            errors.append({"video_id": video_id, "error": str(exc)})

        progress.set_postfix(updated=counters["updated"], errors=counters["errors"])

    if executor is not None:
        executor.shutdown(wait=True)

    report = {
        "selected_features": sorted(selected),
        "files_considered": len(feature_paths),
        "elapsed_seconds": time.time() - started,
        "counters": counters,
        "errors": errors,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
