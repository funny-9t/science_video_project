"""Populate the DeepSeek scientific-audit cache concurrently."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import CFG
from pipeline.step_llm_knowledge import LLMKnowledgeExtractor
from pipeline.step_speech_features import load_transcript


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--metadata", type=Path, default=CFG.metadata_csv)
    parser.add_argument("--feature-dir", type=Path, default=CFG.feature_dir)
    parser.add_argument("--transcript-dir", type=Path, default=CFG.transcript_dir)
    parser.add_argument("--model", default=CFG.llm_model_name)
    parser.add_argument(
        "--video-id",
        action="append",
        dest="video_ids",
        help="Only prefetch the specified video ID; repeat for multiple samples.",
    )
    parser.add_argument("--report", type=Path, default=CFG.log_dir / "llm_prefetch_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY must be set.")

    metadata = pd.read_csv(args.metadata, dtype={"video_id": str}).fillna("")
    row_map = {str(row["video_id"]): row.to_dict() for _, row in metadata.iterrows()}
    available_ids = {path.stem for path in args.feature_dir.glob("*.pt")}
    if args.video_ids:
        requested_ids = {str(video_id).strip() for video_id in args.video_ids}
        missing_ids = sorted(requested_ids - available_ids)
        if missing_ids:
            raise FileNotFoundError(f"Feature files not found for video IDs: {missing_ids}")
        video_ids = sorted(requested_ids)
    else:
        video_ids = sorted(available_ids)
    extractor = LLMKnowledgeExtractor(
        api_key=api_key,
        model=args.model,
        api_base=CFG.llm_api_base,
        cache_dir=CFG.llm_cache_dir,
    )

    tasks = []
    for video_id in video_ids:
        row = row_map.get(video_id, {})
        transcript = load_transcript(args.transcript_dir / f"{video_id}.json")
        text = "\n".join(
            part for part in [
                str(row.get("title", "") or "").strip(),
                str(row.get("tags", "") or "").strip(),
                transcript.text.strip() if transcript is not None else "",
            ] if part
        )
        tasks.append((video_id, text))

    counters = {"valid": 0, "invalid": 0, "errors": 0}
    invalid_video_ids = []
    errors = []

    def run_one(item: tuple[str, str]):
        video_id, text = item
        return video_id, extractor.extract_details(text)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run_one, item): item[0] for item in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="DeepSeek cache"):
            video_id = futures[future]
            try:
                _, result = future.result()
                counters["valid" if result.is_valid else "invalid"] += 1
                if not result.is_valid:
                    invalid_video_ids.append(video_id)
            except Exception as exc:
                counters["errors"] += 1
                errors.append({"video_id": video_id, "error": str(exc)})

    report = {
        "model": args.model,
        "workers": args.workers,
        "counters": counters,
        "invalid_video_ids": sorted(invalid_video_ids),
        "errors": errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
