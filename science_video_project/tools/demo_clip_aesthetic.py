"""
Demo: CLIP prompt scoring for aesthetic evaluation.

Compares CLIP prompt scoring against the existing model's aesthetic_score.

Usage:
    python tools/demo_clip_aesthetic.py --video <path_to_video> --checkpoint <path_to_checkpoint>
    python tools/demo_clip_aesthetic.py --frame_dir <path_to_frames>  # standalone
"""

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.step_aesthetic_clip import CLIPAestheticScorer, DEFAULT_PROMPTS, QUICK_PROMPTS


def parse_args():
    parser = argparse.ArgumentParser(description="CLIP prompt aesthetic scoring demo")
    parser.add_argument("--frame_dir", type=str, default=None, help="Path to extracted frames")
    parser.add_argument("--video", type=str, default=None, help="Path to video (will extract frames)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional model checkpoint for comparison")
    parser.add_argument("--quick", action="store_true", help="Use quick prompt set (2 dims)")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve frame dir
    if args.frame_dir:
        frame_dir = Path(args.frame_dir)
    elif args.video:
        from pipeline.step_extract import VideoExtractor
        from pipeline.config import CFG

        video_path = Path(args.video)
        video_id = video_path.stem
        frame_dir = CFG.frame_dir / video_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        extractor = VideoExtractor()
        extractor.extract_frames(video_path, frame_dir, fps=CFG.frame_fps)
        print(f"Extracted frames to {frame_dir}")
    else:
        print("Please provide --frame_dir or --video")
        sys.exit(1)

    # Init scorer
    prompts = QUICK_PROMPTS if args.quick else DEFAULT_PROMPTS
    scorer = CLIPAestheticScorer(prompts=prompts, device="cuda")

    # Score
    result = scorer.score_frames_batched(frame_dir)

    # Pretty print
    print("\n" + "=" * 60)
    print("  CLIP Prompt Aesthetic Scoring Results")
    print("=" * 60)
    print(f"  Frames evaluated:  {result['num_frames']}")
    print(f"  Overall aesthetic: {result['aesthetic_score']:.4f}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Dimension breakdown:")
    for name, score in result["dimensions"].items():
        bar_len = int((score + 2) / 4 * 30)  # score ∈ [-2, 2] → [0, 30] chars
        bar = "█" * max(0, min(bar_len, 30))
        print(f"    {name:20s}  {score:+.4f}  {bar}")
    print("=" * 60)

    # Comparison with existing model (if checkpoint provided)
    if args.checkpoint and args.video:
        print("\n[Comparison with trained model]")
        from inference.infer import main as infer_main
        import subprocess
        # Run existing inference
        result_path = str(Path(args.output or "tmp_infer_result.json"))
        subprocess.run([
            sys.executable, str(PROJECT_ROOT / "inference" / "infer.py"),
            "--video", args.video,
            "--checkpoint", args.checkpoint,
            "--output", result_path,
        ])
        with open(result_path) as f:
            model_result = json.load(f)
        print(f"  Model aesthetic_score:  {model_result.get('aesthetic_score', 'N/A')}")
        print(f"  CLIP aesthetic_score:   {result['aesthetic_score']:.4f}")

    # Save
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
