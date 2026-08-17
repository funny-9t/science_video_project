import sys
from pathlib import Path
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import CFG
from pipeline.step_cover import COVERFeatureExtractor
from pipeline.utils_io import load_metadata


def main() -> None:
    root = PROJECT_ROOT / "outputs" / "features"
    expected = {
        "text_feat": 768,
        "video_feat": 512,
        "clip_video_feat": 768,
        "audio_feat": 384,
        "meta_feat": 16,
        "aes_feat": 7,
        "dnsmos_feat": 3,
        "speech_rhythm_feat": 6,
        "sci_hand_feat": 5,
        "llm_knowledge_feat": 4,
        "llm_analysis_feat": 768,
        "cover_feat": 3,
    }

    effective_ids = set(load_metadata(CFG.metadata_csv)["video_id"].astype(str))
    files = sorted(path for path in root.glob("*.pt") if path.stem in effective_ids)
    bad = []
    for path in files:
        sample = torch.load(path, map_location="cpu", weights_only=False)
        for key, dim in expected.items():
            value = sample.get(key)
            if value is None:
                bad.append((path.name, key, "missing"))
                continue
            shape = getattr(value, "shape", None)
            if shape is None:
                bad.append((path.name, key, "no-shape"))
                continue
            got = shape[-1]
            if got != dim:
                bad.append((path.name, key, f"{got} != {dim}"))
            elif not np.isfinite(np.asarray(value)).all():
                bad.append((path.name, key, "contains non-finite values"))
        cover_version = sample.get("cover_feature_version", "")
        if cover_version != COVERFeatureExtractor.FEATURE_VERSION:
            bad.append((path.name, "cover_feature_version", repr(cover_version)))
        frame_features = sample.get("frame_features")
        if frame_features is None or getattr(frame_features, "ndim", 0) != 2:
            bad.append((path.name, "frame_features", "missing or not 2D"))
        elif frame_features.shape[-1] != 768 or not (1 <= frame_features.shape[0] <= 40):
            bad.append((path.name, "frame_features", str(tuple(frame_features.shape))))

    print(f"Checked {len(files)} trainable feature files from {len(effective_ids)} metadata IDs")
    if bad:
        print("Mismatches:")
        for item in bad:
            print(" -", item)
    else:
        print("All feature shapes OK")


if __name__ == "__main__":
    main()
