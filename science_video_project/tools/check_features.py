from pathlib import Path
import torch


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "outputs" / "features"
    expected = {
        "text_feat": 768,
        "video_feat": 512,
        "audio_feat": 384,
        "meta_feat": 16,
    }

    files = sorted(root.glob("*.pt"))
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

    print(f"Checked {len(files)} files")
    if bad:
        print("Mismatches:")
        for item in bad:
            print(" -", item)
    else:
        print("All feature shapes OK")


if __name__ == "__main__":
    main()
