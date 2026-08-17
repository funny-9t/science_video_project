import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import CFG
from pipeline.utils_io import load_metadata


FEATURE_GROUPS = {
    "text": ["text_feat"],
    "video": ["video_feat"],
    "native_clip": ["clip_video_feat"],
    "cover": ["cover_feat"],
    "audio": ["audio_feat"],
    "meta": ["meta_feat"],
    "aesthetic": ["aes_feat"],
    "dnsmos": ["dnsmos_feat"],
    "speech": ["wpm", "speech_rhythm_feat"],
    "llm_scores": ["llm_knowledge_feat"],
    "llm_analysis_only": ["llm_analysis_only_feat"],
    "llm_reasoning_analysis": ["llm_reasoning_analysis_feat"],
    "science": ["sci_hand_feat", "llm_knowledge_feat", "llm_analysis_feat"],
    "all": [
        "text_feat",
        "video_feat",
        "clip_video_feat",
        "cover_feat",
        "audio_feat",
        "meta_feat",
        "aes_feat",
        "dnsmos_feat",
        "wpm",
        "speech_rhythm_feat",
        "sci_hand_feat",
        "llm_knowledge_feat",
        "llm_analysis_feat",
    ],
}

DEFAULT_FEATURES = {
    "text_feat": np.zeros(768, dtype=np.float32),
    "video_feat": np.zeros(512, dtype=np.float32),
    "clip_video_feat": np.zeros(768, dtype=np.float32),
    "cover_feat": np.zeros(3, dtype=np.float32),
    "audio_feat": np.zeros(384, dtype=np.float32),
    "meta_feat": np.zeros(16, dtype=np.float32),
    "aes_feat": np.zeros(7, dtype=np.float32),
    "dnsmos_feat": np.zeros(3, dtype=np.float32),
    "speech_rhythm_feat": np.zeros(6, dtype=np.float32),
    "sci_hand_feat": np.zeros(5, dtype=np.float32),
    "llm_knowledge_feat": np.zeros(4, dtype=np.float32),
    "llm_analysis_feat": np.zeros(768, dtype=np.float32),
    "llm_analysis_only_feat": np.zeros(768, dtype=np.float32),
    "llm_reasoning_analysis_feat": np.zeros(768, dtype=np.float32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose extracted features and simple baselines.")
    parser.add_argument("--metadata", default=str(CFG.metadata_csv))
    parser.add_argument("--feature_dir", default=str(CFG.feature_dir))
    parser.add_argument("--output", default=str(CFG.log_dir / "feature_diagnostics.json"))
    return parser.parse_args()


def _as_1d_array(value, default: np.ndarray | None = None) -> np.ndarray:
    if value is None:
        return np.array(default, dtype=np.float32) if default is not None else np.zeros(0, dtype=np.float32)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def _load_dataset(metadata_path: str, feature_dir: str) -> tuple[pd.DataFrame, dict[str, dict]]:
    metadata = load_metadata(metadata_path)
    feature_dir = Path(feature_dir)
    samples: dict[str, dict] = {}
    rows = []
    for _, row in metadata.iterrows():
        video_id = str(row["video_id"])
        pt_path = feature_dir / f"{video_id}.pt"
        if not pt_path.exists():
            continue
        sample = torch.load(pt_path, map_location="cpu", weights_only=False)
        samples[video_id] = sample
        rows.append(row)
    if not rows:
        raise ValueError("No metadata rows have matching feature files.")
    return pd.DataFrame(rows).reset_index(drop=True), samples


def _feature_matrix(df: pd.DataFrame, samples: dict[str, dict], keys: list[str]) -> np.ndarray:
    vectors = []
    for _, row in df.iterrows():
        sample = samples[str(row["video_id"])]
        parts = []
        for key in keys:
            if key == "wpm":
                parts.append(np.array([float(sample.get("wpm", 0.0))], dtype=np.float32))
            else:
                parts.append(_as_1d_array(sample.get(key), DEFAULT_FEATURES.get(key)))
        vectors.append(np.concatenate(parts, axis=0))
    return np.stack(vectors, axis=0)


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, score))
    except Exception:
        return 0.0


def _effect_summary(x: np.ndarray, y: np.ndarray) -> dict:
    pos = x[y == 1]
    neg = x[y == 0]
    pos_mean = pos.mean(axis=0)
    neg_mean = neg.mean(axis=0)
    pooled = np.sqrt((pos.var(axis=0) + neg.var(axis=0)) / 2.0) + 1e-8
    cohen = np.abs((pos_mean - neg_mean) / pooled)
    mean_score = x.mean(axis=1)
    auc = _safe_auc(y, mean_score)
    return {
        "dim": int(x.shape[1]),
        "mean_abs_cohen_d": float(np.mean(cohen)),
        "max_abs_cohen_d": float(np.max(cohen)),
        "mean_score_auc": auc,
        "mean_pos": float(mean_score[y == 1].mean()),
        "mean_neg": float(mean_score[y == 0].mean()),
    }


def _baseline_scores(x: np.ndarray, y: np.ndarray) -> dict:
    counts = np.bincount(y.astype(int))
    min_class = int(counts.min()) if counts.size == 2 else 0
    if min_class < 2:
        return {"error": "Need at least two samples per class for baseline."}

    n_splits = min(5, min_class)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CFG.seed)
    models = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=CFG.seed),
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            class_weight="balanced",
            random_state=CFG.seed,
        ),
    }
    out = {}
    for name, model in models.items():
        prob = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
        pred = (prob >= 0.5).astype(int)
        out[name] = {
            "auc": _safe_auc(y, prob),
            "accuracy": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, zero_division=0)),
        }
    return out


def _pair_stats(df: pd.DataFrame) -> dict:
    label_by_cat = df.groupby(["category", "label"]).size().unstack(fill_value=0)
    valid = label_by_cat[(label_by_cat.get(0, 0) > 0) & (label_by_cat.get(1, 0) > 0)]
    same_category_pairs = int((valid.get(0, 0) * valid.get(1, 0)).sum()) if not valid.empty else 0
    total_pos = int((df["label"] == 1).sum())
    total_neg = int((df["label"] == 0).sum())
    return {
        "num_categories": int(df["category"].nunique()),
        "valid_pair_categories": int(len(valid)),
        "same_category_pairs": same_category_pairs,
        "global_pairs": total_pos * total_neg,
        "top_categories": label_by_cat.sort_index().head(20).to_dict(orient="index"),
    }


def _feature_coverage(df: pd.DataFrame, samples: dict[str, dict]) -> dict:
    coverage = {}
    for key, default in DEFAULT_FEATURES.items():
        present = 0
        nonzero = 0
        for _, row in df.iterrows():
            sample = samples[str(row["video_id"])]
            if key not in sample:
                continue
            arr = _as_1d_array(sample.get(key), default)
            present += 1
            if arr.size > 0 and bool(np.any(np.abs(arr) > 1e-8)):
                nonzero += 1
        coverage[key] = {
            "present": present,
            "nonzero": nonzero,
            "present_ratio": float(present / max(len(df), 1)),
            "nonzero_ratio": float(nonzero / max(len(df), 1)),
        }
    wpm_nonzero = 0
    for _, row in df.iterrows():
        sample = samples[str(row["video_id"])]
        if abs(float(sample.get("wpm", 0.0))) > 1e-8:
            wpm_nonzero += 1
    coverage["wpm"] = {
        "present": int(sum("wpm" in samples[str(row["video_id"])] for _, row in df.iterrows())),
        "nonzero": int(wpm_nonzero),
        "present_ratio": float(sum("wpm" in samples[str(row["video_id"])] for _, row in df.iterrows()) / max(len(df), 1)),
        "nonzero_ratio": float(wpm_nonzero / max(len(df), 1)),
    }
    return coverage


def main() -> None:
    args = parse_args()
    df, samples = _load_dataset(args.metadata, args.feature_dir)
    y = df["label"].astype(int).to_numpy()

    report = {
        "num_samples": int(len(df)),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "pair_stats": _pair_stats(df),
        "feature_coverage": _feature_coverage(df, samples),
        "feature_groups": {},
        "baselines": {},
    }

    for group, keys in FEATURE_GROUPS.items():
        x = _feature_matrix(df, samples, keys)
        report["feature_groups"][group] = _effect_summary(x, y)
        if group in {
            "meta",
            "aesthetic",
            "cover",
            "dnsmos",
            "speech",
            "llm_scores",
            "llm_analysis_only",
            "llm_reasoning_analysis",
            "science",
            "all",
        }:
            report["baselines"][group] = _baseline_scores(x, y)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved diagnostics to: {output}")


if __name__ == "__main__":
    main()
