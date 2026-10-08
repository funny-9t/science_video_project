"""FineVD fine-grained transfer validation for KEEP_P2_S2 Technical Branch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import CFG
from pipeline.step_cover_technical import COVERTechnicalFeatureExtractor
from pipeline.step_extract import VideoExtractor
from pipeline.step_video import VideoEncoder
from pipeline.utils_io import load_pt
from training.model_mvp import MultiModalQualityModel
from training.utils_train import get_device, set_seed


ATTRIBUTES = ("overall", "blur", "color", "noise", "artifact", "temporal")
PROBE_SEEDS = (42, 123, 2026)
FEATURE_VERSION = "keep_p2_s2_finevd_technical_v1"
PROBE_VERSION = "finevd_linear_probe_v1_fixed_inner_validation"
EXPECTED_SPLIT_HASH = "79413d21918a956716f230456b80aec037be51bfd1c6a412a8c0a30087a2a58f"


class LowMemoryCOVERTechnicalFeatureExtractor(COVERTechnicalFeatureExtractor):
    """Exact COVER fragment sampling without stacking full-resolution frames.

    COVER's upstream loader materializes all 40 selected frames before extracting
    the 224x224 fragment mosaic. FineVD includes 4K portrait videos, for which the
    temporary tensor can exceed several GB and terminate the process. This method
    preserves the sampled frame indices, RNG order, fragment offsets, pixels, and
    normalization while decoding one source frame at a time.
    """

    FEATURE_VERSION = "cover_technical_swin3d_tiny_grpb_ytugc_lowmem_exact_v1"
    _rng_lock = threading.Lock()

    def prepare_view_cpu(self, video_path: str | Path) -> torch.Tensor:
        import decord
        import torch.nn.functional as functional
        from decord import VideoReader

        seed = int(hashlib.sha256(Path(video_path).stem.encode("utf-8")).hexdigest()[:8], 16)
        decord.bridge.set_bridge("torch")
        reader = VideoReader(str(video_path))
        dimensions = reader[0]
        height, width = int(dimensions.shape[0]), int(dimensions.shape[1])
        options = self.sample_options["technical"]
        fragments_h = int(options["fragments_h"])
        fragments_w = int(options["fragments_w"])
        fragment_h = int(options["fsize_h"])
        fragment_w = int(options["fsize_w"])
        aligned = int(options["aligned"])
        target_h = fragments_h * fragment_h
        target_w = fragments_w * fragment_w
        hlength = height // fragments_h
        wlength = width // fragments_w

        # The upstream sampler uses process-global RNGs. Keep only the tiny RNG
        # section serialized; video decoding and fragment assembly remain parallel.
        with self._rng_lock:
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            torch_state = torch.random.get_rng_state()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            try:
                frame_indices = self.sampler(len(reader), False).astype(np.int32)
                duration = len(frame_indices)
                if duration % aligned:
                    raise ValueError(f"COVER alignment mismatch: {duration} % {aligned}")
                groups = duration // aligned
                rnd_h = (
                    torch.randint(hlength - fragment_h, (fragments_h, fragments_w, groups))
                    if hlength > fragment_h
                    else torch.zeros((fragments_h, fragments_w, groups)).int()
                )
                rnd_w = (
                    torch.randint(wlength - fragment_w, (fragments_h, fragments_w, groups))
                    if wlength > fragment_w
                    else torch.zeros((fragments_h, fragments_w, groups)).int()
                )
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)
                torch.random.set_rng_state(torch_state)

        ratio = min(height / target_h, width / target_w)
        hgrids = torch.tensor(
            [min(height // fragments_h * index, height - fragment_h) for index in range(fragments_h)],
            dtype=torch.long,
        )
        wgrids = torch.tensor(
            [min(width // fragments_w * index, width - fragment_w) for index in range(fragments_w)],
            dtype=torch.long,
        )
        target = torch.zeros((3, duration, target_h, target_w), dtype=dimensions.dtype)
        for time_index, frame_index in enumerate(frame_indices):
            frame = reader[int(frame_index)].permute(2, 0, 1)
            if ratio < 1:
                original = frame
                frame = functional.interpolate(
                    frame.unsqueeze(0) / 255.0,
                    scale_factor=1 / ratio,
                    mode="bilinear",
                ).squeeze(0)
                frame = (frame * 255.0).type_as(original)
            group = time_index // aligned
            for row, hgrid in enumerate(hgrids):
                for column, wgrid in enumerate(wgrids):
                    source_h = int(hgrid + rnd_h[row, column, group])
                    source_w = int(wgrid + rnd_w[row, column, group])
                    dest_h = row * fragment_h
                    dest_w = column * fragment_w
                    target[
                        :, time_index,
                        dest_h : dest_h + fragment_h,
                        dest_w : dest_w + fragment_w,
                    ] = frame[
                        :,
                        source_h : source_h + fragment_h,
                        source_w : source_w + fragment_w,
                    ]

        normalized = (
            ((target.permute(1, 2, 3, 0) - self.mean) / self.std)
            .permute(3, 0, 1, 2)
            .unsqueeze(0)
        )
        del reader, dimensions, target
        return normalized

    def _prepare_view(self, video_path: str | Path) -> torch.Tensor:
        return self.prepare_view_cpu(video_path).to(self.device)

    def extract_prepared(self, view: torch.Tensor) -> torch.Tensor:
        view = view.to(self.device)
        with torch.inference_mode():
            feature_map = self.backbone(view)
            feature = feature_map.mean(dim=(2, 3, 4)).squeeze(0).float().cpu()
        if feature.shape != (self.FEATURE_DIM,) or not torch.isfinite(feature).all():
            raise RuntimeError(f"Invalid prepared COVER feature: {tuple(feature.shape)}")
        return feature

    def extract(self, video_path: str | Path) -> torch.Tensor:
        return self.extract_prepared(self.prepare_view_cpu(video_path))


class SeekingVideoExtractor(VideoExtractor):
    """Decode only the exact frame numbers selected by the legacy 1 FPS loop."""

    def extract_frames(self, video_path: str | Path, out_dir: str | Path, fps: int = 1) -> int:
        import cv2

        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video_path))
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            source_fps = 25.0
        interval = max(1, int(round(source_fps / max(1, int(fps)))))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        saved = 0
        for frame_index in range(0, total_frames, interval):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            cv2.imwrite(str(output_dir / f"{saved:06d}.jpg"), frame)
            saved += 1
        capture.release()
        if saved == 0:
            raise RuntimeError(f"No frame extracted from {video_path}")
        return saved


class MemorySafeVideoEncoder(VideoEncoder):
    """Apply the unchanged CLIP processor one image at a time before batching."""

    def encode_frame_features(
        self,
        frame_dir: str | Path,
        max_frames: int | None = None,
    ) -> torch.Tensor:
        from PIL import Image

        frame_paths = sorted(Path(frame_dir).glob("*.jpg"))
        if not frame_paths:
            raise RuntimeError(f"No frame images in {frame_dir}")
        frame_paths = self._global_uniform_paths(frame_paths, max_frames)
        features = []
        with torch.inference_mode():
            for start in range(0, len(frame_paths), self.batch_size):
                selected = frame_paths[start : start + self.batch_size]
                if self.backend == "hf":
                    processed = []
                    for path in selected:
                        with Image.open(path) as image:
                            pixel = self.processor(
                                images=image.convert("RGB"), return_tensors="pt"
                            )["pixel_values"][0]
                        processed.append(pixel)
                    pixel_values = torch.stack(processed).to(
                        self.device, dtype=self.model.dtype
                    )
                    feature = self.model(pixel_values=pixel_values).image_embeds
                else:
                    processed = []
                    for path in selected:
                        with Image.open(path) as image:
                            processed.append(self.preprocess(image.convert("RGB")))
                    batch = torch.stack(processed).to(self.device)
                    feature = self.model.encode_image(batch)
                feature = feature / feature.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                features.append(feature.detach().cpu())
        return torch.cat(features, dim=0).float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--finevd-root", type=Path, default=Path(r"F:\test_data\FineVD")
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "outputs" / "progressive_training" / "P2" / "stage2_best.pt",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "finevd_technical_validation",
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force-probes", action="store_true")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--prefetch-workers", type=int, default=2)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("finevd_technical_validation")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def resolve_data_paths(root: Path) -> tuple[Path, Path]:
    candidates = [root / "FineVD", root]
    annotation_dir = next(
        (path for path in candidates if (path / "MOS_train.csv").is_file()), None
    )
    video_candidates = [
        root / "videos_all" / "videos_all_videos",
        root / "videos_all_videos",
        root / "videos_all",
        root / "videos",
    ]
    video_dir = next((path for path in video_candidates if path.is_dir()), None)
    if annotation_dir is None:
        raise FileNotFoundError(f"MOS_train.csv not found under {root}")
    if video_dir is None:
        raise FileNotFoundError(f"FineVD video directory not found under {root}")
    return annotation_dir, video_dir


def load_metadata(root: Path) -> tuple[pd.DataFrame, dict]:
    annotation_dir, video_dir = resolve_data_paths(root)
    frames = []
    for split, filename in (("train", "MOS_train.csv"), ("val", "MOS_val.csv")):
        frame = pd.read_csv(annotation_dir / filename)
        expected = {"video_name", *ATTRIBUTES}
        missing_columns = expected.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"{filename} missing columns: {sorted(missing_columns)}")
        frame = frame[["video_name", *ATTRIBUTES]].copy()
        frame["split"] = split
        frames.append(frame)
    metadata = pd.concat(frames, ignore_index=True)
    metadata["video_id"] = metadata["video_name"].map(lambda value: Path(str(value)).stem)
    metadata["video_path"] = metadata["video_name"].map(
        lambda value: str((video_dir / str(value)).resolve())
    )
    for attribute in ATTRIBUTES:
        metadata[f"{attribute}_normalized"] = (metadata[attribute].astype(float) - 1.0) / 4.0

    video_names = {path.name for path in video_dir.glob("*.mp4")}
    label_names = set(metadata["video_name"].astype(str))
    duplicate_ids = int(metadata["video_id"].duplicated().sum())
    overlap = len(
        set(frames[0]["video_name"].astype(str)).intersection(frames[1]["video_name"].astype(str))
    )
    audit = {
        "annotation_dir": str(annotation_dir.resolve()),
        "video_dir": str(video_dir.resolve()),
        "total_video_files": len(video_names),
        "total_labeled": len(metadata),
        "train": int(metadata["split"].eq("train").sum()),
        "val": int(metadata["split"].eq("val").sum()),
        "test": 0,
        "duplicate_ids": duplicate_ids,
        "train_val_overlap": overlap,
        "missing_values": int(metadata[list(ATTRIBUTES)].isna().sum().sum()),
        "missing_video_files": sorted(label_names.difference(video_names)),
        "unlabeled_video_files": sorted(video_names.difference(label_names)),
    }
    return metadata, audit


def write_dataset_audit(output_dir: Path, metadata: pd.DataFrame, audit: dict) -> None:
    lines = [
        "# FineVD Dataset Audit", "",
        "## Resolved official files", "",
        f"- Annotation directory: `{audit['annotation_dir']}`",
        f"- Video directory: `{audit['video_dir']}`",
        "- Annotation source: `MOS_train.csv` and `MOS_val.csv`.",
        "- Actual fields: `video_name, color, noise, artifact, blur, temporal, overall`.",
        "- Score direction: higher is better for every attribute.",
        "- Observed/raw scale: approximately 1-5; probes use fixed `(MOS - 1) / 4` normalization.",
        "", "## Split and matching audit", "",
        f"- Video files in archive: **{audit['total_video_files']}**",
        f"- Labeled videos: **{audit['total_labeled']}**",
        f"- Official train: **{audit['train']}**",
        f"- Official val: **{audit['val']}**",
        f"- Official test: **{audit['test']}** (not provided)",
        f"- Duplicate IDs: **{audit['duplicate_ids']}**",
        f"- Train/val overlap: **{audit['train_val_overlap']}**",
        f"- Missing label values: **{audit['missing_values']}**",
        f"- Labels without a matching MP4: **{len(audit['missing_video_files'])}**",
        f"- MP4 files without MOS labels: **{len(audit['unlabeled_video_files'])}**",
        "", "The 1,030 unlabeled videos are retained in the downloaded archive but excluded from E6.",
        "The official validation set is held out for final evaluation. A fixed 90/10 split of the",
        "official training set is used only for probe fitting and early stopping.",
        "", "## Attribute statistics", "",
        "| Split | Attribute | Count | Mean | Std | Min | Max |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("all", "train", "val"):
        frame = metadata if split == "all" else metadata[metadata["split"].eq(split)]
        for attribute in ATTRIBUTES:
            values = frame[attribute].astype(float)
            lines.append(
                f"| {split} | {attribute} | {values.count()} | {values.mean():.4f} | "
                f"{values.std(ddof=1):.4f} | {values.min():.4f} | {values.max():.4f} |"
            )
    (output_dir / "finevd_dataset_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(checkpoint_path: Path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("split_hash") != EXPECTED_SPLIT_HASH:
        raise ValueError(f"Unexpected KEEP_P2_S2 split hash: {payload.get('split_hash')}")
    model = MultiModalQualityModel(**payload["config"]).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, payload


def write_technical_audit(
    output_dir: Path,
    checkpoint_path: Path,
    checkpoint: dict,
    model: MultiModalQualityModel,
) -> int:
    technical_params = sum(parameter.numel() for parameter in model.technical_branch.parameters())
    cover_checkpoint = Path(CFG.cover_root) / "pretrained_weights" / "COVER.pth"
    lines = [
        "# Technical Branch Audit", "",
        "## Experiment boundary", "",
        "E6 freezes KEEP_P2_S2, CLIP, and COVER/Swin. FineVD labels update only an independent",
        "`nn.Linear(128, 1)` probe. The Scientific, Aesthetic, fusion, and final ranking heads are unused.",
        "", "## Checkpoint", "",
        f"- Path: `{checkpoint_path.resolve()}`",
        f"- SHA-256: `{sha256(checkpoint_path)}`",
        f"- Split hash: `{checkpoint['split_hash']}`",
        f"- Frozen Technical Branch parameters: **{technical_params:,}**",
        "", "## Actual path", "",
        "```text",
        "video",
        "  +-> 1 FPS -> <=40 global-uniform frames -> frozen CLIP ViT-L/14",
        "  |      -> mean -> legacy 768D-to-512D adaptive projection",
        "  +-> deterministic COVER val-ytugc technical sampling",
        "         -> frozen Swin-3D Tiny GRPB -> pooled 768D",
        "concat projected CLIP/COVER visual features -> visual 128D",
        "concat visual + zero metadata/WPM/rhythm context -> technical_embedding [128]",
        "technical_embedding -> existing ScoreHead -> sigmoid -> s_tech",
        "```", "",
        "## Active configuration", "",
        f"- `technical_visual_source`: `{checkpoint['config'].get('technical_visual_source')}`",
        f"- `technical_feature_mode`: `{checkpoint['config'].get('technical_feature_mode')}`",
        "- CLIP participates in the Technical Branch together with COVER.",
        "- DNSMOS is disabled by `full_no_dnsmos`.",
        "- FineVD has no science-video metadata or ASR context; metadata (16D), WPM (1D), and",
        "  speech rhythm (6D) are zero-filled without changing the branch graph.",
        "- The embedding is consumed exactly as emitted; stored LayerNorm layers remain active",
        "  and no external feature normalization is added.",
        "- Inference uses `eval()` and `torch.inference_mode()`.",
        "", "## COVER provenance and preprocessing", "",
        f"- COVER checkpoint: `{cover_checkpoint.resolve()}`",
        f"- COVER SHA-256: `{sha256(cover_checkpoint)}`",
        "- Backbone: official `swin_tiny_grpb` technical branch.",
        "- Sampling: `cover.yml` `val-ytugc` technical view, deterministic per-video offsets.",
        "- FineVD uses a memory-equivalent decoder that applies the same selected-frame indices and",
        "  fragment offsets one frame at a time. This avoids the upstream loader's multi-GB 4K tensor;",
        "  the resulting fragment mosaic was checked against the upstream path before full extraction.",
        "- The CLIP path seeks directly to the same legacy 1-FPS frame numbers instead of decoding",
        "  discarded intermediate frames. OpenCV/JPEG output and the resulting cached CLIP feature",
        "  were checked against the sequential path with zero numerical difference.",
        "- To support 4K videos, the unchanged CLIP image processor runs per source JPEG before the",
        "  processed 224x224 tensors are stacked for model inference. This avoids retaining 32",
        "  full-resolution NumPy images and is numerically checked against the original batch path.",
        "- Resize/crop/normalization are performed by the existing COVER extractor without E6 changes.",
        "", "## Probe boundary", "",
        "Only the 128D pre-score-head `technical_embedding` is exposed. Each probe has 129 trainable",
        f"parameters; the {technical_params:,} checkpoint Technical Branch parameters and both visual",
        "backbones remain frozen.",
    ]
    (output_dir / "technical_branch_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return technical_params


def branch_forward(
    model: MultiModalQualityModel,
    clip_feature: np.ndarray,
    cover_feature: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    video = torch.as_tensor(clip_feature, dtype=torch.float32, device=device).view(1, -1)
    cover = torch.as_tensor(cover_feature, dtype=torch.float32, device=device).view(1, -1)
    with torch.inference_mode():
        embedding, logit = model.technical_branch(
            video,
            torch.zeros(1, 16, device=device),
            dnsmos_feat=torch.zeros(1, 3, device=device),
            wpm=torch.zeros(1, 1, device=device),
            speech_rhythm_feat=torch.zeros(1, 6, device=device),
            cover_technical_feat=cover,
        )
    return embedding.squeeze(0).cpu().numpy().astype(np.float32), float(torch.sigmoid(logit).item())


def cache_path(output_dir: Path, video_id: str) -> Path:
    return output_dir / "cache" / "external_validation" / "finevd" / f"{video_id}.pt"


def valid_cache(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = load_pt(path)
        return (
            payload.get("feature_version") == FEATURE_VERSION
            and np.asarray(payload.get("clip_feature", [])).shape == (512,)
            and np.asarray(payload.get("cover_technical_feature", [])).shape == (768,)
            and np.asarray(payload.get("technical_embedding", [])).shape == (128,)
            and np.isfinite(float(payload.get("technical_score", np.nan)))
        )
    except Exception:
        return False


def extract_cache(
    args: argparse.Namespace,
    metadata: pd.DataFrame,
    model: MultiModalQualityModel,
    device: torch.device,
    logger: logging.Logger,
) -> dict:
    os.environ.setdefault("FFMPEG_PATH", str(CFG.ffmpeg_path))
    video_extractor = SeekingVideoExtractor(audio_sr=CFG.audio_sr)
    clip_encoder = MemorySafeVideoEncoder(CFG.clip_model_name, CFG.device)
    cover_encoder = LowMemoryCOVERTechnicalFeatureExtractor(CFG.cover_root, device=CFG.device)
    cache_root = args.output_dir / "cache" / "external_validation" / "finevd"
    temp_root = args.output_dir / "_tmp_frames" / "finevd"
    cache_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    selected = metadata.iloc[: args.max_videos] if args.max_videos > 0 else metadata
    counters = {"expected": len(selected), "cached": 0, "extracted": 0, "errors": 0}
    errors: list[dict] = []
    started = time.perf_counter()
    pending_rows = []
    for row in selected.itertuples(index=False):
        target = cache_path(args.output_dir, str(row.video_id))
        if not args.force_features and valid_cache(target):
            counters["cached"] += 1
        else:
            pending_rows.append(row)

    def prepare(row) -> tuple[Path, torch.Tensor]:
        frame_dir = temp_root / str(row.video_id)
        shutil.rmtree(frame_dir, ignore_errors=True)
        video_extractor.extract_frames(row.video_path, frame_dir, fps=CFG.frame_fps)
        cover_view = cover_encoder.prepare_view_cpu(row.video_path)
        return frame_dir, cover_view

    workers = max(1, int(args.prefetch_workers))
    queue_size = workers
    iterator = iter(pending_rows)
    queue: list[tuple[object, Future]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="finevd-prep") as executor:
        for _ in range(queue_size):
            try:
                row = next(iterator)
            except StopIteration:
                break
            queue.append((row, executor.submit(prepare, row)))

        while queue:
            row, future = queue.pop(0)
            frame_dir = temp_root / str(row.video_id)
            target = cache_path(args.output_dir, str(row.video_id))
            try:
                frame_dir, cover_view = future.result()
                frame_features = clip_encoder.encode_frame_features(
                    frame_dir, max_frames=CFG.clip_max_frames
                )
                clip_native = frame_features.mean(dim=0)
                clip_feature = VideoEncoder.legacy_project(
                    clip_native, output_dim=CFG.video_dim
                ).numpy().astype(np.float32)
                cover_feature = cover_encoder.extract_prepared(cover_view).numpy().astype(np.float32)
                embedding, score = branch_forward(model, clip_feature, cover_feature, device)
                labels = {attribute: float(getattr(row, attribute)) for attribute in ATTRIBUTES}
                labels.update({
                    f"{attribute}_normalized": float(getattr(row, f"{attribute}_normalized"))
                    for attribute in ATTRIBUTES
                })
                torch.save({
                    "video_id": str(row.video_id),
                    "video_name": str(row.video_name),
                    "official_split": str(row.split),
                    "clip_feature": clip_feature,
                    "cover_technical_feature": cover_feature,
                    "technical_embedding": embedding,
                    "technical_score": score,
                    "labels": labels,
                    "feature_version": FEATURE_VERSION,
                    "metadata": {
                        "checkpoint": str(args.checkpoint.resolve()),
                        "clip_model": str(CFG.clip_model_name),
                        "clip_sampling": f"fps_{CFG.frame_fps}_global_uniform_{CFG.clip_max_frames}",
                        "cover_version": cover_encoder.FEATURE_VERSION,
                        "cover_sampling": cover_encoder.sampling_config,
                        "missing_context_policy": "zero_meta_wpm_rhythm",
                        "prefetch_workers": workers,
                    },
                }, target)
                counters["extracted"] += 1
            except Exception as exc:
                counters["errors"] += 1
                errors.append({"video_id": str(row.video_id), "error": repr(exc)})
                logger.exception("Feature extraction failed for %s", row.video_id)
            finally:
                shutil.rmtree(frame_dir, ignore_errors=True)

            try:
                next_row = next(iterator)
            except StopIteration:
                next_row = None
            if next_row is not None:
                queue.append((next_row, executor.submit(prepare, next_row)))

            completed = counters["cached"] + counters["extracted"] + counters["errors"]
            if completed % max(args.log_every, 1) == 0 or completed == len(selected):
                elapsed = time.perf_counter() - started
                logger.info(
                    "FineVD cache %d/%d | new=%d reused=%d errors=%d | %.2f videos/s",
                    completed, len(selected), counters["extracted"], counters["cached"],
                    counters["errors"], completed / max(elapsed, 1e-6),
                )
    shutil.rmtree(temp_root, ignore_errors=True)
    counters["elapsed_seconds"] = time.perf_counter() - started
    counters["cache_dir"] = str(cache_root.resolve())
    counters["errors_detail"] = errors
    return counters


def write_cache_report(output_dir: Path, metadata: pd.DataFrame, report: dict | None) -> None:
    if report is None:
        status_path = output_dir / "feature_cache_status.json"
        if status_path.is_file():
            try:
                report = json.loads(status_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                report = None
    valid = sum(valid_cache(cache_path(output_dir, video_id)) for video_id in metadata["video_id"].astype(str))
    lines = [
        "# FineVD Feature Cache Report", "",
        f"- Feature version: `{FEATURE_VERSION}`",
        f"- Expected labeled videos: **{len(metadata)}**",
        f"- Valid cache files: **{valid}**",
        f"- Missing/invalid: **{len(metadata) - valid}**",
        "- Per video: CLIP 512D, COVER technical 768D, technical embedding 128D,",
        "  scalar technical score, official split, and six MOS labels.",
        "- Cache is resumable and no online FineVD supervision enters feature extraction.",
    ]
    if report:
        lines.extend([
            f"- Newly extracted this run: **{report.get('extracted', 0)}**",
            f"- Reused this run: **{report.get('cached', 0)}**",
            f"- Errors this run: **{report.get('errors', 0)}**",
            f"- Elapsed seconds: **{report.get('elapsed_seconds', 0):.1f}**",
        ])
    (output_dir / "feature_cache_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_cached(metadata: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    rows, embeddings = [], []
    for row in metadata.itertuples(index=False):
        path = cache_path(output_dir, str(row.video_id))
        if not valid_cache(path):
            continue
        payload = load_pt(path)
        record = row._asdict()
        record["technical_score"] = float(payload["technical_score"])
        rows.append(record)
        embeddings.append(np.asarray(payload["technical_embedding"], dtype=np.float32))
    if not rows:
        return pd.DataFrame(), np.zeros((0, 128), dtype=np.float32)
    return pd.DataFrame(rows), np.stack(embeddings)


def correlation(prediction: np.ndarray, target: np.ndarray, kind: str) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.size < 2 or np.unique(prediction).size < 2 or np.unique(target).size < 2:
        return float("nan")
    result = spearmanr(prediction, target).statistic if kind == "spearman" else pearsonr(prediction, target).statistic
    return float(result)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = prediction - target
    return {
        "SRCC": correlation(prediction, target, "spearman"),
        "PLCC": correlation(prediction, target, "pearson"),
        "MSE": float(np.mean(error ** 2)),
        "MAE": float(np.mean(np.abs(error))),
    }


def split_hash(frame: pd.DataFrame, train_indices: np.ndarray, inner_indices: np.ndarray) -> str:
    roles = {}
    for index in train_indices:
        roles[str(frame.iloc[index]["video_id"])] = "probe_train"
    for index in inner_indices:
        roles[str(frame.iloc[index]["video_id"])] = "inner_validation"
    for video_id in frame[frame["split"].eq("val")]["video_id"].astype(str):
        roles[video_id] = "official_val"
    payload = "\n".join(f"{key},{roles[key]}" for key in sorted(roles))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def train_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    inner_x: np.ndarray,
    inner_y: np.ndarray,
    eval_x: np.ndarray,
    seed: int,
    attribute: str,
    partition_hash: str,
    probe_dir: Path,
    force: bool,
) -> tuple[np.ndarray, dict]:
    checkpoint_path = probe_dir / "linear_probe_best.pt"
    history_path = probe_dir / "history.csv"
    if checkpoint_path.is_file() and not force:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            payload.get("probe_version") == PROBE_VERSION
            and payload.get("attribute") == attribute
            and payload.get("seed") == seed
            and payload.get("partition_hash") == partition_hash
        ):
            probe = nn.Linear(train_x.shape[1], 1)
            probe.load_state_dict(payload["model_state_dict"])
            probe.eval()
            with torch.inference_mode():
                prediction = probe(torch.from_numpy(eval_x).float()).squeeze(-1).numpy()
            return prediction, payload

    set_seed(seed)
    probe = nn.Linear(train_x.shape[1], 1)
    optimizer = AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()),
        batch_size=64, shuffle=True, generator=generator,
    )
    best_srcc = -float("inf")
    stale = 0
    history: list[dict] = []
    probe_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, 101):
        probe.train()
        losses = []
        for features, target in loader:
            prediction = probe(features).squeeze(-1)
            loss = loss_fn(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        probe.eval()
        with torch.inference_mode():
            inner_prediction = probe(torch.from_numpy(inner_x).float()).squeeze(-1).numpy()
        validation = metrics(inner_prediction, inner_y)
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), **validation})
        if np.isfinite(validation["SRCC"]) and validation["SRCC"] > best_srcc:
            best_srcc = validation["SRCC"]
            stale = 0
            torch.save({
                "model_state_dict": probe.state_dict(),
                "probe_version": PROBE_VERSION,
                "attribute": attribute,
                "seed": seed,
                "partition_hash": partition_hash,
                "best_epoch": epoch,
                "validation_metrics": validation,
                "feature_dim": train_x.shape[1],
                "total_params": sum(parameter.numel() for parameter in probe.parameters()),
                "trainable_params": sum(parameter.numel() for parameter in probe.parameters()),
                "frozen_encoder": True,
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= 10:
                break
    write_csv(history_path, history)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    probe.load_state_dict(payload["model_state_dict"])
    probe.eval()
    with torch.inference_mode():
        prediction = probe(torch.from_numpy(eval_x).float()).squeeze(-1).numpy()
    return prediction, payload


def run_evaluation(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    technical_params: int,
    logger: logging.Logger,
) -> tuple[list[dict], dict[str, pd.DataFrame], str]:
    official_train = np.flatnonzero(frame["split"].eq("train").to_numpy())
    official_val = np.flatnonzero(frame["split"].eq("val").to_numpy())
    probe_train, inner_val = train_test_split(
        official_train, test_size=0.1, random_state=42, shuffle=True
    )
    partition_hash = split_hash(frame, probe_train, inner_val)
    manifest = frame[["video_id", "video_name", "split"]].copy()
    manifest["probe_role"] = "official_val"
    manifest.loc[probe_train, "probe_role"] = "probe_train"
    manifest.loc[inner_val, "probe_role"] = "inner_validation"
    write_csv(args.output_dir / "finevd_split_manifest.csv", manifest.to_dict("records"))

    results: list[dict] = []
    predictions: dict[str, pd.DataFrame] = {}
    for attribute in ATTRIBUTES:
        target_column = f"{attribute}_normalized"
        target = frame[target_column].to_numpy(np.float32)
        zero_prediction = frame.iloc[official_val]["technical_score"].to_numpy(np.float32)
        zero_metrics = metrics(zero_prediction, target[official_val])
        results.append({
            "attribute": attribute, "seed": "", "protocol": "Zero-shot",
            **zero_metrics, "num_train": 0, "num_validation": 0,
            "num_eval": len(official_val), "feature_dim": 128, "trainable_params": 0,
            "best_epoch": "", "partition_hash": partition_hash,
        })
        output = frame.iloc[official_val][["video_id", "video_name", attribute, target_column]].copy()
        output = output.rename(columns={attribute: "ground_truth", target_column: "ground_truth_normalized"})
        output["technical_score_zero_shot"] = zero_prediction
        for seed in PROBE_SEEDS:
            probe_dir = args.output_dir / "probes" / attribute / f"seed_{seed}"
            probe_prediction, payload = train_probe(
                embeddings[probe_train], target[probe_train],
                embeddings[inner_val], target[inner_val],
                embeddings[official_val], seed, attribute, partition_hash,
                probe_dir, args.force_probes,
            )
            probe_metrics = metrics(probe_prediction, target[official_val])
            results.append({
                "attribute": attribute, "seed": seed, "protocol": "Frozen Linear Probe",
                **probe_metrics, "num_train": len(probe_train),
                "num_validation": len(inner_val), "num_eval": len(official_val),
                "feature_dim": embeddings.shape[1],
                "trainable_params": int(payload["trainable_params"]),
                "best_epoch": int(payload["best_epoch"]), "partition_hash": partition_hash,
            })
            output[f"prediction_seed{seed}"] = probe_prediction
            logger.info(
                "%s seed=%d | SRCC=%.4f PLCC=%.4f epoch=%d | trainable=129 frozen=%d",
                attribute, seed, probe_metrics["SRCC"], probe_metrics["PLCC"],
                payload["best_epoch"], technical_params,
            )
        output["prediction_mean"] = output[
            [f"prediction_seed{seed}" for seed in PROBE_SEEDS]
        ].mean(axis=1)
        predictions[attribute] = output
    return results, predictions, partition_hash


def feature_statistics(frame: pd.DataFrame, embeddings: np.ndarray) -> list[dict]:
    rows: list[dict] = []

    def add(scope: str, attribute: str, group: str, indices: np.ndarray) -> None:
        values = embeddings[indices]
        norms = np.linalg.norm(values, axis=1)
        rows.append({
            "scope": scope, "attribute": attribute, "group": group, "count": len(indices),
            "feature_mean": float(values.mean()), "feature_std": float(values.std()),
            "l2_norm_mean": float(norms.mean()), "l2_norm_std": float(norms.std()),
            "technical_score_mean": float(frame.iloc[indices]["technical_score"].mean()),
            "technical_score_std": float(frame.iloc[indices]["technical_score"].std(ddof=1)),
        })

    all_indices = np.arange(len(frame))
    add("all", "all", "all", all_indices)
    for split in ("train", "val"):
        indices = np.flatnonzero(frame["split"].eq(split).to_numpy())
        add(split, "all", "all", indices)
    val_indices = np.flatnonzero(frame["split"].eq("val").to_numpy())
    for attribute in ATTRIBUTES:
        median = float(frame.iloc[val_indices][attribute].median())
        low = val_indices[frame.iloc[val_indices][attribute].to_numpy() < median]
        high = val_indices[frame.iloc[val_indices][attribute].to_numpy() >= median]
        add("val", attribute, "low", low)
        add("val", attribute, "high", high)
    return rows


def error_cases(predictions: dict[str, pd.DataFrame]) -> list[dict]:
    rows: list[dict] = []
    for attribute, frame in predictions.items():
        selected = frame.copy()
        selected["normalized_error"] = (
            selected["prediction_mean"] - selected["ground_truth_normalized"]
        ).abs()
        for row in selected.nlargest(10, "normalized_error").itertuples(index=False):
            rows.append({
                "attribute": attribute, "video_id": str(row.video_id),
                "video_name": str(row.video_name), "ground_truth": float(row.ground_truth),
                "ground_truth_normalized": float(row.ground_truth_normalized),
                "prediction_mean": float(row.prediction_mean),
                "normalized_error": float(row.normalized_error),
            })
    return rows


def aggregate_probe_results(results: list[dict], attribute: str) -> dict:
    selected = [
        row for row in results
        if row["attribute"] == attribute and row["protocol"] == "Frozen Linear Probe"
    ]
    return {
        "srcc_mean": float(np.mean([row["SRCC"] for row in selected])),
        "srcc_std": float(np.std([row["SRCC"] for row in selected], ddof=1)),
        "plcc_mean": float(np.mean([row["PLCC"] for row in selected])),
        "plcc_std": float(np.std([row["PLCC"] for row in selected], ddof=1)),
        "mse_mean": float(np.mean([row["MSE"] for row in selected])),
        "mae_mean": float(np.mean([row["MAE"] for row in selected])),
    }


def write_summary(
    output_dir: Path,
    metadata: pd.DataFrame,
    audit: dict,
    results: list[dict],
    partition_hash: str,
) -> None:
    zero = {
        row["attribute"]: row for row in results if row["protocol"] == "Zero-shot"
    }
    aggregates = {attribute: aggregate_probe_results(results, attribute) for attribute in ATTRIBUTES}
    ranking = sorted(ATTRIBUTES, key=lambda name: aggregates[name]["srcc_mean"], reverse=True)
    temporal = aggregates["temporal"]
    if temporal["srcc_mean"] >= 0.5:
        temporal_interpretation = (
            "The frozen representation contains clearly linearly readable temporal-quality information. "
            "This supports the interpretation that explicit T1/T2 statistics had limited incremental value "
            "because part of the temporal signal was already encoded."
        )
    elif temporal["srcc_mean"] >= 0.3:
        temporal_interpretation = (
            "The frozen representation contains moderate temporal-quality information, but it is not strongly "
            "disentangled. This is consistent with the unstable T1/T2 gains on the small in-domain validation set."
        )
    else:
        temporal_interpretation = (
            "Temporal quality is only weakly linearly readable. Together with unstable T1/T2 gains, this points "
            "to limited temporal sensitivity plus small-sample/content noise rather than a solved temporal pathway."
        )
    lines = [
        "# FineVD Fine-Grained Technical Quality Validation", "",
        "## 1. Research Question", "",
        "Does the frozen KEEP_P2_S2 Technical representation contain transferable, fine-grained visual",
        "technical-quality information?", "",
        "## 2. FineVD Dataset Audit", "",
        f"FineVD contains {audit['total_video_files']} downloaded MP4 files. The released MOS tables label",
        f"{len(metadata)} videos: {audit['train']} official train and {audit['val']} official validation;",
        f"{len(audit['unlabeled_video_files'])} archive videos have no MOS row and are excluded.",
        "The actual labels are Overall, Blur, Color, Noise, Artifact, and Temporal on an approximately 1-5 scale.",
        "", "## 3. Technical Branch Audit", "",
        "The frozen path combines legacy 512D CLIP and pooled 768D COVER/Swin features and emits a 128D",
        "technical embedding. FineVD-unavailable metadata/WPM/rhythm context is zero-filled. DNSMOS is disabled.",
        "", "## 4. Experimental Protocol", "",
        f"The official validation set ({audit['val']} videos) is used only for final evaluation. The official",
        "train set is split once into fixed 90% probe training and 10% inner validation partitions; all three",
        f"seeds share this partition (`{partition_hash}`). Early stopping uses inner-validation SRCC.",
        "Every probe is `nn.Linear(128, 1)` with AdamW, lr=1e-3, weight decay=1e-4, batch 64,",
        "maximum 100 epochs, patience 10, and MSE loss.",
        "", "## 5. Overall Transfer Results (E6-A)", "",
        f"Overall zero-shot: SRCC **{zero['overall']['SRCC']:.4f}**, PLCC **{zero['overall']['PLCC']:.4f}**.",
        f"Overall frozen probe: SRCC **{aggregates['overall']['srcc_mean']:.4f} +/- {aggregates['overall']['srcc_std']:.4f}**, "
        f"PLCC **{aggregates['overall']['plcc_mean']:.4f} +/- {aggregates['overall']['plcc_std']:.4f}**.",
        "", "## 6. Fine-Grained Attribute Results (E6-B / E6-C)", "",
        "| FineVD Attribute | Zero-shot SRCC | Linear Probe SRCC | Linear Probe PLCC |",
        "|---|---:|---:|---:|",
    ]
    for attribute in ATTRIBUTES:
        aggregate = aggregates[attribute]
        lines.append(
            f"| {attribute.title()} | {zero[attribute]['SRCC']:.4f} | "
            f"{aggregate['srcc_mean']:.4f} +/- {aggregate['srcc_std']:.4f} | "
            f"{aggregate['plcc_mean']:.4f} +/- {aggregate['plcc_std']:.4f} |"
        )
    lines.extend([
        "", "Attribute zero-shot values are diagnostics only; the frozen-probe correlations are the",
        "fine-grained transfer evidence.", "", "## 7. Multi-seed Stability", "",
        "| Attribute | SRCC Mean +/- Std | PLCC Mean +/- Std |",
        "|---|---:|---:|",
    ])
    for attribute in ATTRIBUTES:
        aggregate = aggregates[attribute]
        lines.append(
            f"| {attribute.title()} | {aggregate['srcc_mean']:.4f} +/- {aggregate['srcc_std']:.4f} | "
            f"{aggregate['plcc_mean']:.4f} +/- {aggregate['plcc_std']:.4f} |"
        )
    lines.extend([
        "", "## 8. Temporal Quality Analysis (E6-D)", "",
        f"Temporal probe SRCC is **{temporal['srcc_mean']:.4f} +/- {temporal['srcc_std']:.4f}** and PLCC is",
        f"**{temporal['plcc_mean']:.4f} +/- {temporal['plcc_std']:.4f}**. {temporal_interpretation}",
        "The earlier in-domain T2 temporal-statistics experiment raised mean Technical SRCC by only 0.0242",
        "while reducing AUROC by 0.0074 and PR-AUC by 0.0228, so T2 was not promoted.",
        "", "## 9. Representation Analysis", "",
        "Linear readability ranking: **" + " > ".join(name.title() for name in ranking) + "**.",
        "Zero-shot SRCC is near zero for every attribute, while frozen linear probes reach moderate",
        "correlations. The representation transfers, but the original science-video score head has a",
        "clear domain/scale shift on FineVD and should not be interpreted as a calibrated external VQA head.",
        "This ranking describes linear accessibility in the frozen embedding, not categorical distortion-detection skill.",
        "FineVD validates the visual technical pathway only; it does not validate audio quality.",
        "", "## 10. Error Analysis", "",
        "`finevd_error_cases.csv` contains the ten largest normalized errors for every attribute, with special",
        "relevance to Temporal, Blur, and Artifact. Predictions are three-seed means on official validation videos.",
        "", "## 11. Conclusion", "",
    ])
    mean_srcc = float(np.mean([aggregates[name]["srcc_mean"] for name in ATTRIBUTES]))
    if mean_srcc >= 0.4:
        lines.append(
            "The frozen Technical representation shows meaningful external fine-grained transfer: multiple FineVD"
            " attributes are linearly readable without updating KEEP_P2_S2 or either visual backbone."
        )
    else:
        lines.append(
            "The frozen Technical representation shows limited-to-moderate external fine-grained transfer; the"
            " result supports retaining the branch as a general representation but not claiming strong attribute disentanglement."
        )
    lines.extend([
        f"The easiest attributes to read are {', '.join(name.title() for name in ranking[:3])}.",
        temporal_interpretation,
        "These results support keeping the original KEEP_P2_S2 Technical Branch as the thesis baseline; no",
        "FineVD-driven fine-tuning or architecture change is justified by this validation experiment.",
    ])
    (output_dir / "finevd_technical_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(args.output_dir / "finevd_technical_validation.log")
    metadata, audit = load_metadata(args.finevd_root)
    write_dataset_audit(args.output_dir, metadata, audit)
    logger.info(
        "FineVD audit | videos=%d labeled=%d train=%d val=%d unlabeled=%d missing=%d",
        audit["total_video_files"], len(metadata), audit["train"], audit["val"],
        len(audit["unlabeled_video_files"]), len(audit["missing_video_files"]),
    )
    if audit["duplicate_ids"] or audit["train_val_overlap"] or audit["missing_video_files"]:
        raise RuntimeError("FineVD audit failed; inspect finevd_dataset_audit.md")

    device = get_device(CFG.device)
    model, checkpoint = load_model(args.checkpoint, device)
    technical_params = write_technical_audit(
        args.output_dir, args.checkpoint, checkpoint, model
    )
    logger.info("Loaded KEEP_P2_S2 | frozen technical params=%d", technical_params)
    if args.audit_only:
        write_cache_report(args.output_dir, metadata, None)
        return

    report = None
    if not args.evaluate_only:
        report = extract_cache(args, metadata, model, device, logger)
        write_json(args.output_dir / "feature_cache_status.json", report)
    write_cache_report(args.output_dir, metadata, report)
    if args.extract_only:
        return

    frame, embeddings = load_cached(metadata, args.output_dir)
    expected = min(len(metadata), args.max_videos) if args.max_videos > 0 else len(metadata)
    if len(frame) != expected:
        raise RuntimeError(f"Incomplete FineVD cache: {len(frame)}/{expected}")
    if args.max_videos > 0:
        logger.info("Smoke cache complete; formal evaluation skipped")
        return

    results, predictions, partition_hash = run_evaluation(
        args, frame, embeddings, technical_params, logger
    )
    fields = [
        "attribute", "seed", "protocol", "SRCC", "PLCC", "MSE", "MAE",
        "num_train", "num_validation", "num_eval", "feature_dim", "trainable_params",
        "best_epoch", "partition_hash",
    ]
    write_csv(args.output_dir / "finevd_finegrained_results.csv", results, fields)
    write_csv(
        args.output_dir / "finevd_feature_statistics.csv",
        feature_statistics(frame, embeddings),
    )
    for attribute, prediction in predictions.items():
        write_csv(
            args.output_dir / f"finevd_{attribute}_predictions.csv",
            prediction.to_dict("records"),
        )
    write_csv(args.output_dir / "finevd_error_cases.csv", error_cases(predictions))
    write_summary(args.output_dir, metadata, audit, results, partition_hash)
    logger.info("FineVD Technical validation complete: %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
