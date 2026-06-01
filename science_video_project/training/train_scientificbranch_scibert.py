import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import CFG
from pipeline.utils_io import ensure_dir, load_metadata
from training.dataset_scientificbranch_scibert import (
    ScientificBranchTextDataset,
    collate_scientificbranch_text,
)
from training.model_scientificbranch_scibert import SciBERTScientificBranchModel
from training.utils_train import build_logger, get_device, move_batch_to_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train scientificbranch with SciBERT (text+meta -> label)",
    )
    p.add_argument("--metadata", default=str(CFG.metadata_csv))

    p.add_argument(
        "--encoder",
        default="allenai/scibert_scivocab_uncased",
        help="HuggingFace model id or local path (e.g. allenai/scibert_scivocab_uncased)",
    )
    p.add_argument("--local_files_only", action="store_true", default=False)

    p.add_argument("--subtitle_dir", default="", help="Optional dir containing {video_id}.txt subtitles")
    p.add_argument("--max_length", type=int, default=256)

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)

    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--meta_dim", type=int, default=CFG.meta_dim)

    p.add_argument("--seed", type=int, default=CFG.seed)
    p.add_argument("--device", default=CFG.device)
    p.add_argument(
        "--out_dir",
        default=str(CFG.checkpoint_dir / "scientificbranch_scibert"),
        help="Output directory for best checkpoint",
    )
    return p.parse_args()


def _ensure_both_labels(train_df: pd.DataFrame, val_df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    for label in (0, 1):
        if (val_df["label"] == label).sum() == 0:
            candidates = train_df[train_df["label"] == label]
            if candidates.empty:
                raise ValueError("Both positive and negative samples are required for training.")
            moved = candidates.sample(n=1, random_state=seed)
            train_df = train_df.drop(moved.index)
            val_df = pd.concat([val_df, moved], ignore_index=True)

    for label in (0, 1):
        if (train_df["label"] == label).sum() == 0:
            raise ValueError("Training split lacks required label after adjustment.")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


@torch.no_grad()
def evaluate(
    model: SciBERTScientificBranchModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        labels = batch.pop("labels")
        batch = move_batch_to_device(batch, device)
        labels = labels.to(device)

        out = model(**batch)
        probs = out["prob"].detach().float().cpu().numpy()
        y = labels.detach().float().cpu().numpy()

        all_probs.append(probs)
        all_labels.append(y)

    if not all_probs:
        return {"accuracy": 0.0, "f1": 0.0, "auc": 0.0}

    prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = (prob >= 0.5).astype(np.int64)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    auc = 0.0
    try:
        if len(np.unique(y_true)) >= 2:
            auc = float(roc_auc_score(y_true, prob))
    except Exception:
        auc = 0.0

    return {"accuracy": acc, "f1": f1, "auc": auc}


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(CFG.log_dir)

    logger = build_logger(CFG.log_dir / "train_scientificbranch_scibert.log", name="train_scibert")

    df = load_metadata(args.metadata)
    if df.empty:
        raise ValueError("metadata is empty")

    # Keep only minimal required cols to avoid accidental leakage.
    keep = ["video_id", "label", "category", "title", "tags", "duration", "verified", "publish_time"]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Split
    try:
        train_df, val_df = train_test_split(
            df,
            test_size=float(CFG.val_ratio),
            random_state=int(args.seed),
            stratify=df["label"],
        )
        train_df, val_df = _ensure_both_labels(train_df, val_df, seed=int(args.seed))
    except Exception:
        train_df, val_df = train_test_split(df, test_size=float(CFG.val_ratio), random_state=int(args.seed))
        train_df, val_df = _ensure_both_labels(train_df, val_df, seed=int(args.seed))

    subtitle_dir = args.subtitle_dir.strip() or None

    # Model + datasets
    device = get_device(str(args.device))
    model = SciBERTScientificBranchModel(
        encoder_name_or_path=str(args.encoder),
        meta_dim=int(args.meta_dim),
        hidden_dim=int(args.hidden_dim),
        local_files_only=bool(args.local_files_only),
    ).to(device)

    train_ds = ScientificBranchTextDataset(
        train_df,
        tokenizer=model.tokenizer,
        max_length=int(args.max_length),
        subtitle_dir=subtitle_dir,
        meta_dim=int(args.meta_dim),
    )
    val_ds = ScientificBranchTextDataset(
        val_df,
        tokenizer=model.tokenizer,
        max_length=int(args.max_length),
        subtitle_dir=subtitle_dir,
        meta_dim=int(args.meta_dim),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_scientificbranch_text(b, tokenizer=model.tokenizer),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_scientificbranch_text(b, tokenizer=model.tokenizer),
    )

    # Loss (handle imbalance)
    pos = float((train_df["label"] == 1).sum())
    neg = float((train_df["label"] == 0).sum())
    if pos <= 0 or neg <= 0:
        raise ValueError("Need both positive and negative samples.")

    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    total_steps = int(math.ceil(len(train_loader) * int(args.epochs)))
    warmup_steps = int(total_steps * float(args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_key = -1.0
    best_metrics: dict[str, float] | None = None

    logger.info("Train size=%d val size=%d pos=%d neg=%d pos_weight=%.4f", len(train_df), len(val_df), int(pos), int(neg), float(pos_weight.item()))
    logger.info("Encoder=%s local_files_only=%s device=%s", str(args.encoder), str(args.local_files_only), str(device))

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            batch = move_batch_to_device(batch, device)

            out = model(**batch)
            loss = criterion(out["logits"], labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item())

            if step % 20 == 0:
                logger.info("epoch=%d step=%d/%d loss=%.5f", epoch, step, len(train_loader), total_loss / step)

        metrics = evaluate(model, val_loader, device)
        avg_loss = total_loss / max(1, len(train_loader))

        # Prefer AUC when valid; fallback to F1.
        key = float(metrics.get("auc", 0.0) if metrics.get("auc", 0.0) > 0.0 else metrics.get("f1", 0.0))
        logger.info(
            "epoch=%d train_loss=%.5f val_acc=%.4f val_f1=%.4f val_auc=%.4f",
            epoch,
            avg_loss,
            metrics.get("accuracy", 0.0),
            metrics.get("f1", 0.0),
            metrics.get("auc", 0.0),
        )

        if key > best_key:
            best_key = key
            best_metrics = dict(metrics)
            best_dir = out_dir / "best"
            ensure_dir(best_dir)
            model.save(best_dir)
            logger.info("Saved best checkpoint to %s (key=%.4f)", str(best_dir), best_key)

    if best_metrics is None:
        best_metrics = {"accuracy": 0.0, "f1": 0.0, "auc": 0.0}

    logger.info("Done. Best metrics: %s", best_metrics)


if __name__ == "__main__":
    main()
