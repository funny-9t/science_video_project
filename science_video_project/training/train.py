import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import CFG
from pipeline.utils_io import ensure_dir, load_metadata
from training.dataloader_pair import collate_pair
from training.dataset_pair import PairwiseVideoDataset
from training.losses import branch_consistency_loss, pairwise_ranking_loss
from training.metrics import compute_pair_metrics
from training.model_mvp import MultiModalQualityModel
from training.utils_train import build_logger, get_device, move_batch_to_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pairwise ranking model for science video quality")
    parser.add_argument("--metadata", default=str(CFG.metadata_csv))
    parser.add_argument("--feature_dir", default=str(CFG.feature_dir))
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--lr", type=float, default=CFG.lr)
    parser.add_argument("--margin", type=float, default=CFG.margin)
    parser.add_argument("--same_category", action="store_true", default=CFG.same_category_pair)
    parser.add_argument("--checkpoint", default=str(CFG.checkpoint_dir / "best.pt"))
    return parser.parse_args()


def evaluate(
    model: MultiModalQualityModel,
    loader: DataLoader,
    device: torch.device,
    margin: float,
    threshold: float,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    pos_probs = []
    neg_probs = []

    with torch.no_grad():
        for pos_batch, neg_batch in loader:
            pos_batch = move_batch_to_device(pos_batch, device)
            neg_batch = move_batch_to_device(neg_batch, device)

            pos_out = model(**pos_batch)
            neg_out = model(**neg_batch)

            rank_loss = pairwise_ranking_loss(pos_out["overall_score"], neg_out["overall_score"], margin=margin)
            reg_loss = branch_consistency_loss(
                pos_out["overall_score"],
                pos_out["scientific_score"],
                pos_out["technical_score"],
                pos_out["aesthetic_score"],
            )
            loss = rank_loss + 0.1 * reg_loss
            total_loss += float(loss.item())

            pos_probs.append(pos_out["probability"].squeeze(-1).detach().cpu().numpy())
            neg_probs.append(neg_out["probability"].squeeze(-1).detach().cpu().numpy())

    if not pos_probs or not neg_probs:
        return total_loss, {"accuracy": 0.0, "f1": 0.0, "auc": 0.0, "ranking_accuracy": 0.0}

    pos_prob = np.concatenate(pos_probs, axis=0)
    neg_prob = np.concatenate(neg_probs, axis=0)
    metrics = compute_pair_metrics(pos_prob, neg_prob, threshold=threshold)
    return total_loss, metrics


def main() -> None:
    args = parse_args()
    ensure_dir(CFG.checkpoint_dir)
    ensure_dir(CFG.log_dir)

    logger = build_logger(CFG.log_dir / "train.log", name="train")
    set_seed(CFG.seed)

    def _ensure_both_labels(train_df, val_df):
        for label in (0, 1):
            if (val_df["label"] == label).sum() == 0:
                candidates = train_df[train_df["label"] == label]
                if candidates.empty:
                    raise ValueError("Both positive and negative samples are required for training.")
                moved = candidates.sample(n=1, random_state=CFG.seed)
                train_df = train_df.drop(moved.index)
                val_df = pd.concat([val_df, moved], ignore_index=True)

        for label in (0, 1):
            if (train_df["label"] == label).sum() == 0:
                raise ValueError("Training split lacks required label after adjustment.")
        return train_df, val_df

    def _filter_with_features(metadata_df, feature_dir: str | Path):
        feature_dir = Path(feature_dir)
        available = {p.stem for p in feature_dir.glob("*.pt")}
        filtered = metadata_df[metadata_df["video_id"].astype(str).isin(available)].copy()
        if filtered.empty:
            raise ValueError("No metadata rows have matching feature files.")
        return filtered

    try:
        metadata = load_metadata(args.metadata)
        metadata = _filter_with_features(metadata, args.feature_dir)
        train_df, val_df = train_test_split(
            metadata,
            test_size=CFG.val_ratio,
            random_state=CFG.seed,
            stratify=metadata["label"],
        )
        train_df, val_df = _ensure_both_labels(train_df, val_df)
    except ValueError:
        metadata = load_metadata(args.metadata)
        metadata = _filter_with_features(metadata, args.feature_dir)
        train_df, val_df = train_test_split(metadata, test_size=CFG.val_ratio, random_state=CFG.seed)
        train_df, val_df = _ensure_both_labels(train_df, val_df)
    except Exception as exc:
        logger.exception("Failed to load/split metadata: %s", exc)
        raise

    train_ds = PairwiseVideoDataset.from_metadata(train_df, args.feature_dir, same_category=args.same_category)
    val_ds = PairwiseVideoDataset.from_metadata(val_df, args.feature_dir, same_category=args.same_category)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        collate_fn=collate_pair,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        collate_fn=collate_pair,
    )

    device = get_device(CFG.device)
    model = MultiModalQualityModel(
        text_dim=CFG.text_dim,
        video_dim=CFG.video_dim,
        audio_dim=CFG.audio_dim,
        meta_dim=CFG.meta_dim,
        aes_dim=CFG.aes_dim,
        hidden_dim=CFG.hidden_dim,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_rank_acc = -1.0
    ckpt_path = Path(args.checkpoint)
    ensure_dir(ckpt_path.parent)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for pos_batch, neg_batch in train_loader:
            pos_batch = move_batch_to_device(pos_batch, device)
            neg_batch = move_batch_to_device(neg_batch, device)

            pos_out = model(**pos_batch)
            neg_out = model(**neg_batch)

            rank_loss = pairwise_ranking_loss(pos_out["overall_score"], neg_out["overall_score"], margin=args.margin)
            reg_loss = branch_consistency_loss(
                pos_out["overall_score"],
                pos_out["scientific_score"],
                pos_out["technical_score"],
                pos_out["aesthetic_score"],
            )
            loss = rank_loss + 0.1 * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        val_loss, val_metrics = evaluate(model, val_loader, device, args.margin, CFG.threshold)
        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | rank_acc=%.4f | acc=%.4f | f1=%.4f | auc=%.4f",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            val_metrics["ranking_accuracy"],
            val_metrics["accuracy"],
            val_metrics["f1"],
            val_metrics["auc"],
        )

        if val_metrics["ranking_accuracy"] > best_rank_acc:
            best_rank_acc = val_metrics["ranking_accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "text_dim": CFG.text_dim,
                        "video_dim": CFG.video_dim,
                        "audio_dim": CFG.audio_dim,
                        "meta_dim": CFG.meta_dim,
                        "hidden_dim": CFG.hidden_dim,
                    },
                },
                ckpt_path,
            )
            logger.info("Saved best checkpoint to %s", ckpt_path)

    logger.info("Training finished. best_ranking_accuracy=%.4f", best_rank_acc)


if __name__ == "__main__":
    main()
