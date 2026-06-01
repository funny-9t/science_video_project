"""
研究版训练脚本 (§13 实验系统)

基于 train.py 增量升级，支持:
  - ResearchModel 全模块训练
  - 多损失联合优化 (Ranking + Consistency + Diversity + Engagement)
  - 自动消融实验配置
  - 向后兼容原有 checkpoint
"""

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
from training.losses import pairwise_ranking_loss, branch_consistency_loss
from training.losses_research import compute_research_total_loss
from training.metrics import compute_pair_metrics
from training.model_research import ResearchModel
from training.utils_train import build_logger, get_device, move_batch_to_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research training: Multi-dimensional quality assessment")
    parser.add_argument("--metadata", default=str(CFG.metadata_csv))
    parser.add_argument("--feature_dir", default=str(CFG.feature_dir))
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--lr", type=float, default=CFG.lr)
    parser.add_argument("--margin", type=float, default=CFG.margin)
    parser.add_argument("--same_category", action="store_true", default=CFG.same_category_pair)
    parser.add_argument("--checkpoint", default=str(CFG.checkpoint_dir / "research_best.pt"))
    parser.add_argument("--pretrained", default=str(CFG.checkpoint_dir / "best.pt"), help="MVP checkpoint for warm-start")
    parser.add_argument("--use_mvp_loss", action="store_true", default=False, help="Use original MVP loss (for baseline)")
    # Experiment switches
    parser.add_argument("--quality_head", type=int, default=1, help="1=enable QualityHead, 0=disable")
    parser.add_argument("--consistency_loss", type=int, default=1)
    parser.add_argument("--engagement_branch", type=int, default=1)
    parser.add_argument("--science_features", type=int, default=1)
    parser.add_argument("--aesthetic_mlp", type=int, default=1)
    parser.add_argument("--temporal_encoder", type=int, default=0, help="Requires frame_features in .pt")
    parser.add_argument("--cross_modal_attn", type=int, default=1)
    parser.add_argument("--diversity_loss", type=int, default=1)
    return parser.parse_args()


def _load_pretrained_weights(model: ResearchModel, pretrained_path: str, device: torch.device) -> int:
    """加载 MVP checkpoint 中匹配的权重（向后兼容）。"""
    ckpt = torch.load(pretrained_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model_dict = model.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(matched)
    model.load_state_dict(model_dict)
    return len(matched)


def evaluate_research(
    model: ResearchModel,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[float, dict[str, float]]:
    """研究版评估：支持完整的多输出。"""
    model.eval()
    total_loss = 0.0
    pos_probs = []
    neg_probs = []

    with torch.no_grad():
        for pos_batch, neg_batch in loader:
            pos_batch = move_batch_to_device(pos_batch, device)
            neg_batch = move_batch_to_device(neg_batch, device)

            # 提取额外特征
            pos_eng = pos_batch.get("engagement_target", torch.zeros(pos_batch["text_feat"].size(0), 1, device=device))
            pos_kwargs = _build_kwargs(pos_batch)
            neg_kwargs = _build_kwargs(neg_batch)

            pos_out = model(**pos_kwargs)
            neg_out = model(**neg_kwargs)

            if args.use_mvp_loss:
                rank_loss = pairwise_ranking_loss(pos_out["overall_score"], neg_out["overall_score"], margin=args.margin)
                reg_loss = branch_consistency_loss(
                    pos_out["overall_score"], pos_out["scientific_score"],
                    pos_out["technical_score"], pos_out["aesthetic_score"],
                )
                loss = rank_loss + 0.1 * reg_loss
            else:
                loss, _ = compute_research_total_loss(
                    pos_out, neg_out,
                    margin=args.margin,
                    lambda_consistency=CFG.lambda_consistency if args.consistency_loss else 0.0,
                    lambda_diversity=CFG.lambda_diversity if args.diversity_loss else 0.0,
                    engagement_target=pos_eng,
                )
            total_loss += float(loss.item())

            pos_probs.append(pos_out["probability"].squeeze(-1).detach().cpu().numpy())
            neg_probs.append(neg_out["probability"].squeeze(-1).detach().cpu().numpy())

    if not pos_probs or not neg_probs:
        return total_loss, {"accuracy": 0.0, "f1": 0.0, "auc": 0.0, "ranking_accuracy": 0.0}

    pos_prob = np.concatenate(pos_probs, axis=0)
    neg_prob = np.concatenate(neg_probs, axis=0)
    metrics = compute_pair_metrics(pos_prob, neg_prob, threshold=CFG.threshold)
    return total_loss, metrics


def _build_kwargs(batch: dict) -> dict:
    """构建模型 forward 所需参数。"""
    kwargs: dict = {
        "text_feat": batch["text_feat"],
        "video_feat": batch["video_feat"],
        "audio_feat": batch["audio_feat"],
        "meta_feat": batch["meta_feat"],
    }
    if "aes_feat" in batch:
        kwargs["aes_feat"] = batch["aes_feat"]
    if "sci_hand_feat" in batch:
        kwargs["sci_hand_feat"] = batch["sci_hand_feat"]
    if "frame_features" in batch and batch["frame_features"].numel() > 0:
        kwargs["frame_features"] = batch["frame_features"]
    return kwargs


def main() -> None:
    args = parse_args()
    ensure_dir(CFG.checkpoint_dir)
    ensure_dir(CFG.log_dir)

    logger = build_logger(CFG.log_dir / "train_research.log", name="research")
    set_seed(CFG.seed)

    # --- 数据准备 ---
    metadata = load_metadata(args.metadata)
    feature_dir = Path(args.feature_dir)
    available = {p.stem for p in feature_dir.glob("*.pt")}
    metadata = metadata[metadata["video_id"].astype(str).isin(available)].copy()

    train_df, val_df = train_test_split(
        metadata, test_size=CFG.val_ratio, random_state=CFG.seed, stratify=metadata["label"]
    )
    # 确保两者都有正负样本
    for df in (train_df, val_df):
        for lbl in (0, 1):
            if (df["label"] == lbl).sum() == 0:
                raise ValueError("Both positive and negative samples required.")

    train_ds = PairwiseVideoDataset.from_metadata(train_df, args.feature_dir, same_category=args.same_category)
    val_ds = PairwiseVideoDataset.from_metadata(val_df, args.feature_dir, same_category=args.same_category)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=CFG.num_workers, collate_fn=collate_pair)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=CFG.num_workers, collate_fn=collate_pair)

    # --- 模型 ---
    device = get_device(CFG.device)
    model = ResearchModel(
        text_dim=CFG.text_dim, video_dim=CFG.video_dim, audio_dim=CFG.audio_dim,
        meta_dim=CFG.meta_dim, aes_dim=CFG.aes_dim, sci_hand_dim=CFG.sci_hand_dim,
        hidden_dim=CFG.hidden_dim,
        use_quality_head=bool(args.quality_head),
        use_engagement_branch=bool(args.engagement_branch),
        use_science_features=bool(args.science_features),
        use_aesthetic_mlp=bool(args.aesthetic_mlp),
        use_temporal_encoder=bool(args.temporal_encoder),
        use_cross_modal_attention=bool(args.cross_modal_attn),
        temporal_dim=CFG.temporal_dim,
        temporal_arch=CFG.temporal_arch,
        temporal_num_layers=CFG.temporal_num_layers,
        temporal_num_heads=CFG.temporal_num_heads,
        cross_modal_num_heads=CFG.cross_modal_num_heads,
        cross_modal_dropout=CFG.cross_modal_dropout,
    ).to(device)

    # Warm-start from MVP checkpoint
    pretrained_path = Path(args.pretrained)
    if pretrained_path.exists():
        n_matched = _load_pretrained_weights(model, str(pretrained_path), device)
        logger.info("Loaded %d matched weights from %s", n_matched, pretrained_path)

    optimizer = Adam(model.parameters(), lr=args.lr)

    best_rank_acc = -1.0
    ckpt_path = Path(args.checkpoint)
    ensure_dir(ckpt_path.parent)

    config_summary = (
        f"QH={args.quality_head} CL={args.consistency_loss} EG={args.engagement_branch} "
        f"SF={args.science_features} AM={args.aesthetic_mlp} TE={args.temporal_encoder} "
        f"CM={args.cross_modal_attn} DV={args.diversity_loss}"
    )
    logger.info("Research training config: %s", config_summary)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for pos_batch, neg_batch in train_loader:
            pos_batch = move_batch_to_device(pos_batch, device)
            neg_batch = move_batch_to_device(neg_batch, device)

            pos_eng = pos_batch.get("engagement_target",
                torch.zeros(pos_batch["text_feat"].size(0), 1, device=device))
            pos_kwargs = _build_kwargs(pos_batch)
            neg_kwargs = _build_kwargs(neg_batch)

            pos_out = model(**pos_kwargs)
            neg_out = model(**neg_kwargs)

            if args.use_mvp_loss:
                rank_loss = pairwise_ranking_loss(pos_out["overall_score"], neg_out["overall_score"], margin=args.margin)
                reg_loss = branch_consistency_loss(
                    pos_out["overall_score"], pos_out["scientific_score"],
                    pos_out["technical_score"], pos_out["aesthetic_score"],
                )
                loss = rank_loss + 0.1 * reg_loss
            else:
                loss, loss_dict = compute_research_total_loss(
                    pos_out, neg_out,
                    margin=args.margin,
                    lambda_consistency=CFG.lambda_consistency if args.consistency_loss else 0.0,
                    lambda_diversity=CFG.lambda_diversity if args.diversity_loss else 0.0,
                    engagement_target=pos_eng,
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        val_loss, val_metrics = evaluate_research(model, val_loader, device, args)
        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | "
            "rank_acc=%.4f | acc=%.4f | f1=%.4f | auc=%.4f",
            epoch, args.epochs, train_loss, val_loss,
            val_metrics["ranking_accuracy"], val_metrics["accuracy"],
            val_metrics["f1"], val_metrics["auc"],
        )

        if val_metrics["ranking_accuracy"] > best_rank_acc:
            best_rank_acc = val_metrics["ranking_accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "text_dim": CFG.text_dim, "video_dim": CFG.video_dim,
                        "audio_dim": CFG.audio_dim, "meta_dim": CFG.meta_dim,
                        "aes_dim": CFG.aes_dim, "hidden_dim": CFG.hidden_dim,
                        "quality_head": bool(args.quality_head),
                        "engagement_branch": bool(args.engagement_branch),
                        "science_features": bool(args.science_features),
                        "cross_modal_attn": bool(args.cross_modal_attn),
                        "temporal_encoder": bool(args.temporal_encoder),
                    },
                },
                ckpt_path,
            )
            logger.info("Saved best checkpoint to %s", ckpt_path)

    logger.info("Training finished. best_ranking_accuracy=%.4f", best_rank_acc)


if __name__ == "__main__":
    main()
