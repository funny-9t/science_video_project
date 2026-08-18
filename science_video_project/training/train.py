import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score
from scipy.stats import spearmanr
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import CFG
from pipeline.utils_io import ensure_dir, load_metadata
from training.dataloader_pair import collate_pair
from training.dataset_pair import PairwiseVideoDataset
from training.losses import (
    branch_consistency_loss,
    branch_supervision_loss,
    focal_pairwise_ranking_loss,
    pairwise_ranking_loss,
    ranknet_pairwise_loss,
    weighted_pairwise_ranking_loss,
)
from training.metrics import compute_pair_metrics, compute_video_metrics
from training.model_mvp import MultiModalQualityModel
from training.utils_train import (
    build_logger,
    build_model_config_from_cfg,
    get_device,
    move_batch_to_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pairwise ranking model for science video quality")
    parser.add_argument("--metadata", default=str(CFG.metadata_csv))
    parser.add_argument("--feature_dir", default=str(CFG.feature_dir))
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--lr", type=float, default=CFG.lr)
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--split_seed", type=int, default=CFG.seed)
    parser.add_argument("--disable_audio", action="store_true")
    parser.add_argument("--use_cover_features", action="store_true")
    parser.add_argument("--use_native_clip", action="store_true")
    parser.add_argument("--disable_cross_gating", action="store_true")
    parser.add_argument("--fusion_mode", choices=["learned", "average"], default=CFG.fusion_mode)
    parser.add_argument(
        "--branch_weight_floor",
        type=float,
        default=CFG.branch_weight_floor,
        help="Minimum learned fusion weight assigned to each quality branch.",
    )
    parser.add_argument("--margin", type=float, default=CFG.margin)
    parser.add_argument("--same_category", action="store_true", default=CFG.same_category_pair)
    parser.add_argument(
        "--pair_scope",
        choices=["same_category", "global"],
        default="same_category" if CFG.same_category_pair else "global",
        help="Pair positive/negative samples within the same category or globally.",
    )
    parser.add_argument(
        "--loss_type",
        choices=["ranknet", "focal", "weighted", "margin"],
        default="ranknet",
    )
    parser.add_argument("--early_stop_patience", type=int, default=CFG.early_stop_patience)
    parser.add_argument("--lambda_supervision", type=float, default=CFG.lambda_supervision)
    parser.add_argument("--lambda_consistency", type=float, default=0.1)
    parser.add_argument("--lambda_pointwise", type=float, default=0.1)
    parser.add_argument("--branch_pretrain_epochs", type=int, default=0)
    parser.add_argument("--branch_pretrain_supervision", type=float, default=1.0)
    parser.add_argument("--pos_weight", type=float, default=CFG.pos_weight)
    parser.add_argument(
        "--threshold_metric",
        choices=["f1", "balanced_accuracy", "youden"],
        default="f1",
        help="Metric used to choose the classification threshold on validation pairs.",
    )
    parser.add_argument(
        "--science_feature_mode",
        choices=[
            "none",
            "scores",
            "analysis_concat",
            "analysis_scores_concat",
            "analysis_scores_ifg",
            "full_ifg",
        ],
        default="analysis_scores_concat",
    )
    parser.add_argument(
        "--llm_text_source",
        choices=["analysis", "reasoning_and_analysis"],
        default="reasoning_and_analysis",
    )
    parser.add_argument("--checkpoint", default=str(CFG.checkpoint_dir / "best.pt"))
    return parser.parse_args()


def configure_science_features(
    batch: dict[str, torch.Tensor],
    mode: str,
    text_source: str,
) -> dict[str, torch.Tensor]:
    source_key = (
        "llm_analysis_only_feat"
        if text_source == "analysis"
        else "llm_reasoning_analysis_feat"
    )
    batch["llm_analysis_feat"] = batch[source_key]

    if mode in {"none", "scores"}:
        batch["llm_analysis_feat"] = torch.zeros_like(batch["llm_analysis_feat"])
    if mode in {"none", "analysis_concat"}:
        batch["llm_knowledge_feat"] = torch.zeros_like(batch["llm_knowledge_feat"])
    if mode != "full_ifg":
        batch["sci_hand_feat"] = torch.zeros_like(batch["sci_hand_feat"])
    return batch


def configure_audio_feature(
    batch: dict[str, torch.Tensor],
    use_audio_feature: bool,
) -> dict[str, torch.Tensor]:
    if not use_audio_feature:
        batch["audio_feat"] = torch.zeros_like(batch["audio_feat"])
    return batch


def pretrain_branches(
    model: MultiModalQualityModel,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    supervision_weight: float,
    science_feature_mode: str,
    llm_text_source: str,
    use_audio_feature: bool,
    logger,
) -> None:
    """COVER-style limited-view pretraining for independently useful branches."""
    if epochs <= 0:
        return
    branch_parameters = list(model.scientific_branch.parameters())
    branch_parameters += list(model.technical_branch.parameters())
    branch_parameters += list(model.aesthetic_branch.parameters())
    optimizer = AdamW(branch_parameters, lr=lr, weight_decay=CFG.weight_decay)
    cross_gating_enabled = model.use_cross_gating
    model.use_cross_gating = False

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for pos_batch, neg_batch in loader:
            pos_batch = move_batch_to_device(pos_batch, device)
            neg_batch = move_batch_to_device(neg_batch, device)
            pos_batch = configure_science_features(
                pos_batch, science_feature_mode, llm_text_source
            )
            neg_batch = configure_science_features(
                neg_batch, science_feature_mode, llm_text_source
            )
            pos_batch = configure_audio_feature(pos_batch, use_audio_feature)
            neg_batch = configure_audio_feature(neg_batch, use_audio_feature)
            pos_out = model(**pos_batch)
            neg_out = model(**neg_batch)

            branch_rank = sum(
                ranknet_pairwise_loss(pos_out[key], neg_out[key])
                for key in ("scientific_score", "technical_score", "aesthetic_score")
            ) / 3.0
            supervision = (
                branch_supervision_loss(
                    pos_out["scientific_score"], pos_out["technical_score"],
                    pos_out["aesthetic_score"], pos_batch["sci_target"],
                    pos_batch["tech_target"], pos_batch["aes_target"],
                )
                + branch_supervision_loss(
                    neg_out["scientific_score"], neg_out["technical_score"],
                    neg_out["aesthetic_score"], neg_batch["sci_target"],
                    neg_batch["tech_target"], neg_batch["aes_target"],
                )
            ) / 2.0
            loss = branch_rank + supervision_weight * supervision
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        logger.info(
            "Branch pretrain %d/%d | loss=%.4f",
            epoch, epochs, total / max(len(loader), 1),
        )
    model.use_cross_gating = cross_gating_enabled


def evaluate(
    model: MultiModalQualityModel,
    loader: DataLoader,
    device: torch.device,
    margin: float,
    threshold: float,
    lambda_supervision: float = 0.1,
    lambda_consistency: float = 0.1,
    loss_type: str = "focal",
    pos_weight: float = 5.0,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.85,
    science_feature_mode: str = "full_ifg",
    llm_text_source: str = "reasoning_and_analysis",
    use_audio_feature: bool = True,
    lambda_pointwise: float = 0.0,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    pos_probs = []
    neg_probs = []

    with torch.no_grad():
        for pos_batch, neg_batch in loader:
            pos_batch = move_batch_to_device(pos_batch, device)
            neg_batch = move_batch_to_device(neg_batch, device)
            pos_batch = configure_science_features(pos_batch, science_feature_mode, llm_text_source)
            neg_batch = configure_science_features(neg_batch, science_feature_mode, llm_text_source)
            pos_batch = configure_audio_feature(pos_batch, use_audio_feature)
            neg_batch = configure_audio_feature(neg_batch, use_audio_feature)

            pos_out = model(**pos_batch)
            neg_out = model(**neg_batch)

            if loss_type == "ranknet":
                rank_loss = ranknet_pairwise_loss(
                    pos_out["overall_score"], neg_out["overall_score"]
                )
            elif loss_type == "focal":
                rank_loss = focal_pairwise_ranking_loss(
                    pos_out["overall_score"], neg_out["overall_score"],
                    margin=margin, gamma=focal_gamma, alpha=focal_alpha,
                )
            elif loss_type == "weighted":
                rank_loss = weighted_pairwise_ranking_loss(
                    pos_out["overall_score"], neg_out["overall_score"],
                    pos_weight=pos_weight, margin=margin,
                )
            else:
                rank_loss = pairwise_ranking_loss(
                    pos_out["overall_score"], neg_out["overall_score"],
                    margin=margin,
                )
            reg_loss = branch_consistency_loss(
                pos_out["overall_score"],
                pos_out["scientific_score"],
                pos_out["technical_score"],
                pos_out["aesthetic_score"],
            )
            point_loss = (
                F.binary_cross_entropy_with_logits(
                    pos_out["overall_score"], torch.ones_like(pos_out["overall_score"])
                )
                + F.binary_cross_entropy_with_logits(
                    neg_out["overall_score"], torch.zeros_like(neg_out["overall_score"])
                )
            ) / 2.0
            loss = rank_loss + lambda_consistency * reg_loss + lambda_pointwise * point_loss

            # ── 细粒度分支监督 loss (eval 阶段同样计入，观察过拟合) ──
            sup_loss_pos = branch_supervision_loss(
                pos_out["scientific_score"], pos_out["technical_score"], pos_out["aesthetic_score"],
                pos_batch["sci_target"], pos_batch["tech_target"], pos_batch["aes_target"],
            )
            sup_loss_neg = branch_supervision_loss(
                neg_out["scientific_score"], neg_out["technical_score"], neg_out["aesthetic_score"],
                neg_batch["sci_target"], neg_batch["tech_target"], neg_batch["aes_target"],
            )
            loss = loss + lambda_supervision * (sup_loss_pos + sup_loss_neg) / 2.0
            total_loss += float(loss.item())

            pos_probs.append(pos_out["probability"].squeeze(-1).detach().cpu().numpy())
            neg_probs.append(neg_out["probability"].squeeze(-1).detach().cpu().numpy())

    if not pos_probs or not neg_probs:
        return total_loss, {"accuracy": 0.0, "f1": 0.0, "auc": 0.0, "ranking_accuracy": 0.0}

    pos_prob = np.concatenate(pos_probs, axis=0)
    neg_prob = np.concatenate(neg_probs, axis=0)
    metrics = compute_pair_metrics(pos_prob, neg_prob, threshold=threshold)
    return total_loss, metrics


def _search_best_threshold(
    pos_prob: np.ndarray,
    neg_prob: np.ndarray,
    metric: str = "f1",
) -> tuple[float, float]:
    """搜索最佳分类阈值，最大化验证集 F1。

    对 [0.1, 0.9] 范围内以 0.05 步长搜索，返回 (best_threshold, best_f1)。
    """
    y_true = np.concatenate([np.ones_like(pos_prob), np.zeros_like(neg_prob)], axis=0).astype(int)
    y_score = np.concatenate([pos_prob, neg_prob], axis=0)

    best_thr = 0.5
    best_f1 = 0.0
    for thr in np.arange(0.1, 0.95, 0.05):
        y_pred = (y_score >= thr).astype(int)
        if metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, y_pred)
        elif metric == "youden":
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            score = tpr - fpr
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thr = thr
    return best_thr, best_f1


def main() -> None:
    args = parse_args()
    ensure_dir(CFG.checkpoint_dir)
    ensure_dir(CFG.log_dir)

    logger = build_logger(CFG.log_dir / "train.log", name="train")
    set_seed(args.seed)

    def _ensure_both_labels(train_df, val_df):
        for label in (0, 1):
            if (val_df["label"] == label).sum() == 0:
                candidates = train_df[train_df["label"] == label]
                if candidates.empty:
                    raise ValueError("Both positive and negative samples are required for training.")
                moved = candidates.sample(n=1, random_state=args.split_seed)
                train_df = train_df.drop(moved.index)
                val_df = pd.concat([val_df, moved], ignore_index=True)

        for label in (0, 1):
            if (train_df["label"] == label).sum() == 0:
                raise ValueError("Training split lacks required label after adjustment.")
        return train_df, val_df

    def _filter_with_features(metadata_df, feature_dir: str | Path):
        feature_dir = Path(feature_dir)
        available = {p.stem for p in feature_dir.glob("*.pt")}
        metadata_ids = metadata_df["video_id"].astype(str)
        has_features = metadata_ids.isin(available)
        filtered = metadata_df[has_features].copy()
        if filtered.empty:
            raise ValueError("No metadata rows have matching feature files.")

        missing = metadata_df[~has_features]
        coverage = len(filtered) / len(metadata_df)
        filtered_labels = filtered["label"].value_counts().sort_index().to_dict()
        missing_labels = missing["label"].value_counts().sort_index().to_dict()
        logger.info(
            "Training feature coverage: %d/%d (%.1f%%) | labels=%s | feature_files=%d",
            len(filtered),
            len(metadata_df),
            coverage * 100.0,
            filtered_labels,
            len(available),
        )
        if not missing.empty:
            logger.warning(
                "Excluded %d metadata videos without feature files | labels=%s. "
                "Run pipeline/run_pipeline.py to resume feature extraction.",
                len(missing),
                missing_labels,
            )
        orphan_count = len(available - set(metadata_ids))
        if orphan_count:
            logger.warning(
                "Feature directory contains %d files outside the cleaned training metadata.",
                orphan_count,
            )
        return filtered

    try:
        metadata = load_metadata(args.metadata)
        metadata = _filter_with_features(metadata, args.feature_dir)
        train_df, val_df = train_test_split(
            metadata,
            test_size=CFG.val_ratio,
            random_state=args.split_seed,
            stratify=metadata["label"],
        )
        train_df, val_df = _ensure_both_labels(train_df, val_df)
    except ValueError:
        metadata = load_metadata(args.metadata)
        metadata = _filter_with_features(metadata, args.feature_dir)
        train_df, val_df = train_test_split(metadata, test_size=CFG.val_ratio, random_state=args.split_seed)
        train_df, val_df = _ensure_both_labels(train_df, val_df)
    except Exception as exc:
        logger.exception("Failed to load/split metadata: %s", exc)
        raise

    same_category_pair = args.pair_scope == "same_category"

    train_ds = PairwiseVideoDataset.from_metadata(
        train_df,
        args.feature_dir,
        same_category=same_category_pair,
        deterministic_pairs=False,
        use_native_clip=args.use_native_clip,
        use_cover_features=args.use_cover_features,
    )
    val_ds = PairwiseVideoDataset.from_metadata(
        val_df,
        args.feature_dir,
        same_category=same_category_pair,
        deterministic_pairs=True,
        use_native_clip=args.use_native_clip,
        use_cover_features=args.use_cover_features,
    )

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
    model_config = build_model_config_from_cfg(CFG)
    model_config["use_knowledge_gate"] = args.science_feature_mode not in {
        "analysis_concat",
        "analysis_scores_concat",
    }
    model_config["use_cover_features"] = args.use_cover_features
    if args.use_native_clip:
        model_config["video_dim"] = CFG.clip_video_dim
    model_config["fusion_mode"] = args.fusion_mode
    model_config["branch_weight_floor"] = args.branch_weight_floor
    if args.disable_cross_gating:
        model_config["use_cross_gating"] = False
    model = MultiModalQualityModel(**model_config).to(device)
    use_audio_feature = not args.disable_audio
    logger.info(
        "Science feature mode=%s | llm_text_source=%s | knowledge_gate=%s | audio=%s | cover=%s | native_clip=%s | fusion=%s | branch_floor=%.2f | cross_gate=%s",
        args.science_feature_mode,
        args.llm_text_source,
        model_config["use_knowledge_gate"],
        use_audio_feature,
        args.use_cover_features,
        args.use_native_clip,
        args.fusion_mode,
        args.branch_weight_floor,
        model_config["use_cross_gating"],
    )
    pretrain_branches(
        model=model,
        loader=train_loader,
        device=device,
        epochs=args.branch_pretrain_epochs,
        lr=args.lr,
        supervision_weight=args.branch_pretrain_supervision,
        science_feature_mode=args.science_feature_mode,
        llm_text_source=args.llm_text_source,
        use_audio_feature=use_audio_feature,
        logger=logger,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=CFG.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=CFG.scheduler_factor,
        patience=CFG.scheduler_patience, min_lr=1e-6,
    )

    best_rank_acc = -1.0
    best_epoch = 0
    best_checkpoint_threshold = 0.5
    patience = args.early_stop_patience
    no_improve = 0
    ckpt_path = Path(args.checkpoint)
    ensure_dir(ckpt_path.parent)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for pos_batch, neg_batch in train_loader:
            pos_batch = move_batch_to_device(pos_batch, device)
            neg_batch = move_batch_to_device(neg_batch, device)
            pos_batch = configure_science_features(pos_batch, args.science_feature_mode, args.llm_text_source)
            neg_batch = configure_science_features(neg_batch, args.science_feature_mode, args.llm_text_source)
            pos_batch = configure_audio_feature(pos_batch, use_audio_feature)
            neg_batch = configure_audio_feature(neg_batch, use_audio_feature)

            pos_out = model(**pos_batch)
            neg_out = model(**neg_batch)

            if args.loss_type == "ranknet":
                rank_loss = ranknet_pairwise_loss(
                    pos_out["overall_score"], neg_out["overall_score"]
                )
            elif args.loss_type == "focal":
                rank_loss = focal_pairwise_ranking_loss(
                    pos_out["overall_score"], neg_out["overall_score"],
                    margin=args.margin, gamma=CFG.focal_gamma, alpha=CFG.focal_alpha,
                )
            elif args.loss_type == "weighted":
                rank_loss = weighted_pairwise_ranking_loss(
                    pos_out["overall_score"], neg_out["overall_score"],
                    pos_weight=args.pos_weight, margin=args.margin,
                )
            else:
                rank_loss = pairwise_ranking_loss(
                    pos_out["overall_score"], neg_out["overall_score"],
                    margin=args.margin,
                )
            reg_loss = branch_consistency_loss(
                pos_out["overall_score"],
                pos_out["scientific_score"],
                pos_out["technical_score"],
                pos_out["aesthetic_score"],
            )

            # ── 细粒度分支监督 (降低权重，避免主导训练) ──
            lambda_sup = args.lambda_supervision
            sup_loss_pos = branch_supervision_loss(
                pos_out["scientific_score"], pos_out["technical_score"], pos_out["aesthetic_score"],
                pos_batch["sci_target"], pos_batch["tech_target"], pos_batch["aes_target"],
            )
            sup_loss_neg = branch_supervision_loss(
                neg_out["scientific_score"], neg_out["technical_score"], neg_out["aesthetic_score"],
                neg_batch["sci_target"], neg_batch["tech_target"], neg_batch["aes_target"],
            )
            point_loss = (
                F.binary_cross_entropy_with_logits(
                    pos_out["overall_score"], torch.ones_like(pos_out["overall_score"])
                )
                + F.binary_cross_entropy_with_logits(
                    neg_out["overall_score"], torch.zeros_like(neg_out["overall_score"])
                )
            ) / 2.0
            loss = (
                rank_loss
                + args.lambda_consistency * reg_loss
                + lambda_sup * (sup_loss_pos + sup_loss_neg) / 2.0
                + args.lambda_pointwise * point_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        val_loss, val_metrics = evaluate(
            model, val_loader, device, args.margin, CFG.threshold,
            lambda_supervision=args.lambda_supervision,
            lambda_consistency=args.lambda_consistency,
            loss_type=args.loss_type, pos_weight=args.pos_weight,
            focal_gamma=CFG.focal_gamma, focal_alpha=CFG.focal_alpha,
            science_feature_mode=args.science_feature_mode,
            llm_text_source=args.llm_text_source,
            use_audio_feature=use_audio_feature,
            lambda_pointwise=args.lambda_pointwise,
        )

        # Search the threshold over unique videos, not repeated validation pairs.
        model.eval()
        pos_scores_by_id, neg_scores_by_id = {}, {}
        branch_records = {
            "scientific": {}, "technical": {}, "aesthetic": {},
        }
        fusion_weight_records = {}
        with torch.no_grad():
            for pos_batch, neg_batch in val_loader:
                pos_batch = move_batch_to_device(pos_batch, device)
                neg_batch = move_batch_to_device(neg_batch, device)
                pos_batch = configure_science_features(pos_batch, args.science_feature_mode, args.llm_text_source)
                neg_batch = configure_science_features(neg_batch, args.science_feature_mode, args.llm_text_source)
                pos_batch = configure_audio_feature(pos_batch, use_audio_feature)
                neg_batch = configure_audio_feature(neg_batch, use_audio_feature)
                pos_out = model(**pos_batch)
                neg_out = model(**neg_batch)
                pos_values = pos_out["probability"].squeeze(-1).detach().cpu().numpy()
                neg_values = neg_out["probability"].squeeze(-1).detach().cpu().numpy()
                pos_scores_by_id.update(zip(pos_batch["video_id"], pos_values.tolist()))
                neg_scores_by_id.update(zip(neg_batch["video_id"], neg_values.tolist()))
                for batch, output in ((pos_batch, pos_out), (neg_batch, neg_out)):
                    weights = output["branch_weights"].detach().cpu().numpy()
                    fusion_weight_records.update(
                        {
                            video_id: weight.astype(np.float32)
                            for video_id, weight in zip(batch["video_id"], weights)
                        }
                    )
                for prefix, score_key, target_key in (
                    ("scientific", "scientific_score", "sci_target"),
                    ("technical", "technical_score", "tech_target"),
                    ("aesthetic", "aesthetic_score", "aes_target"),
                ):
                    for batch, output in ((pos_batch, pos_out), (neg_batch, neg_out)):
                        predictions = torch.sigmoid(output[score_key]).squeeze(-1).detach().cpu().numpy()
                        targets = batch[target_key].squeeze(-1).detach().cpu().numpy()
                        branch_records[prefix].update(
                            {
                                video_id: (float(prediction), float(target))
                                for video_id, prediction, target in zip(
                                    batch["video_id"], predictions, targets
                                )
                            }
                        )
        if pos_scores_by_id and neg_scores_by_id:
            pos_prob_val = np.asarray(list(pos_scores_by_id.values()), dtype=np.float32)
            neg_prob_val = np.asarray(list(neg_scores_by_id.values()), dtype=np.float32)
            epoch_best_threshold, best_thr_metric = _search_best_threshold(
                pos_prob_val,
                neg_prob_val,
                metric=args.threshold_metric,
            )
            y_true_unique = np.concatenate([
                np.ones_like(pos_prob_val), np.zeros_like(neg_prob_val)
            ]).astype(int)
            y_score_unique = np.concatenate([pos_prob_val, neg_prob_val])
            unique_metrics = compute_video_metrics(
                y_true_unique, y_score_unique, threshold=epoch_best_threshold
            )
            val_metrics_thr = dict(val_metrics)
            val_metrics_thr.update(unique_metrics)
            for prefix, records in branch_records.items():
                values = list(records.values())
                predictions = np.asarray([value[0] for value in values], dtype=np.float32)
                targets = np.asarray([value[1] for value in values], dtype=np.float32)
                valid = targets >= 0
                correlation = spearmanr(predictions[valid], targets[valid]).statistic
                val_metrics_thr[f"{prefix}_srcc"] = (
                    float(correlation) if np.isfinite(correlation) else 0.0
                )
            if fusion_weight_records:
                mean_weights = np.stack(list(fusion_weight_records.values())).mean(axis=0)
                for prefix, weight in zip(
                    ("scientific", "technical", "aesthetic"), mean_weights
                ):
                    val_metrics_thr[f"{prefix}_fusion_weight"] = float(weight)
        else:
            epoch_best_threshold = CFG.threshold
            val_metrics_thr = val_metrics
            best_thr_metric = 0.0
        model.train()

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | rank_acc=%.4f | acc=%.4f | f1=%.4f | auc=%.4f | "
            "best_thr=%.2f | thr_metric=%.4f | lr=%.2e",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            val_metrics_thr["ranking_accuracy"],
            val_metrics_thr["accuracy"],
            val_metrics_thr["f1"],
            val_metrics_thr["auc"],
            epoch_best_threshold,
            best_thr_metric,
            optimizer.param_groups[0]["lr"],
        )

        # ── ReduceLROnPlateau：基于 val_loss 调整学习率 ──
        scheduler.step(val_loss)

        if val_metrics_thr["ranking_accuracy"] > best_rank_acc:
            best_rank_acc = val_metrics_thr["ranking_accuracy"]
            best_epoch = epoch
            best_checkpoint_threshold = epoch_best_threshold
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_threshold": best_checkpoint_threshold,
                    "config": model_config,
                    "science_feature_mode": args.science_feature_mode,
                    "llm_text_source": args.llm_text_source,
                    "use_audio_feature": use_audio_feature,
                    "seed": args.seed,
                    "split_seed": args.split_seed,
                    "training_config": {
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "seed": args.seed,
                        "split_seed": args.split_seed,
                        "use_audio_feature": use_audio_feature,
                        "use_native_clip": args.use_native_clip,
                        "use_cover_features": args.use_cover_features,
                        "use_cross_gating": model_config["use_cross_gating"],
                        "margin": args.margin,
                        "pair_scope": args.pair_scope,
                        "loss_type": args.loss_type,
                        "early_stop_patience": args.early_stop_patience,
                        "lambda_supervision": args.lambda_supervision,
                        "lambda_consistency": args.lambda_consistency,
                        "lambda_pointwise": args.lambda_pointwise,
                        "branch_pretrain_epochs": args.branch_pretrain_epochs,
                        "branch_pretrain_supervision": args.branch_pretrain_supervision,
                    },
                    "best_ranking_accuracy": best_rank_acc,
                    "best_epoch": best_epoch,
                    "validation_metrics": val_metrics_thr,
                },
                ckpt_path,
            )
            logger.info("Saved best checkpoint to %s (threshold=%.2f)", ckpt_path, best_checkpoint_threshold)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, patience)
                break

    logger.info("Training finished. best_ranking_accuracy=%.4f @ epoch %d, best_threshold=%.2f",
                best_rank_acc, best_epoch, best_checkpoint_threshold)


if __name__ == "__main__":
    main()
