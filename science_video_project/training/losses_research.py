"""
§5 + §11 研究版损失函数

新增:
  - quality_consistency_loss (§5):  L_cons = MSE(quality_score, overall_score)
  - branch_diversity_loss  (§11): 最小化分支隐向量的余弦相似度
  - engagement_aux_loss   (§6):  BCE 传播力辅助损失

总损失配方 (§11):
  L_total = L_rank + λ_cons * L_cons + λ_div * L_div (+ λ_eng * L_eng)
"""

import torch
import torch.nn.functional as F


# ============================================================================
#  保留原有损失
# ============================================================================

from training.losses import pairwise_ranking_loss, branch_consistency_loss  # noqa: F401


# ============================================================================
#  §5 Quality Consistency Loss
# ============================================================================

def quality_consistency_loss(
    quality_score: torch.Tensor,
    overall_score: torch.Tensor,
) -> torch.Tensor:
    """约束 quality_score ≈ overall_score，使排序结果具备质量解释性。

    Args:
        quality_score: (B, 1) — QualityHead 输出
        overall_score: (B, 1) — 模型 overall_head 输出

    Returns:
        scalar loss
    """
    return F.mse_loss(quality_score, overall_score.detach())


# ============================================================================
#  §11 Branch Diversity Loss
# ============================================================================

def branch_diversity_loss(
    sci_hidden: torch.Tensor,
    tech_hidden: torch.Tensor,
    aes_hidden: torch.Tensor,
) -> torch.Tensor:
    """最小化三个分支隐向量之间的余弦相似度，鼓励分支学不同内容。

    最小化: cos(sci, tech) + cos(sci, aes) + cos(tech, aes)

    Args:
        sci_hidden:  (B, D)
        tech_hidden: (B, D)
        aes_hidden:  (B, D)

    Returns:
        scalar loss — 越小表示分支越多样化
    """
    # 归一化
    sci_norm = F.normalize(sci_hidden, dim=-1)
    tech_norm = F.normalize(tech_hidden, dim=-1)
    aes_norm = F.normalize(aes_hidden, dim=-1)

    cos_st = (sci_norm * tech_norm).sum(dim=-1).mean()
    cos_sa = (sci_norm * aes_norm).sum(dim=-1).mean()
    cos_ta = (tech_norm * aes_norm).sum(dim=-1).mean()

    return (cos_st + cos_sa + cos_ta) / 3.0


# ============================================================================
#  §6 Engagement Auxiliary Loss
# ============================================================================

def engagement_aux_loss(
    engagement_score: torch.Tensor,
    engagement_target: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        engagement_score:  (B, 1) — EngagementBranch 预测
        engagement_target: (B, 1) — 标签（是否为高传播样本）

    Returns:
        BCE loss
    """
    return F.binary_cross_entropy(engagement_score, engagement_target)


# ============================================================================
#  总损失计算
# ============================================================================

def compute_research_total_loss(
    pos_out: dict[str, torch.Tensor],
    neg_out: dict[str, torch.Tensor],
    margin: float = 1.0,
    lambda_consistency: float = 0.2,
    lambda_diversity: float = 0.05,
    lambda_engagement: float = 0.0,
    engagement_target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """计算研究版总损失。

    总损失 = L_rank
            + λ_cons * (L_cons_pos + L_cons_neg) / 2
            + λ_div * L_div
            [+ λ_eng * L_eng]

    Returns:
        total_loss, loss_dict (用于日志)
    """
    # 1) Pairwise Ranking Loss
    L_rank = pairwise_ranking_loss(
        pos_out["overall_score"], neg_out["overall_score"], margin=margin
    )

    loss_dict = {"rank": float(L_rank.item())}

    # 2) Quality Consistency Loss
    L_cons = torch.tensor(0.0, device=L_rank.device)
    if "quality_score" in pos_out and "quality_score" in neg_out:
        L_cons_pos = quality_consistency_loss(pos_out["quality_score"], pos_out["overall_score"])
        L_cons_neg = quality_consistency_loss(neg_out["quality_score"], neg_out["overall_score"])
        L_cons = (L_cons_pos + L_cons_neg) / 2.0
        loss_dict["consistency"] = float(L_cons.item())

    # 3) Branch Diversity Loss
    L_div = torch.tensor(0.0, device=L_rank.device)
    if "scientific_hidden" in pos_out:
        L_div = branch_diversity_loss(
            pos_out["scientific_hidden"],
            pos_out["technical_hidden"],
            pos_out["aesthetic_hidden"],
        )
        loss_dict["diversity"] = float(L_div.item())

    total = L_rank + lambda_consistency * L_cons + lambda_diversity * L_div

    # 4) Engagement Aux Loss
    L_eng = torch.tensor(0.0, device=L_rank.device)
    if "engagement_score" in pos_out and engagement_target is not None:
        L_eng = engagement_aux_loss(pos_out["engagement_score"], engagement_target)
        total = total + lambda_engagement * L_eng
        loss_dict["engagement"] = float(L_eng.item())

    loss_dict["total"] = float(total.item())
    return total, loss_dict
