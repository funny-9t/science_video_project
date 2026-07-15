import torch
import torch.nn.functional as F


def pairwise_ranking_loss(pos_score: torch.Tensor, neg_score: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    target = torch.ones_like(pos_score)
    return F.margin_ranking_loss(pos_score, neg_score, target, margin=margin)


def weighted_pairwise_ranking_loss(
    pos_score: torch.Tensor,
    neg_score: torch.Tensor,
    pos_weight: float = 5.0,
    margin: float = 0.5,
) -> torch.Tensor:
    """加权排序损失：对正样本被误判为低分施加更高惩罚。

    标准 margin ranking: max(0, margin - (pos - neg))
    增强版：对 pos < neg + margin 的情况额外加权 pos_weight 倍
    """
    diff = pos_score - neg_score
    # 基础 loss
    base_loss = F.relu(margin - diff)
    # 正样本偏低时额外加权
    pos_low_mask = (pos_score < 0.5).float()  # sigmoid < 0.5 → 偏向负类
    weighted = base_loss * (1.0 + pos_weight * pos_low_mask)
    return weighted.mean()


def focal_pairwise_ranking_loss(
    pos_score: torch.Tensor,
    neg_score: torch.Tensor,
    margin: float = 0.5,
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> torch.Tensor:
    """Focal Ranking Loss：对极不平衡的正负样本对进行自适应加权。

    核心思路源自 Focal Loss (Lin et al., 2017):
      - 当模型已经能正确排序 (pos - neg > margin) 时，loss 趋近于 0
      - 当模型排序错误时，给予更高的权重
      - alpha 控制正样本的整体权重

    公式:
      L = alpha * (relu(margin - (pos - neg)))^gamma

    当 gamma=0 时退化为普通 margin ranking loss（带 alpha 加权）.
    当 gamma=2 时，难样本的 loss 是简单样本的 ~4-9 倍。

    参数:
        pos_score: (B, 1) 正样本的 overall_score (logit, 未经过 sigmoid)
        neg_score: (B, 1) 负样本的 overall_score (logit, 未经过 sigmoid)
        margin: margin 值, 默认 0.5
        gamma: focal 指数, 越大对难样本越关注, 推荐 1.5~3.0
        alpha: 正样本权重系数, 推荐 0.75~0.95（越大越关注正样本）
    """
    diff = pos_score - neg_score
    # relu(margin - diff) 是基础 ranking loss
    base_loss = F.relu(margin - diff)
    # focal 调制因子: (loss)^gamma
    focal_weight = base_loss.pow(gamma)
    # alpha 加权：对每个 pair 用 alpha 增强正样本侧
    weighted = alpha * focal_weight * base_loss
    return weighted.mean()


def branch_consistency_loss(
    overall_score: torch.Tensor,
    scientific_score: torch.Tensor,
    technical_score: torch.Tensor,
    aesthetic_score: torch.Tensor,
) -> torch.Tensor:
    avg_branch = (scientific_score + technical_score + aesthetic_score) / 3.0
    return F.mse_loss(overall_score, avg_branch)


def branch_supervision_loss(
    scientific_score: torch.Tensor,
    technical_score: torch.Tensor,
    aesthetic_score: torch.Tensor,
    sci_target: torch.Tensor,
    tech_target: torch.Tensor,
    aes_target: torch.Tensor,
) -> torch.Tensor:
    """用七个细粒度评分监督三条分支输出。

    三个分支输出的 score 经过 sigmoid 归一化到 [0,1] 后与人工标注的
    聚合目标 (sci_target / tech_target / aes_target) 计算 MSE。
    对正样本和负样本都施加此损失。

    参数:
        scientific_score: (B, 1) 模型输出的科学分支原始 logit
        technical_score:  (B, 1) 模型输出的技术分支原始 logit
        aesthetic_score:  (B, 1) 模型输出的美学分支原始 logit
        sci_target:  (B, 1) 人工标注的科学分聚合值 [0,1]
        tech_target: (B, 1) 人工标注的技术分聚合值 [0,1]
        aes_target:  (B, 1) 人工标注的美学分聚合值 [0,1]

    返回:
        scalar loss，仅对有效目标 (>0 after sigmoid check, target >= 0) 的样本计算
    """
    sci_prob = torch.sigmoid(scientific_score)
    tech_prob = torch.sigmoid(technical_score)
    aes_prob = torch.sigmoid(aesthetic_score)

    # 只对提供了人工标注的样本 (target >= 0) 计算 loss
    sci_mask = (sci_target >= 0).float()
    tech_mask = (tech_target >= 0).float()
    aes_mask = (aes_target >= 0).float()

    sci_loss = (sci_mask * (sci_prob - sci_target) ** 2).sum() / (sci_mask.sum() + 1e-8)
    tech_loss = (tech_mask * (tech_prob - tech_target) ** 2).sum() / (tech_mask.sum() + 1e-8)
    aes_loss = (aes_mask * (aes_prob - aes_target) ** 2).sum() / (aes_mask.sum() + 1e-8)

    return (sci_loss + tech_loss + aes_loss) / 3.0
