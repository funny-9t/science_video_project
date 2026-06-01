import torch
import torch.nn.functional as F


def pairwise_ranking_loss(pos_score: torch.Tensor, neg_score: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    target = torch.ones_like(pos_score)
    return F.margin_ranking_loss(pos_score, neg_score, target, margin=margin)


def branch_consistency_loss(
    overall_score: torch.Tensor,
    scientific_score: torch.Tensor,
    technical_score: torch.Tensor,
    aesthetic_score: torch.Tensor,
) -> torch.Tensor:
    avg_branch = (scientific_score + technical_score + aesthetic_score) / 3.0
    return F.mse_loss(overall_score, avg_branch)
