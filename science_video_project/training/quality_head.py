"""
§4 质量空间头 (QualityHead)

输入: 三个分支的原始得分 [scientific, technical, aesthetic]  shape (B, 3)
输出: quality_score  shape (B, 1)

学习目标: 自动发现质量维度的最优组合规律，而非固定平均。
"""

import torch
import torch.nn as nn


class QualityHead(nn.Module):
    """MLP(3 → 32 → 16 → 1) — 学习质量维度的非线性组合规律。

    设计理念:
        不同场景下科学性/技术质量/美学的重要性不同。
        例如: 医学科普更重科学性, 自然风光类更重美学。
        QualityHead 自动学习这种权重关系, 而非简单求均值。
    """

    def __init__(self, input_dim: int = 3, hidden1: int = 32, hidden2: int = 16, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, branch_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            branch_scores: (B, 3) — [scientific_score, technical_score, aesthetic_score]

        Returns:
            quality_score: (B, 1)
        """
        return self.net(branch_scores)
