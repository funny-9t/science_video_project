"""
§6 传播力辅助分支 (EngagementBranch)

输入: meta_feat (16维，包含统计传播特征)
输出: engagement_score

约束: engagement_score 仅作为辅助任务，不直接参与最终分类，避免标签泄露。
"""

import torch
import torch.nn as nn

from training.model_mvp import MLPBlock


class EngagementBranch(nn.Module):
    """传播力预测辅助分支。

    meta_feat(16) → MLP → engagement_score

    该分支仅在训练时作为辅助监督信号使用。
    推理时 engagement_score 不参与 overall 计算，仅作为参考输出。
    """

    def __init__(self, meta_dim: int = 16, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.mlp = MLPBlock(meta_dim, hidden_dim, hidden_dim)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # 预测"高传播"概率
        )

    def forward(self, meta_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            meta_feat: (B, 16)

        Returns:
            hidden: (B, hidden_dim)
            engagement_score: (B, 1) — 0-1 之间，高传播概率
        """
        h = self.mlp(meta_feat)
        score = self.score_head(h)
        return h, score
