"""
§12 跨门控融合 (Cross-Gating Fusion)

参考 COVER (CVPR 2024) 的 CrossGatingBlock 设计思想，适配为向量级特征融合。

COVER 原始设计中，语义分支 (CLIP-IQA+) 的特征图门控技术分支 (Swin-3D) 和
美学分支 (ConvNeXt-3D) 的特征图：y = y * GELU(Linear(x)) + shortcut

本实现适配为向量级 (B, hidden_dim) 的交叉门控：
  - sci_h (科学性分支 hidden) 门控 tech_h (技术分支 hidden)
  - sci_h (科学性分支 hidden) 门控 aes_h (美学分支 hidden)

核心直觉：科学性理解"内容在讲什么"→ 调节对技术质量/美学质量的敏感度
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossGatingBlock(nn.Module):
    """向量级交叉门控块 (Vector-Level Cross-Gating Block)。

    使用门控信号 x (通常为科学性/语义分支的 hidden state) 调制目标 y
    (通常为技术分支或美学分支的 hidden state)。

    适配自 COVER cover/models/evaluator.py CrossGatingBlock，将原始的
    5D (n,c,t,h,w) 操作简化为 2D (B, dim) 向量操作。

    Args:
        dim: 特征维度，x 和 y 必须同维度
        dropout: 门控输出的 dropout 率
        use_residual: 是否使用残差连接 (推荐 True)
    """

    def __init__(self, dim: int, dropout: float = 0.1, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual

        # 门控信号投影: x → gate
        self.gate_linear = nn.Linear(dim, dim)
        self.gate_activation = nn.GELU()

        # 目标 y 的投影
        self.y_transform = nn.Linear(dim, dim)

        # 门控后的输出投影
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 门控信号 (B, dim) — 通常来自科学性/语义分支
            y: 被门控目标 (B, dim) — 通常来自技术/美学分支

        Returns:
            gated_y: (B, dim) 被 x 调制后的 y 特征
        """
        shortcut = y

        # x → gate signal (GELU 激活，与 COVER 一致)
        gate = self.gate_linear(x)
        gate = self.gate_activation(gate)  # (B, dim)

        # y 投影
        y_transformed = self.y_transform(y)  # (B, dim)

        # 交叉门控: y = y * gate(x)
        y_gated = y_transformed * gate  # 逐元素乘法

        # 输出投影 + dropout
        y_out = self.out_proj(y_gated)
        y_out = self.dropout(y_out)

        if self.use_residual:
            y_out = y_out + shortcut

        return y_out


class DualCrossGating(nn.Module):
    """双路交叉门控：科学性 hidden 同时门控技术 hidden 和美学 hidden。

    这是 COVER 中 smtc_gate_tech + smtc_gate_aesc 的向量级等价实现。

    Args:
        dim: 三个分支的 hidden 维度 (必须相同)
        dropout: dropout 率
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate_tech = CrossGatingBlock(dim, dropout=dropout)
        self.gate_aes = CrossGatingBlock(dim, dropout=dropout)

    def forward(
        self,
        sci_h: torch.Tensor,
        tech_h: torch.Tensor,
        aes_h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sci_h:  科学性分支 hidden (B, dim)
            tech_h: 技术分支 hidden   (B, dim)
            aes_h:  美学分支 hidden   (B, dim)

        Returns:
            tech_gated: 被科学性门控后的技术 hidden
            aes_gated:  被科学性门控后的美学 hidden
        """
        tech_gated = self.gate_tech(sci_h, tech_h)
        aes_gated = self.gate_aes(sci_h, aes_h)
        return tech_gated, aes_gated
