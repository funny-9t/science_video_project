"""
§10 跨模态注意力融合 (Cross-Modal Attention)

在三个分支之前，对 text/video/audio 特征进行跨模态 Multi-Head Attention 融合:

  text_feat, video_feat, audio_feat
         ↓
  MultiHeadAttention (交叉注意力)
         ↓
  cross_modal_fused features → 各分支的增强输入

保留原有 Gate 机制，形成:
  CrossModalAttention → Branch Encoding → Gate → Overall Head
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """跨模态注意力融合层。

    将 text/video/audio 三种模态的特征通过多头注意力相互增强。

    Args:
        text_dim: 文本特征维度 (768)
        video_dim: 视频特征维度 (512)
        audio_dim: 音频特征维度 (384)
        num_heads: 注意力头数
        dropout: 注意力 dropout
    """

    def __init__(
        self,
        text_dim: int = 768,
        video_dim: int = 512,
        audio_dim: int = 384,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 投影到统一维度 (取最大维度 768)
        self.unified_dim = max(text_dim, video_dim, audio_dim)
        self.text_proj = nn.Linear(text_dim, self.unified_dim)
        self.video_proj = nn.Linear(video_dim, self.unified_dim)
        self.audio_proj = nn.Linear(audio_dim, self.unified_dim)

        # 跨模态多头注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.unified_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 层归一化
        self.norm = nn.LayerNorm(self.unified_dim)

        # 输出投影（回到各模态原始维度）
        self.text_out = nn.Linear(self.unified_dim, text_dim)
        self.video_out = nn.Linear(self.unified_dim, video_dim)
        self.audio_out = nn.Linear(self.unified_dim, audio_dim)

    def forward(
        self,
        text_feat: torch.Tensor,
        video_feat: torch.Tensor,
        audio_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            text_feat:  (B, 768)
            video_feat: (B, 512)
            audio_feat: (B, 384)

        Returns:
            enhanced_text, enhanced_video, enhanced_audio (各自原始维度)
        """
        B = text_feat.size(0)

        # 投影到统一维度
        t = self.text_proj(text_feat).unsqueeze(1)   # (B, 1, D)
        v = self.video_proj(video_feat).unsqueeze(1) # (B, 1, D)
        a = self.audio_proj(audio_feat).unsqueeze(1) # (B, 1, D)

        # 拼接三模态 token
        modalities = torch.cat([t, v, a], dim=1)     # (B, 3, D)

        # 跨模态自注意力
        attended, _ = self.cross_attn(modalities, modalities, modalities)  # (B, 3, D)
        attended = self.norm(attended + modalities)  # 残差连接

        # 拆分回各模态
        t_enhanced = attended[:, 0, :]  # (B, D)
        v_enhanced = attended[:, 1, :]  # (B, D)
        a_enhanced = attended[:, 2, :]  # (B, D)

        # 投影回原始维度
        t_out = self.text_out(t_enhanced)   # (B, 768)
        v_out = self.video_out(v_enhanced)  # (B, 512)
        a_out = self.audio_out(a_enhanced)  # (B, 384)

        return t_out, v_out, a_out
