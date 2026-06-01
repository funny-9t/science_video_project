"""
§9 时序编码器 (TemporalEncoder)

升级视频表征: 从逐帧 CLIP 均值 → 时序感知序列建模

支持两种架构:
  - BiGRU: 双向 GRU，简单高效
  - Transformer: 多层 TransformerEncoder，捕捉长程依赖

保留原有 CLIP mean 作为 fallback，通过 use_temporal_encoder 配置控制。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """时序视频编码器 — BiGRU / TransformerEncoder over per-frame CLIP features.

    Args:
        input_dim: 每帧 CLIP 特征维度 (512)
        hidden_dim: 隐藏层维度
        num_layers: 层数
        arch: "bigru" | "transformer"
        num_heads: Transformer 头数 (仅 transformer 模式)
        dropout: Dropout 率
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        arch: str = "bigru",
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.arch = arch

        if arch == "bigru":
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim // 2,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.out_proj = nn.Linear(hidden_dim, input_dim)
        elif arch == "transformer":
            self.pos_encoder = PositionalEncoding(input_dim, dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.out_proj = nn.Identity()  # Transformer 保持维度
        else:
            raise ValueError(f"Unknown temporal architecture: {arch}")

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_features: (B, N_frames, 512) — 每帧的 CLIP 特征

        Returns:
            video_feature: (B, 512) — 时序聚合后的视频表征
        """
        if frame_features.dim() == 2:
            # 单帧 fallback
            return frame_features

        if self.arch == "bigru":
            outputs, _ = self.gru(frame_features)  # (B, N, hidden_dim)
            # 取双向 GRU 的末步+均值池化
            last = outputs[:, -1, :]                # (B, hidden_dim)
            mean = outputs.mean(dim=1)              # (B, hidden_dim)
            pooled = (last + mean) / 2.0
            return self.out_proj(pooled)            # (B, 512)
        else:  # transformer
            x = self.pos_encoder(frame_features)    # 加位置编码
            outputs = self.transformer(x)           # (B, N, 512)
            return outputs.mean(dim=1)              # (B, 512)


class PositionalEncoding(nn.Module):
    """可学习的位置编码 (用于 Transformer 时序编码器)。"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
