"""
研究版主模型 (ResearchModel)

基于 MultiModalQualityModel 增量升级，实现 new_prompt.md §4-§11 的全部创新点:

  §4  QualityHead          — 质量空间层
  §5  Consistency Loss      — 质量一致性约束
  §6  EngagementBranch      — 传播力辅助分支
  §7  Scientific Features   — 手工科学性特征
  §8  Aesthetic MLP          — 美学特征学习
  §9  Temporal Encoder      — 时序视频编码
  §10 Cross-Modal Attention — 跨模态注意力融合
  §11 Branch Diversity Loss — 分支多样性正则

所有新模块通过 Config 开关控制，保持向后兼容。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from training.model_mvp import (
    MLPBlock,
    ScoreHead,
    ScientificBranch,
    TechnicalBranch,
    AestheticBranch,
    MultiModalQualityModel,
)


# ============================================================================
#  §8 Aesthetic MLP — 学习 prompt score 的非线性映射
# ============================================================================

class AestheticMLP(nn.Module):
    """aes_feat(7) → MLP → aesthetic_score (替代原始 prompt 得分均值)。"""

    def __init__(self, aes_dim: int = 7, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(aes_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, aes_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            aes_feat: (B, 7)

        Returns:
            aesthetic_score: (B, 1)
        """
        return self.net(aes_feat)


# ============================================================================
#  §7 Scientific Feature Projection — 融合手工特征
# ============================================================================

class ScienceFeatureProjection(nn.Module):
    """BERT embedding + 手工科学性特征 → 增强的文本表征。"""

    def __init__(self, bert_dim: int = 768, sci_hand_dim: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(bert_dim + sci_hand_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, bert_dim),
            nn.ReLU(),
        )

    def forward(self, text_feat: torch.Tensor, sci_hand_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_feat:     (B, 768) BERT embedding
            sci_hand_feat: (B, 5)   手工科学性特征

        Returns:
            enhanced_text: (B, 768)
        """
        combined = torch.cat([text_feat, sci_hand_feat], dim=-1)
        return self.fuse(combined)


# ============================================================================
#  研究版主模型
# ============================================================================

class ResearchModel(nn.Module):
    """面向科普短视频的多维质量评估研究模型。

    在 MultiModalQualityModel 基础上增量添加:
      - CrossModalAttention (§10)
      - ScienceFeatureProjection (§7)
      - AestheticMLP (§8)
      - TemporalEncoder (§9) — 仅在推理阶段外部注入
      - QualityHead (§4)
      - EngagementBranch (§6)

    所有模块通过 CFG 开关控制。旧 checkpoint 可直接加载其子模型权重。
    """

    def __init__(
        self,
        text_dim: int = 768,
        video_dim: int = 512,
        audio_dim: int = 384,
        meta_dim: int = 16,
        aes_dim: int = 7,
        sci_hand_dim: int = 5,
        hidden_dim: int = 128,
        use_quality_head: bool = True,
        use_engagement_branch: bool = True,
        use_science_features: bool = True,
        use_aesthetic_mlp: bool = True,
        use_temporal_encoder: bool = True,
        use_cross_modal_attention: bool = True,
        temporal_dim: int = 256,
        temporal_arch: str = "bigru",
        temporal_num_layers: int = 2,
        temporal_num_heads: int = 4,
        cross_modal_num_heads: int = 4,
        cross_modal_dropout: float = 0.1,
    ):
        super().__init__()

        # --- 核心三分支模型 (保持原有键名，兼容旧 checkpoint) ---
        self.scientific_branch = ScientificBranch(text_dim, meta_dim, hidden_dim)
        self.technical_branch = TechnicalBranch(video_dim, audio_dim, meta_dim, hidden_dim)
        self.aesthetic_branch = AestheticBranch(video_dim, text_dim, audio_dim, aes_dim, hidden_dim)

        # Gate + Fusion + Overall Head (保持原有结构)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 3),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.overall_head = nn.Linear(hidden_dim, 1)

        # --- §10 Cross-Modal Attention ---
        self.use_cross_modal_attention = use_cross_modal_attention
        if use_cross_modal_attention:
            from training.cross_modal_attention import CrossModalAttention
            self.cross_modal_attn = CrossModalAttention(
                text_dim=text_dim,
                video_dim=video_dim,
                audio_dim=audio_dim,
                num_heads=cross_modal_num_heads,
                dropout=cross_modal_dropout,
            )
        else:
            self.cross_modal_attn = None

        # --- §7 Science Feature Projection ---
        self.use_science_features = use_science_features
        if use_science_features:
            self.sci_feat_proj = ScienceFeatureProjection(
                bert_dim=text_dim, sci_hand_dim=sci_hand_dim, hidden_dim=hidden_dim
            )
        else:
            self.sci_feat_proj = None

        # --- §8 Aesthetic MLP ---
        self.use_aesthetic_mlp = use_aesthetic_mlp
        if use_aesthetic_mlp:
            self.aesthetic_mlp = AestheticMLP(aes_dim=aes_dim, hidden_dim=hidden_dim)
        else:
            self.aesthetic_mlp = None

        # --- §9 Temporal Encoder ---
        self.use_temporal_encoder = use_temporal_encoder
        if use_temporal_encoder:
            from training.step_temporal import TemporalEncoder
            self.temporal_encoder = TemporalEncoder(
                input_dim=video_dim,
                hidden_dim=temporal_dim,
                num_layers=temporal_num_layers,
                arch=temporal_arch,
                num_heads=temporal_num_heads,
                dropout=0.2,
            )
        else:
            self.temporal_encoder = None

        # --- §4 Quality Head ---
        self.use_quality_head = use_quality_head
        if use_quality_head:
            from training.quality_head import QualityHead
            self.quality_head = QualityHead(input_dim=3, hidden1=32, hidden2=16)
        else:
            self.quality_head = None

        # --- §6 Engagement Branch ---
        self.use_engagement_branch = use_engagement_branch
        if use_engagement_branch:
            from training.engagement_branch import EngagementBranch
            self.engagement_branch = EngagementBranch(meta_dim=meta_dim, hidden_dim=hidden_dim)
        else:
            self.engagement_branch = None

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        text_feat: torch.Tensor,
        video_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        meta_feat: torch.Tensor,
        aes_feat: torch.Tensor | None = None,
        sci_hand_feat: torch.Tensor | None = None,
        frame_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            text_feat:     (B, 768)
            video_feat:    (B, 512)  — 或 (B, N, 512) 当 temporal encoder 开启
            audio_feat:    (B, 384)
            meta_feat:     (B, 16)
            aes_feat:      (B, 7)    — 可选，缺失时填零
            sci_hand_feat: (B, 5)    — 可选
            frame_features:(B, N, 512) — 可选，时序编码器的帧序列

        Returns:
            dict with keys in §12 of new_prompt.md:
              scientific_score, technical_score, aesthetic_score,
              quality_score (if enabled), engagement_score (if enabled),
              overall_score, probability, gate_weights,
              scientific_hidden, technical_hidden, aesthetic_hidden
        """
        B = text_feat.size(0)
        device = text_feat.device

        # --- 默认值 ---
        if aes_feat is None:
            aes_feat = torch.zeros(B, 7, device=device)

        # --- §9 Temporal Encoder: 帧序列 → 时序视频表征 ---
        if self.use_temporal_encoder and self.temporal_encoder is not None:
            if frame_features is not None and frame_features.dim() == 3:
                video_feat = self.temporal_encoder(frame_features)
            # else: 使用预计算的 video_feat (B, 512) 不变

        # --- §10 Cross-Modal Attention: 跨模态增强 ---
        if self.use_cross_modal_attention and self.cross_modal_attn is not None:
            t_cross, v_cross, a_cross = self.cross_modal_attn(text_feat, video_feat, audio_feat)
        else:
            t_cross, v_cross, a_cross = text_feat, video_feat, audio_feat

        # --- §7 Science Feature Projection: BERT + 手工特征 ---
        if self.use_science_features and self.sci_feat_proj is not None and sci_hand_feat is not None:
            text_enhanced = self.sci_feat_proj(t_cross, sci_hand_feat)
        else:
            text_enhanced = t_cross

        # --- 三分支编码 ---
        sci_h, sci_s = self.scientific_branch(text_enhanced, meta_feat)
        tech_h, tech_s = self.technical_branch(v_cross, a_cross, meta_feat)

        # AestheticBranch 始终接收原始 7-dim aes_feat 用于内部处理
        aes_h, aes_s = self.aesthetic_branch(v_cross, text_enhanced, a_cross, aes_feat)

        # §8 Aesthetic MLP — 学习非线性美学评分，覆盖分支原始得分
        if self.use_aesthetic_mlp and self.aesthetic_mlp is not None:
            aes_s = self.aesthetic_mlp(aes_feat)  # (B, 1) — 覆盖分支得分

        # --- §4 Quality Head: 质量空间 ---
        quality_score: torch.Tensor | None = None
        if self.use_quality_head and self.quality_head is not None:
            branch_scores = torch.cat([sci_s, tech_s, aes_s], dim=-1)  # (B, 3)
            quality_score = self.quality_head(branch_scores)

        # --- §6 Engagement Branch: 传播力 (仅辅助) ---
        engagement_h: torch.Tensor | None = None
        engagement_score: torch.Tensor | None = None
        if self.use_engagement_branch and self.engagement_branch is not None:
            engagement_h, engagement_score = self.engagement_branch(meta_feat)

        # --- Gate + Fusion (保持原有逻辑) ---
        gate_in = torch.cat([sci_h, tech_h, aes_h], dim=-1)
        gate_weights = torch.softmax(self.gate(gate_in), dim=-1)

        fused_h = (
            gate_weights[:, 0:1] * sci_h
            + gate_weights[:, 1:2] * tech_h
            + gate_weights[:, 2:3] * aes_h
        )
        fusion_in = torch.cat([fused_h, sci_s, tech_s, aes_s], dim=-1)
        fusion_h = self.fusion(fusion_in)
        overall = self.overall_head(fusion_h)
        prob = torch.sigmoid(overall)

        # --- 构建输出 ---
        out: dict[str, torch.Tensor] = {
            "scientific_score": sci_s,
            "technical_score": tech_s,
            "aesthetic_score": aes_s,
            "overall_score": overall,
            "probability": prob,
            "gate_weights": gate_weights,
            "scientific_hidden": sci_h,
            "technical_hidden": tech_h,
            "aesthetic_hidden": aes_h,
        }

        if quality_score is not None:
            out["quality_score"] = quality_score

        if engagement_score is not None:
            out["engagement_score"] = engagement_score

        return out
