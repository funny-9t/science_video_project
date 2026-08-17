import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScoreHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScientificBranch(nn.Module):
    """
    科学性分支：使用 IFG 融合 RoBERTa 语义、LLM 分析文本/评分和手工特征。

    输入:
        text_feat:          BERT 语义特征 (batch, text_dim)
        meta_feat:          元信息特征 (batch, meta_dim)
        llm_knowledge_feat: LLM 四维科学性评分 (batch, llm_dim)

    架构:
        text_feat      → MLP(text_dim → hidden → hidden) ─┐
        meta_feat      → MLP(meta_dim → hidden → hidden) ─┤
        llm_knowledge  → MLP(llm_dim  → hidden → hidden) ─┘→ concat → fuse → score

    输出:
        h: 隐藏表示 (batch, hidden)
        s: 科学性评分 (batch, 1)
    """

    def __init__(
        self,
        text_dim: int,
        meta_dim: int,
        hidden_dim: int,
        llm_knowledge_dim: int = 4,
        llm_analysis_dim: int = 768,
        sci_hand_dim: int = 5,
        use_knowledge_gate: bool = True,
    ):
        super().__init__()
        self.llm_knowledge_dim = llm_knowledge_dim
        self.llm_analysis_dim = llm_analysis_dim
        self.sci_hand_dim = sci_hand_dim
        self.use_knowledge_gate = use_knowledge_gate
        self.text_proj = MLPBlock(text_dim, hidden_dim, hidden_dim)
        self.meta_proj = MLPBlock(meta_dim, hidden_dim, hidden_dim)
        self.llm_score_proj = MLPBlock(llm_knowledge_dim, hidden_dim, hidden_dim)
        self.llm_analysis_proj = MLPBlock(llm_analysis_dim, hidden_dim, hidden_dim)
        self.sci_hand_proj = MLPBlock(sci_hand_dim, hidden_dim, hidden_dim)
        self.knowledge_fuse = MLPBlock(hidden_dim * 3, hidden_dim, hidden_dim)
        self.knowledge_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.knowledge_residual = nn.Linear(hidden_dim * 2, hidden_dim)
        self.knowledge_norm = nn.LayerNorm(hidden_dim)
        self.fuse = MLPBlock(hidden_dim * 2, hidden_dim, hidden_dim)
        self.score = ScoreHead(hidden_dim)

    @staticmethod
    def _fit_last_dim(value: torch.Tensor, dim: int) -> torch.Tensor:
        if value.size(-1) == dim:
            return value
        if value.size(-1) > dim:
            return value[..., :dim]
        padding = value.new_zeros(*value.shape[:-1], dim - value.size(-1))
        return torch.cat([value, padding], dim=-1)

    def forward(
        self,
        text_feat: torch.Tensor,
        meta_feat: torch.Tensor,
        llm_knowledge_feat: torch.Tensor | None = None,
        llm_analysis_feat: torch.Tensor | None = None,
        sci_hand_feat: torch.Tensor | None = None,
        return_gate: bool = False,
    ):
        t = self.text_proj(text_feat)
        m = self.meta_proj(meta_feat)
        if llm_knowledge_feat is None:
            llm_knowledge_feat = torch.zeros(
                text_feat.size(0), self.llm_knowledge_dim,
                device=text_feat.device, dtype=text_feat.dtype,
            )
        if llm_analysis_feat is None:
            llm_analysis_feat = torch.zeros(
                text_feat.size(0), self.llm_analysis_dim,
                device=text_feat.device, dtype=text_feat.dtype,
            )
        if sci_hand_feat is None:
            sci_hand_feat = torch.zeros(
                text_feat.size(0), self.sci_hand_dim,
                device=text_feat.device, dtype=text_feat.dtype,
            )

        llm_knowledge_feat = self._fit_last_dim(llm_knowledge_feat, self.llm_knowledge_dim)
        llm_analysis_feat = self._fit_last_dim(llm_analysis_feat, self.llm_analysis_dim)
        sci_hand_feat = self._fit_last_dim(sci_hand_feat, self.sci_hand_dim)

        knowledge = self.knowledge_fuse(
            torch.cat(
                [
                    self.llm_score_proj(llm_knowledge_feat),
                    self.llm_analysis_proj(llm_analysis_feat),
                    self.sci_hand_proj(sci_hand_feat),
                ],
                dim=-1,
            )
        )
        semantic_knowledge = torch.cat([t, knowledge], dim=-1)
        if self.use_knowledge_gate:
            gate = torch.sigmoid(self.knowledge_gate(semantic_knowledge))
            content = gate * t + (1.0 - gate) * knowledge
            content = content + self.knowledge_residual(semantic_knowledge)
        else:
            gate = torch.full_like(t, 0.5)
            content = self.knowledge_residual(semantic_knowledge)
        content = self.knowledge_norm(content)
        h = self.fuse(torch.cat([content, m], dim=-1))
        s = self.score(h)
        if return_gate:
            return h, s, gate
        return h, s


class TechnicalBranch(nn.Module):
    def __init__(self, video_dim: int, meta_dim: int, dnsmos_dim: int, wpm_dim: int,
                 rhythm_dim: int, hidden_dim: int, cover_dim: int = 3,
                 use_cover_features: bool = False):
        super().__init__()
        self.cover_dim = cover_dim
        self.use_cover_features = use_cover_features
        self.video_proj = MLPBlock(video_dim, hidden_dim, hidden_dim)
        self.meta_proj = MLPBlock(meta_dim, hidden_dim, hidden_dim)
        self.dnsmos_proj = MLPBlock(dnsmos_dim, hidden_dim, hidden_dim)
        self.wpm_proj = MLPBlock(wpm_dim, hidden_dim, hidden_dim)
        self.rhythm_proj = MLPBlock(rhythm_dim, hidden_dim, hidden_dim)
        if use_cover_features:
            self.cover_proj = MLPBlock(cover_dim, hidden_dim, hidden_dim)
        else:
            self.cover_proj = None
        fuse_inputs = 6 if use_cover_features else 5
        self.fuse = MLPBlock(hidden_dim * fuse_inputs, hidden_dim, hidden_dim)
        self.score = ScoreHead(hidden_dim)

    def forward(
        self,
        video_feat: torch.Tensor,
        meta_feat: torch.Tensor,
        dnsmos_feat: torch.Tensor | None = None,
        wpm: torch.Tensor | None = None,
        speech_rhythm_feat: torch.Tensor | None = None,
        cover_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.video_proj(video_feat)
        m = self.meta_proj(meta_feat)
        if dnsmos_feat is None:
            dnsmos_feat = torch.zeros(video_feat.size(0), 3, device=video_feat.device, dtype=video_feat.dtype)
        d = self.dnsmos_proj(dnsmos_feat)
        if wpm is None:
            wpm = torch.zeros(video_feat.size(0), 1, device=video_feat.device, dtype=video_feat.dtype)
        wp = self.wpm_proj(wpm)
        if speech_rhythm_feat is None:
            speech_rhythm_feat = torch.zeros(video_feat.size(0), 6, device=video_feat.device, dtype=video_feat.dtype)
        rh = self.rhythm_proj(speech_rhythm_feat)
        parts = [v, m, d, wp, rh]
        if self.use_cover_features and self.cover_proj is not None:
            if cover_feat is None:
                cover_feat = torch.zeros(
                    video_feat.size(0), self.cover_dim,
                    device=video_feat.device, dtype=video_feat.dtype,
                )
            parts.append(self.cover_proj(cover_feat[..., : self.cover_dim]))
        h = self.fuse(torch.cat(parts, dim=-1))
        s = self.score(h)
        return h, s


class AestheticBranch(nn.Module):
    def __init__(self, video_dim: int, text_dim: int, audio_dim: int, aes_dim: int,
                 hidden_dim: int, cover_dim: int = 3,
                 use_cover_features: bool = False):
        super().__init__()
        self.cover_dim = cover_dim
        self.use_cover_features = use_cover_features
        self.video_proj = MLPBlock(video_dim, hidden_dim, hidden_dim)
        self.text_proj = MLPBlock(text_dim, hidden_dim, hidden_dim)
        self.audio_proj = MLPBlock(audio_dim, hidden_dim, hidden_dim)
        self.aes_proj = MLPBlock(aes_dim, hidden_dim, hidden_dim)
        if use_cover_features:
            self.cover_proj = MLPBlock(cover_dim, hidden_dim, hidden_dim)
        else:
            self.cover_proj = None
        fuse_inputs = 5 if use_cover_features else 4
        self.fuse = MLPBlock(hidden_dim * fuse_inputs, hidden_dim, hidden_dim)
        self.score = ScoreHead(hidden_dim)

    def forward(
        self,
        video_feat: torch.Tensor,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        aes_feat: torch.Tensor,
        cover_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.video_proj(video_feat)
        t = self.text_proj(text_feat)
        a = self.audio_proj(audio_feat)
        ae = self.aes_proj(aes_feat)
        parts = [v, t, a, ae]
        if self.use_cover_features and self.cover_proj is not None:
            if cover_feat is None:
                cover_feat = torch.zeros(
                    video_feat.size(0), self.cover_dim,
                    device=video_feat.device, dtype=video_feat.dtype,
                )
            parts.append(self.cover_proj(cover_feat[..., : self.cover_dim]))
        h = self.fuse(torch.cat(parts, dim=-1))
        s = self.score(h)
        return h, s


class MultiModalQualityModel(nn.Module):
    def __init__(self, text_dim: int, video_dim: int, audio_dim: int, meta_dim: int,
                 aes_dim: int = 7, dnsmos_dim: int = 3, wpm_dim: int = 1, rhythm_dim: int = 6,
                 llm_knowledge_dim: int = 4, llm_analysis_dim: int = 768,
                 sci_hand_dim: int = 5, use_knowledge_gate: bool = True,
                 hidden_dim: int = 128,
                 use_cross_gating: bool = True, cross_gating_dropout: float = 0.1,
                 cover_dim: int = 3, use_cover_features: bool = False,
                 fusion_mode: str = "learned"):
        super().__init__()
        if fusion_mode not in {"learned", "average"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.use_cover_features = use_cover_features
        self.scientific_branch = ScientificBranch(text_dim, meta_dim, hidden_dim,
                                                  llm_knowledge_dim=llm_knowledge_dim,
                                                  llm_analysis_dim=llm_analysis_dim,
                                                  sci_hand_dim=sci_hand_dim,
                                                  use_knowledge_gate=use_knowledge_gate)
        self.technical_branch = TechnicalBranch(
            video_dim, meta_dim, dnsmos_dim, wpm_dim, rhythm_dim, hidden_dim,
            cover_dim=cover_dim, use_cover_features=use_cover_features,
        )
        self.aesthetic_branch = AestheticBranch(
            video_dim, text_dim, audio_dim, aes_dim, hidden_dim,
            cover_dim=cover_dim, use_cover_features=use_cover_features,
        )

        # --- Cross-Gating Fusion (§12) ---
        # 科学性 hidden 门控技术/美学 hidden (参考 COVER CrossGatingBlock)
        self.use_cross_gating = use_cross_gating
        if use_cross_gating:
            from training.cross_gating import DualCrossGating
            self.cross_gating = DualCrossGating(dim=hidden_dim, dropout=cross_gating_dropout)
        else:
            self.cross_gating = None

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

    def forward(
        self,
        text_feat: torch.Tensor,
        video_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        meta_feat: torch.Tensor,
        aes_feat: torch.Tensor | None = None,
        dnsmos_feat: torch.Tensor | None = None,
        wpm: torch.Tensor | None = None,
        speech_rhythm_feat: torch.Tensor | None = None,
        llm_knowledge_feat: torch.Tensor | None = None,
        llm_analysis_feat: torch.Tensor | None = None,
        sci_hand_feat: torch.Tensor | None = None,
        cover_feat: torch.Tensor | None = None,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        sci_h, sci_s, knowledge_gate = self.scientific_branch(
            text_feat,
            meta_feat,
            llm_knowledge_feat=llm_knowledge_feat,
            llm_analysis_feat=llm_analysis_feat,
            sci_hand_feat=sci_hand_feat,
            return_gate=True,
        )
        tech_h, tech_s = self.technical_branch(
            video_feat, meta_feat,
            dnsmos_feat=dnsmos_feat, wpm=wpm, speech_rhythm_feat=speech_rhythm_feat,
            cover_feat=cover_feat,
        )
        if aes_feat is None:
            aes_feat = torch.zeros(video_feat.size(0), 7, device=video_feat.device)
        aes_h, aes_s = self.aesthetic_branch(
            video_feat, text_feat, audio_feat, aes_feat, cover_feat=cover_feat
        )

        # --- Cross-Gating: 科学性 hidden 门控技术/美学 hidden ---
        if self.use_cross_gating and self.cross_gating is not None:
            tech_h, aes_h = self.cross_gating(sci_h, tech_h, aes_h)
            # COVER-style gating should affect each visual branch score, not
            # only the hidden representation used by learned fusion.
            if self.use_cover_features:
                tech_s = self.technical_branch.score(tech_h)
                aes_s = self.aesthetic_branch.score(aes_h)

        if self.fusion_mode == "average":
            weights = torch.full(
                (sci_h.size(0), 3), 1.0 / 3.0,
                device=sci_h.device, dtype=sci_h.dtype,
            )
            overall = (sci_s + tech_s + aes_s) / 3.0
        else:
            gate_in = torch.cat([sci_h, tech_h, aes_h], dim=-1)
            weights = torch.softmax(self.gate(gate_in), dim=-1)
            fused_h = (
                weights[:, 0:1] * sci_h
                + weights[:, 1:2] * tech_h
                + weights[:, 2:3] * aes_h
            )
            fusion_in = torch.cat([fused_h, sci_s, tech_s, aes_s], dim=-1)
            fusion_h = self.fusion(fusion_in)
            overall = self.overall_head(fusion_h)
        prob = torch.sigmoid(overall)
        return {
            "scientific_score": sci_s,
            "technical_score": tech_s,
            "aesthetic_score": aes_s,
            "overall_score": overall,
            "probability": prob,
            "knowledge_gate_weights": knowledge_gate,
            "branch_weights": weights,
        }
