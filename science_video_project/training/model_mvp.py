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
    def __init__(self, text_dim: int, meta_dim: int, hidden_dim: int):
        super().__init__()
        self.text_proj = MLPBlock(text_dim, hidden_dim, hidden_dim)
        self.meta_proj = MLPBlock(meta_dim, hidden_dim, hidden_dim)
        self.fuse = MLPBlock(hidden_dim * 2, hidden_dim, hidden_dim)
        self.score = ScoreHead(hidden_dim)

    def forward(self, text_feat: torch.Tensor, meta_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.text_proj(text_feat)
        m = self.meta_proj(meta_feat)
        h = self.fuse(torch.cat([t, m], dim=-1))
        s = self.score(h)
        return h, s


class TechnicalBranch(nn.Module):
    def __init__(self, video_dim: int, audio_dim: int, meta_dim: int, hidden_dim: int):
        super().__init__()
        self.video_proj = MLPBlock(video_dim, hidden_dim, hidden_dim)
        self.audio_proj = MLPBlock(audio_dim, hidden_dim, hidden_dim)
        self.meta_proj = MLPBlock(meta_dim, hidden_dim, hidden_dim)
        self.fuse = MLPBlock(hidden_dim * 3, hidden_dim, hidden_dim)
        self.score = ScoreHead(hidden_dim)

    def forward(
        self,
        video_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        meta_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.video_proj(video_feat)
        a = self.audio_proj(audio_feat)
        m = self.meta_proj(meta_feat)
        h = self.fuse(torch.cat([v, a, m], dim=-1))
        s = self.score(h)
        return h, s


class AestheticBranch(nn.Module):
    def __init__(self, video_dim: int, text_dim: int, audio_dim: int, aes_dim: int, hidden_dim: int):
        super().__init__()
        self.video_proj = MLPBlock(video_dim, hidden_dim, hidden_dim)
        self.text_proj = MLPBlock(text_dim, hidden_dim, hidden_dim)
        self.audio_proj = MLPBlock(audio_dim, hidden_dim, hidden_dim)
        self.aes_proj = MLPBlock(aes_dim, hidden_dim, hidden_dim)
        self.fuse = MLPBlock(hidden_dim * 4, hidden_dim, hidden_dim)
        self.score = ScoreHead(hidden_dim)

    def forward(
        self,
        video_feat: torch.Tensor,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        aes_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.video_proj(video_feat)
        t = self.text_proj(text_feat)
        a = self.audio_proj(audio_feat)
        ae = self.aes_proj(aes_feat)
        h = self.fuse(torch.cat([v, t, a, ae], dim=-1))
        s = self.score(h)
        return h, s


class MultiModalQualityModel(nn.Module):
    def __init__(self, text_dim: int, video_dim: int, audio_dim: int, meta_dim: int,
                 aes_dim: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.scientific_branch = ScientificBranch(text_dim, meta_dim, hidden_dim)
        self.technical_branch = TechnicalBranch(video_dim, audio_dim, meta_dim, hidden_dim)
        self.aesthetic_branch = AestheticBranch(video_dim, text_dim, audio_dim, aes_dim, hidden_dim)

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
    ) -> dict[str, torch.Tensor]:
        sci_h, sci_s = self.scientific_branch(text_feat, meta_feat)
        tech_h, tech_s = self.technical_branch(video_feat, audio_feat, meta_feat)
        if aes_feat is None:
            aes_feat = torch.zeros(video_feat.size(0), 7, device=video_feat.device)
        aes_h, aes_s = self.aesthetic_branch(video_feat, text_feat, audio_feat, aes_feat)

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
        }
