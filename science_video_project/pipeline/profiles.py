"""Named extraction and training profiles for reproducible experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineProfile:
    name: str
    llm_cache_policy: str
    use_cover: bool
    use_cover_technical: bool
    use_dnsmos: bool
    technical_feature_mode: str
    technical_visual_source: str
    technical_clip_projection_dim: int
    technical_cover_projection_dim: int
    science_fusion_mode: str
    aesthetic_backend: str


PROFILES = {
    "main_v2": PipelineProfile(
        name="main_v2",
        llm_cache_policy="require",
        use_cover=False,
        use_cover_technical=False,
        use_dnsmos=False,
        technical_feature_mode="full_no_dnsmos",
        technical_visual_source="clip",
        technical_clip_projection_dim=128,
        technical_cover_projection_dim=256,
        science_fusion_mode="concat",
        aesthetic_backend="shared_clip",
    ),
    "main_v3": PipelineProfile(
        name="main_v3",
        llm_cache_policy="require",
        use_cover=False,
        use_cover_technical=True,
        use_dnsmos=False,
        technical_feature_mode="full_no_dnsmos",
        technical_visual_source="clip_cover",
        technical_clip_projection_dim=256,
        technical_cover_projection_dim=256,
        science_fusion_mode="concat",
        aesthetic_backend="shared_clip",
    ),
    "cover_ablation": PipelineProfile(
        name="cover_ablation",
        llm_cache_policy="require",
        use_cover=True,
        use_cover_technical=False,
        use_dnsmos=False,
        technical_feature_mode="full_no_dnsmos",
        technical_visual_source="clip",
        technical_clip_projection_dim=128,
        technical_cover_projection_dim=256,
        science_fusion_mode="concat",
        aesthetic_backend="shared_clip",
    ),
    "legacy_full": PipelineProfile(
        name="legacy_full",
        llm_cache_policy="require",
        use_cover=True,
        use_cover_technical=False,
        use_dnsmos=True,
        technical_feature_mode="full",
        technical_visual_source="clip",
        technical_clip_projection_dim=128,
        technical_cover_projection_dim=256,
        science_fusion_mode="ifg",
        aesthetic_backend="separate_clip",
    ),
}


def get_profile(name: str) -> PipelineProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown profile {name!r}; choose from {sorted(PROFILES)}") from exc
