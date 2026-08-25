import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parents[1]

    data_dir: Path = project_root / "data"
    video_dir: Path = Path(os.environ.get("SCIENCE_VIDEO_DIR", r"F:\Data\172.16.29.65"))
    cover_root: Path = Path(os.environ.get("COVER_ROOT", r"D:\Projects\COVER"))
    metadata_csv: Path = Path(
        os.environ.get("SCIENCE_VIDEO_METADATA", str(data_dir / "parsed_metadata_filtered.csv"))
    )

    output_dir: Path = Path(
        os.environ.get(
            "SCIENCE_VIDEO_OUTPUT_DIR",
            r"D:\Projects\science_video_ranker_mvp\science_video_project\outputs",
        )
    )
    frame_dir: Path = Path(os.environ.get("SCIENCE_VIDEO_FRAME_DIR", str(output_dir / "frames")))
    audio_dir: Path = Path(os.environ.get("SCIENCE_VIDEO_AUDIO_DIR", str(output_dir / "audio")))
    transcript_dir: Path = Path(
        os.environ.get("SCIENCE_VIDEO_TRANSCRIPT_DIR", str(output_dir / "transcripts"))
    )
    feature_dir: Path = Path(os.environ.get("SCIENCE_VIDEO_FEATURE_DIR", str(output_dir / "features")))
    checkpoint_dir: Path = Path(
        os.environ.get("SCIENCE_VIDEO_CHECKPOINT_DIR", str(output_dir / "checkpoints"))
    )
    log_dir: Path = Path(os.environ.get("SCIENCE_VIDEO_LOG_DIR", str(output_dir / "logs")))

    # extraction
    frame_fps: int = 1
    clip_max_frames: int = 40      # global-uniform temporal samples for native CLIP sequence
    audio_sr: int = 16000
    skip_existing: bool = True
    ffmpeg_path: str = r"D:\Projects\ffmpeg-8.0.1-essentials_build\bin"

    # models
    text_model_name: str = os.environ.get(
        "SCIENCE_VIDEO_TEXT_MODEL",
        r"D:\Projects\science_video_ranker_mvp\chinese-robeta-wwm-ext",
    )
    clip_model_name: str = os.environ.get(
        "SCIENCE_VIDEO_CLIP_MODEL",
        r"D:\Projects\science_video_ranker_mvp\openaiclip-vit-large-patch14",
    )
    asr_model_path: str = os.environ.get(
        "SCIENCE_VIDEO_ASR_MODEL",
        r"D:\Projects\science_video_ranker_mvp\faster-whisper-large-v2",
    )
    audio_model_path: str = os.environ.get("SCIENCE_VIDEO_AUDIO_MODEL", asr_model_path)
    whisper_model_name: str = "large-v2"
    device: str = "cuda"
    local_files_only: bool = True

    # dimensions
    text_dim: int = 768
    video_dim: int = 512
    clip_video_dim: int = 768       # native ViT-L/14 projection dimension
    audio_dim: int = 384
    meta_dim: int = 16
    aes_dim: int = 7             # CLIP prompt scoring dimensions (DEFAULT_PROMPTS)
    sci_hand_dim: int = 5         # handcrafted scientific features
    llm_knowledge_dim: int = 4    # factual, logical, evidence, uncertainty scores
    llm_analysis_dim: int = 768   # RoBERTa encoding of DeepSeek analysis/reasoning text
    cover_dim: int = 3            # frozen COVER semantic/technical/aesthetic scores
    dnsmos_dim: int = 3           # DNSMOS 音频质量评分 (ovrl/sig/bak)
    wpm_dim: int = 1              # 每分钟字数 (Words Per Minute)
    speech_rhythm_dim: int = 6    # 段级语速节奏 (mean/std/min/max WPM + pause_ratio + speech_density)
    temporal_dim: int = 256       # temporal encoder output dim
    hidden_dim: int = 128

    # ====== Research Flags (§4-§13 of new_prompt.md) ======
    use_quality_head: bool = True              # §4: quality_score ← MLP(sci,tech,aes)
    use_consistency_loss: bool = True          # §5: L_cons = MSE(quality, overall)
    use_engagement_branch: bool = True         # §6: engagement_score (auxiliary)
    use_science_features: bool = True          # §7: handcrafted scientific features
    use_llm_knowledge: bool = True             # §7+: LLM 知识科学性特征 (DeepSeek)
    use_aesthetic_mlp: bool = True             # §8: MLP over aes_feat
    use_temporal_encoder: bool = True          # §9: BiGRU/Transformer over frames
    use_cross_modal_attention: bool = True     # §10: MultiHeadAttention fusion
    use_diversity_loss: bool = True            # §11: Branch diversity regularization
    use_frame_features: bool = True            # Save per-frame CLIP features (for temporal)
    use_cross_gating: bool = False             # §12: Cross-Gating fusion (参考 COVER) — Baseline 关闭
    use_cover_features: bool = False            # frozen COVER priors for visual branches
    fusion_mode: str = "learned"               # learned | average
    branch_weight_floor: float = 0.2            # preserve technical/aesthetic branches in learned fusion
    technical_feature_mode: str = "full_no_dnsmos"
    use_knowledge_gate: bool = False            # IFG remains available through legacy/ablation modes
    science_fusion_mode: str = "concat"         # concat for main_v2; ifg for ablation

    # loss weights
    lambda_consistency: float = 0.2
    lambda_diversity: float = 0.05
    lambda_supervision: float = 0.05     # 降低（原0.2→0.05），避免分支监督主导初期训练
    pos_weight: float = 15.0             # 加权排序中正样本被误判的额外惩罚倍数（1:19 → 提高至 15）
    focal_gamma: float = 2.0             # Focal Ranking Loss 的 gamma（越大对难样本越关注）
    focal_alpha: float = 0.85            # Focal Ranking Loss 的正样本权重系数
    use_focal_ranking: bool = True       # 是否使用 Focal Ranking Loss（替代 weighted）

    # temporal encoder config
    temporal_arch: str = "bigru"
    temporal_num_layers: int = 2
    temporal_num_heads: int = 4

    # cross-modal attention config
    cross_modal_num_heads: int = 4
    cross_modal_dropout: float = 0.1

    # cross-gating config (§12, 参考 COVER)
    cross_gating_dropout: float = 0.1

    # ====== LLM 知识科学性特征配置 (§7+) ======
    # DeepSeek V4 Pro API 配置
    use_llm_knowledge: bool = True          # 是否启用 LLM 知识科学性特征
    llm_api_base: str = "https://api.deepseek.com"
    llm_model_name: str = "deepseek-v4-pro"
    llm_api_key: str = ""                    # 从环境变量 DEEPSEEK_API_KEY 或参数传入
    llm_temperature: float = 0.1             # 低温度提高评分稳定性
    llm_cache_dir: str = os.environ.get(
        "SCIENCE_VIDEO_LLM_CACHE",
        str(project_root / "cache" / "llm_knowledge"),
    )  # LLM 特征缓存目录
    llm_include_reasoning: bool = True

    # train
    batch_size: int = 8
    num_workers: int = 0
    epochs: int = 30                    # 延长训练（原10→30），配合 scheduler 慢慢学
    lr: float = 5e-5                    # full-dataset ablation optimum
    weight_decay: float = 1e-3          # AdamW L2 正则化，防止过拟合
    margin: float = 0.3                 # 进一步降低 margin（原0.5→0.3），降低排序难度
    seed: int = 42
    same_category_pair: bool = False
    threshold: float = 0.5
    val_ratio: float = 0.2
    scheduler_patience: int = 5          # ReduceLROnPlateau 耐心值（val_loss 不降则降 lr）
    scheduler_factor: float = 0.5        # lr 衰减因子
    early_stop_patience: int = 10        # 早停耐心值
    pos_aug_noise: float = 0.05          # 正样本特征级增强：高斯噪声标准差

    # ====== Regression Training (train_regression.py) ======
    regression_epochs: int = 50
    regression_batch_size: int = 16
    regression_lr: float = 3e-4
    regression_weight_decay: float = 1e-3
    regression_lambda_consistency: float = 0.1
    regression_scheduler_patience: int = 8
    regression_scheduler_factor: float = 0.5
    regression_early_stop_patience: int = 15
    regression_val_ratio: float = 0.15

    # experiment config
    experiment_name: str = "full"               # baseline | quality_head | consistency | temporal | cross_modal | science_feat | full | regression


CFG = Config()
