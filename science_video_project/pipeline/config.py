from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parents[1]

    data_dir: Path = project_root / "data"
    video_dir: Path = data_dir / "videos"
    metadata_csv: Path = data_dir / "parsed_metadata_filtered.csv"

    output_dir: Path = Path(r"D:\Projects\science_video_ranker_mvp\science_video_project\outputs")
    frame_dir: Path = output_dir / "frames"
    audio_dir: Path = output_dir / "audio"
    feature_dir: Path = output_dir / "features"
    checkpoint_dir: Path = output_dir / "checkpoints"
    log_dir: Path = output_dir / "logs"

    # extraction
    frame_fps: int = 1
    audio_sr: int = 16000
    skip_existing: bool = True
    ffmpeg_path: str = r"D:\Projects\ffmpeg-8.0.1-essentials_build\bin"

    # models
    text_model_name: str = r"D:\Projects\science_video_ranker_mvp\chinese-robeta-wwm-ext"
    clip_model_name: str = r"D:\Projects\science_video_ranker_mvp\openaiclip-vit-large-patch14"
    asr_model_path: str = str(project_root.parent / "faster-whisper-large-v2")
    audio_model_path: str = str(project_root.parent / "faster-whisper-large-v2")
    whisper_model_name: str = "large-v2"
    device: str = "cuda"
    local_files_only: bool = True

    # dimensions
    text_dim: int = 768
    video_dim: int = 512
    audio_dim: int = 384
    meta_dim: int = 16
    aes_dim: int = 7             # CLIP prompt scoring dimensions (DEFAULT_PROMPTS)
    sci_hand_dim: int = 5         # handcrafted scientific features
    temporal_dim: int = 256       # temporal encoder output dim
    hidden_dim: int = 128

    # ====== Research Flags (§4-§13 of new_prompt.md) ======
    use_quality_head: bool = True              # §4: quality_score ← MLP(sci,tech,aes)
    use_consistency_loss: bool = True          # §5: L_cons = MSE(quality, overall)
    use_engagement_branch: bool = True         # §6: engagement_score (auxiliary)
    use_science_features: bool = True          # §7: handcrafted scientific features
    use_aesthetic_mlp: bool = True             # §8: MLP over aes_feat
    use_temporal_encoder: bool = True          # §9: BiGRU/Transformer over frames
    use_cross_modal_attention: bool = True     # §10: MultiHeadAttention fusion
    use_diversity_loss: bool = True            # §11: Branch diversity regularization
    use_frame_features: bool = True            # Save per-frame CLIP features (for temporal)

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

    # train
    batch_size: int = 8
    num_workers: int = 0
    epochs: int = 30                    # 延长训练（原10→30），配合 scheduler 慢慢学
    lr: float = 3e-4                    # 降低学习率（原1e-3→3e-4），防止剧烈震荡
    weight_decay: float = 1e-3          # AdamW L2 正则化，防止过拟合
    margin: float = 0.3                 # 进一步降低 margin（原0.5→0.3），降低排序难度
    seed: int = 42
    same_category_pair: bool = True
    threshold: float = 0.5
    val_ratio: float = 0.2
    scheduler_patience: int = 5          # ReduceLROnPlateau 耐心值（val_loss 不降则降 lr）
    scheduler_factor: float = 0.5        # lr 衰减因子
    early_stop_patience: int = 10        # 早停耐心值（延长，给 scheduler 更多机会）
    pos_aug_noise: float = 0.05          # 正样本特征级增强：高斯噪声标准差

    # experiment config
    experiment_name: str = "full"               # baseline | quality_head | consistency | temporal | cross_modal | science_feat | full


CFG = Config()
