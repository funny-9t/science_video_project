from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parents[1]

    data_dir: Path = project_root / "data"
    video_dir: Path = data_dir / "videos"
    metadata_csv: Path = data_dir / "parsed_metadata.csv"

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
    whisper_model_name: str = "base"
    device: str = "cuda"
    local_files_only: bool = True

    # dimensions
    text_dim: int = 768
    video_dim: int = 512
    audio_dim: int = 384
    meta_dim: int = 16
    aes_dim: int = 7             # CLIP prompt scoring dimensions (DEFAULT_PROMPTS)
    hidden_dim: int = 128

    # train
    batch_size: int = 8
    num_workers: int = 0
    epochs: int = 10
    lr: float = 1e-3
    margin: float = 1.0
    seed: int = 42
    same_category_pair: bool = True
    threshold: float = 0.5
    val_ratio: float = 0.2


CFG = Config()
