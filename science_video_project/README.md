# science_video_project

Multimodal weak-supervision ranking MVP for no-reference quality evaluation of science short videos.

## Project Overview

Input modalities:
- video frames
- audio waveform
- text (title + tags + ASR subtitle)
- metadata (non-propagation features only)

Output:
- scientific_score
- technical_score
- aesthetic_score
- overall_score
- prediction ("上榜" or "未上榜")

Training target uses pairwise ranking:
- positive sample: label = 1
- negative sample: label = 0
- constraint: q(pos) > q(neg)

## Directory Structure

```text
science_video_project/
├── data/
│   ├── videos/
│   └── metadata.csv
├── outputs/
│   ├── frames/
│   ├── audio/
│   ├── features/
│   ├── checkpoints/
│   └── logs/
├── pipeline/
│   ├── __init__.py
│   ├── config.py
│   ├── utils_io.py
│   ├── step_extract.py
│   ├── step_text.py
│   ├── step_video.py
│   ├── step_audio.py
│   ├── step_meta.py
│   ├── build_sample.py
│   └── run_pipeline.py
├── training/
│   ├── __init__.py
│   ├── model_mvp.py
│   ├── dataset_pair.py
│   ├── dataloader_pair.py
│   ├── losses.py
│   ├── metrics.py
│   ├── utils_train.py
│   └── train.py
├── inference/
│   ├── __init__.py
│   └── infer.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Environment Setup

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Install `ffmpeg` and make sure command `ffmpeg` is available in PATH.

3. Prepare data:
- put videos in `data/videos/`
- edit `data/metadata.csv`

## metadata.csv Example

Required columns:
- `video_id`
- `label`
- `category`
- `title`
- `tags`
- `duration`

Optional columns:
- `verified`
- `publish_time`

Example:

```csv
video_id,label,category,title,tags,duration,verified,publish_time
demo,1,physics,Demo science video,experiment|education,35,1,2025-08-01 14:30:00
```

Important constraints:
- `video_id` must match filename stem in `data/videos/`
- do not include propagation fields (likes/comments/shares/favorites/recommendation volume)

## Run Pipeline

```bash
python pipeline/run_pipeline.py
```

Outputs:
- `outputs/features/{video_id}.pt`
- `outputs/audio/{video_id}.wav`
- `outputs/frames/{video_id}/...`

## Train Model

```bash
python training/train.py
```

Default checkpoint:
- `outputs/checkpoints/best.pt`

Training log:
- `outputs/logs/train.log`

## Train Scientific Branch With SciBERT

```bash
python training/train_scientificbranch_scibert.py --encoder allenai/scibert_scivocab_uncased
```

This trainer fine-tunes a SciBERT text encoder for the scientific branch using `title + tags` by default.
If you also have subtitle text files, pass `--subtitle_dir path/to/txt_dir` and the script will append `{video_id}.txt` to each sample.

Outputs:
- `outputs/checkpoints/scientificbranch_scibert/best/`
- `outputs/logs/train_scientificbranch_scibert.log`

## Inference

```bash
python inference/infer.py --video data/videos/demo.mp4 --checkpoint outputs/checkpoints/best.pt
```

Output format:

```json
{
  "scientific_score": 0.0,
  "technical_score": 0.0,
  "aesthetic_score": 0.0,
  "overall_score": 0.0,
  "probability": 0.0,
  "prediction": "未上榜"
}
```

## Common Errors

1. ffmpeg not found
- symptom: pipeline fails before extraction
- fix: install ffmpeg and add to PATH

2. CUDA unavailable
- symptom: torch reports no cuda or OOM
- fix: set `device = "cpu"` in `pipeline/config.py`

3. model download failure
- symptom: transformers/whisper/clip model download timeout
- fix: retry with stable network or pre-download model weights

4. missing features for training
- symptom: pairwise dataset has no positive/negative data
- fix: run pipeline first and verify both labels exist
