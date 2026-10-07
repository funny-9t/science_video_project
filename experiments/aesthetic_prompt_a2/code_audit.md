# Aesthetic Prompt A2 Code Audit

## Audit scope

This audit was completed before changing the aesthetic prompt implementation. It
traces the active `KEEP_P2_S2` path rather than assuming that the standalone
scorer is the training path.

## Prompt definitions and call sites

- `science_video_project/pipeline/step_aesthetic_clip.py`
  - `DEFAULT_PROMPTS` defines the seven A1 dimensions: `clarity`,
    `cleanliness`, `composition`, `appeal`, `lighting`, `text_readability`, and
    `professional`.
  - `CLIPAestheticScorer` is the legacy standalone path. Its default backbone is
    OpenAI CLIP `ViT-B/32`; it reads image files and encodes them itself.
  - `SharedCLIPAestheticScorer` is the active main-path scorer. It consumes
    cached frame embeddings and loads only the text tower from the configured
    local CLIP model.
- `science_video_project/pipeline/run_pipeline.py` and
  `science_video_project/tools/backfill_features.py` instantiate the shared
  scorer with `DEFAULT_PROMPTS` and cache the result.
- `science_video_project/inference/full_infer.py` and
  `science_video_project/inference/infer.py` use the same prompt list for
  inference.
- `science_video_project/tools/demo_clip_aesthetic.py` is a legacy/demo caller.
- No downstream training code hard-codes the seven A1 dimension names. The
  model depends only on a finite vector of shape `(7,)`.

## Active KEEP_P2_S2 data flow

1. `pipeline/config.py` configures the local
   `openaiclip-vit-large-patch14` model and `clip_max_frames=40`.
2. The feature cache stores `frame_features` with observed shape `(40, 768)`.
3. `SharedCLIPAestheticScorer.score_frame_features` L2-normalizes frame and
   prompt embeddings, computes `sim_positive - sim_negative`, and applies mean
   aggregation over frames. No sigmoid, softmax, or z-score is applied.
4. The A1 result is stored as `aes_shared_clip_feat` with version
   `shared_clip_vitl14_prompts_v1` inside each main feature `.pt` file.
5. `training/dataset_pair.py::PairwiseVideoDataset.from_metadata` validates the
   `(7,)` shape and cache version, then assigns it to `sample["aes_feat"]`.
6. `training/dataloader_pair.py::collate_pair` batches the tensor.
7. `training/model_mvp.py::AestheticBranch` projects `video_feat`, `text_feat`,
   `audio_feat`, and `aes_feat`, fuses them with an MLP, and emits
   `aesthetic_score`.

Therefore, the active backbone is the frozen local ViT-L/14 shared CLIP path,
not the legacy class's default ViT-B/32 path.

## Target and supervision

- `science_video_project/data/convert_annotations.py` constructs
  `aes_target = video_aesthetics / 5.0`.
- `training/dataset_pair.py` reads the normalized `aes_target` from metadata.
- `training/losses.py::branch_supervision_loss` applies MSE to the sigmoid of
  the aesthetic branch logit.
- `tools/run_progressive_training.py` implements P2-S2:
  - Stage 1: branch MSE plus `0.05 * branch RankNet`; only branch parameters are
    trainable; selection uses macro branch SRCC.
  - Stage 2: overall RankNet + `0.1` pointwise BCE + `0.1` consistency + `0.05`
    branch MSE + `0.05` branch RankNet; all parameters are trainable; selection
    uses ranking accuracy.
  - AdamW learning rate `5e-5`, configured weight decay, batch size 8,
    branch-weight floor 0.2, maximum epochs 15/30, patience 5/10.

## Reproducibility identifiers

- Main baseline: `KEEP_P2_S2`, also described as main_v3-D3 P2 Stage 2.
- Training seed: 42 for the first comparison.
- Split seed: 42.
- Split implementation: stratified `train_test_split` with the configured 20%
  validation ratio.
- Expected clean feature coverage: 615 videos, split 492/123.
- Expected validation split hash:
  `79413d21918a956716f230456b80aec037be51bfd1c6a412a8c0a30087a2a58f`.
- Historical A1 checkpoint:
  `science_video_project/outputs/progressive_training/P2/stage2_best.pt`.

## Cache and experiment boundary

The historical A1 cache must remain untouched because it lives in the shared
per-video feature files. A2 should be written to a separate versioned directory
and loaded as an experiment-only override. The only changed tensor will be the
seven-dimensional `aes_feat`; all other sample fields, model code, split,
initialization, optimizer, scheduler, losses, and training stages stay fixed.

## Observed metadata details

The full filtered metadata contains 692 rows. Its `video_aesthetics` values span
0 through 5 and map to `aes_target` values from 0 through 1. The active clean
training set is the intersection with valid feature files (615 rows). The value
0 is retained as a valid target by the current dataset path.
