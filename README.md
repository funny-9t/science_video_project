# 面向科普短视频的多模态质量评估系统

本项目是一个面向毕业设计的科普短视频质量评估原型系统。系统以视频、音频、文本和元数据为输入，通过多模态特征提取与弱监督 pairwise ranking 训练，输出科普短视频的多维质量分数和上榜概率。

当前主线代码位于 `science_video_project/`，核心模型是 `MultiModalQualityModel`。项目同时保留了研究版扩展模块，放在 `science_video_project/training/research/`。

## 1. 项目目标

短视频平台通常只有“上榜/未上榜”等弱标签，而没有人工 MOS 绝对质量分数。本项目将任务建模为无参考视频质量评估与成对排序学习：

- 正样本：`label = 1`，代表上榜或高质量样本
- 负样本：`label = 0`，代表未上榜或低质量样本
- 训练约束：`score(pos) > score(neg)`

模型输出包括：

- `scientific_score`：科学内容质量
- `technical_score`：技术制作质量
- `aesthetic_score`：视觉美学质量
- `overall_score`：综合质量 logit
- `probability`：上榜概率
- `prediction`：根据阈值判断 `top` / `not_top`

## 2. 当前代码概览

### 特征提取

`science_video_project/pipeline/` 负责把原始视频转换为 `.pt` 特征文件。

主要特征包括：

- 文本特征：Whisper ASR 字幕 + 标题 + 标签，经中文 RoBERTa 编码，维度 768
- 视频特征：CLIP ViT-L/14 对全视频最多 40 帧做全局均匀采样；保存原生 768 维均值与逐帧序列，同时保留 512 维兼容特征用于旧 checkpoint
- COVER 特征：冻结官方预训练模型，提取 semantic / technical / aesthetic 三个分支分数，维度 3
- 音频特征：Whisper encoder 音频表示，维度 384
- 元数据特征：时长、标题长度、标签数量、认证状态、发布时间、类别编码，维度 16
- 美学特征：CLIP prompt 对比打分，维度 7
- 客观音频质量：DNSMOS，维度 3
- 语速节奏：WPM 与分段语速/停顿/语音密度特征，维度 1 + 6
- 科学性手工特征：术语密度、实体比例、数字密度、平均句长、关键词覆盖率，维度 5
- LLM 知识评分：事实一致性、逻辑连贯性、证据充分性、不确定性意识，维度 4
- LLM 分析语义：DeepSeek V4-Pro 的结构化分析/推理文本经中文 RoBERTa 编码，维度 768

### 模型结构

`science_video_project/training/model_mvp.py` 中的 `MultiModalQualityModel` 采用三分支结构：

```text
文本 + 元数据 + LLM评分 + LLM分析语义 + 科学性手工特征
        -> ScientificBranch -> scientific_score

视频 + 元数据 + DNSMOS + WPM + 语速节奏 + COVER技术/语义质量
        -> TechnicalBranch  -> technical_score

视频 + 文本 + 音频 + CLIP美学特征 + COVER美学/语义质量
        -> AestheticBranch  -> aesthetic_score

ScientificBranch hidden 对技术/美学 hidden 做 Cross-Gating
        -> 学习式融合或三个分支严格等权融合
        -> overall_score / probability
```

可选模块：

- `ScientificBranch`：支持直接拼接和 IFG（信息特征门控）两种知识融合方式
- `pipeline/step_cover.py`：官方 COVER 权重的冻结适配器，按视频 ID 固定 Python / NumPy / Torch 采样随机数
- `training/cross_gating.py`：科学性表示先门控技术/美学表示，再计算增强分支分数
- `fusion_mode=average`：科学性分支直接贡献固定为 `1/3`，用于防止学习式门控把创新分支稀释
- `training/research/`：研究版扩展，包括 cross-modal attention、temporal encoder、quality head、engagement branch 等

### 训练逻辑

`science_video_project/training/train.py` 使用 pairwise ranking：

- 训练集随机采样正负样本对
- 验证集使用固定正负样本对，保证指标可复现
- 损失函数支持 RankNet、focal/weighted/margin ranking、点式 BCE 校准和细粒度分支监督
- 可选 COVER 式两阶段训练：先独立预训练三个分支，再恢复 Cross-Gating 联合训练
- 排序指标基于固定正负 pair；AUC、PR-AUC、F1 和阈值基于去重后的唯一视频
- checkpoint 保存模型权重、最佳阈值和完整模型结构配置

### 推理逻辑

`science_video_project/inference/infer.py` 与 `full_infer.py` 已与训练特征保持一致。推理时会重新提取训练阶段使用的完整特征，并从 checkpoint 中读取：

- 模型结构配置
- `best_threshold`
- `model_state_dict`

这样可以避免训练和推理模型参数不一致、特征缺失或阈值漂移。

## 3. 目录结构

```text
science_video_ranker_mvp/
├── README.md
├── 1_QUICK_START.md
├── 2_README_FRONTEND.md
├── 3_DEPLOYMENT_GUIDE.md
├── 架构.md
├── prompt.md
├── new_prompt.md
├── chinese-robeta-wwm-ext/
├── faster-whisper-large-v2/
├── openaiclip-vit-large-patch14/
└── science_video_project/
    ├── data/
    │   ├── convert_annotations.py
    │   ├── filter_metadata.py
    │   ├── normalize_urls.py
    │   └── resolve_short_links.py
    ├── pipeline/
    │   ├── config.py
    │   ├── run_pipeline.py
    │   ├── run_pipeline_research.py
    │   ├── build_sample.py
    │   ├── step_text.py
    │   ├── step_video.py
    │   ├── step_audio.py
    │   ├── step_meta.py
    │   ├── step_aesthetic_clip.py
    │   ├── step_dnsmos.py
    │   ├── step_science_features.py
    │   └── step_llm_knowledge.py
    ├── training/
    │   ├── model_mvp.py
    │   ├── train.py
    │   ├── dataset_pair.py
    │   ├── dataloader_pair.py
    │   ├── losses.py
    │   ├── metrics.py
    │   ├── utils_train.py
    │   ├── cross_gating.py
    │   └── research/
    ├── inference/
    │   ├── infer.py
    │   ├── full_infer.py
    │   └── batch_infer.py
    ├── frontend/
    ├── tools/
    ├── outputs/
    ├── requirements.txt
    └── README.md
```

## 4. 环境准备

推荐环境：

- Python 3.10+
- CUDA GPU，可选但推荐
- ffmpeg

安装依赖：

```bash
cd science_video_project
pip install -r requirements.txt
```

Windows 下若要让 `faster-whisper` 使用 NVIDIA GPU，还需要与本机 CUDA 兼容的
`nvidia-cublas-cu12` 和 `nvidia-cudnn-cu12`。没有 GPU 时会自动使用 CPU，结果一致但速度较慢。

模型路径和数据路径集中在：

```text
science_video_project/pipeline/config.py
```

重点检查：

```python
video_dir = Path(r"F:\Data\172.16.29.65")
metadata_csv = data_dir / "parsed_metadata_filtered.csv"
text_model_name = r"D:\Projects\science_video_ranker_mvp\chinese-robeta-wwm-ext"
clip_model_name = r"D:\Projects\science_video_ranker_mvp\openaiclip-vit-large-patch14"
asr_model_path = str(project_root.parent / "faster-whisper-large-v2")
ffmpeg_path = r"D:\Projects\ffmpeg-8.0.1-essentials_build\bin"
```

## 5. 数据格式

元数据 CSV 至少需要包含：

```csv
video_id,label,category,title,tags,duration
```

推荐字段：

```csv
video_id,label,category,title,tags,duration,verified,publish_time
```

说明：

- `video_id` 需要能和视频文件名匹配
- `label` 为 `1` 或 `0`
- `category` 用于同类别内正负配对
- `title`、`tags` 会与 ASR 字幕拼接后进入文本编码器
- 传播数据如点赞、评论、转发不进入主模型，避免标签泄漏

## 6. 特征提取

进入子项目目录：

```bash
cd science_video_project
```

运行主线特征流水线：

```bash
python pipeline/run_pipeline.py
```

输出位置：

```text
outputs/audio/{video_id}.wav
outputs/frames/{video_id}/*.jpg
outputs/features/{video_id}.pt
outputs/logs/pipeline.log
```

如果需要研究版逐帧特征和传播辅助标签：

```bash
python pipeline/run_pipeline_research.py
```

对已有 `.pt` 样本补齐 Whisper encoder 音频表示、DNSMOS、WPM 和 speech rhythm：

```bash
python tools/backfill_features.py --features audio,dnsmos,speech --force
```

`audio_feat` 直接复用 `outputs/audio` 中的 WAV，不会重新执行 ASR。编码器使用本地
`faster-whisper-large-v2`，对每个视频均匀抽取最多 8 个 30 秒窗口，池化并归一化为 384 维向量。

DeepSeek 科学性知识特征分为“API 预取”和“本地 RoBERTa 编码”两步，支持 JSONL 断点缓存：

```powershell
$env:DEEPSEEK_API_KEY="替换为新申请的 API Key"
python tools/prefetch_llm_cache.py --workers 4
python tools/backfill_features.py --features llm --llm-cache-only --force
```

最终每个样本会保存 4 维评分、仅分析文本的 768 维向量，以及“推理过程 + 分析文本”的
768 维向量，便于做公平消融。API Key 仅通过环境变量传入，不要写入代码、配置或缓存。

补齐原生 CLIP 全局稀疏帧序列与冻结 COVER 三分支特征：

```powershell
python tools/backfill_features.py --features clip --report outputs/logs/clip_backfill_report.json
python tools/backfill_features.py --features cover --cover-root D:\Projects\COVER --report outputs/logs/cover_backfill_report.json
python tools/check_features.py
```

COVER 使用 `D:\Projects\COVER\pretrained_weights\COVER.pth` 与官方 `val-ytugc`
视图配置。技术分支采用空间 fragments，语义/美学分支采用全局稀疏采样；特征版本不匹配时
`--use_cover_features` 会直接终止训练，避免旧随机结果或零向量混入实验。

## 7. 模型训练

主线 MVP 训练：

```bash
cd science_video_project
python training/train.py
```

常用参数：

```bash
python training/train.py \
  --metadata data/parsed_metadata_filtered.csv \
  --feature_dir outputs/features \
  --epochs 30 \
  --batch_size 8 \
  --lr 3e-4 \
  --margin 0.3
```

科学性分支可通过以下参数切换：

```text
--science_feature_mode none|scores|analysis_concat|analysis_scores_concat|analysis_scores_ifg|full_ifg
--llm_text_source analysis|reasoning_and_analysis
```

运行完整科学性特征消融：

```bash
python tools/run_science_ablation.py
```

运行 COVER 结构、采样、融合、Cross-Gating、科学性特征与分阶段训练的受控消融：

```bash
python tools/run_cover_ablation.py --seeds 42 --epochs 15 --patience 5 --lr 1e-5
```

建议先用单 seed 筛选方案，再对候选方案执行 `--seeds 42,43,44`。报告会同时保存
pairwise accuracy、唯一视频 AUC/PR-AUC、三个分支 SRCC 和科学性融合权重。

当前固定划分下的最高分训练配置：

```bash
python training/train.py \
  --epochs 15 \
  --early_stop_patience 5 \
  --lr 1e-5 \
  --seed 42 \
  --split_seed 42 \
  --disable_audio \
  --pair_scope global \
  --loss_type focal \
  --science_feature_mode analysis_concat \
  --llm_text_source analysis \
  --checkpoint outputs/checkpoints/best_llm_optimized.pt
```

默认输出：

```text
outputs/checkpoints/best.pt
outputs/logs/train.log
```

checkpoint 内容：

- `model_state_dict`
- `best_threshold`
- `config`：完整模型结构配置，用于推理阶段恢复同构模型
- `science_feature_mode`、`llm_text_source`：推理时自动恢复同样的特征源和掩码
- `training_config`、`seed`、`split_seed`：用于复现实验

## 8. 单视频推理

```bash
cd science_video_project
python inference/infer.py \
  --video path/to/video.mp4 \
  --checkpoint outputs/checkpoints/best_llm_optimized.pt \
  --metadata data/parsed_metadata_filtered.csv
```

也可以使用完整推理引擎：

```bash
python inference/full_infer.py \
  --video path/to/video.mp4 \
  --checkpoint outputs/checkpoints/best_llm_optimized.pt \
  --title "视频标题" \
  --tags "物理|科普|实验" \
  --category "physics"
```

输出示例：

```json
{
  "scientific_score": 0.12,
  "technical_score": 0.08,
  "aesthetic_score": 0.21,
  "overall_score": 0.34,
  "probability": 0.5842,
  "threshold": 0.55,
  "prediction": "top"
}
```

## 9. 最近修复与当前状态

当前代码已完成以下关键修复：

- checkpoint 适配逻辑只处理 MVP 模型参数，不再错误注入 research-only 参数
- 训练和推理统一使用 checkpoint 中保存的模型配置
- 推理阶段补齐训练使用的 DNSMOS、WPM、语速节奏、科学性手工特征、LLM 知识特征
- DeepSeek 模型更新为 `deepseek-v4-pro`，启用 thinking mode、JSON 输出、失败重试与断点缓存
- LLM 输出长度提高到 8192 token，避免高强度推理过程占满预算后截断最终 JSON
- 科学性分支新增 RoBERTa 分析文本编码，以及论文思路对应的 IFG + residual 融合
- 元数据加载会排除同一 `video_id` 标签互相冲突的样本，并按 `video_id` 去重，防止训练/验证泄漏
- `seed` 与 `split_seed` 已拆分，可在固定验证集上独立评估初始化稳定性
- 推理会从 checkpoint 恢复 LLM 文本源和科学性特征掩码，避免训练/推理特征不一致
- 验证集 pair 改为确定性生成，实验指标更容易复现
- 推理模块降低顶层重依赖，单独加载 checkpoint 不会因为 `clip` 或 `soundfile` 缺失而失败

当前数据状态：

```text
筛选 CSV：692 行（正类 183，负类 509）
唯一视频：629 个
排除 14 个标签冲突 ID，并合并同 ID 重复标注后：615 个（正类 164，负类 451）
唯一特征文件：652（含 37 个清洗元数据外的历史文件）
实际训练覆盖：615/615（正类 164，负类 451）
DeepSeek 有效缓存：652/652
Whisper audio / DNSMOS / WPM / speech rhythm / 两套 LLM 分析向量：615/615
COVER 官方三分支特征：615/615
```

2026-08-18 已断点补齐原先缺失的 273 个训练样本。`tools/check_features.py` 检查 615 个训练文件后，
所有字段维度、有限值和 COVER 版本均通过。另有 37 个历史特征文件不属于清洗后的训练元数据，训练时自动排除。

## 10. 科学性实验结果

以下结果使用完整 615 个视频、同一数据划分（`split_seed=42`）、全局 pair、`seed=42` 和 `lr=5e-5`：

| 科学性方案 | 验证 pairwise Acc / AUC |
|---|---:|
| 无 LLM | 0.7721 |
| 仅 4 维 LLM 评分 | 0.7663 |
| 仅最终分析文本 RoBERTa 向量，直接拼接 | 0.7923 |
| 最终分析向量 + 4 维评分，直接拼接 | 0.7859 |
| **推理全文 + 最终分析 + 评分，直接拼接** | **0.8091** |
| 最终分析向量 + 评分，IFG | 0.7852 |
| 推理全文 + 最终分析 + 评分，IFG | 0.8020 |
| 推理全文 + 评分 + 手工科学特征，完整 IFG | 0.8051 |

最佳训练目标为 RankNet + pointwise calibration。加入 20% 分支权重下限后，seed 42 达到 **0.8438**，
三种子均值为 `0.7987 +/- 0.0377`。该约束避免 learned fusion 完全塌缩到科学分支，同时保留了最佳性能。

最终主模型为 `outputs/checkpoints/best_full_dataset_balanced.pt`；完整实验解释见
`EXPERIMENT_RESULTS_FULL_DATASET.md`。

完整结果位于：

```text
outputs/logs/feature_diagnostics.json
outputs/logs/feature_diagnostics_with_audio.json
outputs/logs/science_ablation.json
outputs/logs/cover_ablation.json
outputs/logs/cover_ablation_multiseed.json
outputs/logs/branch_floor_multiseed.json
outputs/checkpoints/science_ablation/
outputs/checkpoints/cover_ablation/
```

## 11. 常见问题

### 1. ffmpeg 找不到

检查 `pipeline/config.py` 中的 `ffmpeg_path`，或将 ffmpeg 加入系统 PATH。

### 2. CUDA 不可用或显存不足

可以在 `pipeline/config.py` 中将：

```python
device = "cpu"
```

或者减小 `batch_size`。

### 3. LLM 知识特征没有 API Key

主流水线未设置 `DEEPSEEK_API_KEY` 时会使用 `DummyLLMKnowledgeExtractor` 保持流程可运行，
但该占位输出不能用于最终训练或论文实验。正式生成应使用 `prefetch_llm_cache.py`，它会在缺少 Key 时直接报错。

Windows PowerShell 设置方式：

```powershell
$env:DEEPSEEK_API_KEY="你的 API Key"
```

### 4. 训练时报没有正负样本

需要确认：

- 元数据中同时存在 `label=1` 和 `label=0`
- `outputs/features/` 中存在对应 `video_id.pt`
- `video_id` 与视频文件名/元数据一致

### 5. 推理结果和训练结果不一致

优先检查是否使用了最新训练出的 checkpoint。当前推理脚本会读取 checkpoint 中保存的模型结构和阈值，旧 checkpoint 若缺少完整配置，会回退到当前 `CFG`。

## 12. 毕设论文可对应的技术点

可以在论文中对应展开：

- 无参考科普短视频质量评估任务定义
- 弱监督 pairwise ranking 建模
- 科学性、技术性、美学性三分支结构
- ASR + BERT + CLIP + Whisper + DNSMOS 多模态特征融合
- LLM 知识特征增强科学性评估
- 门控融合与可解释分支分数
- 验证集固定 pair 的实验可复现设计
