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

视频 + 元数据 + WPM + 语速节奏（DNSMOS/COVER 仅消融）
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
python pipeline/run_pipeline.py --profile main_v2
```

`main_v2` 默认使用 DeepSeek 离线缓存、Shared CLIP，且不加载 COVER/DNSMOS。正式 `main_v3`
在此基础上增加冻结 COVER technical backbone 的 768D 离线缓存，不运行完整 COVER score：

```bash
python pipeline/run_pipeline.py --profile main_v3
```

完整 COVER 消融与旧完整路径分别使用：

```bash
python pipeline/run_pipeline.py --profile cover_ablation
python pipeline/run_pipeline.py --profile legacy_full
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

当前高效主路径训练配置：

```bash
python training/train.py \
  --profile main_v2 \
  --epochs 30 \
  --early_stop_patience 10 \
  --lr 5e-5 \
  --seed 42 \
  --split_seed 42 \
  --pair_scope global \
  --loss_type ranknet \
  --lambda_pointwise 0.1 \
  --branch_weight_floor 0.2 \
  --technical_feature_mode full_no_dnsmos \
  --aesthetic_feature_backend shared_clip \
  --science_feature_mode analysis_scores_concat \
  --llm_text_source reasoning_and_analysis \
  --checkpoint outputs/checkpoints/best_main_v2.pt
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
- 推理阶段按 checkpoint 配置恢复 WPM、语速节奏、科学性和可选 DNSMOS 特征
- DeepSeek 模型更新为 `deepseek-v4-pro`，启用 thinking mode、JSON 输出、失败重试与断点缓存
- LLM 输出长度提高到 8192 token，避免高强度推理过程占满预算后截断最终 JSON
- 科学性分支新增 RoBERTa 分析文本编码，以及论文思路对应的 IFG + residual 融合
- 元数据加载会排除同一 `video_id` 标签互相冲突的样本，并按 `video_id` 去重，防止训练/验证泄漏
- `seed` 与 `split_seed` 已拆分，可在固定验证集上独立评估初始化稳定性
- 推理会从 checkpoint 恢复 LLM 文本源和科学性特征掩码，避免训练/推理特征不一致
- 验证集 pair 改为确定性生成，实验指标更容易复现
- 推理模块降低顶层重依赖，单独加载 checkpoint 不会因为 `clip` 或 `soundfile` 缺失而失败
- 新增 `main_v2`、`cover_ablation`、`legacy_full` 三种 profile；COVER 不再进入默认路径
- 常规提取和推理强制 DeepSeek cache-only，只有 `prefetch_llm_cache.py` 或显式
  `--llm-online-enroll` 可以访问 API
- 主科学性融合显式设为 concat，IFG 仅保留为消融方案
- DNSMOS 已从 `main_v2` 技术分支移除，历史特征和消融入口继续保留
- Shared CLIP 直接复用 ViT-L/14 帧向量生成美学特征，不再执行第二套图像编码

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
outputs/logs/technical_ablation.json
outputs/logs/aesthetic_ablation.json
outputs/logs/shared_aesthetic_backfill.json
outputs/checkpoints/aesthetic_ablation/
outputs/main_v3_validation/
```

## 10.1 高效主路径消融（2026-08-22）

技术分支三随机种子结果：

| 技术输入 | ROC-AUC mean +/- std | PR-AUC | Technical SRCC |
|---|---:|---:|---:|
| visual only | 0.7825 +/- 0.0151 | 0.5792 | 0.0625 |
| visual + DNSMOS | 0.7873 +/- 0.0209 | 0.5728 | 0.0359 |
| DNSMOS only | 0.7836 +/- 0.0201 | 0.5984 | 0.0336 |
| full | 0.7859 +/- 0.0256 | 0.5831 | 0.0712 |
| **full without DNSMOS** | 0.7809 +/- 0.0179 | 0.5789 | **0.1083** |

DNSMOS 对完整技术分支的平均 AUC 仅增加 0.0050，却使 Technical SRCC 下降 0.0371，
因此 `main_v2` 使用 `full_no_dnsmos`，即保留视觉、元信息、WPM 和 speech rhythm。

Shared CLIP 三随机种子结果：

| 美学特征 | ROC-AUC mean +/- std | PR-AUC | Aesthetic SRCC |
|---|---:|---:|---:|
| legacy ViT-B/32 | 0.7809 +/- 0.0179 | 0.5789 | 0.2655 |
| **shared ViT-L/14** | **0.7853 +/- 0.0230** | **0.6047** | **0.2704** |

652 个共享美学特征全部回填成功；利用现有帧向量计算 652 条特征只需 3.12 秒。
历史最高性能 checkpoint `best_full_dataset_balanced.pt` 仍保留；高效路径 checkpoint 为
`outputs/checkpoints/best_main_v2.pt`。

## 10.2 COVER Technical main_v3 稳定性验证（2026-09-12）

固定 `split_seed=42`，在训练 seeds `42/123/2026` 上严格比较 main_v2 与仅替换技术视觉表征的
main_v3-candidate：

| 模型 | ROC-AUC mean +/- std | PR-AUC | Technical SRCC |
|---|---:|---:|---:|
| main_v2 | 0.7721 +/- 0.0169 | 0.5584 | 0.1109 |
| CLIP+COVER D1 (128+256) | 0.7706 +/- 0.0340 | 0.5841 | 0.2709 |
| **CLIP+COVER D3 (256+256)** | **0.7804 +/- 0.0178** | **0.5850** | **0.2900** |

D3 相对 main_v2 的 AUC、PR-AUC、Technical SRCC 分别提高 `0.0083`、`0.0266`、`0.1791`；
三个 seed 的 Technical SRCC 均提高。技术分支权重仍约为 0.20，说明旧技术表征偏弱并不是
fusion 偏向科学分支的唯一原因。正式 `main_v3` 使用 D3 的 CLIP 256D + COVER 256D，训练参数
1,516,423 个；COVER backbone 的 28,078,620 个参数冻结且只参与离线特征提取。

完整原始结果与报告位于 `outputs/main_v3_validation/`。

## 10.3 COVER-inspired Progressive Training（2026-09-14）

在 main_v3-D3 上固定数据、特征、`split_seed=42` 和验证划分，比较直接联合训练 J0 与
branch warm-up、branch RankNet、总体排序集成和融合校准。最终采用不含额外 Stage 3 的
P2-S2（Stage1 branch warm-up + Stage2 overall ranking，branch RankNet 权重 0.05）：

| 模型 | ROC-AUC mean +/- std | PR-AUC | Accuracy | F1 | Technical SRCC |
|---|---:|---:|---:|---:|---:|
| J0 joint | 0.7782 +/- 0.0171 | 0.5885 | 0.7696 | 0.6238 | 0.2495 |
| **P2-S2** | **0.7998 +/- 0.0160** | **0.6174** | **0.7940** | 0.6235 | 0.2475 |

P2-S2 在 seeds `42/123/2026` 上逐一提高 AUC，平均 AUC、PR-AUC、Accuracy 分别提升
`0.0215`、`0.0289`、`0.0244`，F1 基本持平。Technical/Aesthetic SRCC 没有稳定提高，且
Technical SRCC 从 Stage1 到 Stage2 的三种子平均漂移为 `-0.0624`；因此结论是分支级排序正则
改善总体排序泛化，而不是已经消除梯度竞争或 Technical branch collapse。

最终决策为 `USE_PROGRESSIVE_BRANCH_RANK`。完整逐 seed 结果、epoch history、target-gap 统计、
checkpoints 与报告位于 `outputs/progressive_training/`。

## 10.4 Fine-grained Attribute Supervision（2026-09-14）

在固定 main_v3-D3 representation、615 个清洗样本、相同 492/123 划分和 validation hash 的
条件下，新增 7 个轻量属性头与一个可选连续质量头。所有 head 仅用于辅助监督，不作为 fusion
或 ranking head 的输入；本轮没有重新提取 CLIP、COVER、DeepSeek、Whisper 或 DNSMOS 特征。

| 模型（seed 42） | ROC-AUC | PR-AUC | Branch macro SRCC | Attribute mean SRCC | Quality SRCC |
|---|---:|---:|---:|---:|---:|
| A0 Ranking only | 0.7798 | 0.6651 | 0.1019 | - | - |
| A1 Branch + consistency | 0.7923 | 0.6444 | 0.2971 | - | - |
| **A2 7 attributes** | **0.8118** | **0.6624** | 0.2957 | **0.3328** | - |
| A3 Branch + attributes | 0.7896 | 0.6517 | 0.2882 | 0.3185 | - |
| A4 Branch + attributes + quality | 0.7976 | 0.6360 | 0.2883 | 0.3117 | 0.3997 |

A1 与入选 A2 的 seeds `42/123/2026` 复验中，A2 的 ROC-AUC 为
`0.7847 +/- 0.0215`，相对 A1 的 `0.7791 +/- 0.0183` 提高 `0.0056`；PR-AUC 提高
`0.0229`。最终决策为 `PROMOTE_ATTRIBUTE_SUPERVISION`：7 项原始评分比 3 项聚合 target
更适合作为后续监督信号。但 A2 尚未超过 progressive P2-S2 的 `0.7998 +/- 0.0160` AUC，
因此不直接替换当前最佳 checkpoint，下一步应在 P2 Stage2 内验证 attribute loss。

复现实验：

```bash
python tools/run_fine_grained_supervision.py --experiments F0,F1,F2,F3,A2,A3,A4
python tools/run_fine_grained_supervision.py --seed 123 --experiments A1,A2
python tools/run_fine_grained_supervision.py --seed 2026 --experiments A1,A2
```

可用开关包括 `--use-branch-supervision`、`--use-attribute-supervision`、
`--use-quality-supervision`、`--lambda-attr` 和 `--lambda-quality`。完整结果、逐属性指标、
checkpoints 与标签审计位于 `outputs/fine_grained_supervision/`。

## 10.5 Progressive Training × Attribute Supervision（2026-09-14）

进一步将 7 属性监督接入当前最强 P2-S2。P0 复用原 P2-S2；P1 增加 attribute MSE；P2 再增加
continuous quality MSE。模型结构、特征、数据划分、branch floor 和两阶段训练日程均保持不变。

| 模型（seed 42） | ROC-AUC | PR-AUC | F1 | Branch macro SRCC | Attribute SRCC | Quality SRCC |
|---|---:|---:|---:|---:|---:|---:|
| P0 P2-S2 | 0.8114 | **0.6522** | 0.6230 | 0.3192 | - | - |
| P1 + attributes | 0.8178 | 0.6481 | 0.6250 | 0.3149 | 0.3420 | - |
| **P2 + attributes + quality** | **0.8185** | 0.6495 | **0.6400** | 0.3197 | 0.3322 | 0.4363 |
| P1-Simple, no branch MSE | 0.8027 | 0.6013 | 0.6329 | 0.3481 | 0.3426 | - |

seed=42 触发了 P1-Simple，但删除 branch MSE 后 AUC/PR-AUC 分别比 P1 下降
`0.0152/0.0468`，因此保留 branch supervision。P1 和 P2 均晋级 seeds `42/123/2026`：

| 模型 | ROC-AUC mean +/- std | PR-AUC mean +/- std | F1 mean +/- std |
|---|---:|---:|---:|
| P0 | 0.7998 +/- 0.0160 | 0.6174 +/- 0.0357 | **0.6235 +/- 0.0151** |
| **P1** | **0.8000 +/- 0.0186** | **0.6216 +/- 0.0314** | 0.6082 +/- 0.0199 |
| P2 | 0.7923 +/- 0.0268 | 0.5770 +/- 0.0643 | 0.6057 +/- 0.0503 |

P1 相对 P0 的 AUC/PR-AUC/F1 分别变化 `+0.0002/+0.0042/-0.0153`，attribute mean SRCC 为
`0.3138 +/- 0.0246`。它满足原实验的机械晋级门槛，但提升小于随机种子波动，且 branch macro
与 q_rank-quality 分别下降 `0.0118/0.0366`。
P2 相对 P0 的 AUC/PR-AUC 为 `-0.0075/-0.0404`；虽然 quality head 保持
`0.3993 +/- 0.0374` SRCC，适合作为诊断输出，但不进入最终排名模型。综合稳定性、F1、分支语义
和连续质量相关性后，正式主模型退回 P0，最终决策为 `KEEP_P2_S2`；P1 仅保留为属性可解释性消融。

```bash
python tools/run_progressive_attribute.py --models P0,P1,P2
python tools/run_progressive_attribute.py --models P1-Simple
python tools/run_progressive_attribute.py --seed 123 --models P0,P1,P2
python tools/run_progressive_attribute.py --seed 2026 --models P0,P1,P2
```

完整结果、stage diagnostics、epoch loss 和 checkpoints 位于 `outputs/progressive_attribute/`。

## 10.6 Temporal Quality Statistics（2026-09-20）

在最终 `KEEP_P2_S2` 上只替换 COVER technical 路径的时间聚合方式。原缓存是对
`[B,768,T,H,W]` 同时做时空均值得到的 768D 向量；本轮使用同一 COVER checkpoint、采样和预处理，
一次性缓存空间池化后的 `[20,768]` 序列。615/615 成功，无短序列；序列均值复现旧特征的最大误差
为 `4.77e-7`。

| seed 42 | AUC | PR-AUC | F1 | Overall SRCC | Tech SRCC | Tech PLCC | Tech MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0 pooled | 0.8114 | 0.6522 | 0.6230 | 0.4548 | 0.2542 | 0.2090 | 0.0438 |
| T1 mean+std | **0.8148** | **0.6520** | **0.6349** | 0.4529 | 0.2616 | 0.2480 | **0.0405** |
| T2 mean+std+diff | 0.8077 | 0.6235 | 0.6216 | **0.4649** | **0.3602** | **0.3727** | 0.0499 |

T2 因技术分支提升明显而晋级三种子，但收益不稳定：

| Variant | AUC mean +/- std | PR-AUC | Tech SRCC | Tech PLCC | Tech MSE |
|---|---:|---:|---:|---:|---:|
| **T0** | **0.7998 +/- 0.0160** | **0.6174 +/- 0.0357** | 0.2475 +/- 0.0366 | 0.2362 +/- 0.0636 | **0.0486 +/- 0.0070** |
| T2 | 0.7924 +/- 0.0147 | 0.5946 +/- 0.0258 | **0.2717 +/- 0.0911** | **0.2894 +/- 0.0994** | 0.0657 +/- 0.0327 |

T2 的逐 seed Tech SRCC 变化为 `+0.1060/-0.0297/-0.0036`，仅 seed 42 提升；平均
AUC/PR-AUC 分别下降 `0.0074/0.0228`，Tech MSE 增加 `0.0171`。最终决策为 `KEEP_T0`，
继续使用原 `KEEP_P2_S2`。时间统计保留为诊断特征，不进入主模型。

```bash
python tools/backfill_cover_temporal.py --metadata <metadata.csv> --feature-dir <features>
python tools/run_temporal_quality_statistics.py --metadata <metadata.csv> --feature-dir <features> --variants T0_KEEP_P2_S2,T1_MEAN_STD,T2_MEAN_STD_DIFF --seed 42
python tools/run_temporal_quality_statistics.py --metadata <metadata.csv> --feature-dir <features> --variants T0_KEEP_P2_S2,T2_MEAN_STD_DIFF --seed 123
python tools/run_temporal_quality_statistics.py --metadata <metadata.csv> --feature-dir <features> --variants T0_KEEP_P2_S2,T2_MEAN_STD_DIFF --seed 2026
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
- ASR + BERT + Shared CLIP + Whisper 多模态特征融合，DNSMOS/COVER 可选
- LLM 知识特征增强科学性评估
- 门控融合与可解释分支分数
- 验证集固定 pair 的实验可复现设计
