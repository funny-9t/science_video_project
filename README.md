# 面向科普短视频的多维质量评估系统

> **Multi-dimensional Quality Assessment Framework for Science Short Videos**

基于多模态特征融合与弱监督排序学习，构建科普类短视频无参考质量评估系统。支持 **MVP 版**（Pairwise Ranking）和 **研究版**（多维质量空间建模）两种模式。

---

## 目录

- [1. 项目概览](#1-项目概览)
- [2. 模型架构](#2-模型架构)
- [3. 目录结构](#3-目录结构)
- [4. 快速开始](#4-快速开始)
- [5. 特征提取流水线](#5-特征提取流水线)
- [6. 训练](#6-训练)
- [7. 推理](#7-推理)
- [8. Web 服务](#8-web-服务)
- [9. 研究版升级](#9-研究版升级)
- [10. 消融实验](#10-消融实验)
- [11. 配置说明](#11-配置说明)
- [12. 常见问题](#12-常见问题)

---

## 1. 项目概览

### 问题定义

无参考视频质量评估（NR-VQA）任务。给定科普短视频及其元数据，输出多维质量评分和"上榜/未上榜"预测。

**输入**：$\mathbf{x} = \{\text{video}, \text{audio}, \text{text}, \text{meta}\}$

**输出**：
- `scientific_score` — 科学内容质量
- `technical_score` — 技术制作质量
- `aesthetic_score` — 视觉美学质量
- `quality_score` — 学习到的质量空间得分（研究版）
- `overall_score` — 综合评分
- `probability` — 上榜概率

### 关键约束

| 约束 | 说明 |
|------|------|
| 无 MOS 绝对评分 | 仅有"上榜/未上榜"二值标签 |
| 传播性指标隔离 | 点赞/评论/转发等不参与主干建模，避免信息泄露 |
| Pairwise Ranking | $q(x_{pos}) > q(x_{neg})$ |

---

## 2. 模型架构

### 2.1 MVP 版：`MultiModalQualityModel`

```
输入特征                    三分支评分                     门控融合
─────────────────────────────────────────────────────────────────
text(768) ──┐           ┌─ ScientificBranch ─→ sci_score ─┐
meta(16)  ──┘           │                                  │
                        │                                  │
video(512)──┐           ├─ TechnicalBranch  ─→ tech_score ─┼─→ Gate → Fusion → overall_score
audio(384)──┼───────────┤                                  │              → probability
meta(16)   ─┘           │                                  │
                        │                                  │
video(512)──┐           └─ AestheticBranch  ─→ aes_score  ─┘
text(768) ──┼─ AestheticBranch
audio(384)──┤
aes(7)    ──┘
```

### 2.2 研究版：`ResearchModel`（新增模块以 ★ 标注）

```
                          ┌─────────────────────────────────┐
                          │       §10 CrossModalAttention ★  │
                          │  MultiHeadAttention(text,video,  │
                          │                    audio)        │
                          └──────────┬──────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ ScientificBranch │      │ TechnicalBranch  │      │ AestheticBranch  │
│ text(768+5★)+meta│      │ video+audio+meta │      │ video+text+audio │
│ §7 手工科学特征 ★│      │                  │      │ +aes+§8 MLP ★    │
│ sci_score        │      │ tech_score       │      │ aes_score        │
└────────┬────────┘      └────────┬────────┘      └────────┬────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │     §4 QualityHead ★       │
                    │  MLP(sci,tech,aes)→quality │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │    Gate(Fusion) + Fusion   │
                    └─────────────┬─────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                    ▼
      overall_score         probability        §6 EngagementBranch ★
                                                (辅助,不参与分类)
```

**新增创新模块**（详见 [§9 研究版升级](#9-研究版升级)）：

| 模块 | 章节 | 参数量 | 说明 |
|------|------|--------|------|
| QualityHead | §4 | +1.7K | MLP(3→32→16→1) 质量空间建模 |
| EngagementBranch | §6 | +37K | 传播力辅助预测 |
| ScienceFeatureProjection | §7 | +99K | BERT+手工科学性特征融合 |
| AestheticMLP | §8 | +17K | Prompt 得分非线性学习 |
| TemporalEncoder | §9 | +2.1M | BiGRU/Transformer 时序编码 |
| CrossModalAttention | §10 | +2.9M | 跨模态注意力融合 |
![alt text](image.png)
---

## 3. 目录结构

```
science_video_ranker_mvp/
│
├── README.md                          ← 本文件
├── 1_QUICK_START.md                   ← 快速启动指南
├── 2_README_FRONTEND.md               ← 前端说明
├── 3_DEPLOYMENT_GUIDE.md              ← 部署指南
├── 架构.md                            ← 架构文档
├── start_service.bat / .sh            ← 一键启动脚本
│
├── chinese-robeta-wwm-ext/            ← 中文 RoBERTa 模型权重
├── faster-whisper-large-v2/           ← Whisper ASR 模型权重
├── openaiclip-vit-large-patch14/      ← CLIP 视觉模型权重
│
└── science_video_project/             ← 主项目目录
    │
    ├── data/
    │   ├── videos/                    ← 原始视频
    │   ├── metadata.csv               ← 元数据标签
    │   └── parse.py                   ← 元数据解析
    │
    ├── outputs/
    │   ├── frames/                    ← 抽帧输出
    │   ├── audio/                     ← 音频提取
    │   ├── features/                  ← 特征 .pt 文件
    │   ├── checkpoints/               ← 模型检查点
    │   └── logs/                      ← 训练/流水线日志
    │
    ├── pipeline/                      ← 特征提取流水线
    │   ├── config.py                  ← 全局配置
    │   ├── utils_io.py                ← IO 工具
    │   ├── step_extract.py            ← ffmpeg/OpenCV 抽帧
    │   ├── step_text.py               ← ASR + BERT 文本编码
    │   ├── step_video.py              ← CLIP 视频编码
    │   ├── step_audio.py              ← Whisper 音频编码
    │   ├── step_meta.py               ← 元数据特征构建
    │   ├── step_aesthetic_clip.py     ← CLIP Prompt 美学评分
    │   ├── step_science_features.py   ← ★ 手工科学性特征
    │   ├── build_sample.py            ← 特征组装
    │   ├── run_pipeline.py            ← MVP 流水线入口
    │   └── run_pipeline_research.py   ← ★ 研究版流水线入口
    │
    ├── training/                      ← 训练模块
    │   ├── model_mvp.py               ← MVP 三分支模型
    │   ├── model_research.py          ← ★ 研究版主模型
    │   ├── quality_head.py            ← ★ 质量空间头
    │   ├── cross_modal_attention.py   ← ★ 跨模态注意力
    │   ├── engagement_branch.py       ← ★ 传播力分支
    │   ├── step_temporal.py           ← ★ 时序编码器
    │   ├── losses.py                  ← MVP 损失函数
    │   ├── losses_research.py         ← ★ 研究版损失函数
    │   ├── metrics.py                 ← 评估指标
    │   ├── dataset_pair.py            ← Pairwise 数据集
    │   ├── dataloader_pair.py         ← DataLoader 整理
    │   ├── utils_train.py             ← 训练工具 + checkpoint 适配
    │   ├── train.py                   ← MVP 训练入口
    │   ├── train_research.py          ← ★ 研究版训练入口
    │   └── experiments.py             ← ★ 消融实验系统
    │
    ├── inference/                     ← 推理模块
    │   ├── infer.py                   ← 单视频推理
    │   ├── full_infer.py              ← 完整推理引擎
    │   └── batch_infer.py             ← 批量推理
    │
    ├── frontend/                      ← Web 前端
    │   ├── index.html
    │   └── static/
    │
    ├── tools/                         ← 诊断工具
    ├── app.py                         ← Flask Web 服务
    ├── requirements.txt               ← Python 依赖
    └── README.md                      ← 子项目文档
```

---

## 4. 快速开始

### 4.1 环境要求

- Python 3.10+
- CUDA 11.8+（可选，CPU 模式可用但较慢）
- ffmpeg（用于音视频提取）

### 4.2 安装依赖

```bash
cd science_video_project
pip install -r requirements.txt
```

`requirements.txt` 内容：
```
torch
transformers
safetensors
faster-whisper
openai-whisper
openai-clip
opencv-python
pillow
numpy
pandas
scikit-learn
tqdm
flask
flask-cors
```

### 4.3 准备数据

1. 将视频文件放入 `data/videos/`
2. 编辑 `data/metadata.csv`，至少包含：

```csv
video_id,label,category,title,tags,duration
demo_001,1,物理,量子纠缠科普,量子|物理|科普,120
demo_002,0,生物,细胞分裂过程,细胞|生物|分裂,90
```

- `video_id`：视频文件名（不含扩展名）
- `label`：1=上榜，0=未上榜
- **禁止**包含点赞/评论/转发等传播性字段

### 4.4 配置模型路径

编辑 `pipeline/config.py`，修改以下路径：

```python
text_model_name = r"<你的路径>/chinese-robeta-wwm-ext"
clip_model_name = r"<你的路径>/openaiclip-vit-large-patch14"
asr_model_path  = r"<你的路径>/faster-whisper-large-v2"
ffmpeg_path     = r"<你的路径>/ffmpeg/bin"
```

---

## 5. 特征提取流水线

### MVP 版流水线

```bash
python pipeline/run_pipeline.py
```

处理流程：
```
原始视频 → ffmpeg提取音频(16kHz) + OpenCV逐帧抽取(1fps)
        → Whisper ASR转写字幕
        → BERT 提取文本embedding (768维)
        → CLIP 提取视频embedding (512维, 逐帧均值池化)
        → Whisper Encoder 提取音频embedding (384维)
        → MetaFeatureBuilder 构建元数据特征 (16维)
        → CLIPAestheticScorer 美学评分 (7维)
        → 组装保存为 outputs/features/{video_id}.pt
```

### 研究版流水线（扩展特征）

```bash
python pipeline/run_pipeline_research.py
```

在 MVP 基础上新增：
- **手工科学性特征**（5维）：术语密度、实体数量、数字密度、平均句长、关键词覆盖率
- **逐帧 CLIP 特征**（N×512）：供时序编码器使用
- **传播力标签**：用于 EngagementBranch 辅助训练

---

## 6. 训练

### 6.1 MVP 训练

```bash
python training/train.py
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 10 | 训练轮数 |
| `--batch_size` | 8 | 批大小 |
| `--lr` | 1e-3 | 学习率 |
| `--margin` | 1.0 | Ranking margin |
| `--same_category` | True | 同类别内配对 |

**损失函数**：
$$L = L_{rank} + 0.1 \cdot L_{reg}$$

- $L_{rank}$：MarginRankingLoss，正样本得分 > 负样本得分
- $L_{reg}$：MSE(overall_score, mean(sci, tech, aes))

**评估指标**：ranking_accuracy、accuracy、F1、AUC

### 6.2 研究版训练

```bash
# 全模块训练
python training/train_research.py

# 仅启用部分模块
python training/train_research.py --quality_head 1 --consistency_loss 1 --cross_modal_attn 0

# 从 MVP checkpoint 热启动
python training/train_research.py --pretrained outputs/checkpoints/best.pt

# 使用 MVP 原始损失（基线对比）
python training/train_research.py --use_mvp_loss
```

**总损失函数**：
$$L_{total} = L_{rank} + 0.2 \cdot L_{cons} + 0.05 \cdot L_{div}$$

| 损失 | 公式 | 作用 |
|------|------|------|
| $L_{rank}$ | MarginRankingLoss | 成对排序 |
| $L_{cons}$ | MSE(quality_score, overall_score) | 质量一致性 |
| $L_{div}$ | mean(cos(sci_h, tech_h) + cos(sci_h, aes_h) + cos(tech_h, aes_h)) | 分支多样性 |

---

## 7. 推理

### 7.1 命令行推理

```bash
# MVP 模型
python inference/infer.py \
    --video data/videos/demo.mp4 \
    --checkpoint outputs/checkpoints/best.pt

# 完整推理引擎
python inference/full_infer.py \
    --mode video \
    --video data/videos/demo.mp4 \
    --checkpoint outputs/checkpoints/research_best.pt
```

### 7.2 输出格式

```json
{
  "scientific_score": 2.31,
  "technical_score": 1.87,
  "aesthetic_score": 2.54,
  "quality_score": 2.19,
  "engagement_score": 0.72,
  "overall_score": 2.45,
  "probability": 0.92,
  "gate_weights": {
    "scientific": 0.38,
    "technical": 0.27,
    "aesthetic": 0.35
  },
  "prediction": "上榜"
}
```

---

## 8. Web 服务

### 启动服务

```bash
# MVP 模式
python app.py

# 研究版模式
set RESEARCH_MODE=1 && python app.py    # Windows
RESEARCH_MODE=1 python app.py           # Linux/Mac
```

访问 `http://localhost:5000`

### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 前端页面 |
| `/api/infer` | POST | 推理接口（支持 JSON + 文件上传） |
| `/api/health` | GET | 健康检查 |

---

## 9. 研究版升级

以下是相对于 MVP 的完整创新点清单：

### §4 QualityHead — 质量空间层

```
sci_score, tech_score, aes_score (B,3)
         ↓
MLP(3 → 32 → 16 → 1)
         ↓
    quality_score (B,1)
```

**创新点**：自动学习三维质量的最优组合规律，替代固定平均。不同场景下科学性/技术/美学的权重不同（如医学科普更重科学性）。

### §5 Consistency Loss — 质量一致性约束

$$L_{cons} = \text{MSE}(quality\_score, overall\_score)$$

**创新点**：使排序结果具备质量语义解释——"为什么上榜"而非仅仅"是否上榜"。

### §6 EngagementBranch — 传播力辅助分支

```
meta_feat(16) → MLP → engagement_score (B,1)
```

**约束**：仅在训练时作为辅助监督，不参与 final classification，避免标签泄露。

### §7 ScienceFeatureExtractor — 手工科学性特征

从文本中提取 5 维语言学特征：

| 特征 | 计算方式 |
|------|----------|
| term_density | 科普关键词命中数 / 总词数 |
| entity_count | 长词（≥3字）比例 |
| number_density | 含数字词比例 |
| avg_sentence_length | 平均句长（归一化） |
| keyword_coverage | 关键词覆盖率 |

与 BERT 768 维 embedding 拼接后经 `ScienceFeatureProjection` 融合。

### §8 AestheticMLP — 美学特征学习

```
aes_feat(7) → MLP(7→128→64→1) → aesthetic_score
```

学习 CLIP Prompt 得分的非线性映射，替代简单的维度均值。

### §9 TemporalEncoder — 时序视频编码

```
frame_features (N, 512)
         ↓
BiGRU(2层, 双向) 或 TransformerEncoder(2层)
         ↓
    video_feature (512)
```

**创新点**：捕捉视频帧间时序动态（镜头切换、运动变化），替代 CLIP 逐帧均值池化。

### §10 CrossModalAttention — 跨模态注意力融合

```
text(768), video(512), audio(384)
         ↓
  投影到统一维度(768)
         ↓
MultiHeadAttention(3 tokens, 4 heads)
         ↓
  残差连接 + LayerNorm
         ↓
enhanced_text, enhanced_video, enhanced_audio
```

在分支编码之前进行跨模态信息交换，形成"先交叉→再分支"的增强架构。

### §11 Branch Diversity Loss — 分支多样性正则

$$L_{div} = \frac{1}{3}\sum \cos(h_i, h_j), \quad i \neq j \in \{sci, tech, aes\}$$

**创新点**：最小化分支隐向量余弦相似度，鼓励三个分支学习互补而非冗余的质量维度。

---

## 10. 消融实验

```bash
# 查看实验矩阵
python training/experiments.py --print_only

# 运行全部 7 组实验
python training/experiments.py --epochs 10 --batch_size 8

# 不跳过已有 checkpoint
python training/experiments.py --no_skip
```

### 实验矩阵

| 实验 | QualityHead | Consistency | Temporal | CrossModal | ScienceFeat | Engagement | Diversity |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exp1** Baseline | — | — | — | — | — | — | — |
| **Exp2** +QualityHead | ✓ | — | — | — | — | — | — |
| **Exp3** +Consistency | ✓ | ✓ | — | — | — | — | — |
| **Exp4** +Temporal | ✓ | ✓ | ✓ | — | — | — | — |
| **Exp5** +CrossModal | ✓ | ✓ | ✓ | ✓ | — | — | — |
| **Exp6** +ScienceFeat | ✓ | ✓ | ✓ | ✓ | ✓ | — | — |
| **Exp7** Full Model | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## 11. 配置说明

`pipeline/config.py` 中的关键配置：

```python
# ===== 模型维度 =====
text_dim: int = 768        # BERT 输出维度
video_dim: int = 512       # CLIP 输出维度
audio_dim: int = 384       # Whisper 输出维度
meta_dim: int = 16         # 元数据特征维度
aes_dim: int = 7           # CLIP Prompt 美学维度
sci_hand_dim: int = 5      # 手工科学性特征维度
temporal_dim: int = 256    # 时序编码器隐层维度
hidden_dim: int = 128      # 通用隐层维度

# ===== 研究版开关 =====
use_quality_head: bool = True              # §4 质量空间层
use_consistency_loss: bool = True          # §5 一致性损失
use_engagement_branch: bool = True         # §6 传播力分支
use_science_features: bool = True          # §7 手工科学特征
use_aesthetic_mlp: bool = True             # §8 美学MLP
use_temporal_encoder: bool = True          # §9 时序编码器
use_cross_modal_attention: bool = True     # §10 跨模态注意力
use_diversity_loss: bool = True            # §11 多样性正则

# ===== 损失权重 =====
lambda_consistency: float = 0.2            # 一致性损失权重
lambda_diversity: float = 0.05            # 多样性损失权重

# ===== 时序编码器 =====
temporal_arch: str = "bigru"              # "bigru" | "transformer"
temporal_num_layers: int = 2
temporal_num_heads: int = 4               # Transformer 模式

# ===== 跨模态注意力 =====
cross_modal_num_heads: int = 4
cross_modal_dropout: float = 0.1
```

---

## 12. 常见问题

<details>
<summary><b>Q: ffmpeg 找不到</b></summary>

设置环境变量 `FFMPEG_PATH` 或在 `config.py` 中配置：
```python
ffmpeg_path: str = r"D:\你的路径\ffmpeg\bin"
```
</details>

<details>
<summary><b>Q: CUDA out of memory</b></summary>

减小 `batch_size`，或将 `device = "cpu"`。
</details>

<details>
<summary><b>Q: 模型权重下载失败</b></summary>

提前下载模型权重到本地目录，并设置 `local_files_only = True`。
</details>

<details>
<summary><b>Q: 训练时正负样本不平衡</b></summary>

数据集自动确保正负样本都存在。如果某类别仅有单一标签，该类别会被跳过。
</details>

<details>
<summary><b>Q: 如何从 MVP checkpoint 热启动研究版训练？</b></summary>

```bash
python training/train_research.py --pretrained outputs/checkpoints/best.pt
```

新模块权重自动随机初始化，MVP 权重完全复用（70/70 keys matched）。
</details>

<details>
<summary><b>Q: ResearchModel 和 MultiModalQualityModel 的关系？</b></summary>

`ResearchModel` 是 `MultiModalQualityModel` 的**严格超集**。所有 MVP 功能完全保留，新模块通过配置开关控制。关闭所有开关后，行为与 MVP 完全一致。
</details>

---

## 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@misc{science-video-ranker,
  title  = {Multi-dimensional Quality Assessment Framework for Science Short Videos},
  author = {Science Video Project},
  year   = {2026},
  note   = {https://github.com/funny-9t/science_video_project}
}
```
