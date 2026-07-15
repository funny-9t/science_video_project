# Role

你是一位 CVPR / ACM MM / TMM 方向研究员，同时也是该项目的核心维护者。

请不要重写项目。

请基于现有 science_video_project 代码库进行增量升级（Incremental Research Upgrade）。

目标：

在保持现有训练流程、推理接口、Checkpoint兼容性、特征提取流水线不变的前提下，

将当前 MVP 模型升级为：

“面向科普短视频的多模态质量评估系统（Research Version）”。

---

# 一、现有项目现状

当前系统：

MultiModalQualityModel

包含：

ScientificBranch
TechnicalBranch
AestheticBranch

三分支结构。

训练目标：

Pairwise Ranking。

输出：

overall_score
probability

---

# 二、现有架构问题

请不要删除现有模块。

但需要解决以下问题：

## 问题1

当前模型学习的是：

```text
是否上榜
```

而不是：

```text
视频质量
```

即：

overall_score

缺乏明确质量语义。

---

## 问题2

Scientific Score

Technical Score

Aesthetic Score

仅作为辅助分支。

没有形成显式质量空间。

---

## 问题3

Gate只能学习：

哪个分支更重要

但无法学习：

质量维度之间的关系。

---

## 问题4

Pairwise Ranking

只能学习：

正样本 > 负样本

无法学习：

为什么更好。

---

# 三、升级目标

保留：

ScientificBranch

TechnicalBranch

AestheticBranch

同时新增：

Quality Space Layer

---

# 四、质量空间设计

新增：

```python
quality_scores = [
scientific_score,
technical_score,
aesthetic_score
]
```

shape:

```python
(B,3)
```

---

新增：

QualityHead

结构：

```python
MLP(
3 → 32 → 16 → 1
)
```

输出：

```python
quality_score
```

意义：

学习：

```text
质量维度组合规律
```

而不是固定平均。

---

# 五、Consistency Learning

新增：

Consistency Loss

约束：

```python
quality_score
≈
overall_score
```

公式：

```python
L_cons =
MSE(
quality_score,
overall_score
)
```

作用：

让排序结果具有质量解释。

---

# 六、Metadata升级

当前：

meta_feat

只有：

- 时长
- 标签数
- 标题长度
- 发布时间编码

等基础特征。

---

新增：

统计传播特征：

```python
log(fans+1)
log(likes+1)
log(comments+1)
log(shares+1)
log(favorites+1)
```

---

新增：

EngagementBranch

结构：

```python
meta_feat
↓
MLP
↓
engagement_score
```

输出：

```python
engagement_score
```

---

注意：

engagement_score

不得直接参与最终分类。

只能：

作为辅助任务。

避免标签泄露。

---

# 七、Scientific Branch升级

当前：

text_feat

来自：

RoBERTa CLS

---

新增：

Handcrafted Scientific Features

例如：

```python
term_density

entity_count

number_density

avg_sentence_length

keyword_coverage
```

---

与：

RoBERTa embedding

拼接：

```python
text_feature =
concat(
bert_feat,
science_features
)
```

---

重新训练：

ScientificBranch

---

# 八、Aesthetic Branch升级

当前：

CLIP Prompt Scorer

输出：

7维美学特征

---

保留。

新增：

Aesthetic MLP

学习：

```python
aes_feature
↓
MLP
↓
aesthetic_score
```

而非直接使用Prompt得分。

---

# 九、Temporal Modeling

当前：

video_feat

来自：

CLIP逐帧均值

即：

```python
mean(frame_features)
```

---

升级：

增加：

TemporalEncoder

可选：

```python
TransformerEncoder
```

或：

```python
BiGRU
```

结构：

```python
frame_features

↓

TemporalEncoder

↓

video_feature
```

---

保留原有实现。

通过配置控制：

```yaml
use_temporal_encoder: true
```

---

# 十、Cross-Modal Fusion升级

当前：

Gate

只对：

三个分支隐向量

进行加权。

---

升级：

新增：

CrossModalAttention

结构：

```python
text_feature

video_feature

audio_feature

↓

MultiHeadAttention

↓

fusion_feature
```

---

保留Gate。

形成：

```text
CrossModalAttention

↓

Gate Fusion

↓

Overall Head
```

---

# 十一、训练目标升级

保留：

Pairwise Ranking Loss

---

新增：

Quality Consistency Loss

```python
L_cons
```

---

新增：

Branch Diversity Loss

避免：

三个分支学到相同内容。

例如：

```python
cosine_similarity(
sci_h,
tech_h
)
```

最小化。

---

总损失：

```python
Loss =
L_rank
+
0.2 * L_cons
+
0.05 * L_div
```

---

# 十二、解释性增强

训练结束后输出：

```python
{
    scientific_score,
    technical_score,
    aesthetic_score,
    engagement_score,
    quality_score,
    overall_score,
    gate_weights
}
```

推理接口同步升级。

---

# 十三、实验系统

自动生成：

Exp1 Baseline

Exp2 + QualityHead

Exp3 + Consistency Loss

Exp4 + Temporal Encoder

Exp5 + CrossModalAttention

Exp6 + Scientific Features

Exp7 Full Model

---

# 十四、代码要求

不要删除现有模块。

不要修改已有API。

不要破坏Checkpoint兼容性。

采用：

Backward Compatible Design。

新增模块必须：

- 独立文件
- 配置驱动
- 可开关

例如：

```yaml
use_quality_head: true

use_temporal_encoder: true

use_cross_modal_attention: true

use_science_features: true
```

---

# 最终目标

将当前 science_video_project

从：

Pairwise Ranking MVP

升级为：

具备论文创新点的

Multi-dimensional Quality Assessment Framework

并保持现有代码库可继续训练与部署。