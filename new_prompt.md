你是一个专业的 AI 算法工程师，需要根据以下需求生成一个完整、可运行的 Python 项目代码。该项目用于训练和评估一个**面向科普短视频的多模态质量评价模型**。要求代码结构清晰、模块化，包含数据加载、模型定义、训练循环、评估指标和推理脚本。

### 一、项目背景与目标

现有来自抖音的科普短视频数据集，包含：
- **视频文件**（mp4）
- **结构化标注文件**（Excel/CSV），包含 7 个细粒度质量指标（1~5 分）以及“是否上榜”的二分类标签，同时还包含博主粉丝数、点赞/评论/转发/收藏量等统计特征。
- 部分样本只有“是否上榜”标签（粗粒度），部分样本同时有 7 个细粒度指标和上榜标签。

**目标**：构建一个多模态无参考视频质量评估模型，能够：
1. 对输入视频预测 7 个细粒度质量分数（科普信息量、科普通俗性、选题重要性、内容趣味性、低层视觉质量、听觉质量、视频美学质量）。
2. 同时输出一个综合排序分数，用于判断视频是否能够“上榜”（二分类排序）。
3. 利用 **pairwise ranking loss** 充分挖掘粗粒度数据中的相对顺序信息（上榜 > 未上榜），并结合细粒度数据的回归损失，实现多任务学习。

### 二、数据集说明（基于用户提供的 Excel 样例）

Excel 文件列名如下（示例数据已提供），你需要将其转换为 CSV 或直接在代码中读取 Excel：

| 列名 | 含义 | 备注 |
|------|------|------|
| `链接` | 视频链接（或本地视频文件名） | 实际项目中应替换为本地视频路径 |
| `是否上榜` | 是否上榜（是/否） | 二分类标签，用于构造 pair |
| `科普信息量` | 1~5 分 | 细粒度标签，仅部分样本有值 |
| `选题重要性` | 1~5 分 | 细粒度标签 |
| `科普通俗性` | 1~5 分 | 细粒度标签 |
| `内容趣味性` | 1~5 分 | 细粒度标签 |
| `低层视觉质量` | 1~5 分 | 细粒度标签 |
| `听觉质量` | 1~5 分 | 细粒度标签 |
| `视频美学质量` | 1~5 分 | 细粒度标签 |
| `粉丝量` | 整数 | 博主粉丝数，用于 Engagement 分支 |
| `点赞量` | 整数 | 用于 Engagement 分支 |
| `评论量` | 整数 | 用于 Engagement 分支 |
| `转发量` | 整数 | 用于 Engagement 分支 |
| `收藏量` | 整数 | 用于 Engagement 分支 |

**数据规模**：
- 细粒度样本：105 正（上榜） + 315 负（未上榜）（后续可扩至 315 + 945）
- 粗粒度样本：365 正 + 1095 负（仅包含“是否上榜”标签）

**视频文件存放路径**：假设为 `/data/videos/`，Excel 中的“链接”列包含文件名（如 `7640403274843065642.mp4`）。

### 三、模型架构要求

参考 COVER 的多分支思想，设计如下架构（可适当简化）：

#### 输入模态：
- 视频帧序列（从 mp4 均匀采样 8 帧）
- 音频波形（采样率 16kHz，取 5 秒或完整长度）
- 视频标题+字幕（从视频 OCR 提取，若无则用 Excel 中“内容（视频描述）”列代替）
- 统计特征（粉丝量、点赞量、评论量、转发量、收藏量）

#### 分支设计：

| 分支 | 输入 | 骨干网络 | 输出 |
|------|------|----------|------|
| Technical | 视频帧 | Swin-T (预训练) | 视觉质量分数 |
| Aesthetic | 视频帧 | CLIP ViT-B/16 (冻结图像编码器) | 美学分数 |
| Acoustics | 音频 Mel 谱 | Whisper tiny encoder (冻结) | 听觉质量分数 |
| Scientific | 标题+字幕 | Chinese-RoBERTa-wwm-ext (微调) | 信息量、通俗性、重要性 三个分数 |
| Engagement | 统计特征 | 2 层 MLP | 趣味性分数 |

**注**：趣味性分数也作为细粒度标签的一个输出，与标注中的“内容趣味性”对齐。

#### 综合排序头：
所有分支输出的特征向量（非标量）先分别通过各自的回归头得到分数，然后将**所有特征向量拼接**，再经过一个 2 层 MLP 输出一个**综合排序得分**（标量）。

### 四、损失函数与训练策略

#### 损失组成：
1. **回归损失**（仅对细粒度样本计算）：  
   对 7 个细粒度分数（信息量、通俗性、重要性、趣味性、视觉质量、听觉质量、美学质量）分别计算 MSE 或 L1Loss。  
   `loss_reg = sum(MSE(pred_i, target_i))`

2. **Pairwise Ranking 损失**（对所有样本计算）：  
   利用“是否上榜”标签构造正负对（上榜视频得分应高于未上榜视频）。  
   采样策略：每个 batch 包含 B 个视频，从中动态组合所有可能的 (正, 负) 对，计算 margin ranking loss：  
   `loss_rank = max(0, - (score_pos - score_neg) + margin)`  
   margin 设为 0.1。

3. **总损失**：  
   `total_loss = loss_reg + lambda_rank * loss_rank`  
   lambda_rank 可设为 0.5。

#### 优化器：
- AdamW，学习率 1e-4（骨干网络冻结部分使用更低学习率或单独设置）
- 采用余弦退火学习率调度

#### 训练技巧：
- 冻结预训练模型（CLIP、Whisper、Swin）的前几层，仅微调最后几层或回归头。
- 视频帧数据增强：随机裁剪、水平翻转、色彩抖动。
- 音频增强：添加高斯噪声、时间平移。
- 处理不平衡数据：在构造 pair 时保证每个 batch 中正负样本数量接近。

### 五、代码结构与文件清单

需要生成以下文件（放置在项目根目录下）：

'''
project/
├── config/
│ └── default.yaml # 配置文件（路径、超参数）
├── data/
│ ├── dataset.py # PyTorch Dataset 类，读取 Excel 和视频/音频/文本
│ └── pair_sampler.py # 自定义采样器，动态生成正负对
├── models/
│ ├── backbone_tech.py # Swin Transformer 封装
│ ├── backbone_aesthetic.py # CLIP 图像编码器
│ ├── backbone_acoustic.py # Whisper 音频编码器
│ ├── backbone_scientific.py # RoBERTa 文本编码器
│ ├── backbone_engagement.py # 统计特征 MLP
│ ├── multi_task_model.py # 完整模型，整合所有分支和排序头
│ └── losses.py # 回归损失 + pairwise ranking loss
├── train.py # 主训练脚本
├── evaluate.py # 评估脚本（计算 PLCC、SROCC、AUC、HitRate）
├── inference.py # 对单个视频进行推理，输出 7 维分数和上榜概率
├── utils/
│ ├── video_processing.py # 帧采样、音频提取
│ ├── text_processing.py # 标题+字幕处理
│ └── metrics.py # 评估指标计算
└── requirements.txt # 依赖列表
'''


### 六、需要实现的细节

#### 1. 数据加载
- 读取 Excel 文件（pandas）。
- 根据“链接”列构造视频文件路径。
- 对于每个视频：
  - 使用 `decord` 或 `cv2` 均匀采样 N=8 帧，resize 到 224x224。
  - 使用 `torchaudio` 加载音频，重采样到 16kHz，转换为 Mel 谱（80 bins，时间维度固定到 5 秒，若不足则补零）。
  - 文本字段：将“内容（视频描述）”作为字幕，如果有标题则拼接标题。
  - 统计特征：取粉丝量、点赞量、评论量、转发量、收藏量，进行归一化（使用全局统计量）。
- 返回一个字典：`{'frames': tensor, 'audio_mel': tensor, 'text': str, 'stats': tensor, 'fine_labels': tensor (7,) or None, 'is_ranked': int (0/1)}`

#### 2. 模型实现
- `MultiTaskModel` 类接收配置，初始化各分支。
- 前向传播：
  - 通过技术分支得到视觉质量分数（标量）和特征向量（如 Swin 的 cls token）。
  - 通过美学分支得到美学分数（标量）和特征向量（CLIP 的 cls token）。
  - 通过声学分支持听觉质量分数（标量）和特征向量（Whisper encoder 的均值池化）。
  - 通过科学分支得到三个分数（信息量、通俗性、重要性）和特征向量（RoBERTa 的 cls）。
  - 通过参与分支得到趣味性分数（标量）和特征向量（MLP 中间层）。
  - 将所有分支的特征向量拼接，输入排序头得到综合得分。
  - 返回：`{'scores_7dim': tensor (7,), 'rank_score': scalar, 'features': tensor}`

#### 3. 训练循环
- 每个 epoch：遍历 dataloader，对每个 batch 调用模型。
- 对细粒度样本计算回归损失（MSE）。
- 利用当前 batch 内的“是否上榜”标签构造所有正负对（上榜 vs 未上榜），计算 ranking loss（需要使用 batch 内的 rank_score）。
- 反向传播，更新参数。
- 记录训练损失，每 100 步打印。

#### 4. 评估指标
- **回归任务**（细粒度测试集）：PLCC, SROCC, RMSE（针对每个维度）
- **排序任务**（所有测试集）：AUC, Hit Rate@K (K=10, 20), NDCG@K
- 同时计算综合得分与“是否上榜”的 Spearman 相关系数。

### 七、额外要求

- 代码必须包含详细的注释，说明每个模块的作用。
- 支持 GPU 训练，自动检测 `cuda`。
- 提供 `requirements.txt`，包含 `torch, torchvision, torchaudio, transformers, openai-whisper, decord, opencv-python, pandas, numpy, scikit-learn, pyyaml, librosa`。
- 提供 `config/default.yaml` 示例，允许用户修改数据路径、批大小、学习率等。
- 训练完成后保存最佳模型 checkpoint（根据验证集的 AUC 或 SROCC 选择）。
- 推理脚本 `inference.py` 接受一个视频路径，输出 7 个分数和上榜概率（综合得分经过 sigmoid）。

### 八、参考文件

- 原 COVER 代码结构：https://github.com/taco-group/COVER
- 用户提供的 Excel 数据样例（见对话中表格）

请生成完整的代码，确保可以直接运行（假设数据按描述放置）。