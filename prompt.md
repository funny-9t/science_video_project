请在当前目录中直接生成一个可运行的完整 Python 项目，项目名为：

science_video_project

项目目标：
基于多模态特征融合与弱监督排序学习，构建一个“科普类短视频无参考质量评估系统”。系统输入短视频及其元数据，输出 scientific_score、technical_score、aesthetic_score、overall_score，以及“上榜/未上榜”预测结果。

====================
一、任务背景与问题定义
====================

这是一个无参考视频质量评估（NR-VQA）与多模态排序学习任务。

输入：
x = {video, audio, text, meta}

其中：
- video：视频帧序列
- audio：音频信号
- text：字幕、标题、标签
- meta：元数据（时长、类别等，不包含传播性指标）

输出：
- scientific_score
- technical_score
- aesthetic_score
- overall_score
- prediction（上榜 / 未上榜）

训练约束：
没有 MOS 绝对评分，只有“上榜/未上榜”标签，因此训练目标采用 pairwise ranking：

q(x_pos) > q(x_neg)

损失函数使用 margin ranking loss：

L = max(0, 1 - (q_pos - q_neg))

注意：
- 传播性指标（点赞、评论、转发、收藏、推荐量）不能作为模型输入，避免信息泄漏
- 传播性不参与主干建模
- 该项目先实现 MVP：先用离线特征提取 + 轻量多分支模型跑通完整闭环

====================
二、项目目录结构
====================

请严格创建以下目录结构：

science_video_project/
│
├── data/
│   ├── videos/
│   ├── metadata.csv
│
├── outputs/
│   ├── frames/
│   ├── audio/
│   ├── features/
│   ├── checkpoints/
│   └── logs/
│
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
│
├── training/
│   ├── __init__.py
│   ├── model_mvp.py
│   ├── dataset_pair.py
│   ├── dataloader_pair.py
│   ├── losses.py
│   ├── metrics.py
│   ├── utils_train.py
│   └── train.py
│
├── inference/
│   ├── __init__.py
│   └── infer.py
│
├── requirements.txt
├── README.md
└── .gitignore

====================
三、功能要求
====================

### 1. pipeline 模块：从原始视频自动提取 embedding

实现以下流程：

原始视频
→ 抽帧 + 音频提取
→ ASR（音频转字幕）
→ 文本 / 视频 / 音频 / 元数据 embedding
→ 存储为 outputs/features/{video_id}.pt

要求：

#### 1.1 step_extract.py
- 使用 ffmpeg 从视频提取 wav
- 输出单声道、16k 采样率
- 使用 OpenCV 按固定 fps 抽帧
- 抽帧目录输出到 outputs/frames/{video_id}/

#### 1.2 step_text.py
- 使用 faster-whisper 做 ASR
- 使用 transformers 的 BERT 模型提取文本 embedding
- 文本 embedding 维度固定为 768
- 文本输入由：标题 + 标签 + ASR字幕 拼接得到
- 提供 TextEncoder 类，至少包含：
  - audio_to_text(wav_path)
  - text_embedding(text)

#### 1.3 step_video.py
- 使用 OpenAI CLIP 提取视频帧 embedding
- 逐帧编码后做平均池化
- 输出 video embedding 维度固定为 512
- 提供 VideoEncoder 类，至少包含：
  - encode_frames(frame_dir)

#### 1.4 step_audio.py
- 使用 whisper encoder 提取音频 embedding
- 输出维度固定为 384
- 如果原模型输出维度不是 384，请在代码里加入线性映射或池化映射到 384
- 提供 AudioEncoder 类，至少包含：
  - encode(wav_path)

#### 1.5 step_meta.py
- 生成 meta_feat，维度固定为 16
- 不允许包含传播性指标
- 允许包含以下信息：
  - duration
  - category one-hot / category id
  - title_length
  - tag_count
  - verified flag（如果 metadata.csv 提供）
  - publish time 的简单编码（可选）
- 缺失值要有默认处理

#### 1.6 build_sample.py
- 将 text_feat、video_feat、audio_feat、meta_feat 组装成统一 sample dict：
  {
    "text_feat": ...,
    "video_feat": ...,
    "audio_feat": ...,
    "meta_feat": ...
  }

#### 1.7 run_pipeline.py
- 遍历 data/videos/ 下所有视频
- 读取 data/metadata.csv
- 根据 video_id 对齐元数据
- 支持跳过已处理样本
- 将每个视频的特征保存为 .pt 文件
- 输出处理日志

====================
四、训练模块
====================

### 2. training 模块：多分支排序学习

#### 2.1 输入维度固定
- text_dim = 768
- video_dim = 512
- audio_dim = 384
- meta_dim = 16

#### 2.2 model_mvp.py
实现 MultiModalQualityModel，包含：

- ScientificBranch
  - 输入：text_feat + meta_feat
  - 输出：scientific_score

- TechnicalBranch
  - 输入：video_feat + audio_feat + meta_feat
  - 输出：technical_score

- AestheticBranch
  - 输入：video_feat + text_feat + audio_feat
  - 输出：aesthetic_score

- Fusion Head
  - 输入三个分支 hidden + 三个分支 score
  - 输出：
    - overall_score
    - probability = sigmoid(overall_score)

模型 forward 输出 dict：
{
  "scientific_score": ...,
  "technical_score": ...,
  "aesthetic_score": ...,
  "overall_score": ...,
  "probability": ...
}

#### 2.3 dataset_pair.py
实现 PairwiseVideoDataset：

- 从 metadata.csv 和 outputs/features/*.pt 构建样本
- 根据 label 构造正负 pair：
  - label=1 为上榜
  - label=0 为未上榜
- 默认启用“同类别配对”
- 支持 same_category=True/False
- __getitem__ 返回：
  (pos_sample_dict, neg_sample_dict)

#### 2.4 dataloader_pair.py
实现 collate_pair(batch)，把一批 pair 样本堆叠成：
- pos_batch
- neg_batch

#### 2.5 losses.py
实现：
- pairwise_ranking_loss(pos_score, neg_score, margin=1.0)
- 可选 branch_consistency_loss(overall_score, sci, tech, aes)

#### 2.6 metrics.py
实现：
- accuracy
- f1
- auc（如果可行）
- ranking accuracy（pos_score > neg_score 的比例）

#### 2.7 train.py
实现完整训练脚本：
- 加载样本
- 构造 PairwiseVideoDataset + DataLoader
- 初始化 MultiModalQualityModel
- 训练若干 epoch
- 保存 checkpoint 到 outputs/checkpoints/
- 输出日志到 outputs/logs/
- 在终端打印每轮 loss 与简单指标

要求：
- 代码必须可运行
- 加入 device 选择（cuda / cpu）
- 加入随机种子
- 加入基础异常处理

====================
五、推理模块
====================

### 3. inference 模块

实现 infer.py：

输入：
- 单个视频路径
- metadata.csv 中对应元数据（若存在）

流程：
- 调用 pipeline 中的提特征逻辑
- 加载训练好的 checkpoint
- 输出：
{
  "scientific_score": float,
  "technical_score": float,
  "aesthetic_score": float,
  "overall_score": float,
  "probability": float,
  "prediction": "上榜" 或 "未上榜"
}

要求：
- 提供命令行参数
- 支持：
  python inference/infer.py --video path/to/demo.mp4 --checkpoint outputs/checkpoints/best.pt

====================
六、数据文件约定
====================

metadata.csv 至少支持以下列：

- video_id
- label
- category
- title
- tags
- duration

可选列：
- verified
- publish_time

注意：
- 如果 title/tags 缺失，要做健壮处理
- video_id 必须与视频文件名（不含扩展名）一致
- 不允许把点赞、评论、转发、收藏、推荐量等字段接入输入特征

====================
七、工程要求
====================

1. 所有模块必须写清晰注释
2. 代码必须使用 Python + PyTorch
3. 不写伪代码，必须给出完整实现
4. 导入路径要正确，项目应可直接运行
5. 所有必要目录在运行时自动创建
6. requirements.txt 必须完整列出依赖
7. README.md 必须包含：
   - 项目简介
   - 目录结构
   - 环境安装
   - metadata.csv 示例
   - 如何运行 pipeline
   - 如何训练
   - 如何推理
   - 常见报错说明（ffmpeg / CUDA / model download）

====================
八、运行方式
====================

README 中必须写清：

1. 提取特征
   python pipeline/run_pipeline.py

2. 训练
   python training/train.py

3. 推理
   python inference/infer.py --video data/videos/demo.mp4 --checkpoint outputs/checkpoints/best.pt

====================
九、验收标准
====================

请在生成项目后，自行检查并保证：

- 目录结构完整
- 所有 import 正确
- train.py 可以读取 pipeline 生成的 .pt 文件
- infer.py 可以复用 pipeline 的特征提取逻辑
- 各 embedding 维度严格一致：
  - text=768
  - video=512
  - audio=384
  - meta=16
- 项目中没有把传播性指标作为输入
- 所有关键文件都有实际内容，而不是空壳

====================
十、执行方式
====================

请按以下顺序完成：

1. 先创建目录和全部文件
2. 再逐个写入完整代码
3. 再检查 import 与路径
4. 再完善 README 和 requirements.txt
5. 最后给出一个简短总结，说明：
   - 已生成哪些模块
   - 如何开始运行

现在开始直接生成完整项目代码。