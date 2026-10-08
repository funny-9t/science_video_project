# 章节设计
第一章 绪论
1.1 研究背景与意义
1.2 国内外研究现状
1.3 论文主要工作
1.4 论文结构安排
第二章 相关理论及技术基础
2.1 视频质量评估相关理论
2.2 多模态预训练模型与特征表示
2.3 排序学习与弱监督学习
2.4 大语言模型与结构化语义分析
2.5 Web系统与接口开发技术
2.6 本章小结
第三章 面向科普短视频的领域化多维质量表征方法
3.1 引言
3.2 多维质量表征框架
3.3 多模态质量特征建模（已完成：Visual/Speech/Combined、T0/T1/T2、Shared CLIP）
3.4 科学性质量建模与多分支融合（已完成：Scientific Branch、IFG/Concat、RQ-VQA-style）
3.5 实验验证（已完成：COVER-Direct/COVER-Linear、main_v2/main_v3、Technical Branch对比、融合结构对比）
3.6 本章小结
第四章 面向弱监督小样本的多层次质量学习方法
4.1 引言
4.2 弱监督质量排序学习（已完成：Loss Ablation、RankNet、Pointwise Calibration、多随机种子损失实验）
4.3 渐进式多分支训练（已完成：Joint Training vs Progressive Training、Progressive多随机种子稳定性实验）
4.4 细粒度属性监督与参数适配（已完成：P0/P1/P2、Attribute Supervision、F0/F1 Partial Fine-tuning、过拟合分析）
4.5 实验验证（已完成：P0/P1/P2多随机种子、分支SRCC、Branch Floor、F1稳定性分析；待补充：Scientific Branch消融、四分支逐项消融、RankNet/Pointwise严格对照、Progressive阶段消融、细粒度标签相关性分析）
4.6 本章小结
第五章 科普短视频多模态质量评估系统设计与实现
5.1 系统需求与总体设计
5.2 系统架构与模块划分
5.3 视频分析与模型推理服务
5.4 基于FastAPI的后端服务设计
5.5 前端可视化与交互设计
5.6 大模型辅助质量解释与诊断
5.7 系统实现与功能验证
5.8 本章小结
第六章 总结与展望
6.1 工作总结
6.2 研究不足
6.3 未来展望

# 本周实验

## 11. COVER Technical main_v3 稳定性验证（2026-09-12）

使用固定验证划分（hash `79413d21918a9567...`）和 seeds `42/123/2026`，严格 A/B 只改变
Technical visual representation。预定义 D1（CLIP 128D + COVER 256D）相对 main_v2：AUC
`-0.0015`、PR-AUC `+0.0258`、Technical SRCC `+0.1600`，且三个 seed 的 Technical SRCC
均提高，满足总体可比且分支对齐稳定改善的升级条件。

触发维度消融后，D3（CLIP 256D + COVER 256D）三 seed 达到：

| 模型 | AUC mean +/- std | PR-AUC mean +/- std | Technical SRCC mean +/- std |
|---|---:|---:|---:|
| main_v2 | 0.7721 +/- 0.0169 | 0.5584 +/- 0.0637 | 0.1109 +/- 0.0211 |
| D1 128+256 | 0.7706 +/- 0.0340 | 0.5841 +/- 0.0625 | 0.2709 +/- 0.0791 |
| **D3 256+256** | **0.7804 +/- 0.0178** | **0.5850 +/- 0.0765** | **0.2900 +/- 0.0313** |

最终决策为 `PROMOTE_MAIN_V3`，正式 profile 采用 D3。其训练参数为 1,516,423，较 main_v2
增加 330,752；冻结 COVER technical backbone（28,078,620 参数）仅用于一次性离线提取，成本
1.848 秒/视频、峰值显存 3.849 GiB、缓存约 4.74 KiB/视频。Technical fusion weight 仍为
`0.200253 +/- 0.000436`，因此分支塌缩不能仅由旧 Technical representation 质量不足解释。

## 12. COVER-inspired Progressive Training（2026-09-14）

### 12.1 受控设置

本轮唯一结构基线为 main_v3-D3，继续使用 CLIP 256D + COVER technical 256D、20% branch
floor、相同的 492/123 划分与验证 hash `79413d21918a9567...`。未修改数据、特征、encoder、
融合结构或细粒度 target。

- J0：从随机初始化直接联合优化 RankNet、pointwise、consistency 与 branch MSE。
- P1：先用 branch MSE 预热，再接入与 J0 等价的总体目标，可选 Stage3 fusion calibration。
- P2：在 P1 上增加权重 0.05 的 branch RankNet；归一化 target gap 固定为 0.05。

Sci/Tech/Aes branch pair 的保留率为 98.77%/98.71%/97.06%，因此没有调整 gap。

### 12.2 Seed=42 阶段实验

| 实验 | AUC | PR-AUC | F1 | Sci SRCC | Tech SRCC | Aes SRCC |
|---|---:|---:|---:|---:|---:|---:|
| J0 | 0.7933 | 0.6315 | 0.6579 | 0.3656 | 0.2538 | 0.2741 |
| P1-S2 | 0.8061 | 0.6216 | 0.5679 | 0.4109 | 0.2856 | 0.2193 |
| P1-S3 | 0.8067 | 0.6263 | 0.5897 | 0.4092 | 0.2866 | 0.2214 |
| P2-S2 | 0.8114 | 0.6522 | 0.6230 | 0.4252 | 0.2542 | 0.2784 |
| P2-S3 | 0.8135 | 0.6526 | 0.6230 | 0.4213 | 0.2500 | 0.2787 |

P2-S3 相对 P2-S2 只增加 0.0020 AUC 和 0.0004 PR-AUC，同时 Technical SRCC 下降
0.0042，因此多种子阶段选择更简单的 P2-S2。

### 12.3 Multi-seed 结果

| 指标 | J0 mean +/- std | P2-S2 mean +/- std | Delta |
|---|---:|---:|---:|
| ROC-AUC | 0.7782 +/- 0.0171 | **0.7998 +/- 0.0160** | +0.0215 |
| PR-AUC | 0.5885 +/- 0.0392 | **0.6174 +/- 0.0357** | +0.0289 |
| Accuracy | 0.7696 +/- 0.0205 | **0.7940 +/- 0.0169** | +0.0244 |
| F1 | 0.6238 +/- 0.0524 | 0.6235 +/- 0.0151 | -0.0002 |
| Scientific SRCC | 0.3373 +/- 0.0246 | 0.3852 +/- 0.0914 | +0.0479 |
| Technical SRCC | 0.2495 +/- 0.0334 | 0.2475 +/- 0.0366 | -0.0021 |
| Aesthetic SRCC | 0.2873 +/- 0.0120 | 0.2651 +/- 0.0195 | -0.0222 |

P2-S2 在三个 seed 上都提高 AUC，整体 F1 持平；但 Technical/Aesthetic SRCC 未稳定改善。
Stage1→Stage2 的三种子平均漂移为 Sci `+0.0177`、Tech `-0.0624`、Aes `+0.0348`。
seed=42 的 overall RankNet 梯度均值也表现为 Scientific 明显大于 Technical/Aesthetic，说明
梯度不平衡仍存在，但该诊断不作为因果证明。

最终决策：`USE_PROGRESSIVE_BRANCH_RANK`，采用 P2 Stage1→Stage2，删除低收益 Stage3。
这一结果应解释为 branch-specific ranking regularization 提升总体排序泛化，不应写成已解决
Technical branch collapse。完整报告与原始产物位于 `outputs/progressive_training/`。

## 13. Fine-grained Attribute Supervision 与连续质量建模（2026-09-14）

### 13.1 数据与受控设置

本轮固定 main_v3-D3 encoder、融合结构、20% branch floor、已有特征缓存、492/123 划分和
validation hash `79413d21918a9567...`，只改变监督粒度。615 个清洗样本的 7 项评分均无缺失；
`video_aesthetics=0` 的 12 条样本按有效标注保留。全部 target 使用 `raw_score / 5` 归一化，
Overall Human Quality Score（Pseudo-MOS）为 7 项归一化评分的均值。

- Scientific representation：预测 science_info、topic_importance、science_access、content_interest。
- Technical representation：预测 visual_quality、audio_quality。
- Aesthetic representation：预测 video_aesthetics。
- Fusion representation：可选 quality head；属性与质量输出均不进入 ranking/fusion 输入。

### 13.2 Loss 审计与核心矩阵

F0-F3 说明 consistency 是主要的 branch alignment 来源；加入 branch MSE 后，F3 相对 F1 的
AUC 与 branch macro SRCC 分别变化 `+0.0010` 和 `+0.0118`。

| 模型（seed 42） | AUC | PR-AUC | F1 | Sci SRCC | Tech SRCC | Aes SRCC | Attr mean | Quality SRCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 Ranking only | 0.7798 | 0.6651 | 0.6118 | 0.3306 | -0.2214 | 0.1965 | - | - |
| A1 Branch + consistency | 0.7923 | 0.6444 | 0.6750 | 0.3563 | 0.2553 | 0.2795 | - | - |
| **A2 Attributes** | **0.8118** | **0.6624** | **0.6842** | **0.3786** | 0.2412 | 0.2673 | **0.3328** | - |
| A3 Branch + attributes | 0.7896 | 0.6517 | 0.6420 | 0.3403 | 0.2433 | 0.2810 | 0.3185 | - |
| A4 Branch + attributes + quality | 0.7976 | 0.6360 | 0.6389 | 0.3274 | 0.2515 | 0.2861 | 0.3117 | 0.3997 |

A2 的逐属性 SRCC 为：science_info `0.3537`、topic_importance `0.3764`、science_access
`0.4063`、content_interest `0.4667`、visual_quality `0.1791`、audio_quality `0.2811`、
video_aesthetics `0.2666`。A4 quality head 的 SRCC/PLCC/RMSE/MAE 分别为
`0.3997/0.4964/0.1683/0.1240`；quality AUC 为 `0.8088`。A4 相对 A3 的 AUC 提高
`0.0081`，说明连续质量目标可学，但不足以超过更简单的 A2。

### 13.3 Multi-seed 与结论

| 模型 | AUC mean +/- std | PR-AUC mean +/- std | Branch macro SRCC | Attribute mean SRCC |
|---|---:|---:|---:|---:|
| A1 | 0.7791 +/- 0.0183 | 0.5872 +/- 0.0494 | 0.2935 +/- 0.0052 | - |
| **A2** | **0.7847 +/- 0.0215** | **0.6101 +/- 0.0370** | 0.2920 +/- 0.0049 | 0.3165 +/- 0.0237 |

A2 相对同协议 A1 的平均 AUC/PR-AUC 提高 `0.0056/0.0229`，新增参数仅 903 个。因此本轮
决策为 `PROMOTE_ATTRIBUTE_SUPERVISION`。不过 A2 仍低于上一轮 P2-S2 的 AUC
`0.7998 +/- 0.0160`，不直接覆盖当前最佳 checkpoint；后续应将 7-attribute loss 接入 P2
Stage2，并与原 P2-S2 做唯一变量对照。完整报告、CSV、审计 JSON 和 checkpoints 位于
`outputs/fine_grained_supervision/`。

## 14. Progressive Training × Fine-grained Attribute Supervision（2026-09-14）

### 14.1 实际 P2-S2 配置

本轮固定 main_v3-D3、615 个样本、492/123 划分、validation hash
`79413d21918a9567...`、20% branch floor 和已有 feature cache。P2-S2 的实际实现为：

- Stage1：只训练三个 branch 及 score heads；`branch MSE + 0.05 branch RankNet`；最多 15 epoch，patience 5；按 branch macro SRCC 选模。
- Stage2：全模型解冻并继承 Stage1 最佳 checkpoint；`RankNet + 0.1 pointwise + 0.1 consistency + 0.05 branch MSE + 0.05 branch RankNet`；最多 30 epoch，patience 10；按 ranking accuracy 选模。
- 两阶段均使用 AdamW、learning rate `5e-5`、weight decay `1e-3`。P1/P2 的 attribute MSE 权重为 `0.02`，P2 quality MSE 仅在 Stage2 启用，权重为 `0.05`。

### 14.2 Seed=42 门控实验

| 模型 | AUC | PR-AUC | Accuracy | F1 | Branch macro | Attr mean | Quality SRCC | q_rank-quality |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | 0.8114 | **0.6522** | **0.8130** | 0.6230 | 0.3192 | - | - | **0.4548** |
| P1 | 0.8178 | 0.6481 | 0.8049 | 0.6250 | 0.3149 | **0.3420** | - | 0.4501 |
| **P2** | **0.8185** | 0.6495 | 0.7805 | **0.6400** | 0.3197 | 0.3322 | 0.4363 | 0.4470 |
| P1-Simple | 0.8027 | 0.6013 | 0.7642 | 0.6329 | **0.3481** | 0.3426 | - | 0.4536 |

P1 相对 P0 的 AUC/PR-AUC/F1/branch macro 变化为 `+0.0064/-0.0042/+0.0020/-0.0043`，
满足简化实验门控。P1-Simple 删除 branch MSE 后，AUC/PR-AUC 比 P1 下降
`0.0152/0.0468`，因此结论为 `KEEP_BRANCH_SUPERVISION`。P2 相对 P1 的 AUC/PR-AUC 仅提高
`0.0007/0.0015`，但具备 SRCC/PLCC/RMSE/MAE 为 `0.4363/0.5149/0.1555/0.1123`
的连续质量输出，因此作为 seed=42 候选进入多种子验证。

### 14.3 Multi-seed 结果

| 指标 | P0 mean +/- std | P1 mean +/- std | P1-P0 | P2 mean +/- std |
|---|---:|---:|---:|---:|
| ROC-AUC | 0.7998 +/- 0.0160 | **0.8000 +/- 0.0186** | +0.0002 | 0.7923 +/- 0.0268 |
| PR-AUC | 0.6174 +/- 0.0357 | **0.6216 +/- 0.0314** | +0.0042 | 0.5770 +/- 0.0643 |
| F1 | **0.6235 +/- 0.0151** | 0.6082 +/- 0.0199 | -0.0153 | 0.6057 +/- 0.0503 |
| Branch macro SRCC | **0.2993 +/- 0.0275** | 0.2874 +/- 0.0240 | -0.0118 | 0.2941 +/- 0.0225 |
| q_rank-quality SRCC | **0.4299 +/- 0.0697** | 0.3933 +/- 0.0586 | -0.0366 | 0.3989 +/- 0.0484 |

P1 的 attribute mean SRCC 为 `0.3138 +/- 0.0246`。其七项属性 SRCC 分别为：
science_info `0.3247 +/- 0.0814`、topic_importance `0.3824 +/- 0.0491`、
science_access `0.3613 +/- 0.0458`、content_interest `0.3619 +/- 0.0553`、
visual_quality `0.1884 +/- 0.0386`、audio_quality `0.3225 +/- 0.0372`、
video_aesthetics `0.2556 +/- 0.0048`。

P2 的 attribute mean SRCC 为 `0.3114 +/- 0.0245`；quality SRCC/PLCC/RMSE/MAE 为
`0.3993 +/- 0.0374`、`0.4482 +/- 0.0694`、`0.1694 +/- 0.0129`、
`0.1269 +/- 0.0199`。辅助 loss 没有数值主导：seed=42 Stage2 平均加权 attribute/quality
贡献均约 `0.001`，平均 RankNet loss 约 `0.163`。但 seed=123 的融合权重变为约
`0.301/0.371/0.328`，导致跨 seed 排名方差与 PR-AUC 明显恶化。

### 14.4 最终结论

最终决策：`KEEP_P2_S2`。

P1 的平均 AUC 不低于 P0、平均 PR-AUC 高于 P0，且 attribute mean SRCC 为正，满足原实验的机械
晋级门槛；但 AUC/PR-AUC 增益仅为 `0.0002/0.0042`，小于跨 seed 波动，同时 F1、branch macro
和 q_rank-quality 分别下降 `0.0153/0.0118/0.0366`。因此不再追加 P1 实验，也不替换主模型。
P1 作为属性可解释性消融保留，P1-Simple 明显退化，故继续保留 branch supervision；continuous
quality head 仅作为诊断模块。完整 CSV、stage diagnostics、epoch loss、报告与 checkpoints 位于
`outputs/progressive_attribute/`。

## 15. COVER Technical Swin Partial Fine-Tuning（2026-09-14）

### 15.1 受控设置

F0 复用正式 P2-S2 的 frozen COVER cache；F1 的 Stage1 原样复用各 seed 的 P0 checkpoint，
Stage2 才将 raw video 经相同的 7x7 fragments、32x32 fragment、40 frames、interval 2
deterministic view 输入 Swin。只解冻实际层级中的 `layers.3.*` 和 final `norm`，其余 Swin、
CLIP、RoBERTa 与全部模型设计保持不变。固定 615 个清洗样本、492/123 划分和 validation hash
`79413d21918a956716f230456b80aec037be51bfd1c6a412a8c0a30087a2a58f`。

实际层级为 `patch_embed -> pos_drop -> layers.0 -> layers.1 -> layers.2 -> layers.3 -> norm`。
Swin 共 28,078,620 参数，`layers.3 + norm` 含 14,298,960 参数，占 50.92%。这是该实现中
“最后一个 stage + final norm”的真实规模，明显高于任务文档中约 20%-30% 的估计。

### 15.2 Seed 42 与门控

| 模型 | AUC | PR-AUC | Accuracy | F1 | Sci SRCC | Tech SRCC | Visual SRCC | Audio SRCC | Tech-label SRCC | Aes SRCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F0 | 0.8114 | 0.6522 | 0.8130 | 0.6230 | 0.4252 | 0.2542 | 0.1718 | 0.2937 | 0.2734 | 0.2784 |
| F1 | 0.8088 | 0.6507 | 0.7236 | 0.6222 | 0.4060 | 0.2839 | 0.2003 | 0.3319 | 0.3003 | 0.2586 |

F1 的 Technical/Visual SRCC 提高 `+0.0297/+0.0286`，AUC/PR-AUC/F1 仅变化
`-0.0027/-0.0016/-0.0007`，达到预设 `F1_SUCCESS` 并触发多种子。Technical fusion weight
从 0.2048 增至 0.2138，仍接近 20% floor；表征改善没有改变总体标签对 Scientific branch
的偏好。seed 42 最佳 checkpoint 的 train-val AUC/Technical gap 已为 `+0.1882/+0.1572`。

### 15.3 Multi-seed 稳定性

| 指标 | F0 mean +/- std | F1 mean +/- std | Delta |
|---|---:|---:|---:|
| ROC-AUC | 0.7998 +/- 0.0160 | **0.8001 +/- 0.0084** | +0.0003 |
| PR-AUC | **0.6174 +/- 0.0357** | 0.6071 +/- 0.0499 | -0.0103 |
| F1 | **0.6235 +/- 0.0151** | 0.5981 +/- 0.0326 | -0.0254 |
| Technical SRCC | **0.2475 +/- 0.0366** | 0.2461 +/- 0.0369 | -0.0014 |
| Visual SRCC | 0.1649 +/- 0.0389 | **0.1732 +/- 0.0248** | +0.0083 |
| Tech-label SRCC | 0.2694 +/- 0.0263 | **0.2708 +/- 0.0269** | +0.0014 |

seed 123 虽提高 Technical SRCC，但 AUC/PR-AUC 分别下降 `0.0145/0.0664`；seed 2026 的
Technical SRCC 下降约 0.0701。三个最佳 checkpoint 的平均 train-val AUC/Technical gap 为
`+0.1861/+0.2183`。三个 seed 的最后训练 AUC 都达到 1.0，而最后验证 Technical SRCC 分别仅为
0.2348、0.1994、0.1820，符合“训练持续改善、验证技术对齐不升或下降”的过拟合判据。

### 15.4 参数、效率与结论

F0/F1 可训练参数为 1,516,423/15,815,383，增加 14,298,960。task/Swin optimizer groups
使用 `5e-5/5e-6`，weight decay `1e-3`；F1 为 batch 1 x accumulation 8、AMP、clip norm 1.0。
seed 42 的 cached F0/F1 Stage2 平均耗时为 2.5/31.9 秒每 epoch，即 F1 慢 12.8x；F1 还需约
1014.8 秒预热 run-local frozen trunk cache。F0 cached 单步训练峰值显存基准为 0.046 GiB，F1
完整运行峰值为 5.353 GiB。

最终决策：`F1_OVERFIT_RETRY_WITH_LOWER_LR`。当前 F1 不替换 F0，也不进入 F2。若后续重试，
优先将 Swin LR 降至 `1e-6`，或仅解冻 `layers.3` 最后的 block 与 norm，以减少当前一次解冻
50.92% Swin 参数造成的容量跃升。原始结果、训练曲线、日志和 checkpoints 位于
`outputs/swin_partial_ft/`。

## 16. F1.1 Lightweight Swin Block-level Fine-Tuning（2026-09-15）

### 16.1 工作量与受控实现

本轮只新增 F1.1 seed 42 的在线 Stage2：F0、F1 和 P0 Stage1 checkpoint 直接复用，无 Stage1
重训和无 multi-seed 自动扩展。工作量主要是将现有在线 Swin 的 frozen trunk cache 边界后移到
`layers.3.blocks.0` 输出，以及完成一次全量视频预热与最多 30 epoch 的 Stage2。实际固定 615 个
清洗样本、492/123 划分、`split_seed=42`、同一 validation hash、main_v3-D3 结构、P0 loss、
branch floor 和 ranking-based checkpoint selection。

Swin 最后 stage 有 `blocks.0/blocks.1` 两个 block，且没有 downsample。F1.1 仅解冻
`layers.3.blocks.1.*` 与 final `norm.*`；冻结 patch embedding、`layers.0–2` 和
`layers.3.blocks.0`。实际 Swin 参数 28,078,620，可训练 7,150,248，占 25.47%；总模型可训练
8,666,671。Task/Swin optimizer groups 为 `5e-5/2e-6`，weight decay `1e-3`，batch 1 x
accumulation 8、AMP、gradient clipping 1.0。相对原 F1，本轮同时缩小了解冻范围并降低 Swin
LR，不能把两种变化的作用完全分开归因。

缓存是 `blocks.0` 之后、`blocks.1` 之前的 channels-last activation，每视频
`(1,20,7,7,768)`、CPU FP16、run-local；attention mask 复用 COVER 的
`get_window_size/compute_mask`。同一 deterministic view 下，“完整 Swin”与“缓存前缀 + 最后
block”前向最大绝对误差为 `0.000000`，首个训练 batch 也确认 `blocks.1/norm` 有梯度而冻结的
`blocks.0` 没有梯度。旧 768D COVER feature cache 未被覆盖。

### 16.2 Seed 42 结果

| 模型 | Swin FT 参数 | Swin FT % | AUC | PR-AUC | F1 | Tech SRCC | Visual SRCC | Tech-label SRCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F0 frozen | 0 | 0% | 0.8114 | **0.6522** | 0.6230 | 0.2542 | 0.1718 | 0.2734 |
| F1 last stage + norm | 14,298,960 | 50.92% | 0.8088 | 0.6507 | 0.6222 | 0.2839 | 0.2003 | **0.3003** |
| F1.1 last block + norm | 7,150,248 | 25.47% | **0.8175** | 0.6413 | **0.6292** | **0.2905** | **0.2106** | 0.2992 |

F1.1 在第 5 epoch 按原 ranking selection 入选。相对 F0，AUC/PR-AUC/F1 为
`+0.0061/-0.0110/+0.0063`，Technical/Visual/Tech-label SRCC 为
`+0.0364/+0.0388/+0.0258`。Scientific/Aesthetic SRCC 为 `0.4263/0.2554`，平均
S/T/A 融合权重 `0.5733/0.2128/0.2140`；Technical 权重仍接近 20% floor。

### 16.3 过拟合、效率与决策

F1.1 最佳 checkpoint 的完整训练/验证 AUC 为 `0.9963/0.8175`，Technical SRCC 为
`0.4393/0.2905`，对应 gap `0.1788/0.1488`。相对 F1 的 `0.1882/0.1572`，仅缩小
`0.0094/0.0085`，约 5%，未达到“明显减弱”的研究目标。曲线在第 7 epoch 之后多次出现训练
AUC 1.0，同时验证 Technical SRCC 从入选 epoch 的 0.2905 下降至最后 epoch 的 0.2531；
第 15 epoch 触发原 patience 10 早停。因此单 seed 的 Technical 提升不等同于稳定泛化改善。

run-local cache warm-up 为 1064.7 秒，Stage2 训练 454.6 秒、平均 30.3 秒/epoch；F1 平均
31.9 秒/epoch，缩小参数带来的速度收益约 5%，峰值显存从 5.353 降至 5.244 GiB。首次解码
依然占多数成本，参数减半未使全量实验显著更快。

最终决策：`KEEP_F0`，`RECOMMEND_MULTI_SEED=NO`，本轮不做 F2。F1.1 的单 seed
AUC/Technical/Visual 增益保留为有价值的消融观察，但 PR-AUC 下降且过拟合未明显缓解，按预设
晋级标准不能替换正式 P0/F0，也不自动扩展 seed 123/2026。原始 CSV、训练曲线、日志、报告和
checkpoint 位于 `outputs/swin_partial_ft_f1_1/`。

## 17. KEEP_P2_S2 + Temporal Quality Statistics（2026-09-20）

### 17.1 Feature audit 与受控设置

现有 COVER technical cache 为 `[768]`，来自 Swin 输出 `[B,768,T,H,W]` 的时空平均，未保留时间轴。
本轮复用相同 `COVER.pth`、technical sampler、预处理及确定性 fragment offset，仅额外缓存空间平均后的
`[20,768]`。615/615 成功，0 条短序列、0 失败；序列均值与旧 pooled 特征最大误差 `4.77e-7`。

T0 继续使用 pooled 768D；T1 使用 mean+std 1536D；T2 再加入相邻绝对差的 mean+std，得到 3072D。
三者均投影到 COVER 256D，与原 CLIP 256D 路径融合，后续 Technical Branch、三分支 fusion、20%
branch floor、P2-S2 两阶段训练和全部 loss 保持不变。T1/T2 仅 temporal projection 重新初始化，其余
同形参数与相同 seed 的 baseline 初始化严格一致。

### 17.2 Seed=42 screening

| Variant | AUC | PR-AUC | F1 | Overall SRCC | Tech SRCC | Tech PLCC | Tech MSE | Branch macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 | 0.8114 | 0.6522 | 0.6230 | 0.4548 | 0.2542 | 0.2090 | 0.0438 | 0.3192 |
| T1 | **0.8148** | **0.6520** | **0.6349** | 0.4529 | 0.2616 | 0.2480 | **0.0405** | 0.3191 |
| T2 | 0.8077 | 0.6235 | 0.6216 | **0.4649** | **0.3602** | **0.3727** | 0.0499 | **0.3450** |

T1 的 Tech SRCC 仅提高 `0.0074`，没有形成明确技术分支增益。T2 的 Tech SRCC/PLCC 提高
`0.1060/0.1637`，AUC 仅下降 `0.0037`，因此 T2 晋级多种子；T1 停止。

### 17.3 Multi-seed stability

| Metric | T0 mean +/- std | T2 mean +/- std | Delta |
|---|---:|---:|---:|
| ROC-AUC | **0.7998 +/- 0.0160** | 0.7924 +/- 0.0147 | -0.0074 |
| PR-AUC | **0.6174 +/- 0.0357** | 0.5946 +/- 0.0258 | -0.0228 |
| F1 | 0.6235 +/- 0.0151 | **0.6292 +/- 0.0251** | +0.0057 |
| Overall SRCC | **0.4299 +/- 0.0697** | 0.4187 +/- 0.0604 | -0.0112 |
| Tech SRCC | 0.2475 +/- 0.0366 | **0.2717 +/- 0.0911** | +0.0242 |
| Tech PLCC | 0.2362 +/- 0.0636 | **0.2894 +/- 0.0994** | +0.0532 |
| Tech MSE | **0.0486 +/- 0.0070** | 0.0657 +/- 0.0327 | +0.0171 |
| Branch macro | 0.2993 +/- 0.0275 | **0.3044 +/- 0.0392** | +0.0051 |

T2 的逐 seed Tech SRCC 变化为 `+0.1060/-0.0297/-0.0036`，说明均值提升由 seed 42 单独驱动，
不满足稳定性优先标准。其 Tech MSE、AUC、PR-AUC 和 Overall SRCC 也同时恶化。

### 17.4 Feature diagnostics 与结论

验证集 temporal std、diff mean、diff std 的均值分别为 `0.06461/0.05875/0.04690`，无 NaN，
与 tech_target 的 SRCC 为 `0.3361/0.3452/0.3215`。因此时间波动统计本身确实携带技术质量信息，
但投影后的模型收益没有跨 seed 稳定传递。

最终决策：`KEEP_T0`。原 `KEEP_P2_S2` 主模型不变；T1/T2 和 temporal cache 作为真实负结果及
诊断资产保留，不继续扩展 GRU、Transformer、光流或新的时序 backbone。
