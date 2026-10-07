# Semantic-Guided Aesthetic Prompt Experiment

## 1. 实验目的

在 KEEP_P2_S2 中只替换 7 维美学 Prompt 属性空间，比较原始 A1 与领域解耦 A2。

## 2. 当前代码审计结论

主路径使用缓存的 ViT-L/14 逐帧特征和同一模型的文本塔；训练读取 7D PT 缓存，并非在线运行 legacy ViT-B/32。完整审计见 `code_audit.md`。

## 3. A1 / A2 Prompt 设计

| A1 | 潜在问题 | A2 对应处理 |
|---|---|---|
| clarity | 与技术 sharpness/blur 重叠 | 删除 |
| cleanliness | 混入 clutter/noise | 拆除，不直接保留 |
| composition | 合理 | 保留并规范化 |
| appeal | 措辞较泛 | 改为 visual_appeal |
| lighting | 与 color 混合 | 拆为 lighting + color_harmony |
| text_readability | 更接近技术表达 | 删除 |
| professional | 多概念混杂 | 改为 visual_refinement / visual_presentation |

## 4. 控制变量

训练/验证=492/123，seed=42，split_seed=42，validation hash=`79413d21918a956716f230456b80aec037be51bfd1c6a412a8c0a30087a2a58f`。骨干、帧特征、网络、P2-S2 两阶段训练、损失、优化器、scheduler、batch size 和选模规则均保持一致。A1 历史缓存与 checkpoint 未覆盖。

## 5. 特征 sanity check

A2 已写入独立版本缓存；完整方差、共线性、量级和确定性检查见 `feature_sanity.md`。

## 6. 美学分支结果

| Variant | Stage | SRCC_aes | PLCC_aes | MSE_aes | MAE_aes |
|---|---|---:|---:|---:|---:|
| a1_original | stage1 | 0.2405 | 0.2470 | 0.0446 | 0.1729 |
| a1_original | stage2 | 0.2784 | 0.2795 | 0.0389 | 0.1576 |
| a2_domain_semantic | stage1 | 0.2420 | 0.2479 | 0.0445 | 0.1731 |
| a2_domain_semantic | stage2 | 0.3019 | 0.2914 | 0.0402 | 0.1609 |

Stage 1 只训练三个分支，且每个分支由自身标签损失驱动，因此作为 branch-only 诊断；Stage 2 是完整 KEEP_P2_S2 结果。

## 7. 整体模型结果

项目没有名为 AUCF1 的历史指标；为保持可比性，这里报告 ROC-AUC、PR-AUC、F1 与 overall SRCC。

| Variant | ROC-AUC | PR-AUC | F1 | Overall SRCC | S/T/A weight |
|---|---:|---:|---:|---:|---|
| a1_original | 0.8114 | 0.6522 | 0.6230 | 0.4548 | 0.5904/0.2048/0.2048 |
| a2_domain_semantic | 0.8145 | 0.6430 | 0.6410 | 0.4649 | 0.5978/0.2011/0.2011 |

## 8. 维度级相关性

`dimension_correlations.csv` 包含 A1/A2 14 维完整 Pearson 矩阵、每维与 aes_target 的 SRCC/PLCC，以及每维与对应模型 s_tech 的 SRCC。

## 9. A1 vs A2 结果解释

A2-A1: aes SRCC +0.0236，aes PLCC +0.0118，PR-AUC -0.0092，ROC-AUC +0.0030，overall SRCC +0.0101。

## 10. 是否支持实验假设

Primary branch improvement: `True`；overall stability: `True`。

## 11. 是否建议将 A2 纳入主模型

A2 improves both primary aesthetic correlations without a clear overall regression; it is a candidate for multi-seed validation.

## 12. 下一步建议

仅当 seed=42 同时满足美学分支改善与整体稳定时，再运行 seeds 123/2026；否则停止对 A2 调参，避免在固定验证集上搜索 Prompt。

## 13. Multi-seed 稳定性验证与最终决策

固定 seeds 42/123/2026，所有 seed 使用同一 split_seed=42 和同一 validation hash。

| Metric | A1 mean ± std | A2 mean ± std | Delta mean |
|---|---:|---:|---:|
| Aesthetic SRCC | 0.2651 ± 0.0195 | 0.2627 ± 0.0341 | -0.0024 |
| Aesthetic PLCC | 0.2536 ± 0.0225 | 0.2504 ± 0.0394 | -0.0032 |
| Aesthetic MSE | 0.0419 ± 0.0026 | 0.0418 ± 0.0014 | -0.0001 |
| Aesthetic MAE | 0.1635 ± 0.0053 | 0.1632 ± 0.0031 | -0.0003 |
| ROC-AUC | 0.7998 ± 0.0160 | 0.7990 ± 0.0162 | -0.0008 |
| PR-AUC | 0.6174 ± 0.0357 | 0.5970 ± 0.0424 | -0.0204 |
| F1 | 0.6235 ± 0.0151 | 0.6258 ± 0.0225 | +0.0023 |
| Overall SRCC | 0.4299 ± 0.0697 | 0.4097 ± 0.0513 | -0.0202 |

多种子 branch improvement=`False`，overall stability=`False`。

多种子均值未同时支持美学分支改善与整体稳定：主模型继续保留 A1，A2 作为负结果。
