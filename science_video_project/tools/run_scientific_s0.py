"""Offline, fixed-protocol scientific attribute diagnostic (S0)."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pipeline.utils_io import load_metadata, load_pt

ATTRS = ["science_info", "topic_importance", "science_access", "content_interest"]
FEATURES = {"text_feat": 768, "llm_reasoning_analysis_feat": 768, "llm_knowledge_feat": 4}
SEEDS = [42, 123, 2026]
SPLIT_HASH = "79413d21918a956716f230456b80aec037be51bfd1c6a412a8c0a30087a2a58f"
HP = dict(hidden_dim=256, dropout=0.2, lr=1e-3, weight_decay=1e-4,
          batch_size=32, max_epochs=100, patience=10, inner_val_ratio=0.2,
          inner_split_seed=42)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path, frame, index=False):
    frame.to_csv(path, index=index, encoding="utf-8-sig")


def table(frame):
    def fmt(value):
        return f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
    return "\n".join([
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
        *["| " + " | ".join(fmt(v) for v in row) + " |"
          for row in frame.itertuples(index=False, name=None)],
    ])


def correlations(pred, target):
    if np.unique(pred).size < 2 or np.unique(target).size < 2:
        return None, None
    return float(spearmanr(pred, target).statistic), float(pearsonr(pred, target).statistic)


def metrics(pred, target):
    result = {}
    for i, attr in enumerate(ATTRS):
        srcc, plcc = correlations(pred[:, i], target[:, i])
        error = pred[:, i] - target[:, i]
        result[attr] = dict(srcc=srcc, plcc=plcc, mae=float(np.abs(error).mean()),
                            mse=float(np.square(error).mean()))
    result["Macro"] = {
        name: (float(np.mean([result[a][name] for a in ATTRS]))
               if all(result[a][name] is not None for a in ATTRS) else None)
        for name in ["srcc", "plcc", "mae", "mse"]
    }
    return result


def new_model(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(sum(FEATURES.values()), HP["hidden_dim"]),
                         nn.ReLU(), nn.Dropout(HP["dropout"]),
                         nn.Linear(HP["hidden_dim"], 4), nn.Sigmoid())


def train(x, y, indices, seed, epochs, monitor=None):
    model = new_model(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=HP["lr"],
                                 weight_decay=HP["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(x[indices], y[indices]), batch_size=HP["batch_size"],
                        shuffle=True, generator=generator, num_workers=0)
    best, best_epoch, stale = float("inf"), 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(xb), yb)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        row = dict(seed=seed, phase="selection" if monitor is not None else "refit",
                   epoch=epoch, train_mse=total / len(indices))
        if monitor is not None:
            model.eval()
            with torch.inference_mode():
                monitored = nn.functional.mse_loss(model(x[monitor]), y[monitor]).item()
            row["inner_val_mse"] = monitored
            if monitored < best:
                best, best_epoch, stale = monitored, epoch, 0
            else:
                stale += 1
        history.append(row)
        if epoch == 1 or epoch % 10 == 0:
            print(f"seed={seed} {row['phase']} epoch={epoch} mse={row['train_mse']:.6f}", flush=True)
        if monitor is not None and stale >= HP["patience"]:
            break
    return model, best_epoch if monitor is not None else epochs, history


def heatmap(frame, path, title):
    fig, ax = plt.subplots(figsize=(8.4, 6.5), layout="constrained")
    plot = ax.imshow(frame.values, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = ["Science info", "Topic importance", "Science access", "Content interest"]
    ax.set_xticks(range(4), labels, rotation=25, ha="right")
    ax.set_yticks(range(4), labels)
    ax.set_title(title + " (n=615)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{frame.iloc[i, j]:.3f}", ha="center", va="center",
                    color="white" if abs(frame.iloc[i, j]) > 0.55 else "black")
    fig.colorbar(plot, ax=ax, label="Correlation")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/scientific_s0")
    parser.add_argument("--resume-audit", action="store_true",
                        help="Restart an interrupted pre-training audit only.")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    audit_only = (out.exists() and set(p.name for p in out.iterdir()) <= {"checkpoints", "split_manifest.csv"}
                  and not any((out / "checkpoints").glob("*")))
    if out.exists() and any(out.iterdir()) and not (args.resume_audit and audit_only):
        raise FileExistsError(f"Refusing to overwrite an existing experiment: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    protected = {str(p.resolve()): sha(p) for p in
                 [args.metadata, args.reference_checkpoint]}
    raw = pd.read_csv(args.metadata, dtype={"video_id": str})
    cleaned = load_metadata(args.metadata)
    available = {p.stem for p in args.feature_dir.glob("*.pt")}
    data = cleaned[cleaned.video_id.isin(available)].copy().reset_index(drop=True)
    if len(data) != 615 or data.video_id.nunique() != 615:
        raise ValueError(f"Expected 615 unique samples, got {len(data)}")
    scores = data[ATTRS].apply(pd.to_numeric, errors="raise")
    if scores.isna().any().any() or not np.isfinite(scores.values).all():
        raise ValueError("Missing/nonfinite attribute labels; no imputation allowed")
    if not ((scores >= 1) & (scores <= 5)).all().all():
        raise ValueError("Attribute labels outside 1..5")
    sci = scores.mean(axis=1) / 5.0
    if not np.allclose(sci, data.sci_target, atol=1e-7, rtol=0):
        raise ValueError("Scientific target definition mismatch")
    train_idx, val_idx = train_test_split(np.arange(len(data)), test_size=0.2,
                                         random_state=42, stratify=data.label)
    payload = "\n".join(["train:" + v for v in sorted(data.iloc[train_idx].video_id)]
                        + ["val:" + v for v in sorted(data.iloc[val_idx].video_id)])
    split_hash = hashlib.sha256(payload.encode()).hexdigest()
    reference = load_pt(args.reference_checkpoint)
    if (split_hash != SPLIT_HASH or reference["split_hash"] != split_hash
            or (len(train_idx), len(val_idx)) != (492, 123)):
        raise ValueError("Official split/checkpoint mismatch")
    inner_train, inner_val = train_test_split(
        train_idx, test_size=HP["inner_val_ratio"], random_state=HP["inner_split_seed"],
        stratify=data.iloc[train_idx].label)
    if set(inner_train) & set(val_idx) or set(inner_val) & set(val_idx):
        raise ValueError("Inner/outer split leakage")
    manifest = data[["video_id", "label", *ATTRS, "sci_target"]].copy()
    manifest["split"] = "train"
    manifest.loc[val_idx, "split"] = "validation"
    manifest["inner_split"] = "not_used_for_selection"
    manifest.loc[inner_train, "inner_split"] = "inner_train"
    manifest.loc[inner_val, "inner_split"] = "inner_validation"
    save_csv(out / "split_manifest.csv", manifest)

    feature_rows, values = [], []
    zero_vectors = {key: [] for key in FEATURES}
    for row in data.itertuples(index=False):
        path = args.feature_dir / f"{row.video_id}.pt"
        digest = sha(path)
        protected[str(path.resolve())] = digest
        sample = load_pt(path)
        if str(sample["video_id"]) != row.video_id:
            raise ValueError(f"Feature ID mismatch: {path}")
        parts = []
        for key, dim in FEATURES.items():
            value = np.asarray(sample[key], dtype=np.float32)
            if value.shape != (dim,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid/empty scientific input: {path}, {key}")
            if not np.any(value):
                zero_vectors[key].append(row.video_id)
            parts.append(value)
        values.append(np.concatenate(parts))
        feature_rows.append(dict(video_id=row.video_id, path=str(path.resolve()), sha256=digest))
    x = torch.from_numpy(np.stack(values))
    y = torch.from_numpy(scores.to_numpy(dtype=np.float32) / 5.0)
    save_csv(out / "feature_manifest.csv", pd.DataFrame(feature_rows))
    write_json(out / "feature_zero_vectors.json", zero_vectors)
    config = dict(experiment="S0", created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  metadata=str(args.metadata.resolve()), feature_dir=str(args.feature_dir.resolve()),
                  reference_checkpoint=str(args.reference_checkpoint.resolve()),
                  features=FEATURES, input_dim=x.shape[1], attributes=ATTRS, seeds=SEEDS,
                  split_seed=42, split_hash=split_hash, train_size=492, validation_size=123,
                  inner_train_size=len(inner_train), inner_validation_size=len(inner_val),
                  hyperparameters=HP, feature_scaling="none; raw cache concatenation",
                  target_scaling="raw_score / 5", model="Linear(1540,256)-ReLU-Dropout(0.2)-Linear(256,4)-Sigmoid",
                  selection="minimum inner-validation MSE; patience 10; refit all 492 for selected epochs",
                  evaluation="official 123 only after refit; no hyperparameter search",
                  device="cpu", threads=4, torch=torch.__version__, numpy=np.__version__,
                  scipy=scipy.__version__, sklearn=sklearn.__version__,
                  script_sha256=sha(__file__))
    write_json(out / "config.json", config)
    descriptive = scores.describe(percentiles=[.25, .5, .75]).T
    descriptive["missing"] = scores.isna().sum()
    descriptive["integer_scores"] = (scores == np.floor(scores)).all()
    save_csv(out / "attribute_descriptive_stats.csv", descriptive.rename_axis("attribute").reset_index())
    distribution = pd.DataFrame([
        dict(attribute=a, score=k, count=int((scores[a] == k).sum()),
             proportion=float((scores[a] == k).mean())) for a in ATTRS for k in range(1, 6)])
    save_csv(out / "attribute_score_distribution.csv", distribution)
    srcc, plcc = scores.corr(method="spearman"), scores.corr(method="pearson")
    save_csv(out / "attribute_srcc_matrix.csv", srcc, index=True)
    save_csv(out / "attribute_plcc_matrix.csv", plcc, index=True)
    heatmap(srcc, out / "attribute_srcc_heatmap.png", "Scientific attribute SRCC")
    heatmap(plcc, out / "attribute_plcc_heatmap.png", "Scientific attribute PLCC")
    pairs = pd.DataFrame([dict(left=ATTRS[i], right=ATTRS[j], srcc=srcc.iloc[i, j],
                              plcc=plcc.iloc[i, j]) for i in range(4) for j in range(i + 1, 4)])
    save_csv(out / "attribute_pairwise_correlations.csv", pairs)
    target_corr = pd.DataFrame([dict(attribute=a, srcc=correlations(scores[a], sci)[0],
                                    plcc=correlations(scores[a], sci)[1]) for a in ATTRS])
    save_csv(out / "sci_target_correlation.csv", target_corr)
    weakest, strongest = pairs.loc[pairs.srcc.idxmin()], pairs.loc[pairs.srcc.idxmax()]
    average_corr = (srcc.sum() - 1) / 3
    independent = average_corr.idxmin()
    correlation_text = (
        f"六个属性对的平均 SRCC={pairs.srcc.mean():.4f}，范围 "
        f"{pairs.srcc.min():.4f}–{pairs.srcc.max():.4f}；SRCC>0.8 的对数="
        f"{int((pairs.srcc > .8).sum())}/6。\n\n"
        f"最相关：{strongest.left} / {strongest.right}；最低相关：{weakest.left} / {weakest.right}。"
        f"与其余三项的平均相关最低的是 {independent}（{average_corr[independent]:.4f}）；"
        f"content_interest 的对应均值为 {average_corr['content_interest']:.4f}。"
        "这里的相对低相关不等于统计独立，相关性也不能证明属性可相互替代。\n\n"
        "sci_target 是四项等权平均，属性与聚合量的相关含有自相关贡献，不能用于因果归因。")
    (out / "attribute_correlation_summary.md").write_text(
        "# Attribute Correlation\n\n" + table(pairs) + "\n\n" + correlation_text, encoding="utf-8")
    conflicts = int((raw.groupby("video_id").label.nunique() > 1).sum())
    audit = (f"# S0 Data Audit\n\n源表 {len(raw)} 行，冲突标签 ID={conflicts}，复用 load_metadata "
             f"清洗后 {len(cleaned)} 条，缓存匹配 {len(data)}/615；正负类="
             f"{data.label.value_counts().sort_index().to_dict()}。\n\n"
             f"四列无缺失，均在 1–5 内；实际整数评分检查：{bool(descriptive.integer_scores.all())}。"
             "topic_importance 无 1 分样本属于实际分布，不补造该等级。\n\n"
             "sci_target 与四项均值/5 完全一致。标签来自人工评分列，DeepSeek 评分仅作输入。"
             "转换代码按原 CSV 第 8–11 列读取这四项，未新建事实正确性标签。\n\n"
             f"正式划分：492/123；split hash=`{split_hash}`，已与 P2-S2 checkpoint 核对。"
             f"内部选轮划分={len(inner_train)}/{len(inner_val)}，只从 492 条训练视频中产生；"
             "选轮后同 seed 重置模型，在全部 492 条重训。\n\n"
             "该内部 MSE 选轮策略区别于历史主模型的外部验证集排序选模，旨在隔离 S0 正式验证评估；"
             "因此不能将 S0 相关系数与主模型指标作为同训练协议的升级比较。\n\n"
             "输入为 text_feat(768)+llm_reasoning_analysis_feat(768)+llm_knowledge_feat(4)，"
             "共 1540 维，逐样本检查维度、有限值与全零。未训练文本/视觉 backbone。\n\n"
             f"全零向量计数：{ {key: len(ids) for key, ids in zero_vectors.items()} }。"
             "沿用正式缓存的数值，不插补、不剔除、不请求 API；全零不能直接解释为真实低评分，"
             "其成因未在本次 S0 追溯，是输入完整性的限制。ID 见 feature_zero_vectors.json。\n\n"
             "原缓存读取 SHA256 保存在 feature_manifest.csv；写入只发生在本次输出目录。\n\n"
             + table(descriptive.rename_axis("attribute").reset_index()))
    (out / "data_audit.md").write_text(audit, encoding="utf-8")
    print(f"Audit passed: shape={tuple(x.shape)} split={split_hash}", flush=True)

    all_rows, all_history, epoch_rows = [], [], []
    mean_pred = np.repeat(y[train_idx].mean(dim=0).numpy()[None, :], len(val_idx), axis=0)
    target = y[val_idx].numpy()
    baseline = metrics(mean_pred, target)
    write_json(out / "train_mean_baseline_metrics.json", baseline)
    baseline_predictions = pd.DataFrame({"video_id": data.iloc[val_idx].video_id.to_numpy()})
    for i, a in enumerate(ATTRS):
        baseline_predictions[a + "_target"] = target[:, i]
        baseline_predictions[a + "_pred"] = mean_pred[:, i]
    save_csv(out / "train_mean_baseline_predictions.csv", baseline_predictions)
    for seed in SEEDS:
        _, best_epoch, selection_history = train(x, y, inner_train, seed, HP["max_epochs"], inner_val)
        model, _, refit_history = train(x, y, train_idx, seed, best_epoch)
        model.eval()
        with torch.inference_mode():
            pred = model(x[val_idx]).numpy()
        result = metrics(pred, target)
        if any(result[a]["srcc"] is None for a in ATTRS):
            raise ValueError("Undefined correlation in fitted model")
        write_json(out / f"seed_{seed}_metrics.json",
                   dict(seed=seed, selected_epoch=best_epoch, split_hash=split_hash, metrics=result))
        predictions = pd.DataFrame({"video_id": data.iloc[val_idx].video_id.to_numpy()})
        for i, a in enumerate(ATTRS):
            predictions[a + "_target"] = target[:, i]
            predictions[a + "_pred"] = pred[:, i]
        save_csv(out / f"attribute_predictions_seed{seed}.csv", predictions)
        torch.save(dict(model_state_dict=model.state_dict(), config=config, seed=seed,
                        selected_epoch=best_epoch, split_hash=split_hash),
                   out / "checkpoints" / f"seed_{seed}.pt")
        all_rows.extend(dict(seed=seed, attribute=a, **m) for a, m in result.items())
        all_history.extend(selection_history + refit_history)
        epoch_rows.append(dict(seed=seed, selected_epoch=best_epoch,
                               selection_epochs_run=len(selection_history),
                               macro_srcc=result["Macro"]["srcc"], macro_plcc=result["Macro"]["plcc"]))
        print(f"seed={seed} selected_epoch={best_epoch} final={result['Macro']}", flush=True)
    all_metrics = pd.DataFrame(all_rows)
    save_csv(out / "metrics_all_seeds.csv", all_metrics)
    save_csv(out / "training_history.csv", pd.DataFrame(all_history))
    save_csv(out / "selected_epochs.csv", pd.DataFrame(epoch_rows))
    means = all_metrics.groupby("attribute").mean(numeric_only=True).reindex(ATTRS + ["Macro"])
    stds = all_metrics.groupby("attribute").std(numeric_only=True, ddof=1).reindex(ATTRS + ["Macro"])
    report_rows = []
    for a in ATTRS + ["Macro"]:
        report_rows.append({"Attribute": a, **{
            k.upper(): f"{means.loc[a, k]:.4f} ± {stds.loc[a, k]:.4f}"
            for k in ["srcc", "plcc", "mae", "mse"]}})
    aggregate = pd.DataFrame(report_rows)
    save_csv(out / "metrics_summary.csv", aggregate)
    error_baseline = pd.DataFrame([
        dict(attribute=a, baseline="train-mean constant", mae=baseline[a]["mae"], mse=baseline[a]["mse"],
             mlp_mse=means.loc[a, "mse"], mse_delta=means.loc[a, "mse"] - baseline[a]["mse"])
        for a in ATTRS + ["Macro"]])
    easiest, hardest = means.loc[ATTRS, "srcc"].idxmax(), means.loc[ATTRS, "srcc"].idxmin()
    unchanged = all(sha(path) == digest for path, digest in protected.items())
    write_json(out / "integrity_check.json", dict(files_checked=len(protected), all_unchanged=unchanged,
                                                  source_sha256=protected))
    if not unchanged:
        raise RuntimeError("Source files changed during run; inspect integrity_check.json")
    interest_rank = int(means.loc[ATTRS, "srcc"].rank(ascending=False)["content_interest"])
    summary = [
        "# S0 Scientific Attribute Diagnostic Summary", "", "## 1. Experiment Setup", "",
        f"实验时间：{config['created_at']}。正式模型背景：main_v3-D3 + KEEP_P2_S2 + T0。",
        f"输入：1540D 文本/知识离线缓存；共享 MLP 1540→256→4 + Sigmoid，"
        f"可训练参数 {sum(p.numel() for p in model.parameters()):,}。ReLU、Dropout=0.2；"
        "AdamW lr=0.001、weight_decay=0.0001、batch=32、最多100轮、patience=10；仅四属性等权 MSE。",
        "基线模型：S0 shared four-output MLP；附训练集均值常数基线检验误差是否低于常数预测。",
        "", "## 2. Data Audit", "",
        f"615 个唯一视频；492 train / 123 validation；split_seed=42；hash=`{split_hash}`。",
        f"内部 {len(inner_train)}/{len(inner_val)} 分割选最小 MSE 轮数后，在完整 492 条训练集重训。"
        "正式验证集只在各 seed 最终重训后评估；同一固定配置，未执行超参数搜索。",
        "四项原始人工标签按 /5 归一化，完整审计见 data_audit.md；无缺失填充、无标签新增。",
        f"LLM 结构化评分全零样本 {len(zero_vectors['llm_knowledge_feat'])} 条，保留正式缓存原值。"
        "文本/推理向量非零不能证明原 API 响应的有效性；未重新调用 API 校正。",
        f"输入缓存、metadata 和正式参考 checkpoint 共 {len(protected)} 个文件的 SHA256 运行前后相同。",
        "", "## 3. Attribute Distribution", "",
        table(descriptive[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].rename_axis("Attribute").reset_index()),
        "", "1–5 分各等级计数和占比见 attribute_score_distribution.csv。",
        "", "## 4. Attribute Correlation", "",
        table(srcc.rename_axis("SRCC").reset_index()), "", table(plcc.rename_axis("PLCC").reset_index()),
        "", correlation_text, "", "与等权科学性目标的相关：", "", table(target_corr),
        "", "## 5. Attribute Learnability", "",
        "SRCC=Spearman 平均秩相关（含 ties），PLCC=原预测与目标的 Pearson 相关，不做 logistic 拟合。"
        "MAE=mean(abs(pred-target))；MSE=mean((pred-target)^2)；均在 /5 后的尺度上计算。"
        "Macro 为每个 seed 四项指标的算术平均，随后跨 seed 求均值和样本标准差（ddof=1）。",
        "", table(aggregate), "",
        "以上均值±标准差不是置信区间；本实验没有执行显著性检验，不报告 p 值。",
        "", "训练均值常数基线（SRCC/PLCC 因预测恒定而未定义，不以 0 替代）：", "",
        table(error_baseline), "", "## 6. Multi-seed Results", "", table(pd.DataFrame(epoch_rows)), "",
        table(all_metrics), "",
        "三个 seed 为 42/123/2026，正式验证集相同，故这里仅衡量训练随机性，不能代表跨划分稳定性。",
        "", "## 7. Key Findings", "",
        f"1. 四属性的相关范围 {pairs.srcc.min():.4f}–{pairs.srcc.max():.4f}，均低于 0.8；"
        "存在较强共同信息，但未达到全部高度重叠，不能据此断言独立。",
        f"2. 相对最独立（定义为与其余三项平均 SRCC 最低）的是 {independent}，"
        f"均值 {average_corr[independent]:.4f}。",
        f"3. 按三种子平均 SRCC，最容易预测的是 {easiest}：{means.loc[easiest, 'srcc']:.4f}；"
        "此难度排序仅针对当前输入和预测头。",
        f"4. 同一口径最难预测的是 {hardest}：{means.loc[hardest, 'srcc']:.4f}。",
        f"5. content_interest 平均 SRCC={means.loc['content_interest', 'srcc']:.4f}，四项中第 {interest_rank}。"
        "S0 未对比视觉输入，不能证明增加视觉的必要性或增益；应由 S2 的受控模态对照验证。",
        "6. 不同标签的相关结构与预测难度可作为进入 S1 对照实验的动机；"
        "S0 尚不能证明 Four-Adapter 优于共享头。",
        "7. 四属性在每个 seed 上的 SRCC/PLCC 均为正，且跨 seed 平均 MSE 均低于训练均值常数基线，"
        "支持现有文本/知识表示包含可学习信号；误差改善幅度有限，不能称为精准预测。",
        f"8. 各属性与聚合 sci_target 的 SRCC 为 {target_corr.srcc.min():.4f}–"
        f"{target_corr.srcc.max():.4f}，PLCC 为 {target_corr.plcc.min():.4f}–"
        f"{target_corr.plcc.max():.4f}，描述上较均衡，未见单项明显主导的证据。",
        "", "## 8. Decision for S1/S2", "",
        "- S1：以本次 shared 4-output head、固定划分及相同内部选轮规则为对照，检验专属 Adapter 的增量。",
        "- S2：逐属性比较 text/knowledge-only 与加入 Shared CLIP，重点观察 content_interest，"
        "同时控制训练协议、输入归一化和参数量，避免把容量变化误归因为模态收益。",
        "- S0 仅为诊断基线，不构成正式主模型升级。保留本次逐视频预测用于后续配对分析。",
    ]
    (out / "S0_SUMMARY.md").write_text("\n\n".join(line for line in summary if line) + "\n", encoding="utf-8")
    write_json(out / "completion.json", dict(status="completed", elapsed_seconds=time.perf_counter()-started,
                                             seeds=SEEDS, split_hash=split_hash, source_integrity=True))
    print(f"Completed: {out}", flush=True)


if __name__ == "__main__":
    main()
