"""Fixed-protocol S1 shared head versus parameter-matched attribute adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
from tools.run_scientific_s0 import ATTRS, FEATURES, HP, SEEDS, SPLIT_HASH, metrics, sha, table, write_json

BASELINE_HIDDEN = 256
ADAPTER_SHARED_DIM = 224
ADAPTER_DIM = 32
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 7001


class SharedHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(sum(FEATURES.values()), BASELINE_HIDDEN)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(HP["dropout"])
        self.head = nn.Linear(BASELINE_HIDDEN, len(ATTRS))

    def forward(self, value):
        return torch.sigmoid(self.head(self.dropout(self.activation(self.projection(value)))))


class FourAttributeAdapters(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(sum(FEATURES.values()), ADAPTER_SHARED_DIM)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(HP["dropout"])
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ADAPTER_SHARED_DIM, ADAPTER_DIM),
                nn.ReLU(),
                nn.Linear(ADAPTER_DIM, 1),
            )
            for _ in ATTRS
        ])

    def forward(self, value):
        hidden = self.dropout(self.activation(self.projection(value)))
        return torch.sigmoid(torch.cat([adapter(hidden) for adapter in self.adapters], dim=-1))


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def make_model(kind, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if kind == "shared":
        return SharedHead()
    if kind == "adapter":
        return FourAttributeAdapters()
    raise ValueError(f"Unknown model: {kind}")


def train(model_kind, x, y, indices, seed, epochs, monitor=None):
    model = make_model(model_kind, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(x[indices], y[indices]), batch_size=HP["batch_size"],
                        shuffle=True, generator=generator, num_workers=0)
    best_value, best_epoch, stale = float("inf"), 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(xb), yb)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        item = dict(model=model_kind, seed=seed,
                    phase="selection" if monitor is not None else "refit",
                    epoch=epoch, train_mse=total / len(indices))
        if monitor is not None:
            model.eval()
            with torch.inference_mode():
                inner_mse = nn.functional.mse_loss(model(x[monitor]), y[monitor]).item()
            item["inner_val_mse"] = inner_mse
            if inner_mse < best_value:
                best_value, best_epoch, stale = inner_mse, epoch, 0
            else:
                stale += 1
        history.append(item)
        if epoch == 1 or epoch % 10 == 0:
            print(f"{model_kind} seed={seed} {item['phase']} epoch={epoch} mse={item['train_mse']:.6f}", flush=True)
        if monitor is not None and stale >= HP["patience"]:
            break
    return model, (best_epoch if monitor is not None else epochs), history


def aggregate_metrics(prediction, target):
    average_pred = prediction.mean(axis=1)
    average_target = target.mean(axis=1)
    return dict(
        srcc=float(spearmanr(average_pred, average_target).statistic),
        plcc=float(pearsonr(average_pred, average_target).statistic),
        mae=float(np.abs(average_pred - average_target).mean()),
        mse=float(np.square(average_pred - average_target).mean()),
    )


def bootstrap_delta(shared_by_seed, adapter_by_seed, target):
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    sample_count = len(target)
    macro_deltas, aggregate_deltas = [], []
    for _ in range(BOOTSTRAP_SAMPLES):
        indices = generator.integers(0, sample_count, size=sample_count)
        per_seed_macro, per_seed_aggregate = [], []
        for seed in SEEDS:
            shared = metrics(shared_by_seed[seed][indices], target[indices])["Macro"]["srcc"]
            adapter = metrics(adapter_by_seed[seed][indices], target[indices])["Macro"]["srcc"]
            shared_agg = aggregate_metrics(shared_by_seed[seed][indices], target[indices])["srcc"]
            adapter_agg = aggregate_metrics(adapter_by_seed[seed][indices], target[indices])["srcc"]
            per_seed_macro.append(adapter - shared)
            per_seed_aggregate.append(adapter_agg - shared_agg)
        macro_deltas.append(float(np.mean(per_seed_macro)))
        aggregate_deltas.append(float(np.mean(per_seed_aggregate)))
    def interval(values):
        return dict(mean=float(np.mean(values)), ci95_percentile=[float(np.quantile(values, .025)),
                  float(np.quantile(values, .975))], positive_fraction=float(np.mean(np.asarray(values) > 0)))
    return dict(method="paired nonparametric bootstrap over 123 validation videos; statistic is mean delta across three fixed training seeds",
                repetitions=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED,
                adapter_minus_shared=dict(macro_srcc=interval(macro_deltas),
                                           aggregate_srcc=interval(aggregate_deltas)))


def format_mean_std(frame, group, label):
    subset = frame[frame.model == group]
    values = subset.groupby("attribute")[["srcc", "plcc", "mae", "mse"]].agg(["mean", "std"]).reindex(ATTRS + ["Macro"])
    rows = []
    for attr in ATTRS + ["Macro"]:
        rows.append({"Model": label, "Attribute": attr, **{
            metric.upper(): f"{values.loc[attr, (metric, 'mean')]:.4f} ± {values.loc[attr, (metric, 'std')]:.4f}"
            for metric in ("srcc", "plcc", "mae", "mse")
        }})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s0-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/scientific_s1_adapter")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    resumable_startup = (
        out.exists()
        and {item.name for item in out.iterdir()} <= {"checkpoints", "parameter_comparison.csv"}
        and not any((out / "checkpoints").glob("*.pt"))
    )
    if out.exists() and any(out.iterdir()) and not resumable_startup:
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    s0 = json.loads((args.s0_dir / "config.json").read_text(encoding="utf-8"))
    if s0["split_hash"] != SPLIT_HASH or s0["features"] != FEATURES or s0["hyperparameters"] != HP:
        raise ValueError("S0 configuration mismatch")
    metadata_path, feature_dir, reference_checkpoint = map(Path, [s0["metadata"], s0["feature_dir"], s0["reference_checkpoint"]])
    protected = {str(path.resolve()): sha(path) for path in [metadata_path, reference_checkpoint]}
    s0_manifest = pd.read_csv(args.s0_dir / "split_manifest.csv", dtype={"video_id": str})
    if len(s0_manifest) != 615 or s0_manifest.video_id.nunique() != 615:
        raise ValueError("Unexpected S0 manifest")
    data = load_metadata(metadata_path)
    data = data[data.video_id.isin(set(s0_manifest.video_id))].copy().reset_index(drop=True)
    # Recompute split arrays to retain the order returned by S0's train_test_split.
    # The manifest preserves membership but not that order, which affects shuffled batches.
    train_idx, val_idx = train_test_split(
        np.arange(len(data)), test_size=0.2, random_state=42, stratify=data.label
    )
    inner_train, inner_val = train_test_split(
        train_idx, test_size=HP["inner_val_ratio"], random_state=HP["inner_split_seed"],
        stratify=data.iloc[train_idx].label,
    )
    if (
        set(data.iloc[train_idx].video_id) != set(s0_manifest.loc[s0_manifest.split == "train", "video_id"])
        or set(data.iloc[val_idx].video_id) != set(s0_manifest.loc[s0_manifest.split == "validation", "video_id"])
        or set(data.iloc[inner_train].video_id) != set(s0_manifest.loc[s0_manifest.inner_split == "inner_train", "video_id"])
        or set(data.iloc[inner_val].video_id) != set(s0_manifest.loc[s0_manifest.inner_split == "inner_validation", "video_id"])
    ):
        raise ValueError("S0 split membership mismatch")
    if len(train_idx) != 492 or len(val_idx) != 123 or len(inner_train) != 393 or len(inner_val) != 99:
        raise ValueError("S0 split size mismatch")
    if set(train_idx) & set(val_idx) or set(inner_train) & set(val_idx) or set(inner_val) & set(val_idx):
        raise ValueError("Validation leakage")
    values = []
    for video_id in data.video_id:
        feature_path = feature_dir / f"{video_id}.pt"
        protected[str(feature_path.resolve())] = sha(feature_path)
        sample = load_pt(feature_path)
        if str(sample["video_id"]) != video_id:
            raise ValueError(f"Feature ID mismatch for {video_id}")
        values.append(np.concatenate([np.asarray(sample[name], dtype=np.float32) for name in FEATURES]))
    x = torch.from_numpy(np.stack(values))
    y = torch.from_numpy(data[ATTRS].to_numpy(dtype=np.float32) / 5.0)
    target = y[val_idx].numpy()
    shared_params = parameter_count(SharedHead())
    adapter_params = parameter_count(FourAttributeAdapters())
    parameter_delta = (adapter_params - shared_params) / shared_params
    if shared_params != 395524 or abs(parameter_delta) > .10:
        raise ValueError(f"Parameter control failed: shared={shared_params}, adapter={adapter_params}, delta={parameter_delta}")
    parameter_frame = pd.DataFrame([
        dict(model="S1-A Shared Head", shared_dim=BASELINE_HIDDEN, adapter_dim=np.nan,
             params=shared_params, relative_params=1.0, parameter_delta=0.0),
        dict(model="S1-B Four Adapters", shared_dim=ADAPTER_SHARED_DIM, adapter_dim=ADAPTER_DIM,
             params=adapter_params, relative_params=adapter_params/shared_params, parameter_delta=parameter_delta),
    ])
    parameter_frame.to_csv(out / "parameter_comparison.csv", index=False, encoding="utf-8-sig")
    config = dict(experiment="S1", created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  s0_dir=str(args.s0_dir.resolve()), metadata=str(metadata_path.resolve()), feature_dir=str(feature_dir.resolve()),
                  reference_checkpoint=str(reference_checkpoint.resolve()), split_hash=SPLIT_HASH, seeds=SEEDS,
                  features=FEATURES, input_dim=int(x.shape[1]), attributes=ATTRS, hyperparameters=HP,
                  models=dict(shared="1540->256->4 sigmoid", adapter="1540->224 shared; four 224->32->1 MLP adapters + sigmoid"),
                  parameter_control=[
                      dict(model="S1-A Shared Head", shared_dim=BASELINE_HIDDEN, adapter_dim=None,
                           params=shared_params, relative_params=1.0, parameter_delta=0.0),
                      dict(model="S1-B Four Adapters", shared_dim=ADAPTER_SHARED_DIM, adapter_dim=ADAPTER_DIM,
                           params=adapter_params, relative_params=adapter_params / shared_params,
                           parameter_delta=parameter_delta),
                  ],
                  selection="S0 fixed inner 393/99 split; original train_test_split order reconstructed; minimum inner MSE, patience 10; refit all 492; official validation final only",
                  bootstrap=dict(repetitions=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED),
                  torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                  script_sha256=sha(__file__))
    write_json(out / "config.json", config)
    s0_metrics = pd.read_csv(args.s0_dir / "metrics_all_seeds.csv")
    all_rows, history_rows, prediction_rows = [], [], []
    model_predictions = {"shared": {}, "adapter": {}}
    for seed in SEEDS:
        seed_dir = out / f"seed_{seed}"
        seed_dir.mkdir()
        for kind, label in (("shared", "S1-A Shared Head"), ("adapter", "S1-B Four Adapters")):
            _, chosen_epoch, selection_history = train(kind, x, y, inner_train, seed, HP["max_epochs"], inner_val)
            model, _, refit_history = train(kind, x, y, train_idx, seed, chosen_epoch)
            model.eval()
            with torch.inference_mode():
                prediction = model(x[val_idx]).numpy()
            result = metrics(prediction, target)
            aggregate = aggregate_metrics(prediction, target)
            model_predictions[kind][seed] = prediction
            write_json(seed_dir / f"{kind}_metrics.json", dict(seed=seed, model=label, selected_epoch=chosen_epoch,
                                                                   split_hash=SPLIT_HASH, attributes=result, aggregate=aggregate))
            torch.save(dict(model_state_dict=model.state_dict(), model=kind, seed=seed,
                            selected_epoch=chosen_epoch, split_hash=SPLIT_HASH, config=config),
                       out / "checkpoints" / f"{kind}_seed_{seed}.pt")
            all_rows.extend(dict(model=kind, model_label=label, seed=seed, attribute=attribute, **values)
                            for attribute, values in result.items())
            history_rows.extend(selection_history + refit_history)
            for pos, video_id in enumerate(data.iloc[val_idx].video_id):
                row = dict(video_id=video_id, seed=seed, model=kind)
                for index, attr in enumerate(ATTRS):
                    row[f"true_{attr}"] = target[pos, index]
                    row[f"pred_{attr}"] = prediction[pos, index]
                prediction_rows.append(row)
            print(f"completed {kind} seed={seed} epoch={chosen_epoch} macro_srcc={result['Macro']['srcc']:.4f}", flush=True)
    metrics_frame = pd.DataFrame(all_rows)
    metrics_frame.to_csv(out / "metrics_all_seeds.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history_rows).to_csv(out / "training_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(out / "predictions_by_model_seed.csv", index=False, encoding="utf-8-sig")
    paired_rows = []
    for seed in SEEDS:
        shared = model_predictions["shared"][seed]
        adapter = model_predictions["adapter"][seed]
        for pos, video_id in enumerate(data.iloc[val_idx].video_id):
            row = dict(video_id=video_id, seed=seed)
            for index, attr in enumerate(ATTRS):
                row[f"true_{attr}"] = target[pos, index]
                row[f"pred_{attr}_shared"] = shared[pos, index]
                row[f"pred_{attr}_adapter"] = adapter[pos, index]
            paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out / "paired_predictions.csv", index=False, encoding="utf-8-sig")
    aggregate_rows = []
    for kind, label in (("shared", "S1-A Shared Head"), ("adapter", "S1-B Four Adapters")):
        for seed in SEEDS:
            aggregate_rows.append(dict(model=kind, model_label=label, seed=seed,
                                       **aggregate_metrics(model_predictions[kind][seed], target)))
    aggregate_frame = pd.DataFrame(aggregate_rows)
    aggregate_frame.to_csv(out / "aggregate_metrics_all_seeds.csv", index=False, encoding="utf-8-sig")
    summary_rows = format_mean_std(metrics_frame, "shared", "S1-A Shared Head") + format_mean_std(metrics_frame, "adapter", "S1-B Four Adapters")
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(out / "attribute_comparison.csv", index=False, encoding="utf-8-sig")
    aggregate_summary = aggregate_frame.groupby("model")[["srcc", "plcc", "mae", "mse"]].agg(["mean", "std"])
    aggregate_comparison = pd.DataFrame([
        {"Model": label, **{metric.upper(): f"{aggregate_summary.loc[kind, (metric, 'mean')]:.4f} ± {aggregate_summary.loc[kind, (metric, 'std')]:.4f}"
                                           for metric in ("srcc", "plcc", "mae", "mse")}}
        for kind, label in (("shared", "S1-A Shared Head"), ("adapter", "S1-B Four Adapters"))])
    aggregate_comparison.to_csv(out / "aggregate_comparison.csv", index=False, encoding="utf-8-sig")
    paired_bootstrap = bootstrap_delta(model_predictions["shared"], model_predictions["adapter"], target)
    write_json(out / "bootstrap_results.json", paired_bootstrap)
    shared = metrics_frame[metrics_frame.model == "shared"].groupby("attribute")[["srcc", "plcc", "mae", "mse"]].mean()
    adapter = metrics_frame[metrics_frame.model == "adapter"].groupby("attribute")[["srcc", "plcc", "mae", "mse"]].mean()
    delta = (adapter - shared).reindex(ATTRS + ["Macro"])
    per_attribute_delta = delta.reset_index().rename(columns={"attribute": "Attribute"})
    per_attribute_delta.to_csv(out / "per_attribute_delta.csv", index=False, encoding="utf-8-sig")
    s0_shared = s0_metrics.groupby("attribute")[["srcc", "plcc", "mae", "mse"]].mean().reindex(ATTRS + ["Macro"])
    reproducibility = (shared - s0_shared).reset_index().rename(columns={"attribute": "Attribute"})
    reproducibility.to_csv(out / "s0_shared_reproduction_delta.csv", index=False, encoding="utf-8-sig")
    unchanged = all(sha(path) == digest for path, digest in protected.items())
    if not unchanged:
        raise RuntimeError("Source input changed during experiment")
    macro_deltas = delta.loc["Macro"]
    aggregate_delta = aggregate_frame.groupby("model")[["srcc", "plcc", "mae", "mse"]].mean().loc["adapter"] - aggregate_frame.groupby("model")[["srcc", "plcc", "mae", "mse"]].mean().loc["shared"]
    per_seed = pd.DataFrame({"seed": SEEDS,
                             "shared": [metrics(model_predictions["shared"][seed], target)["Macro"]["srcc"] for seed in SEEDS],
                             "adapter": [metrics(model_predictions["adapter"][seed], target)["Macro"]["srcc"] for seed in SEEDS]})
    per_seed["delta"] = per_seed.adapter - per_seed.shared
    positive_seeds = int((per_seed.delta > 0).sum())
    improved_attrs = [attr for attr in ATTRS if delta.loc[attr, "srcc"] > 0]
    degraded_attrs = [attr for attr in ATTRS if delta.loc[attr, "srcc"] < 0]
    no_material_harm = all(delta.loc[attr, "srcc"] > -0.02 for attr in ATTRS)
    case_a = (macro_deltas.srcc > 0 and positive_seeds >= 2 and aggregate_delta.srcc >= 0
              and len(improved_attrs) >= 2 and no_material_harm)
    case_b = (not case_a and macro_deltas.srcc > 0 and positive_seeds >= 2
              and len(improved_attrs) >= 2 and no_material_harm)
    case = "A" if case_a else "B" if case_b else "C"
    summary = [
        "# S1 Attribute-specific Adapter Ablation Summary", "", "## 1. Experiment Setup", "",
        f"时间：{config['created_at']}；输入为 S0 相同的 1540D text/knowledge 缓存；"
        "标签为四项人工评分/5。固定 492/123 划分和三个种子 42/123/2026。",
        "S1-A 与 S1-B 在同一 seed 使用相同训练样本顺序、内部 393/99 选轮划分、损失、优化器和超参数。"
        "先以内部分割最小 MSE 选轮，再完整 492 重训，正式 123 只作最终评估。",
        "", "## 2. S0 Baseline", "",
        "S0 shared head Macro SRCC/PLCC 为 0.3375±0.0102 / 0.3359±0.0125。"
        "本次 S1-A 是独立的严格复现，用于同轮对照；复现差异见 s0_shared_reproduction_delta.csv。",
        "", "## 3. Model Architecture", "",
        "S1-A：1540→256→4 + Sigmoid。S1-B：1540→224 shared representation，"
        "然后四个独立 224→32→1 ReLU MLP Adapter + Sigmoid。无视觉、音频、metadata、注意力、门控、排序或额外损失。",
        "", "## 4. Parameter Control", "", table(parameter_frame), "",
        f"Adapter 相对共享头参数变化={parameter_delta:+.4%}，满足不超过 ±10% 的预注册容量约束；不运行 S1-C。",
        "", "## 5. Multi-seed Results", "", table(summary_frame[summary_frame.Attribute == "Macro"]), "",
        "逐 seed Macro SRCC：", "", table(per_seed),
        "", "## 6. Per-attribute Results", "", table(summary_frame[summary_frame.Attribute != "Macro"]), "",
        "平均差值（Adapter - Shared）：", "", table(per_attribute_delta),
        "", "## 7. Scientific Aggregate Results", "", table(aggregate_comparison),
        "", f"Aggregate 平均差值：SRCC {aggregate_delta.srcc:+.4f}，PLCC {aggregate_delta.plcc:+.4f}，"
        f"MAE {aggregate_delta.mae:+.4f}，MSE {aggregate_delta.mse:+.4f}。该聚合只用于诊断，未参与训练。",
        "", "## 8. Paired Analysis", "",
        "同一 123 视频、同一 seed 的逐样本预测见 paired_predictions.csv。"
        "1000 次 paired bootstrap 对每次重采样先算每 seed 的差值，再取三 seed 平均。",
        f"Macro SRCC Δ 的 95% percentile CI={paired_bootstrap['adapter_minus_shared']['macro_srcc']['ci95_percentile']}；"
        f"Aggregate SRCC Δ 的 95% percentile CI={paired_bootstrap['adapter_minus_shared']['aggregate_srcc']['ci95_percentile']}。"
        "该区间描述固定三训练 seed 下验证视频重采样不确定性，不代替跨数据划分或模型选择校正。",
        "", "## 9. Key Findings", "",
        f"1. Four-Adapter 的 Macro SRCC 相对共享头变化为 {macro_deltas.srcc:+.4f}，"
        f"Macro PLCC/MAE/MSE 为 {macro_deltas.plcc:+.4f}/{macro_deltas.mae:+.4f}/{macro_deltas.mse:+.4f}。",
        f"2. Macro SRCC 在 {positive_seeds}/3 个 seed 为正向变化。",
        f"3. 平均 SRCC 改善属性：{', '.join(improved_attrs) if improved_attrs else '无'}；"
        f"下降属性：{', '.join(degraded_attrs) if degraded_attrs else '无'}。",
        f"4. Adapter 参数变化为 {parameter_delta:+.4%}，结果不能简单归因于更大容量。",
        f"5. Scientific Aggregate SRCC 变化为 {aggregate_delta.srcc:+.4f}。",
        f"6. 按预先定义的判定规则，结论为 Case {case}："
        + ("整体支持 Four-Adapter。" if case == "A" else
           "属性级局部支持；总体聚合增量有限。" if case == "B" else
           "不支持 Four-Adapter。"),
        "", "## 10. Decision for S2", "",
        ("S1 满足 Case A；S2 以 Four-Adapter 作为 text-only baseline，再检验 Shared CLIP 对各属性的增量。"
         if case == "A" else
         "S1 为 Case B；S2 以 Four-Adapter 作为属性级 text-only baseline，并保留 Shared Head 作为聚合对照。"
         if case == "B" else
         "S1 为 Case C；S2 继续以 Shared Head 作为 text-only baseline，并单独检验视觉的增量。"),
        "", "## 11. Verification", "",
        f"运行前后核对 {len(protected)} 个 metadata、参考 checkpoint 与特征缓存文件的 SHA256：{unchanged}。"
        "逐视频预测、最终 checkpoint、训练历史、参数比较和 bootstrap 输出均已保存。",
    ]
    (out / "S1_SUMMARY.md").write_text("\n\n".join(summary) + "\n", encoding="utf-8")
    write_json(out / "integrity_check.json", dict(status="passed", source_files=len(protected), source_sha256=protected))
    write_json(out / "completion.json", dict(status="completed", elapsed_seconds=time.perf_counter() - started,
                                               split_hash=SPLIT_HASH, source_integrity=unchanged,
                                               case=case, case_a_success=bool(case_a)))
    print(f"Completed: {out}", flush=True)


if __name__ == "__main__":
    main()
