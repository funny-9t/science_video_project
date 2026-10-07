"""Fixed-protocol S2 attribute-aware multimodal fusion ablation.

This script deliberately reuses S0/S1's fixed split, text/knowledge cache and
training protocol.  The only candidate feature is the cached Shared ViT-L/14
``clip_video_feat`` representation; no extraction or online LLM call occurs.
"""

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

TEXT_SHARED_DIM = 224
ADAPTER_DIM = 32
VISUAL_DIM = 64
WIDE_TEXT_DIM = 258
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 7002
MODELS = ("text_only", "all_visual", "interest_visual", "capacity_control")
LABELS = {
    "text_only": "S2-A Text-only Four-Adapter",
    "all_visual": "S2-B All-Attribute Visual",
    "interest_visual": "S2-C Interest-only Visual",
    "capacity_control": "S2-D Wider Text-only Capacity Control",
}


class TextOnlyFourAdapters(nn.Module):
    """Exact S1-B architecture; visual input is accepted but deliberately ignored."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(sum(FEATURES.values()), TEXT_SHARED_DIM)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(HP["dropout"])
        self.adapters = nn.ModuleList([
            nn.Sequential(nn.Linear(TEXT_SHARED_DIM, ADAPTER_DIM), nn.ReLU(), nn.Linear(ADAPTER_DIM, 1))
            for _ in ATTRS
        ])

    def forward(self, text, visual):
        del visual
        hidden = self.dropout(self.activation(self.projection(text)))
        return torch.sigmoid(torch.cat([adapter(hidden) for adapter in self.adapters], dim=-1))


class AllAttributeVisual(nn.Module):
    def __init__(self, visual_input_dim):
        super().__init__()
        self.text_projection = nn.Linear(sum(FEATURES.values()), TEXT_SHARED_DIM)
        self.visual_projection = nn.Linear(visual_input_dim, VISUAL_DIM)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(HP["dropout"])
        fused_dim = TEXT_SHARED_DIM + VISUAL_DIM
        self.adapters = nn.ModuleList([
            nn.Sequential(nn.Linear(fused_dim, ADAPTER_DIM), nn.ReLU(), nn.Linear(ADAPTER_DIM, 1))
            for _ in ATTRS
        ])

    def forward(self, text, visual):
        text_hidden = self.dropout(self.activation(self.text_projection(text)))
        visual_hidden = self.dropout(self.activation(self.visual_projection(visual)))
        fused = torch.cat([text_hidden, visual_hidden], dim=-1)
        return torch.sigmoid(torch.cat([adapter(fused) for adapter in self.adapters], dim=-1))


class InterestOnlyVisual(nn.Module):
    def __init__(self, visual_input_dim):
        super().__init__()
        self.text_projection = nn.Linear(sum(FEATURES.values()), TEXT_SHARED_DIM)
        self.visual_projection = nn.Linear(visual_input_dim, VISUAL_DIM)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(HP["dropout"])
        self.text_adapters = nn.ModuleList([
            nn.Sequential(nn.Linear(TEXT_SHARED_DIM, ADAPTER_DIM), nn.ReLU(), nn.Linear(ADAPTER_DIM, 1))
            for _ in ATTRS[:3]
        ])
        self.interest_adapter = nn.Sequential(
            nn.Linear(TEXT_SHARED_DIM + VISUAL_DIM, ADAPTER_DIM), nn.ReLU(), nn.Linear(ADAPTER_DIM, 1)
        )

    def forward(self, text, visual):
        text_hidden = self.dropout(self.activation(self.text_projection(text)))
        visual_hidden = self.dropout(self.activation(self.visual_projection(visual)))
        first_three = [adapter(text_hidden) for adapter in self.text_adapters]
        interest = self.interest_adapter(torch.cat([text_hidden, visual_hidden], dim=-1))
        return torch.sigmoid(torch.cat(first_three + [interest], dim=-1))


class WiderTextOnlyCapacityControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(sum(FEATURES.values()), WIDE_TEXT_DIM)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(HP["dropout"])
        self.adapters = nn.ModuleList([
            nn.Sequential(nn.Linear(WIDE_TEXT_DIM, ADAPTER_DIM), nn.ReLU(), nn.Linear(ADAPTER_DIM, 1))
            for _ in ATTRS
        ])

    def forward(self, text, visual):
        del visual
        hidden = self.dropout(self.activation(self.projection(text)))
        return torch.sigmoid(torch.cat([adapter(hidden) for adapter in self.adapters], dim=-1))


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def make_model(kind, seed, visual_input_dim):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if kind == "text_only":
        return TextOnlyFourAdapters()
    if kind == "all_visual":
        return AllAttributeVisual(visual_input_dim)
    if kind == "interest_visual":
        return InterestOnlyVisual(visual_input_dim)
    if kind == "capacity_control":
        return WiderTextOnlyCapacityControl()
    raise ValueError(f"Unknown model kind: {kind}")


def aggregate_metrics(prediction, target):
    pred_mean, target_mean = prediction.mean(axis=1), target.mean(axis=1)
    return dict(
        srcc=float(spearmanr(pred_mean, target_mean).statistic),
        plcc=float(pearsonr(pred_mean, target_mean).statistic),
        mae=float(np.abs(pred_mean - target_mean).mean()),
        mse=float(np.square(pred_mean - target_mean).mean()),
    )


def train(kind, text, visual, target, indices, seed, epochs, visual_input_dim, monitor=None):
    model = make_model(kind, seed, visual_input_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(text[indices], visual[indices], target[indices]),
        batch_size=HP["batch_size"], shuffle=True, generator=generator, num_workers=0,
    )
    best_value, best_epoch, stale = float("inf"), 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for text_batch, visual_batch, target_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(text_batch, visual_batch), target_batch)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite {kind} loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(text_batch)
        item = dict(model=kind, seed=seed, phase="selection" if monitor is not None else "refit",
                    epoch=epoch, train_mse=total / len(indices))
        if monitor is not None:
            model.eval()
            with torch.inference_mode():
                inner_mse = nn.functional.mse_loss(model(text[monitor], visual[monitor]), target[monitor]).item()
            item["inner_val_mse"] = inner_mse
            if inner_mse < best_value:
                best_value, best_epoch, stale = inner_mse, epoch, 0
            else:
                stale += 1
        history.append(item)
        if epoch == 1 or epoch % 10 == 0:
            print(f"{kind} seed={seed} {item['phase']} epoch={epoch} mse={item['train_mse']:.6f}", flush=True)
        if monitor is not None and stale >= HP["patience"]:
            break
    return model, best_epoch if monitor is not None else epochs, history


def mean_std_table(metrics_frame):
    result = []
    for kind in MODELS:
        grouped = metrics_frame[metrics_frame.model == kind].groupby("attribute")[["srcc", "plcc", "mae", "mse"]].agg(["mean", "std"])
        for attribute in ATTRS + ["Macro"]:
            result.append({"Model": LABELS[kind], "Attribute": attribute, **{
                metric.upper(): f"{grouped.loc[attribute, (metric, 'mean')]:.4f} ± {grouped.loc[attribute, (metric, 'std')]:.4f}"
                for metric in ("srcc", "plcc", "mae", "mse")
            }})
    return pd.DataFrame(result)


def paired_bootstrap(predictions, target):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = {"all_visual_minus_text_only": ("all_visual", "text_only"),
                   "interest_visual_minus_text_only": ("interest_visual", "text_only"),
                   "all_visual_minus_capacity_control": ("all_visual", "capacity_control"),
                   "interest_visual_minus_capacity_control": ("interest_visual", "capacity_control")}
    output = {}
    for name, (candidate, baseline) in comparisons.items():
        values = {"macro_srcc": [], "interest_srcc": [], "aggregate_srcc": []}
        for _ in range(BOOTSTRAP_SAMPLES):
            indices = rng.integers(0, len(target), size=len(target))
            per_seed = {key: [] for key in values}
            for seed in SEEDS:
                cand, base, sampled_target = predictions[candidate][seed][indices], predictions[baseline][seed][indices], target[indices]
                per_seed["macro_srcc"].append(metrics(cand, sampled_target)["Macro"]["srcc"] - metrics(base, sampled_target)["Macro"]["srcc"])
                per_seed["interest_srcc"].append(metrics(cand, sampled_target)["content_interest"]["srcc"] - metrics(base, sampled_target)["content_interest"]["srcc"])
                per_seed["aggregate_srcc"].append(aggregate_metrics(cand, sampled_target)["srcc"] - aggregate_metrics(base, sampled_target)["srcc"])
            for key in values:
                values[key].append(float(np.mean(per_seed[key])))
        output[name] = {key: {
            "mean": float(np.mean(value)),
            "ci95_percentile": [float(np.quantile(value, .025)), float(np.quantile(value, .975))],
            "positive_fraction": float(np.mean(np.asarray(value) > 0)),
        } for key, value in values.items()}
    return dict(method="paired nonparametric bootstrap over 123 validation videos; statistic is mean delta across three fixed training seeds",
                repetitions=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED, comparisons=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s0-dir", type=Path, required=True)
    parser.add_argument("--s1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/scientific_s2_multimodal")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing experiment: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()

    s0_config = json.loads((args.s0_dir / "config.json").read_text(encoding="utf-8"))
    s1_config = json.loads((args.s1_dir / "config.json").read_text(encoding="utf-8"))
    if s0_config["split_hash"] != SPLIT_HASH or s1_config["split_hash"] != SPLIT_HASH:
        raise ValueError("S0/S1 split hash mismatch")
    if s0_config["features"] != FEATURES or s1_config["features"] != FEATURES:
        raise ValueError("S0/S1 text feature specification mismatch")
    metadata_path, feature_dir, reference_checkpoint = map(Path, [s0_config["metadata"], s0_config["feature_dir"], s0_config["reference_checkpoint"]])
    protected = {str(path.resolve()): sha(path) for path in [metadata_path, reference_checkpoint]}
    manifest = pd.read_csv(args.s0_dir / "split_manifest.csv", dtype={"video_id": str})
    data = load_metadata(metadata_path)
    data = data[data.video_id.isin(set(manifest.video_id))].copy().reset_index(drop=True)
    if len(data) != 615 or data.video_id.nunique() != 615:
        raise ValueError("Expected the official 615 unique videos")
    train_idx, val_idx = train_test_split(np.arange(len(data)), test_size=.2, random_state=42, stratify=data.label)
    inner_train, inner_val = train_test_split(train_idx, test_size=HP["inner_val_ratio"], random_state=HP["inner_split_seed"], stratify=data.iloc[train_idx].label)
    for field, indices in (("train", train_idx), ("validation", val_idx)):
        if set(data.iloc[indices].video_id) != set(manifest.loc[manifest.split == field, "video_id"]):
            raise ValueError(f"S0 {field} membership mismatch")
    if set(data.iloc[inner_train].video_id) != set(manifest.loc[manifest.inner_split == "inner_train", "video_id"]):
        raise ValueError("S0 inner training membership mismatch")
    if set(data.iloc[inner_val].video_id) != set(manifest.loc[manifest.inner_split == "inner_validation", "video_id"]):
        raise ValueError("S0 inner validation membership mismatch")
    if (len(train_idx), len(val_idx), len(inner_train), len(inner_val)) != (492, 123, 393, 99):
        raise ValueError("Unexpected fixed split size")

    text_rows, visual_rows, feature_audit = [], [], []
    for row in data.itertuples(index=False):
        path = feature_dir / f"{row.video_id}.pt"
        protected[str(path.resolve())] = sha(path)
        sample = load_pt(path)
        if str(sample["video_id"]) != row.video_id:
            raise ValueError(f"Feature id mismatch in {path}")
        text_parts = []
        for name, dim in FEATURES.items():
            value = np.asarray(sample[name], dtype=np.float32)
            if value.shape != (dim,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid S1 text feature {name}: {path}")
            text_parts.append(value)
        visual = np.asarray(sample["clip_video_feat"], dtype=np.float32)
        if visual.shape != (768,) or not np.isfinite(visual).all() or not np.any(visual):
            raise ValueError(f"Invalid cached Shared CLIP feature: {path}")
        text_rows.append(np.concatenate(text_parts))
        visual_rows.append(visual)
        feature_audit.append(dict(video_id=row.video_id, clip_dim=int(visual.size), clip_norm=float(np.linalg.norm(visual)),
                                  clip_sampling_version=str(sample.get("clip_sampling_version", ""))))
    text = torch.from_numpy(np.stack(text_rows))
    visual = torch.from_numpy(np.stack(visual_rows))
    target = torch.from_numpy(data[ATTRS].to_numpy(dtype=np.float32) / 5.0)
    audit_frame = pd.DataFrame(feature_audit)
    audit_frame.to_csv(out / "visual_feature_audit.csv", index=False, encoding="utf-8-sig")
    visual_dim = int(visual.shape[1])
    versions = audit_frame.clip_sampling_version.value_counts().to_dict()
    if text.shape[1] != 1540 or visual_dim != 768:
        raise ValueError(f"Unexpected cache shape text={tuple(text.shape)}, visual={tuple(visual.shape)}")

    parameter_rows = []
    expected_models = {kind: make_model(kind, 1, visual_dim) for kind in MODELS}
    params = {kind: parameter_count(model) for kind, model in expected_models.items()}
    if params["text_only"] != 374116:
        raise ValueError("S2-A is not the exact S1-B architecture")
    for kind in MODELS:
        parameter_rows.append(dict(model=LABELS[kind], key=kind, text=True,
                                   visual="Shared CLIP" if "visual" in kind else "-",
                                   visual_scope="All attributes" if kind == "all_visual" else "Interest only" if kind == "interest_visual" else "-",
                                   params=params[kind], parameter_delta_vs_all_visual=(params[kind] - params["all_visual"]) / params["all_visual"]))
    parameter_frame = pd.DataFrame(parameter_rows)
    parameter_frame.to_csv(out / "parameter_comparison.csv", index=False, encoding="utf-8-sig")
    capacity_gap = abs(params["capacity_control"] - params["all_visual"]) / params["all_visual"]
    if capacity_gap > .10:
        raise ValueError("Capacity control is not within 10% of all-visual model")

    config = dict(experiment="S2", created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  s0_dir=str(args.s0_dir.resolve()), s1_dir=str(args.s1_dir.resolve()), metadata=str(metadata_path.resolve()),
                  feature_dir=str(feature_dir.resolve()), reference_checkpoint=str(reference_checkpoint.resolve()), split_hash=SPLIT_HASH,
                  seeds=SEEDS, text_features=FEATURES, text_input_dim=1540, visual_feature="clip_video_feat",
                  visual_input_dim=visual_dim, visual_projection_dim=VISUAL_DIM, visual_sampling_versions=versions,
                  attributes=ATTRS, hyperparameters=HP, models={kind: LABELS[kind] for kind in MODELS},
                  architecture=dict(text_only="1540->224; four 224->32->1 adapters", all_visual="1540->224 + 768->64; four concat(288)->32->1 adapters",
                                    interest_visual="1540->224 + 768->64; first three 224->32->1, interest concat(288)->32->1",
                                    capacity_control="1540->258; four 258->32->1 adapters"),
                  parameter_counts=params, capacity_control_relative_gap=capacity_gap,
                  selection="S0 original train_test_split order reconstructed; fixed 393/99 minimum inner MSE; refit all 492; official 123 final only",
                  bootstrap=dict(repetitions=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED), device="cpu", threads=4,
                  torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, script_sha256=sha(__file__))
    write_json(out / "config.json", config)

    all_rows, histories, prediction_rows, predictions = [], [], [], {kind: {} for kind in MODELS}
    official_target = target[val_idx].numpy()
    for seed in SEEDS:
        seed_dir = out / f"seed_{seed}"
        seed_dir.mkdir()
        for kind in MODELS:
            _, epoch, selection_history = train(kind, text, visual, target, inner_train, seed, HP["max_epochs"], visual_dim, inner_val)
            model, _, refit_history = train(kind, text, visual, target, train_idx, seed, epoch, visual_dim)
            model.eval()
            with torch.inference_mode():
                prediction = model(text[val_idx], visual[val_idx]).numpy()
            result, aggregate = metrics(prediction, official_target), aggregate_metrics(prediction, official_target)
            predictions[kind][seed] = prediction
            write_json(seed_dir / f"{kind}_metrics.json", dict(seed=seed, model=LABELS[kind], selected_epoch=epoch,
                                                                  split_hash=SPLIT_HASH, attributes=result, aggregate=aggregate))
            torch.save(dict(model_state_dict=model.state_dict(), model=kind, seed=seed, selected_epoch=epoch,
                            split_hash=SPLIT_HASH, config=config), out / "checkpoints" / f"{kind}_seed_{seed}.pt")
            all_rows.extend(dict(model=kind, model_label=LABELS[kind], seed=seed, attribute=attribute, **value)
                            for attribute, value in result.items())
            histories.extend(selection_history + refit_history)
            for position, video_id in enumerate(data.iloc[val_idx].video_id):
                row = dict(video_id=video_id, seed=seed, model=kind)
                for index, attr in enumerate(ATTRS):
                    row[f"true_{attr}"] = official_target[position, index]
                    row[f"pred_{attr}"] = prediction[position, index]
                prediction_rows.append(row)
            print(f"completed {kind} seed={seed} epoch={epoch} macro_srcc={result['Macro']['srcc']:.4f}", flush=True)

    metrics_frame = pd.DataFrame(all_rows)
    metrics_frame.to_csv(out / "metrics_all_seeds.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(histories).to_csv(out / "training_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(out / "predictions_by_model_seed.csv", index=False, encoding="utf-8-sig")
    summary_frame = mean_std_table(metrics_frame)
    summary_frame.to_csv(out / "attribute_comparison.csv", index=False, encoding="utf-8-sig")
    aggregate_rows = [dict(model=kind, model_label=LABELS[kind], seed=seed, **aggregate_metrics(predictions[kind][seed], official_target))
                      for kind in MODELS for seed in SEEDS]
    aggregate_frame = pd.DataFrame(aggregate_rows)
    aggregate_frame.to_csv(out / "aggregate_metrics_all_seeds.csv", index=False, encoding="utf-8-sig")
    aggregate_summary = aggregate_frame.groupby("model")[["srcc", "plcc", "mae", "mse"]].agg(["mean", "std"])
    aggregate_comparison = pd.DataFrame([
        {"Model": LABELS[kind], **{metric.upper(): f"{aggregate_summary.loc[kind, (metric, 'mean')]:.4f} ± {aggregate_summary.loc[kind, (metric, 'std')]:.4f}" for metric in ("srcc", "plcc", "mae", "mse")}}
        for kind in MODELS
    ])
    aggregate_comparison.to_csv(out / "aggregate_comparison.csv", index=False, encoding="utf-8-sig")

    paired_rows = []
    for seed in SEEDS:
        for position, video_id in enumerate(data.iloc[val_idx].video_id):
            row = dict(video_id=video_id, seed=seed)
            for index, attr in enumerate(ATTRS):
                row[f"true_{attr}"] = official_target[position, index]
                for kind in MODELS:
                    row[f"pred_{attr}_{kind}"] = predictions[kind][seed][position, index]
            paired_rows.append(row)
    pd.DataFrame(paired_rows).to_csv(out / "paired_predictions.csv", index=False, encoding="utf-8-sig")
    model_means = metrics_frame.groupby(["model", "attribute"])[["srcc", "plcc", "mae", "mse"]].mean()
    visual_gain = (model_means.loc["all_visual"] - model_means.loc["text_only"]).reindex(ATTRS + ["Macro"])
    interest_gain = (model_means.loc["interest_visual"] - model_means.loc["text_only"]).reindex(ATTRS + ["Macro"])
    visual_gain_frame = pd.DataFrame({"attribute": ATTRS + ["Macro"],
                                      "all_visual_minus_text_only_srcc": visual_gain.srcc.values,
                                      "interest_visual_minus_text_only_srcc": interest_gain.srcc.values,
                                      "all_visual_minus_text_only_plcc": visual_gain.plcc.values,
                                      "interest_visual_minus_text_only_plcc": interest_gain.plcc.values})
    visual_gain_frame.to_csv(out / "per_attribute_visual_gain.csv", index=False, encoding="utf-8-sig")
    gain_ranking = visual_gain.reindex(ATTRS).sort_values("srcc", ascending=False).reset_index().rename(columns={"attribute": "attribute"})
    gain_ranking.to_csv(out / "visual_gain_ranking.csv", index=False, encoding="utf-8-sig")
    bootstrap = paired_bootstrap(predictions, official_target)
    write_json(out / "bootstrap_results.json", bootstrap)

    s1_metrics = pd.read_csv(args.s1_dir / "metrics_all_seeds.csv")
    s1_adapter = s1_metrics[s1_metrics.model == "adapter"].set_index(["seed", "attribute"])[["srcc", "plcc", "mae", "mse"]].sort_index()
    s2_text = metrics_frame[metrics_frame.model == "text_only"].set_index(["seed", "attribute"])[["srcc", "plcc", "mae", "mse"]].sort_index()
    reproduction_delta = (s2_text - s1_adapter).reset_index()
    reproduction_delta.to_csv(out / "s1_text_only_reproduction_delta.csv", index=False, encoding="utf-8-sig")
    max_reproduction_error = float(np.abs(reproduction_delta[["srcc", "plcc", "mae", "mse"]].to_numpy()).max())
    unchanged = all(sha(path) == digest for path, digest in protected.items())
    if not unchanged:
        raise RuntimeError("Source files changed during S2")
    if max_reproduction_error > 1e-10:
        raise RuntimeError(f"S2-A does not reproduce S1-B: {max_reproduction_error}")

    per_seed = pd.DataFrame({"seed": SEEDS})
    for kind in MODELS:
        per_seed[kind] = [metrics(predictions[kind][seed], official_target)["Macro"]["srcc"] for seed in SEEDS]
        per_seed[f"interest_{kind}"] = [metrics(predictions[kind][seed], official_target)["content_interest"]["srcc"] for seed in SEEDS]
    per_seed["interest_visual_interest_delta"] = per_seed.interest_interest_visual - per_seed.interest_text_only
    per_seed.to_csv(out / "per_seed_macro_and_interest.csv", index=False, encoding="utf-8-sig")
    aggregate_mean = aggregate_frame.groupby("model")[["srcc", "plcc", "mae", "mse"]].mean()
    macro = {kind: model_means.loc[(kind, "Macro"), "srcc"] for kind in MODELS}
    interest = {kind: model_means.loc[(kind, "content_interest"), "srcc"] for kind in MODELS}
    c_interest_positive = int((per_seed.interest_visual_interest_delta > 0).sum())
    c_macro_not_down = macro["interest_visual"] >= macro["text_only"] - 1e-12
    c_aggregate_not_material_down = aggregate_mean.loc["interest_visual", "srcc"] >= aggregate_mean.loc["text_only", "srcc"] - .01
    c_not_worse_than_b = macro["interest_visual"] >= macro["all_visual"] - 1e-12
    capacity_explains_b = macro["all_visual"] <= macro["capacity_control"]
    if interest["interest_visual"] > interest["text_only"] and c_interest_positive >= 2 and c_macro_not_down and c_aggregate_not_material_down and c_not_worse_than_b and not capacity_explains_b:
        decision = "A"
    elif macro["all_visual"] > macro["interest_visual"] and macro["all_visual"] > macro["text_only"] and aggregate_mean.loc["all_visual", "srcc"] >= aggregate_mean.loc["interest_visual", "srcc"] and not capacity_explains_b:
        decision = "B"
    elif (interest["interest_visual"] > interest["text_only"] and c_interest_positive >= 2
          and macro["interest_visual"] >= macro["text_only"] - .005
          and aggregate_mean.loc["interest_visual", "srcc"] >= aggregate_mean.loc["text_only", "srcc"] - .005):
        decision = "C"
    else:
        decision = "D"

    macro_table = summary_frame[summary_frame.Attribute == "Macro"]
    attribute_table = summary_frame[summary_frame.Attribute != "Macro"]
    summary = [
        "# S2 Attribute-aware Multimodal Fusion Summary", "", "## 1. Experiment Setup", "",
        f"时间：{config['created_at']}；615 个唯一视频，固定 492/123 划分、相同 393/99 内部选轮和 seeds={SEEDS}。",
        "S2-A 至 S2-D 固定复用 S1 的文本/知识输入、标签、MSE、AdamW、学习率、batch size、early stopping 与完整训练集重训规则。",
        "", "## 2. S0/S1 Baseline", "", "S2-A 是 S1-B Four-Adapter 的严格复现；差异见 s1_text_only_reproduction_delta.csv。",
        "", "## 3. Visual Feature Audit", "",
        f"唯一新增模态是现有缓存的 Shared ViT-L/14 clip_video_feat（{visual_dim}D）；615/615 条有效，采样版本计数={versions}。未重提取 CLIP、未使用 COVER/aesthetic prompt/audio/metadata，也未调用 DeepSeek。",
        "", "## 4. Model Architecture", "", table(parameter_frame[["model", "visual", "visual_scope", "params", "parameter_delta_vs_all_visual"]]),
        "", "## 5. Parameter Control", "",
        f"S2-D 使用 258D 纯文本共享表示，参数量 {params['capacity_control']}，相对 S2-B {params['all_visual']} 的差异为 {capacity_gap:+.2%}。",
        "", "## 6. Multi-seed Results", "", table(macro_table),
        "", "## 7. Per-attribute Visual Gain", "", table(visual_gain_frame), "", "All-visual gain ranking:", "", table(gain_ranking[["attribute", "srcc"]]),
        "", "## 8. Content Interest Analysis", "", table(per_seed[["seed", "interest_text_only", "interest_all_visual", "interest_interest_visual", "interest_visual_interest_delta"]]),
        "", "## 9. Scientific Aggregate Results", "", table(aggregate_comparison),
        "", "## 10. Paired Bootstrap", "",
        "Bootstrap 使用同一 123 个验证视频、三个固定训练 seed，重采样 1000 次；完整区间见 bootstrap_results.json。",
        json.dumps(bootstrap, ensure_ascii=False, indent=2),
        "", "## 11. Key Findings", "",
        f"1. All-visual 相对 text-only 的 Macro SRCC 变化为 {macro['all_visual'] - macro['text_only']:+.4f}；interest-only 为 {macro['interest_visual'] - macro['text_only']:+.4f}。",
        f"2. All-visual 的最大属性 SRCC 增益为 {gain_ranking.iloc[0]['attribute']} ({gain_ranking.iloc[0]['srcc']:+.4f})。",
        f"3. content_interest：text-only={interest['text_only']:.4f}，all-visual={interest['all_visual']:.4f}，interest-only={interest['interest_visual']:.4f}；interest-only 在 {c_interest_positive}/3 个 seed 正向。",
        f"4. S2-B 与 S2-D Macro SRCC：{macro['all_visual']:.4f} / {macro['capacity_control']:.4f}；容量解释={capacity_explains_b}。",
        f"5. Scientific aggregate SRCC（A/B/C/D）：" + "/".join(f"{aggregate_mean.loc[k, 'srcc']:.4f}" for k in MODELS) + ".",
        f"6. 按预设规则判定为 Case {decision}。",
        "", "## 12. Decision for Scientific Branch", "",
        {"A": "采用 Interest-only Visual：前三个属性保留文本/知识，content_interest 使用 text+Shared CLIP；再进入 S3 总体分支验证。",
         "B": "采用 All-Attribute Visual 并在 S3 中验证总体影响。",
         "C": "视觉仅保留为 content_interest 的辅助输入，不表述为整体 Scientific Branch 升级；是否进入 S3 需以总体消融决定。",
         "D": "不纳入视觉，保留 Four-Adapter text/knowledge-only Scientific Branch；不进入视觉版本的 S3。"}[decision],
    ]
    (out / "S2_SUMMARY.md").write_text("\n\n".join(summary) + "\n", encoding="utf-8")
    write_json(out / "integrity_check.json", dict(status="passed", source_files=len(protected), source_sha256=protected))
    write_json(out / "completion.json", dict(status="completed", elapsed_seconds=time.perf_counter() - started, split_hash=SPLIT_HASH,
                                               source_integrity=unchanged, text_only_reproduction_max_error=max_reproduction_error,
                                               decision_case=decision, parameter_counts=params))
    print(f"Completed: {out}", flush=True)


if __name__ == "__main__":
    main()
