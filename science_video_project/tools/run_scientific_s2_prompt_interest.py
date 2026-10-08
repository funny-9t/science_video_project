"""S2-P: fixed interest-aware CLIP prompt semantic probing on cached frames."""

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
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pipeline.utils_io import load_metadata, load_pt
from tools.run_scientific_s0 import ATTRS, FEATURES, HP, SEEDS, SPLIT_HASH, metrics, sha, table, write_json
from tools.run_scientific_s2_multimodal import (
    InterestOnlyVisual,
    TextOnlyFourAdapters,
    aggregate_metrics,
)

PROMPTS = [
    {"name": "experiment", "positive": "a science explanation showing a clear physical experiment or demonstration", "negative": "a science explanation without any physical experiment or demonstration"},
    {"name": "animation", "positive": "a science explanation using vivid animation or simulation to illustrate a concept", "negative": "a science explanation without animation or simulation"},
    {"name": "visualization", "positive": "a science explanation using diagrams charts or visual graphics to explain information", "negative": "a science explanation without diagrams charts or visual graphics"},
    {"name": "concrete", "positive": "a science explanation using concrete objects or real-world examples to explain an abstract concept", "negative": "a science explanation presenting an abstract concept without concrete visual examples"},
    {"name": "novelty", "positive": "a science video showing a visually surprising unusual or curiosity-provoking phenomenon", "negative": "a science video showing ordinary visuals without surprising or unusual phenomena"},
    {"name": "dynamic", "positive": "a science explanation visually showing a dynamic process change or transformation over time", "negative": "a science explanation with mostly static visuals and no visible process or transformation"},
    {"name": "narrative", "positive": "a science explanation where the visuals actively help tell and explain the scientific story", "negative": "a science explanation where the visuals do not contribute to explaining the scientific content"},
]
PROMPT_NAMES = [item["name"] for item in PROMPTS]
PROMPT_FEATURE_NAMES = [f"{name}_{stat}" for name in PROMPT_NAMES for stat in ("mean", "top20")]
TEXT_SHARED_DIM, GENERIC_VISUAL_DIM, PROMPT_DIM, ADAPTER_DIM = 224, 64, 16, 32
BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, SHUFFLE_SEED = 1000, 7003, 7017
MODELS = ("p0_text", "p1_generic", "p2_prompt", "p3_generic_prompt", "p4_shuffled_prompt")
LABELS = {
    "p0_text": "P0 Text-only Four-Adapter",
    "p1_generic": "P1 Generic CLIP Interest Fusion",
    "p2_prompt": "P2 Interest Prompt Fusion",
    "p3_generic_prompt": "P3 Generic CLIP + Prompt Fusion",
    "p4_shuffled_prompt": "P4 Shuffled Prompt Control",
}


class PromptInterestFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_projection = nn.Linear(sum(FEATURES.values()), TEXT_SHARED_DIM)
        self.prompt_projection = nn.Linear(len(PROMPT_FEATURE_NAMES), PROMPT_DIM)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(HP["dropout"])
        self.text_adapters = nn.ModuleList([
            nn.Sequential(nn.Linear(TEXT_SHARED_DIM, ADAPTER_DIM), nn.ReLU(), nn.Linear(ADAPTER_DIM, 1))
            for _ in ATTRS[:3]
        ])
        self.interest_adapter = nn.Sequential(
            nn.Linear(TEXT_SHARED_DIM + PROMPT_DIM, ADAPTER_DIM), nn.ReLU(), nn.Linear(ADAPTER_DIM, 1)
        )

    def forward(self, text, visual, prompt):
        del visual
        text_hidden = self.dropout(self.activation(self.text_projection(text)))
        prompt_hidden = self.dropout(self.activation(self.prompt_projection(prompt)))
        first_three = [adapter(text_hidden) for adapter in self.text_adapters]
        interest = self.interest_adapter(torch.cat([text_hidden, prompt_hidden], dim=-1))
        return torch.sigmoid(torch.cat(first_three + [interest], dim=-1))


class GenericAndPromptInterestFusion(nn.Module):
    def __init__(self, visual_input_dim):
        super().__init__()
        self.text_projection = nn.Linear(sum(FEATURES.values()), TEXT_SHARED_DIM)
        self.visual_projection = nn.Linear(visual_input_dim, GENERIC_VISUAL_DIM)
        self.prompt_projection = nn.Linear(len(PROMPT_FEATURE_NAMES), PROMPT_DIM)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(HP["dropout"])
        self.text_adapters = nn.ModuleList([
            nn.Sequential(nn.Linear(TEXT_SHARED_DIM, ADAPTER_DIM), nn.ReLU(), nn.Linear(ADAPTER_DIM, 1))
            for _ in ATTRS[:3]
        ])
        self.interest_adapter = nn.Sequential(
            nn.Linear(TEXT_SHARED_DIM + GENERIC_VISUAL_DIM + PROMPT_DIM, ADAPTER_DIM), nn.ReLU(), nn.Linear(ADAPTER_DIM, 1)
        )

    def forward(self, text, visual, prompt):
        text_hidden = self.dropout(self.activation(self.text_projection(text)))
        visual_hidden = self.dropout(self.activation(self.visual_projection(visual)))
        prompt_hidden = self.dropout(self.activation(self.prompt_projection(prompt)))
        first_three = [adapter(text_hidden) for adapter in self.text_adapters]
        interest = self.interest_adapter(torch.cat([text_hidden, visual_hidden, prompt_hidden], dim=-1))
        return torch.sigmoid(torch.cat(first_three + [interest], dim=-1))


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def make_model(kind, seed, visual_dim):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if kind == "p0_text":
        return TextOnlyFourAdapters()
    if kind == "p1_generic":
        return InterestOnlyVisual(visual_dim)
    if kind in {"p2_prompt", "p4_shuffled_prompt"}:
        return PromptInterestFusion()
    if kind == "p3_generic_prompt":
        return GenericAndPromptInterestFusion(visual_dim)
    raise ValueError(kind)


def forward(model, kind, text, visual, prompt):
    if kind == "p0_text":
        return model(text, visual)
    if kind == "p1_generic":
        return model(text, visual)
    return model(text, visual, prompt)


def train(kind, text, visual, prompt, target, indices, seed, epochs, visual_dim, monitor=None):
    model = make_model(kind, seed, visual_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(text[indices], visual[indices], prompt[indices], target[indices]),
                        batch_size=HP["batch_size"], shuffle=True, generator=generator, num_workers=0)
    best, epoch_best, stale, history = float("inf"), 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for text_b, visual_b, prompt_b, target_b in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(forward(model, kind, text_b, visual_b, prompt_b), target_b)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss for {kind}")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(text_b)
        row = {"model": kind, "seed": seed, "phase": "selection" if monitor is not None else "refit", "epoch": epoch,
               "train_mse": total / len(indices)}
        if monitor is not None:
            model.eval()
            with torch.inference_mode():
                value = F.mse_loss(forward(model, kind, text[monitor], visual[monitor], prompt[monitor]), target[monitor]).item()
            row["inner_val_mse"] = value
            if value < best:
                best, epoch_best, stale = value, epoch, 0
            else:
                stale += 1
        history.append(row)
        if epoch == 1 or epoch % 10 == 0:
            print(f"{kind} seed={seed} {row['phase']} epoch={epoch} mse={row['train_mse']:.6f}", flush=True)
        if monitor is not None and stale >= HP["patience"]:
            break
    return model, epoch_best if monitor is not None else epochs, history


def encode_prompt_text(model_path, device):
    tokenizer = CLIPTokenizerFast.from_pretrained(model_path, local_files_only=True)
    text_model = CLIPTextModelWithProjection.from_pretrained(model_path, local_files_only=True).to(device).eval()
    texts = [item[side] for item in PROMPTS for side in ("positive", "negative")]
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
    with torch.inference_mode():
        embeddings = F.normalize(text_model(**{key: value.to(device) for key, value in tokens.items()}).text_embeds, dim=-1)
    pos, neg = embeddings[0::2].cpu(), embeddings[1::2].cpu()
    return pos, neg, tokenizer.__class__.__name__, text_model.config.to_dict()


def prompt_features(frame_features, pos, neg):
    frames = F.normalize(torch.as_tensor(frame_features, dtype=torch.float32), dim=-1)
    response = frames @ pos.T - frames @ neg.T
    top_count = max(1, int(np.ceil(response.shape[0] * .20)))
    top = response.topk(top_count, dim=0).values.mean(dim=0)
    return torch.stack([response.mean(dim=0), top], dim=1).flatten().numpy()


def interval(values):
    return {"mean": float(np.mean(values)), "ci95_percentile": [float(np.quantile(values, .025)), float(np.quantile(values, .975))],
            "positive_fraction": float(np.mean(np.asarray(values) > 0))}


def bootstrap(predictions, target):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = {"p2_minus_p0": ("p2_prompt", "p0_text"), "p2_minus_p1": ("p2_prompt", "p1_generic"),
                   "p3_minus_p2": ("p3_generic_prompt", "p2_prompt"), "p2_minus_p4": ("p2_prompt", "p4_shuffled_prompt")}
    result = {}
    for name, (candidate, baseline) in comparisons.items():
        interest, macro, aggregate = [], [], []
        for _ in range(BOOTSTRAP_SAMPLES):
            indices = rng.integers(0, len(target), size=len(target))
            each_interest, each_macro, each_aggregate = [], [], []
            for seed in SEEDS:
                candidate_pred, baseline_pred, y = predictions[candidate][seed][indices], predictions[baseline][seed][indices], target[indices]
                each_interest.append(metrics(candidate_pred, y)["content_interest"]["srcc"] - metrics(baseline_pred, y)["content_interest"]["srcc"])
                each_macro.append(metrics(candidate_pred, y)["Macro"]["srcc"] - metrics(baseline_pred, y)["Macro"]["srcc"])
                each_aggregate.append(aggregate_metrics(candidate_pred, y)["srcc"] - aggregate_metrics(baseline_pred, y)["srcc"])
            interest.append(float(np.mean(each_interest)))
            macro.append(float(np.mean(each_macro)))
            aggregate.append(float(np.mean(each_aggregate)))
        result[name] = {"interest_srcc": interval(interest), "macro_srcc": interval(macro), "aggregate_srcc": interval(aggregate)}
    return {"method": "paired nonparametric bootstrap over 123 validation videos; mean delta across three fixed training seeds",
            "repetitions": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "comparisons": result}


def formatted_results(frame, model):
    grouped = frame[frame.model == model].groupby("attribute")[["srcc", "plcc", "mae", "mse"]].agg(["mean", "std"])
    return {attr: {metric: (float(grouped.loc[attr, (metric, "mean")]), float(grouped.loc[attr, (metric, "std")]))
                   for metric in ("srcc", "plcc", "mae", "mse")} for attr in ATTRS + ["Macro"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s0-dir", type=Path, required=True)
    parser.add_argument("--s2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/scientific_s2_prompt_interest")
    parser.add_argument("--clip-model", type=Path, default=Path(r"D:\Projects\science_video_ranker_mvp\openaiclip-vit-large-patch14"))
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()

    s0 = json.loads((args.s0_dir / "config.json").read_text(encoding="utf-8"))
    s2 = json.loads((args.s2_dir / "config.json").read_text(encoding="utf-8"))
    if s0["split_hash"] != SPLIT_HASH or s2["split_hash"] != SPLIT_HASH or s0["features"] != FEATURES:
        raise ValueError("S0/S2 protocol mismatch")
    metadata, feature_dir, reference = map(Path, [s0["metadata"], s0["feature_dir"], s0["reference_checkpoint"]])
    protected = {str(path.resolve()): sha(path) for path in [metadata, reference]}
    manifest = pd.read_csv(args.s0_dir / "split_manifest.csv", dtype={"video_id": str})
    data = load_metadata(metadata)
    data = data[data.video_id.isin(set(manifest.video_id))].copy().reset_index(drop=True)
    train_idx, val_idx = train_test_split(np.arange(len(data)), test_size=.2, random_state=42, stratify=data.label)
    inner_train, inner_val = train_test_split(train_idx, test_size=HP["inner_val_ratio"], random_state=HP["inner_split_seed"], stratify=data.iloc[train_idx].label)
    if len(data) != 615 or (len(train_idx), len(val_idx), len(inner_train), len(inner_val)) != (492, 123, 393, 99):
        raise ValueError("Unexpected official S0 split")
    if set(data.iloc[val_idx].video_id) != set(manifest.loc[manifest.split == "validation", "video_id"]):
        raise ValueError("Validation membership mismatch")

    device = torch.device("cpu")
    pos, neg, tokenizer_name, text_config = encode_prompt_text(args.clip_model, device)
    if pos.shape != neg.shape or pos.shape != (len(PROMPTS), 768):
        raise ValueError(f"Prompt text encoder mismatch: pos={tuple(pos.shape)}, neg={tuple(neg.shape)}")
    torch.save({"positive": pos, "negative": neg, "prompt_names": PROMPT_NAMES, "feature_names": PROMPT_FEATURE_NAMES,
                "clip_model": str(args.clip_model.resolve()), "tokenizer": tokenizer_name}, out / "prompt_text_embeddings.pt")
    write_json(out / "prompt_definitions.json", {"version": "s2p_interest_visual_design_v1", "score": "cosine(pos)-cosine(neg)",
                                                    "top_fraction": .20, "prompts": PROMPTS, "feature_order": PROMPT_FEATURE_NAMES})

    texts, visuals, prompt_rows, feature_rows, frame_counts = [], [], [], [], []
    for row in data.itertuples(index=False):
        path = feature_dir / f"{row.video_id}.pt"
        protected[str(path.resolve())] = sha(path)
        sample = load_pt(path)
        text = np.concatenate([np.asarray(sample[name], dtype=np.float32) for name in FEATURES])
        visual = np.asarray(sample["clip_video_feat"], dtype=np.float32)
        frames = np.asarray(sample["frame_features"], dtype=np.float32)
        # Very short videos legitimately have fewer than the 40-frame sampling cap.
        # Use every cached frame and keep Top20% relative to that observed length.
        if (text.shape != (1540,) or visual.shape != (768,) or frames.ndim != 2
                or frames.shape[0] < 1 or frames.shape[1] != 768 or not np.isfinite(frames).all()):
            raise ValueError(f"Invalid cached input at {path}")
        feature = prompt_features(frames, pos, neg)
        texts.append(text)
        visuals.append(visual)
        prompt_rows.append(feature)
        frame_counts.append(int(frames.shape[0]))
        feature_rows.append({"video_id": row.video_id, **{name: float(feature[index]) for index, name in enumerate(PROMPT_FEATURE_NAMES)}})
    text, visual, prompt = torch.from_numpy(np.stack(texts)), torch.from_numpy(np.stack(visuals)), torch.from_numpy(np.stack(prompt_rows))
    target = torch.from_numpy(data[ATTRS].to_numpy(dtype=np.float32) / 5.0)
    pd.DataFrame(feature_rows).to_csv(out / "interest_prompt_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"video_id": data.video_id, "cached_frame_count": frame_counts}).to_csv(out / "visual_frame_feature_audit.csv", index=False, encoding="utf-8-sig")
    if not np.isfinite(prompt.numpy()).all() or np.any(prompt.numpy().std(axis=0) == 0):
        raise ValueError("Prompt features contain nonfinite or constant dimensions")
    stats = pd.DataFrame({"dimension": PROMPT_FEATURE_NAMES, "mean": prompt.numpy().mean(axis=0), "std": prompt.numpy().std(axis=0),
                          "min": prompt.numpy().min(axis=0), "max": prompt.numpy().max(axis=0)})
    stats.to_csv(out / "prompt_feature_stats.csv", index=False, encoding="utf-8-sig")
    audit_rows = []
    for index, name in enumerate(PROMPT_FEATURE_NAMES):
        audit_rows.append({"dimension": name,
                           "srcc_content_interest": float(spearmanr(prompt[:, index].numpy(), data.content_interest).statistic),
                           "srcc_video_aesthetics": float(spearmanr(prompt[:, index].numpy(), data.video_aesthetics).statistic)})
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(out / "prompt_univariate_correlation.csv", index=False, encoding="utf-8-sig")
    corr = pd.DataFrame(prompt.numpy(), columns=PROMPT_FEATURE_NAMES).corr(method="spearman")
    corr.to_csv(out / "prompt_feature_srcc_matrix.csv", encoding="utf-8-sig")
    redundant = [(corr.index[i], corr.columns[j], float(corr.iloc[i, j])) for i in range(len(corr)) for j in range(i + 1, len(corr)) if abs(corr.iloc[i, j]) > .9]
    (out / "prompt_feature_audit.md").write_text(
        "# S2-P Prompt Feature Audit\n\n"
        f"- 615 videos; cached frame features: N×768, N range={min(frame_counts)}–{max(frame_counts)}; prompt features: 14D.\n"
        f"- Nonfinite dimensions: 0; constant dimensions: 0.\n"
        f"- Pairwise |SRCC| > 0.9: {len(redundant)} pairs.\n"
        f"- The univariate correlation table is diagnostic only; no prompt is removed or selected from it.\n\n"
        + ("## High redundancy pairs\n\n" + "\n".join(f"- {a} / {b}: {value:.4f}" for a, b, value in redundant) if redundant else "## High redundancy pairs\n\nNone.\n"),
        encoding="utf-8",
    )
    rng = np.random.default_rng(SHUFFLE_SEED)
    permutation = rng.permutation(len(data))
    while np.any(permutation == np.arange(len(data))):
        permutation = rng.permutation(len(data))
    shuffled_prompt = prompt[permutation]
    pd.DataFrame({"video_id": data.video_id, "source_video_id": data.video_id.iloc[permutation].to_numpy()}).to_csv(out / "shuffled_prompt_mapping.csv", index=False, encoding="utf-8-sig")

    expected = {kind: make_model(kind, 1, visual.shape[1]) for kind in MODELS}
    params = {kind: count_parameters(model) for kind, model in expected.items()}
    if params["p0_text"] != 374116:
        raise ValueError("P0 does not match S1/S2 text-only Four-Adapter")
    parameter_frame = pd.DataFrame([{"Model": LABELS[kind], "key": kind, "Text": True, "Generic CLIP": kind in {"p1_generic", "p3_generic_prompt"},
                                     "Prompt Feature": "shuffled" if kind == "p4_shuffled_prompt" else kind in {"p2_prompt", "p3_generic_prompt"},
                                     "Interest Only": kind != "p0_text", "Params": params[kind]} for kind in MODELS])
    parameter_frame.to_csv(out / "parameter_comparison.csv", index=False, encoding="utf-8-sig")
    config = {"experiment": "S2-P", "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(), "s0_dir": str(args.s0_dir.resolve()),
              "s2_dir": str(args.s2_dir.resolve()), "metadata": str(metadata.resolve()), "feature_dir": str(feature_dir.resolve()),
              "reference_checkpoint": str(reference.resolve()), "clip_model": str(args.clip_model.resolve()), "clip_text_encoder": "CLIPTextModelWithProjection",
              "clip_text_config": text_config, "split_hash": SPLIT_HASH, "seeds": SEEDS, "text_features": FEATURES, "text_input_dim": 1540,
              "visual_feature": "clip_video_feat", "frame_feature": "frame_features", "frame_shape": "N x 768; N <= 40 cached global-uniform samples", "frame_count_range": [min(frame_counts), max(frame_counts)], "prompt_feature_dim": 14,
              "prompt_projection_dim": PROMPT_DIM, "generic_projection_dim": GENERIC_VISUAL_DIM, "prompt_version": "s2p_interest_visual_design_v1",
              "top_fraction": .20, "shuffle_seed": SHUFFLE_SEED, "hyperparameters": HP, "parameter_counts": params,
              "selection": "S0 original order; fixed 393/99 minimum inner MSE; refit all 492; official 123 only", "bootstrap": {"repetitions": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED},
              "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "script_sha256": sha(__file__)}
    write_json(out / "config.json", config)

    official_target = target[val_idx].numpy()
    predictions, all_rows, history_rows, prediction_rows = {kind: {} for kind in MODELS}, [], [], []
    for seed in SEEDS:
        seed_dir = out / f"seed_{seed}"
        seed_dir.mkdir()
        for kind in MODELS:
            candidate_prompt = shuffled_prompt if kind == "p4_shuffled_prompt" else prompt
            _, epoch, selection = train(kind, text, visual, candidate_prompt, target, inner_train, seed, HP["max_epochs"], visual.shape[1], inner_val)
            model, _, refit = train(kind, text, visual, candidate_prompt, target, train_idx, seed, epoch, visual.shape[1])
            model.eval()
            with torch.inference_mode():
                prediction = forward(model, kind, text[val_idx], visual[val_idx], candidate_prompt[val_idx]).numpy()
            result, aggregate = metrics(prediction, official_target), aggregate_metrics(prediction, official_target)
            predictions[kind][seed] = prediction
            write_json(seed_dir / f"{kind}_metrics.json", {"seed": seed, "model": LABELS[kind], "selected_epoch": epoch, "split_hash": SPLIT_HASH,
                                                              "attributes": result, "aggregate": aggregate})
            torch.save({"model_state_dict": model.state_dict(), "model": kind, "seed": seed, "selected_epoch": epoch, "split_hash": SPLIT_HASH,
                        "config": config}, out / "checkpoints" / f"{kind}_seed_{seed}.pt")
            all_rows.extend({"model": kind, "model_label": LABELS[kind], "seed": seed, "attribute": attr, **item} for attr, item in result.items())
            history_rows.extend(selection + refit)
            for position, video_id in enumerate(data.iloc[val_idx].video_id):
                row = {"video_id": video_id, "seed": seed, "model": kind}
                for index, attr in enumerate(ATTRS):
                    row[f"true_{attr}"] = official_target[position, index]
                    row[f"pred_{attr}"] = prediction[position, index]
                prediction_rows.append(row)
            print(f"completed {kind} seed={seed} epoch={epoch} interest_srcc={result['content_interest']['srcc']:.4f}", flush=True)
    metric_frame = pd.DataFrame(all_rows)
    metric_frame.to_csv(out / "metrics_all_seeds.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history_rows).to_csv(out / "training_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(out / "predictions_by_model_seed.csv", index=False, encoding="utf-8-sig")
    aggregate_frame = pd.DataFrame([{"model": kind, "model_label": LABELS[kind], "seed": seed, **aggregate_metrics(predictions[kind][seed], official_target)}
                                    for kind in MODELS for seed in SEEDS])
    aggregate_frame.to_csv(out / "aggregate_metrics_all_seeds.csv", index=False, encoding="utf-8-sig")
    metric_mean = metric_frame.groupby(["model", "attribute"])[["srcc", "plcc", "mae", "mse"]].mean()
    metric_std = metric_frame.groupby(["model", "attribute"])[["srcc", "plcc", "mae", "mse"]].std()
    content = pd.DataFrame([{"Model": LABELS[kind], **{metric.upper(): f"{metric_mean.loc[(kind, 'content_interest'), metric]:.4f} ± {metric_std.loc[(kind, 'content_interest'), metric]:.4f}" for metric in ("srcc", "plcc", "mae", "mse")}}
                            for kind in MODELS])
    content.to_csv(out / "content_interest_results.csv", index=False, encoding="utf-8-sig")
    macro_aggregate = pd.DataFrame([{"Model": LABELS[kind],
        "Macro SRCC": f"{metric_mean.loc[(kind, 'Macro'), 'srcc']:.4f} ± {metric_std.loc[(kind, 'Macro'), 'srcc']:.4f}",
        "Macro PLCC": f"{metric_mean.loc[(kind, 'Macro'), 'plcc']:.4f} ± {metric_std.loc[(kind, 'Macro'), 'plcc']:.4f}",
        "Aggregate SRCC": f"{aggregate_frame[aggregate_frame.model == kind].srcc.mean():.4f} ± {aggregate_frame[aggregate_frame.model == kind].srcc.std():.4f}",
        "Aggregate PLCC": f"{aggregate_frame[aggregate_frame.model == kind].plcc.mean():.4f} ± {aggregate_frame[aggregate_frame.model == kind].plcc.std():.4f}"} for kind in MODELS])
    macro_aggregate.to_csv(out / "aggregate_results.csv", index=False, encoding="utf-8-sig")
    per_seed = pd.DataFrame({"seed": SEEDS})
    for kind in MODELS:
        per_seed[kind] = [metrics(predictions[kind][seed], official_target)["content_interest"]["srcc"] for seed in SEEDS]
    per_seed.to_csv(out / "per_seed_interest_srcc.csv", index=False, encoding="utf-8-sig")
    paired_rows = []
    for seed in SEEDS:
        for position, video_id in enumerate(data.iloc[val_idx].video_id):
            row = {"video_id": video_id, "seed": seed}
            for index, attr in enumerate(ATTRS):
                row[f"true_{attr}"] = official_target[position, index]
                for kind in MODELS:
                    row[f"pred_{attr}_{kind}"] = predictions[kind][seed][position, index]
            paired_rows.append(row)
    pd.DataFrame(paired_rows).to_csv(out / "paired_predictions.csv", index=False, encoding="utf-8-sig")
    boot = bootstrap(predictions, official_target)
    write_json(out / "bootstrap_results.json", boot)

    s2_metrics = pd.read_csv(args.s2_dir / "metrics_all_seeds.csv")
    checks = {"p0_vs_s2a": ("p0_text", "text_only"), "p1_vs_s2c": ("p1_generic", "interest_visual")}
    reproduction = []
    for name, (local, source) in checks.items():
        left = metric_frame[metric_frame.model == local].set_index(["seed", "attribute"])[["srcc", "plcc", "mae", "mse"]].sort_index()
        right = s2_metrics[s2_metrics.model == source].set_index(["seed", "attribute"])[["srcc", "plcc", "mae", "mse"]].sort_index()
        delta = left - right
        reproduction.append({"check": name, "max_abs_error": float(np.abs(delta.to_numpy()).max())})
    reproduction_frame = pd.DataFrame(reproduction)
    reproduction_frame.to_csv(out / "s2_reproduction_checks.csv", index=False, encoding="utf-8-sig")
    if reproduction_frame.max_abs_error.max() > 1e-10:
        raise RuntimeError(f"P0/P1 reproduction failed: {reproduction_frame}")
    unchanged = all(sha(path) == value for path, value in protected.items())
    if not unchanged:
        raise RuntimeError("Source inputs changed during S2-P")
    means = {kind: formatted_results(metric_frame, kind) for kind in MODELS}
    aggregate_means = aggregate_frame.groupby("model")[["srcc", "plcc", "mae", "mse"]].mean()
    p2_interest, p0_interest, p1_interest = means["p2_prompt"]["content_interest"]["srcc"][0], means["p0_text"]["content_interest"]["srcc"][0], means["p1_generic"]["content_interest"]["srcc"][0]
    p2_positive = int((per_seed.p2_prompt > per_seed.p0_text).sum())
    p2_macro_ok = means["p2_prompt"]["Macro"]["srcc"][0] >= means["p0_text"]["Macro"]["srcc"][0] - .005
    p2_aggregate_ok = aggregate_means.loc["p2_prompt", "srcc"] >= aggregate_means.loc["p0_text", "srcc"] - .005
    p2_boot = boot["comparisons"]["p2_minus_p0"]["interest_srcc"]
    p3_better = means["p3_generic_prompt"]["content_interest"]["srcc"][0] > p2_interest
    if p2_interest > p0_interest and p2_interest > p1_interest and p2_positive >= 2 and p2_macro_ok and p2_aggregate_ok and p2_boot["ci95_percentile"][0] > 0:
        decision = "A"
    elif p3_better and means["p3_generic_prompt"]["Macro"]["srcc"][0] >= means["p2_prompt"]["Macro"]["srcc"][0] and aggregate_means.loc["p3_generic_prompt", "srcc"] >= aggregate_means.loc["p2_prompt", "srcc"]:
        decision = "C"
    elif p2_interest > p0_interest and p2_positive >= 2 and p2_macro_ok and p2_aggregate_ok:
        decision = "B"
    else:
        decision = "D"
    summary = ["# S2-P Interest-aware CLIP Prompt Semantic Probing Summary", "", "## 1. Experiment Setup", "",
               f"615 unique videos; fixed 492/123 split; seeds={SEEDS}; identical S0/S1/S2 training protocol.",
               "", "## 2. S2 Motivation", "", "S2 generic Shared CLIP concat did not yield stable gains; S2-P tests fixed task-aligned semantic responses from the same cached frame features.",
               "", "## 3. Prompt Design", "", table(pd.DataFrame(PROMPTS)), "", "## 4. Visual Feature Audit", "",
               "Cached Shared ViT-L/14 frame features only: N×768 per video with N≤40 global-uniform samples. The paired text encoder is loaded locally once; no CLIP image extraction or DeepSeek call occurs.",
               "", "## 5. Prompt Feature Statistics", "", table(stats), "", table(audit),
               "", "## 6. Model Architecture", "", table(parameter_frame), "", "## 7. Parameter Control", "",
               f"P2 has {params['p2_prompt']} parameters versus P1's {params['p1_generic']} ({(params['p2_prompt']-params['p1_generic'])/params['p1_generic']:+.2%}); P4 is a fixed deranged-prompt control.",
               "", "## 8. Content Interest Results", "", table(content), "", "## 9. Multi-seed Results", "", table(per_seed),
               "", "## 10. Macro and Scientific Aggregate", "", table(macro_aggregate), "", "## 11. Paired Bootstrap", "", json.dumps(boot, ensure_ascii=False, indent=2),
               "", "## 12. Prompt vs Aesthetic Diagnostic", "", "The full 14D univariate Interest/Aesthetic SRCC table is saved in prompt_univariate_correlation.csv. It is diagnostic only; no dimension was selected or removed.",
               "", "## 13. Key Findings", "",
               f"1. P2 Interest SRCC={p2_interest:.4f}; P0={p0_interest:.4f}; P1={p1_interest:.4f}.",
               f"2. P2-P0 Interest SRCC bootstrap 95% CI={p2_boot['ci95_percentile']}; positive fraction={p2_boot['positive_fraction']:.3f}.",
               f"3. P2 has positive Interest SRCC change in {p2_positive}/3 seeds; Macro/aggregate hold conditions={p2_macro_ok}/{p2_aggregate_ok}.",
               f"4. P3 better than P2 on Interest SRCC={p3_better}.",
               f"5. Official decision: Case {decision}.",
               "", "## 14. Decision for Scientific Branch", "",
               {"A": "Adopt P2 prompt semantic feature for the Interest Adapter, then enter S3 overall validation.",
                "B": "Keep P2 only as auxiliary Interest evidence; do not claim overall Scientific Branch upgrade before S3.",
                "C": "Assess P3's added complexity in S3; prompt and generic CLIP appear complementary under this fixed protocol.",
                "D": "Do not add prompt visual features. End Scientific Branch visual exploration and retain text/knowledge-only Four-Adapter for S3."}[decision]]
    (out / "S2P_SUMMARY.md").write_text("\n\n".join(summary) + "\n", encoding="utf-8")
    write_json(out / "integrity_check.json", {"status": "passed", "source_files": len(protected), "source_sha256": protected})
    write_json(out / "completion.json", {"status": "completed", "elapsed_seconds": time.perf_counter() - started, "split_hash": SPLIT_HASH,
                                            "source_integrity": unchanged, "decision_case": decision, "params": params,
                                            "reproduction_max_error": float(reproduction_frame.max_abs_error.max())})
    print(f"Completed: {out}", flush=True)


if __name__ == "__main__":
    main()
