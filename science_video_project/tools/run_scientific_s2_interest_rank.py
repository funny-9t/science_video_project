"""S2-R: fixed pointwise-plus-pairwise RankNet experiment for content interest."""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pipeline.utils_io import load_metadata, load_pt
from tools.run_scientific_s0 import ATTRS, FEATURES, HP, SEEDS, SPLIT_HASH, metrics, sha, table, write_json
from tools.run_scientific_s2_multimodal import TextOnlyFourAdapters, aggregate_metrics

MODELS = ("r0_mse", "r1_mse_rank", "r2_rank_only")
LABELS = {
    "r0_mse": "R0 MSE-only",
    "r1_mse_rank": "R1 Interest MSE + RankNet",
    "r2_rank_only": "R2 Interest RankNet-only",
}
RANK_WEIGHT = .1
PAIR_BATCH_SIZE = 32
PAIR_THRESHOLD_RAW = 2
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 7004


def make_model(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return TextOnlyFourAdapters()


def build_pairs(indices, raw_interest):
    """Return unordered pairs whose original 1--5 labels differ by at least two."""
    indices = np.asarray(indices, dtype=np.int64)
    left, right, signs = [], [], []
    for offset, first in enumerate(indices[:-1]):
        for second in indices[offset + 1:]:
            difference = raw_interest[first] - raw_interest[second]
            if abs(difference) >= PAIR_THRESHOLD_RAW:
                left.append(first)
                right.append(second)
                signs.append(1.0 if difference > 0 else -1.0)
    if not left:
        raise ValueError("No high-confidence training pairs")
    return np.asarray(left), np.asarray(right), np.asarray(signs, dtype=np.float32)


def pair_statistics(indices, raw_interest):
    indices = np.asarray(indices, dtype=np.int64)
    counts = {str(delta): 0 for delta in range(1, 5)}
    positive, negative = 0, 0
    for offset, first in enumerate(indices[:-1]):
        for second in indices[offset + 1:]:
            difference = int(raw_interest[first] - raw_interest[second])
            if difference:
                counts[str(abs(difference))] += 1
                positive += difference > 0
                negative += difference < 0
    return {
        "n_samples": int(len(indices)),
        "number_of_delta_1_pairs": counts["1"],
        "number_of_delta_2_pairs": counts["2"],
        "number_of_delta_3_pairs": counts["3"],
        "number_of_delta_4_pairs": counts["4"],
        "total_unequal_pairs": int(sum(counts.values())),
        "high_confidence_pairs": int(counts["2"] + counts["3"] + counts["4"]),
        "direction_positive_i_gt_j": int(positive),
        "direction_negative_i_lt_j": int(negative),
        "direction_positive_fraction": float(positive / (positive + negative)),
        "threshold_raw": PAIR_THRESHOLD_RAW,
    }


def pairwise_accuracy(prediction, raw_target, minimum_delta):
    correct, total, ties = 0.0, 0, 0
    for first in range(len(prediction) - 1):
        for second in range(first + 1, len(prediction)):
            difference = raw_target[first] - raw_target[second]
            if abs(difference) < minimum_delta:
                continue
            pred_difference = prediction[first] - prediction[second]
            if pred_difference == 0:
                correct += .5
                ties += 1
            elif (pred_difference > 0) == (difference > 0):
                correct += 1.0
            total += 1
    return float(correct / total), int(total), int(ties)


def rank_loss(model, text, left, right, signs):
    score_left = model(text[left], torch.empty(0))[:, 3]
    score_right = model(text[right], torch.empty(0))[:, 3]
    return -F.logsigmoid(signs * (score_left - score_right)).mean()


def train(kind, text, target, raw_interest, indices, seed, epochs, monitor=None):
    model = make_model(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(text[indices], target[indices]), batch_size=HP["batch_size"], shuffle=True, generator=generator, num_workers=0)
    pair_left, pair_right, pair_sign = build_pairs(indices, raw_interest)
    best, best_epoch, stale, history = float("inf"), 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        rng = np.random.default_rng((seed * 100003 + epoch) % (2**32))
        total, total_mse, total_rank, batches = 0.0, 0.0, 0.0, 0
        for text_batch, target_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(text_batch, torch.empty(0))
            mse_all = F.mse_loss(prediction, target_batch)
            if kind == "r0_mse":
                loss, sampled_rank = mse_all, torch.zeros((), dtype=prediction.dtype)
            else:
                selection = rng.choice(len(pair_left), size=PAIR_BATCH_SIZE, replace=len(pair_left) < PAIR_BATCH_SIZE)
                left, right, signs = pair_left[selection].copy(), pair_right[selection].copy(), pair_sign[selection].copy()
                swap = rng.integers(0, 2, size=PAIR_BATCH_SIZE).astype(bool)
                left[swap], right[swap], signs[swap] = right[swap], left[swap], -signs[swap]
                sampled_rank = rank_loss(model, text, torch.from_numpy(left), torch.from_numpy(right), torch.from_numpy(signs))
                if kind == "r1_mse_rank":
                    loss = mse_all + RANK_WEIGHT * sampled_rank
                else:
                    mse_first_three = F.mse_loss(prediction[:, :3], target_batch[:, :3])
                    loss = mse_first_three + RANK_WEIGHT * sampled_rank
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite {kind} loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(text_batch)
            total_mse += mse_all.item() * len(text_batch)
            total_rank += sampled_rank.item()
            batches += 1
        row = {"model": kind, "seed": seed, "phase": "selection" if monitor is not None else "refit", "epoch": epoch,
               "train_loss": total / len(indices), "train_attribute_mse": total_mse / len(indices), "train_rank_loss": total_rank / batches}
        if monitor is not None:
            model.eval()
            with torch.inference_mode():
                inner_mse = F.mse_loss(model(text[monitor], torch.empty(0)), target[monitor]).item()
            row["inner_val_macro_mse"] = inner_mse
            if inner_mse < best:
                best, best_epoch, stale = inner_mse, epoch, 0
            else:
                stale += 1
        history.append(row)
        if epoch == 1 or epoch % 10 == 0:
            print(f"{kind} seed={seed} {row['phase']} epoch={epoch} mse={row['train_attribute_mse']:.6f} rank={row['train_rank_loss']:.6f}", flush=True)
        if monitor is not None and stale >= HP["patience"]:
            break
    return model, best_epoch if monitor is not None else epochs, history


def paired_bootstrap(predictions, target, raw_target):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    interest_values, high_pair_values, aggregate_values = [], [], []
    for _ in range(BOOTSTRAP_SAMPLES):
        indices = rng.integers(0, len(target), size=len(target))
        each_interest, each_pair, each_aggregate = [], [], []
        for seed in SEEDS:
            r0, r1 = predictions["r0_mse"][seed][indices], predictions["r1_mse_rank"][seed][indices]
            y, raw = target[indices], raw_target[indices]
            each_interest.append(metrics(r1, y)["content_interest"]["srcc"] - metrics(r0, y)["content_interest"]["srcc"])
            each_pair.append(pairwise_accuracy(r1[:, 3], raw, 2)[0] - pairwise_accuracy(r0[:, 3], raw, 2)[0])
            each_aggregate.append(aggregate_metrics(r1, y)["srcc"] - aggregate_metrics(r0, y)["srcc"])
        interest_values.append(float(np.mean(each_interest)))
        high_pair_values.append(float(np.mean(each_pair)))
        aggregate_values.append(float(np.mean(each_aggregate)))
    def output(values):
        return {"mean": float(np.mean(values)), "ci95_percentile": [float(np.quantile(values, .025)), float(np.quantile(values, .975))],
                "positive_fraction": float(np.mean(np.asarray(values) > 0))}
    return {"method": "paired nonparametric bootstrap over 123 validation videos; mean delta across three fixed training seeds",
            "repetitions": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED,
            "r1_minus_r0": {"interest_srcc": output(interest_values), "pair_accuracy_delta_ge_2": output(high_pair_values), "aggregate_srcc": output(aggregate_values)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s0-dir", type=Path, required=True)
    parser.add_argument("--s1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/scientific_s2_interest_rank")
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
    s1 = json.loads((args.s1_dir / "config.json").read_text(encoding="utf-8"))
    if s0["split_hash"] != SPLIT_HASH or s1["split_hash"] != SPLIT_HASH or s0["features"] != FEATURES:
        raise ValueError("S0/S1 fixed protocol mismatch")
    metadata, feature_dir, reference = map(Path, [s0["metadata"], s0["feature_dir"], s0["reference_checkpoint"]])
    protected = {str(path.resolve()): sha(path) for path in [metadata, reference]}
    manifest = pd.read_csv(args.s0_dir / "split_manifest.csv", dtype={"video_id": str})
    data = load_metadata(metadata)
    data = data[data.video_id.isin(set(manifest.video_id))].copy().reset_index(drop=True)
    train_idx, val_idx = train_test_split(np.arange(len(data)), test_size=.2, random_state=42, stratify=data.label)
    inner_train, inner_val = train_test_split(train_idx, test_size=HP["inner_val_ratio"], random_state=HP["inner_split_seed"], stratify=data.iloc[train_idx].label)
    if len(data) != 615 or (len(train_idx), len(val_idx), len(inner_train), len(inner_val)) != (492, 123, 393, 99):
        raise ValueError("Unexpected S0 split")
    if set(data.iloc[val_idx].video_id) != set(manifest.loc[manifest.split == "validation", "video_id"]):
        raise ValueError("Validation membership mismatch")
    values = []
    for video_id in data.video_id:
        path = feature_dir / f"{video_id}.pt"
        protected[str(path.resolve())] = sha(path)
        sample = load_pt(path)
        value = np.concatenate([np.asarray(sample[name], dtype=np.float32) for name in FEATURES])
        if value.shape != (1540,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid text/knowledge feature: {path}")
        values.append(value)
    text = torch.from_numpy(np.stack(values))
    target = torch.from_numpy(data[ATTRS].to_numpy(dtype=np.float32) / 5.0)
    raw_interest = data.content_interest.to_numpy(dtype=np.int64)
    train_stats, validation_stats = pair_statistics(train_idx, raw_interest), pair_statistics(val_idx, raw_interest)
    write_json(out / "train_pair_statistics.json", train_stats)
    write_json(out / "validation_pair_statistics.json", validation_stats)
    params = sum(item.numel() for item in TextOnlyFourAdapters().parameters())
    if params != 374116:
        raise ValueError("Model differs from S1-B")
    config = {"experiment": "S2-R", "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(), "s0_dir": str(args.s0_dir.resolve()), "s1_dir": str(args.s1_dir.resolve()),
              "metadata": str(metadata.resolve()), "feature_dir": str(feature_dir.resolve()), "reference_checkpoint": str(reference.resolve()), "split_hash": SPLIT_HASH,
              "seeds": SEEDS, "features": FEATURES, "input_dim": 1540, "architecture": "S1-B exact 1540->224; four 224->32->1 adapters", "params": params,
              "rank_attribute": "content_interest", "pair_source": "training labels only", "pair_threshold_raw": PAIR_THRESHOLD_RAW, "pair_batch_size": PAIR_BATCH_SIZE,
              "rank_weight": RANK_WEIGHT, "r2_loss": "mean MSE for first three attributes + 0.1 RankNet for interest", "hyperparameters": HP,
              "selection": "fixed 393/99 minimum four-attribute MSE; refit all 492; official 123 only", "bootstrap": {"repetitions": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED},
              "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "script_sha256": sha(__file__)}
    write_json(out / "config.json", config)
    official_target, official_raw = target[val_idx].numpy(), raw_interest[val_idx]
    predictions, all_rows, histories, prediction_rows, pair_rows = {kind: {} for kind in MODELS}, [], [], [], []
    for seed in SEEDS:
        seed_dir = out / f"seed_{seed}"
        seed_dir.mkdir()
        for kind in MODELS:
            _, selected_epoch, selection_history = train(kind, text, target, raw_interest, inner_train, seed, HP["max_epochs"], inner_val)
            model, _, refit_history = train(kind, text, target, raw_interest, train_idx, seed, selected_epoch)
            model.eval()
            with torch.inference_mode():
                prediction = model(text[val_idx], torch.empty(0)).numpy()
            result, aggregate = metrics(prediction, official_target), aggregate_metrics(prediction, official_target)
            acc_all, n_all, ties_all = pairwise_accuracy(prediction[:, 3], official_raw, 1)
            acc_high, n_high, ties_high = pairwise_accuracy(prediction[:, 3], official_raw, 2)
            predictions[kind][seed] = prediction
            write_json(seed_dir / f"{kind}_metrics.json", {"seed": seed, "model": LABELS[kind], "selected_epoch": selected_epoch, "split_hash": SPLIT_HASH,
                                                              "attributes": result, "aggregate": aggregate,
                                                              "pairwise_accuracy": {"delta_ge_1": acc_all, "delta_ge_2": acc_high, "pairs_ge_1": n_all, "pairs_ge_2": n_high, "ties": {"all": ties_all, "high": ties_high}}})
            torch.save({"model_state_dict": model.state_dict(), "model": kind, "seed": seed, "selected_epoch": selected_epoch, "split_hash": SPLIT_HASH, "config": config},
                       out / "checkpoints" / f"{kind}_seed_{seed}.pt")
            all_rows.extend({"model": kind, "model_label": LABELS[kind], "seed": seed, "attribute": attr, **values} for attr, values in result.items())
            histories.extend(selection_history + refit_history)
            pair_rows.append({"model": kind, "model_label": LABELS[kind], "seed": seed, "pair_acc_delta_ge_1": acc_all, "pair_acc_delta_ge_2": acc_high,
                              "n_delta_ge_1": n_all, "n_delta_ge_2": n_high, "prediction_ties_all": ties_all, "prediction_ties_high": ties_high})
            for position, video_id in enumerate(data.iloc[val_idx].video_id):
                row = {"video_id": video_id, "seed": seed, "model": kind}
                for index, attr in enumerate(ATTRS):
                    row[f"true_{attr}"] = official_target[position, index]
                    row[f"pred_{attr}"] = prediction[position, index]
                prediction_rows.append(row)
            print(f"completed {kind} seed={seed} epoch={selected_epoch} interest_srcc={result['content_interest']['srcc']:.4f} pair_acc_high={acc_high:.4f}", flush=True)
    metric_frame = pd.DataFrame(all_rows)
    pair_frame = pd.DataFrame(pair_rows)
    metric_frame.to_csv(out / "metrics_all_seeds.csv", index=False, encoding="utf-8-sig")
    pair_frame.to_csv(out / "pairwise_accuracy_all_seeds.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(histories).to_csv(out / "training_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(out / "predictions_by_model_seed.csv", index=False, encoding="utf-8-sig")
    grouped_mean = metric_frame.groupby(["model", "attribute"])[["srcc", "plcc", "mae", "mse"]].mean()
    grouped_std = metric_frame.groupby(["model", "attribute"])[["srcc", "plcc", "mae", "mse"]].std()
    interest_summary = pd.DataFrame([{"Model": LABELS[kind], **{metric.upper(): f"{grouped_mean.loc[(kind, 'content_interest'), metric]:.4f} ± {grouped_std.loc[(kind, 'content_interest'), metric]:.4f}" for metric in ("srcc", "plcc", "mae", "mse")}}
                                     for kind in MODELS])
    interest_summary.to_csv(out / "interest_metrics.csv", index=False, encoding="utf-8-sig")
    pair_summary = pair_frame.groupby("model")[["pair_acc_delta_ge_1", "pair_acc_delta_ge_2"]].agg(["mean", "std"])
    pair_table = pd.DataFrame([{"Model": LABELS[kind], "Pair Acc Δ>=1": f"{pair_summary.loc[kind, ('pair_acc_delta_ge_1', 'mean')]:.4f} ± {pair_summary.loc[kind, ('pair_acc_delta_ge_1', 'std')]:.4f}",
                                "Pair Acc Δ>=2": f"{pair_summary.loc[kind, ('pair_acc_delta_ge_2', 'mean')]:.4f} ± {pair_summary.loc[kind, ('pair_acc_delta_ge_2', 'std')]:.4f}"} for kind in MODELS])
    pair_table.to_csv(out / "pairwise_accuracy.csv", index=False, encoding="utf-8-sig")
    aggregate_frame = pd.DataFrame([{"model": kind, "model_label": LABELS[kind], "seed": seed, **aggregate_metrics(predictions[kind][seed], official_target)} for kind in MODELS for seed in SEEDS])
    aggregate_frame.to_csv(out / "aggregate_metrics_all_seeds.csv", index=False, encoding="utf-8-sig")
    aggregate_summary = aggregate_frame.groupby("model")[["srcc", "plcc"]].agg(["mean", "std"])
    macro_aggregate = pd.DataFrame([{"Model": LABELS[kind],
                                     "Macro SRCC": f"{grouped_mean.loc[(kind, 'Macro'), 'srcc']:.4f} ± {grouped_std.loc[(kind, 'Macro'), 'srcc']:.4f}",
                                     "Macro PLCC": f"{grouped_mean.loc[(kind, 'Macro'), 'plcc']:.4f} ± {grouped_std.loc[(kind, 'Macro'), 'plcc']:.4f}",
                                     "Aggregate SRCC": f"{aggregate_summary.loc[kind, ('srcc', 'mean')]:.4f} ± {aggregate_summary.loc[kind, ('srcc', 'std')]:.4f}",
                                     "Aggregate PLCC": f"{aggregate_summary.loc[kind, ('plcc', 'mean')]:.4f} ± {aggregate_summary.loc[kind, ('plcc', 'std')]:.4f}"} for kind in MODELS])
    macro_aggregate.to_csv(out / "aggregate_results.csv", index=False, encoding="utf-8-sig")
    per_seed = pd.DataFrame({"seed": SEEDS, "r0_mse": [metrics(predictions['r0_mse'][seed], official_target)["content_interest"]["srcc"] for seed in SEEDS],
                             "r1_mse_rank": [metrics(predictions['r1_mse_rank'][seed], official_target)["content_interest"]["srcc"] for seed in SEEDS],
                             "r2_rank_only": [metrics(predictions['r2_rank_only'][seed], official_target)["content_interest"]["srcc"] for seed in SEEDS]})
    per_seed["r1_minus_r0"] = per_seed.r1_mse_rank - per_seed.r0_mse
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
    boot = paired_bootstrap(predictions, official_target, official_raw)
    write_json(out / "bootstrap_results.json", boot)
    s1_metrics = pd.read_csv(args.s1_dir / "metrics_all_seeds.csv")
    s1_adapter = s1_metrics[s1_metrics.model == "adapter"].set_index(["seed", "attribute"])[["srcc", "plcc", "mae", "mse"]].sort_index()
    r0 = metric_frame[metric_frame.model == "r0_mse"].set_index(["seed", "attribute"])[["srcc", "plcc", "mae", "mse"]].sort_index()
    reproduction = (r0 - s1_adapter).reset_index()
    reproduction.to_csv(out / "s1_r0_reproduction_delta.csv", index=False, encoding="utf-8-sig")
    reproduction_error = float(np.abs(reproduction[["srcc", "plcc", "mae", "mse"]].to_numpy()).max())
    if reproduction_error > 1e-10:
        raise RuntimeError(f"R0 does not reproduce S1-B: {reproduction_error}")
    unchanged = all(sha(path) == digest for path, digest in protected.items())
    if not unchanged:
        raise RuntimeError("Source input changed during S2-R")
    r0_interest, r1_interest, r2_interest = (grouped_mean.loc[(kind, "content_interest"), "srcc"] for kind in MODELS)
    r1_high_gain = pair_frame[pair_frame.model == "r1_mse_rank"].pair_acc_delta_ge_2.mean() - pair_frame[pair_frame.model == "r0_mse"].pair_acc_delta_ge_2.mean()
    r1_positive = int((per_seed.r1_minus_r0 > 0).sum())
    macro_ok = grouped_mean.loc[("r1_mse_rank", "Macro"), "srcc"] >= grouped_mean.loc[("r0_mse", "Macro"), "srcc"] - .005
    aggregate_ok = aggregate_frame[aggregate_frame.model == "r1_mse_rank"].srcc.mean() >= aggregate_frame[aggregate_frame.model == "r0_mse"].srcc.mean() - .005
    mse_ok = grouped_mean.loc[("r1_mse_rank", "content_interest"), "mse"] <= grouped_mean.loc[("r0_mse", "content_interest"), "mse"] + .002
    if r1_interest > r0_interest and r1_positive >= 2 and r1_high_gain > 0 and macro_ok and aggregate_ok and mse_ok:
        decision = "A" if boot["r1_minus_r0"]["interest_srcc"]["ci95_percentile"][0] > 0 else "B"
    elif r2_interest > max(r0_interest, r1_interest):
        decision = "C"
    else:
        decision = "D"
    summary = ["# S2-R Pairwise Interest Ranking Summary", "", "## 1. Experiment Setup", "",
               f"615 videos; fixed 492/123 split; seeds={SEEDS}; S1-B exact Four-Adapter and 1540D text/knowledge inputs.",
               "", "## 2. Motivation", "", "Tests whether pointwise interest regression benefits from a fixed high-confidence human-label RankNet auxiliary loss, without any new modality or label.",
               "", "## 3. Pair Construction", "", "Training pairs use only training content_interest labels with |Δ raw score|>=2; every pointwise batch samples 32 pairs afresh and randomly swaps order.",
               "", table(pd.DataFrame([train_stats, validation_stats], index=["train", "validation"]).reset_index().rename(columns={"index": "split"})),
               "", "## 4. Model and Loss", "", f"R0: four-attribute MSE. R1: R0 loss + {RANK_WEIGHT}×Interest RankNet. R2: first-three-attribute MSE + {RANK_WEIGHT}×Interest RankNet. Parameter count for every model: {params}.",
               "", "## 5. Pair Statistics", "", "Validation pair ties count as 0.5 correct. Validation pairs are evaluation only.",
               "", "## 6. Content Interest Results", "", table(interest_summary), "", "## 7. Pairwise Accuracy", "", table(pair_table),
               "", "## 8. Multi-seed Results", "", table(per_seed), "", "## 9. Macro and Scientific Aggregate", "", table(macro_aggregate),
               "", "## 10. Paired Bootstrap", "", json.dumps(boot, ensure_ascii=False, indent=2),
               "", "## 11. Key Findings", "",
               f"1. R1-R0 Interest SRCC={r1_interest-r0_interest:+.4f}; positive in {r1_positive}/3 seeds.",
               f"2. R1-R0 high-confidence Pair Accuracy={r1_high_gain:+.4f}.",
               f"3. R1 Macro/aggregate hold conditions={macro_ok}/{aggregate_ok}; Interest MSE hold condition={mse_ok}.",
               f"4. R2 Interest SRCC={r2_interest:.4f}; it is {'higher' if r2_interest > max(r0_interest, r1_interest) else 'not higher'} than both R0 and R1.",
               f"5. Official decision: Case {decision}.",
               "", "## 12. Decision", "",
               {"A": "Adopt Interest MSE+RankNet as the Scientific Branch interest loss for non-visual S3 validation.",
                "B": "Record RankNet as a ranking-oriented auxiliary variant; require S3 confirmation before replacing the MSE-only branch.",
                "C": "RankNet-only is a research finding only; retain MSE-only as formal configuration pending dedicated stability validation.",
                "D": "Retain MSE-only Four-Adapter and do not introduce LLM pairwise teachers; proceed to non-visual S3."}[decision]]
    (out / "S2R_SUMMARY.md").write_text("\n\n".join(summary) + "\n", encoding="utf-8")
    write_json(out / "integrity_check.json", {"status": "passed", "source_files": len(protected), "source_sha256": protected})
    write_json(out / "completion.json", {"status": "completed", "elapsed_seconds": time.perf_counter() - started, "split_hash": SPLIT_HASH, "source_integrity": unchanged,
                                            "decision_case": decision, "r0_reproduction_max_error": reproduction_error, "params": params})
    print(f"Completed: {out}", flush=True)


if __name__ == "__main__":
    main()
