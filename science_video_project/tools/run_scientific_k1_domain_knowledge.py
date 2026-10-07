"""K1: fixed-protocol LLM-generated domain knowledge validation.

This runner deliberately isolates the K1 comparison from the historical
1540-D Scientific Branch cache.  K0 uses only the cached RoBERTa video-text
CLS feature; K1 appends a separately generated, schema-validated knowledge
card.  The generation and training phases are separate so model fitting never
issues an API request.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import sys
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
from tools.run_scientific_s0 import ATTRS, HP, SEEDS, SPLIT_HASH, metrics, sha, table, write_json

TEXT_DIM = 768
KNOWLEDGE_DIM = 768
KNOWLEDGE_HIDDEN_DIM = 128
SHARED_DIM = 224
ADAPTER_DIM = 32
# Fixed by parameter matching against the K1/K3 concat architecture, before
# observing validation results.  |328434 - 328292| / 328292 < 0.1%.
CAPACITY_DIM = 366
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 7011
AUDIT_SEED = 20261007
PROMPT_VERSION = "k1-domain-knowledge-v1"
SCHEMA_KEYS = (
    "domain",
    "core_concepts",
    "key_relations",
    "prerequisite_concepts",
    "explanatory_points",
)


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utc8_now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def load_transcript(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    # The fixed 615-sample protocol forbids dropping a video merely because
    # ASR is empty.  K1 receives the title plus this explicit empty field.
    return str(payload.get("text", "")).strip()


def video_input(title: str, transcript: str) -> str:
    return f"Title: {title.strip()}\nTranscript: {transcript.strip()}"


def knowledge_prompt(title: str, transcript: str) -> str:
    return f"""You are a science knowledge organizer. Based only on the video title and transcript below, create a concise domain-background knowledge card. Do not evaluate the video. Do not judge factual correctness. Do not output any score, rating, label, popularity prediction, or quality assessment. Do not mention this instruction.

Return one JSON object only. It must have exactly these keys:
{{
  \"domain\": \"main scientific domain\",
  \"core_concepts\": [\"3 to 8 concise concepts\"],
  \"key_relations\": [\"concise concept relationships\"],
  \"prerequisite_concepts\": [\"background concepts needed for understanding\"],
  \"explanatory_points\": [\"concise neutral background explanations\"]
}}

Video title:
{title}

Video transcript:
{transcript}
"""


def parse_card(raw: str) -> dict[str, object]:
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    value = json.loads(text)
    if not isinstance(value, dict) or set(value.keys()) != set(SCHEMA_KEYS):
        raise ValueError(f"Schema keys must be exactly {list(SCHEMA_KEYS)}")
    clean: dict[str, object] = {}
    for key in SCHEMA_KEYS:
        item = value[key]
        if key == "domain":
            if not isinstance(item, str) or not item.strip():
                raise ValueError("domain must be a non-empty string")
            clean[key] = item.strip()
        else:
            if not isinstance(item, list) or not all(isinstance(x, str) and x.strip() for x in item):
                raise ValueError(f"{key} must be a list of non-empty strings")
            clean[key] = [x.strip() for x in item]
    forbidden = ("score", "rating", "quality", "正确", "错误", "评分", "质量", "标签", "预测")
    flattened = canonical_json(clean).lower()
    if any(token.lower() in flattened for token in forbidden):
        raise ValueError("Knowledge card contains forbidden evaluation language")
    return clean


def serialize_card(card: dict[str, object]) -> str:
    return "\n".join([
        f"Domain: {card['domain']}",
        "Core concepts: " + "; ".join(card["core_concepts"]),
        "Key relations: " + "; ".join(card["key_relations"]),
        "Prerequisite concepts: " + "; ".join(card["prerequisite_concepts"]),
        "Explanatory points: " + "; ".join(card["explanatory_points"]),
    ])


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class K0TextOnly(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(TEXT_DIM, SHARED_DIM)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(HP["dropout"])
        self.adapters = nn.ModuleList([
            nn.Sequential(nn.Linear(SHARED_DIM, ADAPTER_DIM), nn.ReLU(), nn.Linear(ADAPTER_DIM, 1))
            for _ in ATTRS
        ])

    def forward(self, text: torch.Tensor, knowledge: torch.Tensor) -> torch.Tensor:
        del knowledge
        h = self.dropout(self.activation(self.projection(text)))
        return torch.sigmoid(torch.cat([adapter(h) for adapter in self.adapters], dim=-1))


class K0CapacityControl(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(TEXT_DIM, CAPACITY_DIM)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(HP["dropout"])
        self.adapters = nn.ModuleList([
            nn.Sequential(nn.Linear(CAPACITY_DIM, ADAPTER_DIM), nn.ReLU(), nn.Linear(ADAPTER_DIM, 1))
            for _ in ATTRS
        ])

    def forward(self, text: torch.Tensor, knowledge: torch.Tensor) -> torch.Tensor:
        del knowledge
        h = self.dropout(self.activation(self.projection(text)))
        return torch.sigmoid(torch.cat([adapter(h) for adapter in self.adapters], dim=-1))


class K1KnowledgeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.knowledge_projection = nn.Linear(KNOWLEDGE_DIM, KNOWLEDGE_HIDDEN_DIM)
        self.projection = nn.Linear(TEXT_DIM + KNOWLEDGE_HIDDEN_DIM, SHARED_DIM)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(HP["dropout"])
        self.adapters = nn.ModuleList([
            nn.Sequential(nn.Linear(SHARED_DIM, ADAPTER_DIM), nn.ReLU(), nn.Linear(ADAPTER_DIM, 1))
            for _ in ATTRS
        ])

    def forward(self, text: torch.Tensor, knowledge: torch.Tensor) -> torch.Tensor:
        h_knowledge = self.activation(self.knowledge_projection(knowledge))
        h = self.dropout(self.activation(self.projection(torch.cat([text, h_knowledge], dim=-1))))
        return torch.sigmoid(torch.cat([adapter(h) for adapter in self.adapters], dim=-1))


MODEL_TYPES = {
    "k0": K0TextOnly,
    "k0c": K0CapacityControl,
    "k1": K1KnowledgeModel,
}
MODEL_LABELS = {
    "k0": "K0 Text-only Four-Adapter",
    "k0c": "K0-Capacity Text-only Four-Adapter",
    "k1": "K1 Text + LLM-generated Knowledge",
}


def make_model(kind: str, seed: int) -> nn.Module:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return MODEL_TYPES[kind]()


def split_data(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    train_idx, val_idx = train_test_split(np.arange(len(data)), test_size=0.2, random_state=42,
                                         stratify=data.label)
    payload = "\n".join(["train:" + value for value in sorted(data.iloc[train_idx].video_id)] +
                          ["val:" + value for value in sorted(data.iloc[val_idx].video_id)])
    split_hash = digest_text(payload)
    if split_hash != SPLIT_HASH or (len(train_idx), len(val_idx)) != (492, 123):
        raise ValueError(f"Fixed split mismatch: {split_hash}, sizes={len(train_idx)}/{len(val_idx)}")
    inner_train, inner_val = train_test_split(train_idx, test_size=HP["inner_val_ratio"],
                                              random_state=HP["inner_split_seed"],
                                              stratify=data.iloc[train_idx].label)
    return train_idx, val_idx, inner_train, inner_val, split_hash


def build_dataset(metadata: Path, feature_dir: Path, transcript_dir: Path) -> tuple[pd.DataFrame, list[str], torch.Tensor, torch.Tensor]:
    data = load_metadata(metadata)
    available = {path.stem for path in feature_dir.glob("*.pt")}
    data = data[data.video_id.isin(available)].copy().reset_index(drop=True)
    if len(data) != 615 or data.video_id.nunique() != 615:
        raise ValueError(f"Expected exactly 615 feature-matched videos, got {len(data)}")
    scores = data[ATTRS].apply(pd.to_numeric, errors="raise")
    if scores.isna().any().any() or not np.isfinite(scores.to_numpy()).all() or not ((scores >= 1) & (scores <= 5)).all().all():
        raise ValueError("Scientific labels must be finite human scores in [1, 5]")
    if not np.allclose(scores.mean(axis=1).to_numpy(dtype=np.float32) / 5.0,
                       data.sci_target.to_numpy(dtype=np.float32), atol=1e-7, rtol=0):
        raise ValueError("sci_target is not the required four-attribute mean")
    transcripts, feature_rows = [], []
    for row in data.itertuples(index=False):
        transcript_path = transcript_dir / f"{row.video_id}.json"
        transcripts.append(load_transcript(transcript_path))
        sample = load_pt(feature_dir / f"{row.video_id}.pt")
        vector = np.asarray(sample["text_feat"], dtype=np.float32)
        if str(sample["video_id"]) != row.video_id or vector.shape != (TEXT_DIM,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid text_feat for {row.video_id}")
        feature_rows.append(vector)
    return data, transcripts, torch.from_numpy(np.stack(feature_rows)), torch.from_numpy(scores.to_numpy(dtype=np.float32) / 5.0)


def write_preflight(out: Path, data: pd.DataFrame, transcripts: list[str], metadata: Path, feature_dir: Path,
                    transcript_dir: Path, text: torch.Tensor, target: torch.Tensor) -> dict[str, object]:
    train_idx, val_idx, inner_train, inner_val, split_hash = split_data(data)
    manifest = data[["video_id", "label", *ATTRS, "sci_target"]].copy()
    manifest["split"] = "train"
    manifest.loc[val_idx, "split"] = "validation"
    manifest["inner_split"] = "not_used_for_selection"
    manifest.loc[inner_train, "inner_split"] = "inner_train"
    manifest.loc[inner_val, "inner_split"] = "inner_validation"
    manifest.to_csv(out / "split_manifest.csv", index=False, encoding="utf-8-sig")
    samples = pd.DataFrame({"video_id": data.video_id, "title": data.title.fillna(""),
                            "transcript_sha256": [digest_text(x) for x in transcripts],
                            "transcript_is_empty": [not bool(x) for x in transcripts],
                            "input_sha256": [digest_text(video_input(str(title), transcript)) for title, transcript in zip(data.title.fillna(""), transcripts)]})
    samples.to_csv(out / "generation_input_manifest.csv", index=False, encoding="utf-8-sig")
    audit = """# K1 Input Feature Audit

| Feature | Source | Dim | K0 | K0C | K1 | K2 | K3 |
|---|---|---:|---|---|---|---|---|
| RoBERTa video-text CLS | cached `text_feat`; title + tags + ASR subtitle | 768 | yes | yes | yes | yes | yes |
| Historical DeepSeek reasoning / analysis | old scientific-audit cache | 768 | no | no | no | no | no |
| Historical DeepSeek score vector | old scientific-audit cache | 4 | no | no | no | no | no |
| K1 generated knowledge card CLS | new title + ASR-only, schema-validated cache | 768 -> 128 | no | no | yes | no | no |
| Retrieved knowledge | frozen external corpus | 768 -> 128 | no | no | no | pending | pending |
| Organized retrieved knowledge | retrieval + schema cache | 768 -> 128 | no | no | no | no | pending |

`text_feat` is a local Chinese-RoBERTa CLS representation produced by the pipeline from title, tags and ASR text. The K1 generator receives only title and ASR text: no tags, labels, ranking information, model prediction, or validation membership. Empty ASR is retained as an explicit empty field so every one of the fixed 615 videos remains in the comparison. The prior 1540-D cache is never used because it contains old LLM audit features and scores.
"""
    (out / "input_feature_audit.md").write_text(audit, encoding="utf-8")
    schema = {"prompt_version": PROMPT_VERSION, "schema": {"domain": "string", "core_concepts": ["string"],
              "key_relations": ["string"], "prerequisite_concepts": ["string"], "explanatory_points": ["string"]},
              "forbidden_outputs": ["quality score", "correctness score", "factuality score", "labels", "predictions"]}
    write_json(out / "llm_generation_schema.json", schema)
    params = {name: parameter_count(cls()) for name, cls in MODEL_TYPES.items()}
    if abs(params["k0c"] - params["k1"]) / params["k1"] > .1:
        raise ValueError("K0C parameter matching failed")
    config = {
        "experiment": "K1 Domain Knowledge Enhancement Validation (K0/K0C/K1 stage)",
        "created_at": utc8_now(), "metadata": str(metadata.resolve()), "feature_dir": str(feature_dir.resolve()),
        "transcript_dir": str(transcript_dir.resolve()), "samples": 615, "split_seed": 42,
        "split_hash": split_hash, "train_size": len(train_idx), "validation_size": len(val_idx),
        "inner_train_size": len(inner_train), "inner_validation_size": len(inner_val), "attributes": ATTRS,
        "text_feature": {"name": "text_feat", "dim": TEXT_DIM, "source": "local Chinese-RoBERTa video-text CLS"},
        "knowledge_feature": {"dim": KNOWLEDGE_DIM, "projection_dim": KNOWLEDGE_HIDDEN_DIM,
                              "source": "new K1 knowledge-card CLS only"},
        "models": {name: {"label": MODEL_LABELS[name], "parameters": params[name]} for name in MODEL_TYPES},
        "hyperparameters": HP, "selection": "minimum inner four-attribute mean MSE; patience 10; reset and refit all 492",
        "target_scaling": "raw score / 5", "training_api_requests": 0, "bootstrap": {"repetitions": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED},
    }
    write_json(out / "experiment_manifest.json", config)
    write_json(out / "preflight_integrity.json", {"text_shape": list(text.shape), "target_shape": list(target.shape),
               "split_hash": split_hash, "input_count": len(samples), "empty_transcript_count": int(samples.transcript_is_empty.sum()),
               "k0c_match_ratio": abs(params["k0c"] - params["k1"]) / params["k1"]})
    return {"train_idx": train_idx, "val_idx": val_idx, "inner_train": inner_train, "inner_val": inner_val,
            "split_hash": split_hash, "params": params}


def generate_one(client: object, model: str, row: pd.Series, transcript: str) -> dict[str, object]:
    title = str(row.title or "")
    source = video_input(title, transcript)
    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": knowledge_prompt(title, transcript)}],
                                              temperature=0, response_format={"type": "json_object"}, max_tokens=900)
    raw = str(response.choices[0].message.content or "")
    card = parse_card(raw)
    return {"video_id": str(row.video_id), "title": title, "input_sha256": digest_text(source),
            "prompt_version": PROMPT_VERSION, "model": model, "card": card, "serialized_text": serialize_card(card),
            "card_sha256": digest_text(canonical_json(card))}


def generate_cards(out: Path, data: pd.DataFrame, transcripts: list[str], api_key: str, model: str, workers: int) -> Path:
    cache_path = out / "k1_generated_knowledge_cache.jsonl"
    if cache_path.exists():
        raise FileExistsError(f"Refusing to alter existing K1 cache: {cache_path}")
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    results: dict[str, dict[str, object]] = {}
    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(generate_one, client, model, row, transcript): str(row.video_id)
                   for (_, row), transcript in zip(data.iterrows(), transcripts)}
        for number, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            video_id = futures[future]
            try:
                results[video_id] = future.result()
            except Exception as error:  # Cache must be all-or-nothing.
                errors.append(f"{video_id}: {type(error).__name__}: {error}")
            if number == 1 or number % 25 == 0 or number == len(futures):
                print(f"K1 generation {number}/{len(futures)}; failures={len(errors)}", flush=True)
    if errors:
        raise RuntimeError("K1 generation did not produce a complete valid cache:\n" + "\n".join(errors[:20]))
    with cache_path.open("w", encoding="utf-8", newline="\n") as handle:
        for video_id in data.video_id:
            handle.write(canonical_json(results[str(video_id)]) + "\n")
    cache_hash = sha(cache_path)
    write_json(out / "k1_cache_hash.json", {"path": str(cache_path.resolve()), "sha256": cache_hash, "records": len(results),
               "prompt_version": PROMPT_VERSION, "model": model, "generated_at": utc8_now()})
    return cache_path


def load_cards(cache_path: Path, data: pd.DataFrame, transcripts: list[str]) -> list[str]:
    records = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {str(item["video_id"]): item for item in records}
    if len(records) != 615 or len(by_id) != 615:
        raise ValueError("K1 cache must contain exactly 615 unique records")
    cards = []
    for (_, row), transcript in zip(data.iterrows(), transcripts):
        item = by_id.get(str(row.video_id))
        if item is None or item.get("prompt_version") != PROMPT_VERSION:
            raise ValueError(f"Missing/incorrect K1 card for {row.video_id}")
        if item.get("input_sha256") != digest_text(video_input(str(row.title or ""), transcript)):
            raise ValueError(f"K1 input hash mismatch for {row.video_id}")
        card = parse_card(canonical_json(item["card"]))
        serialized = serialize_card(card)
        if item.get("serialized_text") != serialized:
            raise ValueError(f"K1 serialized text mismatch for {row.video_id}")
        cards.append(serialized)
    return cards


def encode_cards(out: Path, cache_path: Path, data: pd.DataFrame, transcripts: list[str], model_path: Path, batch_size: int) -> Path:
    destination = out / "k1_knowledge_embeddings.pt"
    if destination.exists():
        raise FileExistsError(f"Refusing to alter existing embedding cache: {destination}")
    from transformers import AutoModel, AutoTokenizer
    cards = load_cards(cache_path, data, transcripts)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True).eval()
    all_vectors = []
    with torch.inference_mode():
        for start in range(0, len(cards), batch_size):
            encoded = tokenizer(cards[start:start + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=512)
            vectors = model(**encoded).last_hidden_state[:, 0].detach().cpu().float()
            all_vectors.append(vectors)
            if start == 0 or start + batch_size >= len(cards):
                print(f"K1 encoding {min(start + batch_size, len(cards))}/{len(cards)}", flush=True)
    embeddings = torch.cat(all_vectors)
    if tuple(embeddings.shape) != (615, KNOWLEDGE_DIM) or not torch.isfinite(embeddings).all():
        raise ValueError(f"Invalid K1 embedding tensor: {tuple(embeddings.shape)}")
    torch.save({"video_ids": data.video_id.tolist(), "embeddings": embeddings, "cache_sha256": sha(cache_path),
                "encoder": str(model_path.resolve()), "max_length": 512, "batch_size": batch_size}, destination)
    write_json(out / "k1_embedding_cache_hash.json", {"path": str(destination.resolve()), "sha256": sha(destination),
               "source_cache_sha256": sha(cache_path), "shape": list(embeddings.shape)})
    return destination


def train_one(kind: str, text: torch.Tensor, knowledge: torch.Tensor, target: torch.Tensor, train_idx: np.ndarray,
              monitor_idx: np.ndarray | None, seed: int, epochs: int) -> tuple[nn.Module, int, list[dict[str, object]]]:
    model = make_model(kind, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(text[train_idx], knowledge[train_idx], target[train_idx]), batch_size=HP["batch_size"],
                        shuffle=True, generator=generator, num_workers=0)
    best, best_epoch, stale, history = float("inf"), 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x_text, x_knowledge, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x_text, x_knowledge), y_batch)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite {kind} loss")
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(y_batch)
        item: dict[str, object] = {"model": kind, "seed": seed, "phase": "selection" if monitor_idx is not None else "refit",
                                   "epoch": epoch, "train_mse": total / len(train_idx)}
        if monitor_idx is not None:
            model.eval()
            with torch.inference_mode():
                inner_mse = float(nn.functional.mse_loss(model(text[monitor_idx], knowledge[monitor_idx]), target[monitor_idx]).item())
            item["inner_val_mse"] = inner_mse
            if inner_mse < best:
                best, best_epoch, stale = inner_mse, epoch, 0
            else:
                stale += 1
        history.append(item)
        if epoch == 1 or epoch % 10 == 0:
            print(f"{kind} seed={seed} {item['phase']} epoch={epoch} mse={item['train_mse']:.6f}", flush=True)
        if monitor_idx is not None and stale >= HP["patience"]:
            break
    return model, best_epoch if monitor_idx is not None else epochs, history


def aggregate(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred_mean, target_mean = prediction.mean(axis=1), target.mean(axis=1)
    return {"srcc": float(spearmanr(pred_mean, target_mean).statistic), "plcc": float(pearsonr(pred_mean, target_mean).statistic),
            "mae": float(np.abs(pred_mean - target_mean).mean()), "mse": float(np.square(pred_mean - target_mean).mean())}


def bootstrap(predictions: dict[str, dict[int, np.ndarray]], target: np.ndarray) -> dict[str, object]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = {"macro_srcc": [], "aggregate_srcc": []}
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = rng.integers(0, len(target), size=len(target))
        per_seed = {key: [] for key in values}
        for seed in SEEDS:
            candidate, baseline, y = predictions["k1"][seed][sample], predictions["k0"][seed][sample], target[sample]
            per_seed["macro_srcc"].append(metrics(candidate, y)["Macro"]["srcc"] - metrics(baseline, y)["Macro"]["srcc"])
            per_seed["aggregate_srcc"].append(aggregate(candidate, y)["srcc"] - aggregate(baseline, y)["srcc"])
        for key in values:
            values[key].append(float(np.mean(per_seed[key])))
    return {key: {"mean": float(np.mean(value)), "ci95_percentile": [float(np.quantile(value, .025)), float(np.quantile(value, .975))],
                  "positive_fraction": float(np.mean(np.asarray(value) > 0))} for key, value in values.items()}


def train_models(out: Path, data: pd.DataFrame, text: torch.Tensor, target: torch.Tensor, embedding_path: Path,
                 protocol: dict[str, object]) -> None:
    payload = torch.load(embedding_path, map_location="cpu", weights_only=False)
    if payload["video_ids"] != data.video_id.tolist():
        raise ValueError("K1 embedding cache video ordering mismatch")
    knowledge = payload["embeddings"].float()
    if tuple(knowledge.shape) != (615, KNOWLEDGE_DIM) or not torch.isfinite(knowledge).all():
        raise ValueError("Invalid K1 embeddings")
    train_idx, val_idx = protocol["train_idx"], protocol["val_idx"]
    predictions: dict[str, dict[int, np.ndarray]] = {kind: {} for kind in MODEL_TYPES}
    metric_rows, epoch_rows, history = [], [], []
    for kind in MODEL_TYPES:
        for seed in SEEDS:
            _, selected_epoch, selected_history = train_one(kind, text, knowledge, target, protocol["inner_train"], protocol["inner_val"], seed, HP["max_epochs"])
            fitted, _, refit_history = train_one(kind, text, knowledge, target, train_idx, None, seed, selected_epoch)
            fitted.eval()
            with torch.inference_mode():
                prediction = fitted(text[val_idx], knowledge[val_idx]).numpy()
            result = metrics(prediction, target[val_idx].numpy())
            predictions[kind][seed] = prediction
            metric_rows.extend({"model": kind, "model_label": MODEL_LABELS[kind], "seed": seed, "attribute": attr, **score}
                               for attr, score in result.items())
            epoch_rows.append({"model": kind, "seed": seed, "selected_epoch": selected_epoch})
            history.extend(selected_history + refit_history)
            frame = pd.DataFrame({"video_id": data.iloc[val_idx].video_id.to_numpy()})
            for index, attr in enumerate(ATTRS):
                frame[attr + "_target"] = target[val_idx, index].numpy()
                frame[attr + "_pred"] = prediction[:, index]
            frame.to_csv(out / f"{kind}_predictions_seed{seed}.csv", index=False, encoding="utf-8-sig")
            torch.save({"model_state_dict": fitted.state_dict(), "model": kind, "seed": seed, "selected_epoch": selected_epoch,
                        "split_hash": protocol["split_hash"], "knowledge_embedding_sha256": sha(embedding_path)},
                       out / "checkpoints" / f"{kind}_seed_{seed}.pt")
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(out / "metrics_all_seeds.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(epoch_rows).to_csv(out / "selected_epochs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False, encoding="utf-8-sig")
    summaries, aggregate_rows = [], []
    for kind in MODEL_TYPES:
        subset = metrics_frame[metrics_frame.model == kind]
        for attr in ATTRS + ["Macro"]:
            row = subset[subset.attribute == attr]
            summaries.append({"model": kind, "model_label": MODEL_LABELS[kind], "attribute": attr,
                              **{key: float(row[key].mean()) for key in ("srcc", "plcc", "mae", "mse")},
                              **{key + "_std": float(row[key].std(ddof=1)) for key in ("srcc", "plcc", "mae", "mse")}})
        values = [aggregate(predictions[kind][seed], target[val_idx].numpy()) for seed in SEEDS]
        aggregate_rows.append({"model": kind, "model_label": MODEL_LABELS[kind],
                               **{key: float(np.mean([v[key] for v in values])) for key in values[0]},
                               **{key + "_std": float(np.std([v[key] for v in values], ddof=1)) for key in values[0]}})
    summary_frame, aggregate_frame = pd.DataFrame(summaries), pd.DataFrame(aggregate_rows)
    summary_frame.to_csv(out / "attribute_metrics_summary.csv", index=False, encoding="utf-8-sig")
    aggregate_frame.to_csv(out / "scientific_aggregate_summary.csv", index=False, encoding="utf-8-sig")
    boot = bootstrap(predictions, target[val_idx].numpy())
    write_json(out / "paired_bootstrap_k1_minus_k0.json", {"method": "paired nonparametric bootstrap on the fixed 123 validation videos; mean delta over three training seeds",
               "repetitions": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "comparison": "K1 - K0", "results": boot})
    k0 = summary_frame.query("model == 'k0' and attribute == 'Macro'").iloc[0]
    k1 = summary_frame.query("model == 'k1' and attribute == 'Macro'").iloc[0]
    positive_seeds = int(sum(metrics(predictions["k1"][seed], target[val_idx].numpy())["Macro"]["srcc"] >
                             metrics(predictions["k0"][seed], target[val_idx].numpy())["Macro"]["srcc"] for seed in SEEDS))
    conclusion = "provisional_positive" if k1.srcc > k0.srcc and positive_seeds >= 2 else "no_stable_k1_increment"
    report = ["# K1 Domain Knowledge Enhancement Validation", "", "## Scope", "", "This completed stage evaluates K0, K0-Capacity, and K1 only. K2/K3 require a separately frozen external scientific corpus and are not substituted with generated content.",
              "", "## Model Parameters", "", table(pd.DataFrame([{"model": MODEL_LABELS[name], "parameters": count} for name, count in protocol["params"].items()])),
              "", "## Attribute Metrics", "", table(summary_frame), "", "## Scientific Aggregate", "", table(aggregate_frame), "", "## K1 - K0 Bootstrap", "", "```json", json.dumps(boot, ensure_ascii=False, indent=2), "```",
              "", "## Preliminary Decision", "", f"K1 decision: `{conclusion}`. Macro SRCC K0={k0.srcc:.4f}, K1={k1.srcc:.4f}; positive seeds={positive_seeds}/3. This is a fixed-split controlled comparison, not an independent held-out test."]
    (out / "K1_SUMMARY.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(out / "completion.json", {"status": "completed_k0_k0c_k1", "k1_decision": conclusion, "split_hash": protocol["split_hash"],
               "cache_sha256": sha(out / "k1_generated_knowledge_cache.jsonl"), "embedding_sha256": sha(embedding_path),
               "training_api_requests": 0, "k2_k3_status": "not run: external corpus not yet frozen"})


def write_audit_sample(out: Path, data: pd.DataFrame, cache_path: Path) -> None:
    records = {str(json.loads(line)["video_id"]): json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    rng = np.random.default_rng(AUDIT_SEED)
    selected = rng.choice(data.index.to_numpy(), size=30, replace=False)
    rows = []
    for index in selected:
        row = data.loc[index]
        item = records[str(row.video_id)]
        rows.append({"video_id": row.video_id, "title": row.title, "domain": item["card"]["domain"],
                     "knowledge_card": item["serialized_text"], "manual_topic_relevance": "pending", "manual_unsupported_expansion": "pending"})
    pd.DataFrame(rows).to_csv(out / "k1_knowledge_quality_audit_sample.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--transcript-dir", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs" / "scientific_k1_domain_knowledge")
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--api-model", default="deepseek-v4-pro")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--encoding-batch-size", type=int, default=8)
    parser.add_argument("--phase", choices=("preflight", "generate", "encode", "train", "all"), default="all")
    parser.add_argument("--resume-preflight", action="store_true",
                        help="Reuse only a failed pre-API preflight directory; never permits cache/checkpoint mutation.")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        allowed = {"checkpoints", "split_manifest.csv", "generation_input_manifest.csv", "input_feature_audit.md",
                   "llm_generation_schema.json", "experiment_manifest.json", "preflight_integrity.json"}
        existing = {path.name for path in out.iterdir()}
        if not args.resume_preflight or not existing.issubset(allowed) or any((out / "checkpoints").iterdir()):
            raise FileExistsError(f"Output already exists; do not mutate a frozen experiment: {out}")
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out / "checkpoints").mkdir()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    data, transcripts, text, target = build_dataset(args.metadata, args.feature_dir, args.transcript_dir)
    protocol = write_preflight(out, data, transcripts, args.metadata, args.feature_dir, args.transcript_dir, text, target)
    if args.phase == "preflight":
        print(f"Preflight completed: {out}")
        return
    if not args.api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required to generate the new K1 card cache")
    cache_path = generate_cards(out, data, transcripts, args.api_key, args.api_model, args.workers)
    write_audit_sample(out, data, cache_path)
    if args.phase == "generate":
        print(f"K1 generation completed: {cache_path}")
        return
    embedding_path = encode_cards(out, cache_path, data, transcripts, args.text_model, args.encoding_batch_size)
    if args.phase == "encode":
        print(f"K1 encoding completed: {embedding_path}")
        return
    train_models(out, data, text, target, embedding_path, protocol)
    print(f"K1 stage completed: {out}")


if __name__ == "__main__":
    main()
