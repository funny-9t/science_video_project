from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from pipeline.step_meta import MetaFeatureBuilder


@dataclass
class ScientificBranchTextSample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor | None
    meta_feat: torch.Tensor
    label: torch.Tensor


class ScientificBranchTextDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
        subtitle_dir: str | Path | None = None,
        text_fields: tuple[str, ...] = ("title", "tags"),
        meta_dim: int = 16,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.subtitle_dir = Path(subtitle_dir) if subtitle_dir else None
        self.text_fields = tuple(text_fields)

        self.meta_builder = MetaFeatureBuilder(categories=self.df["category"].tolist(), out_dim=int(meta_dim))

    def __len__(self) -> int:
        return int(len(self.df))

    def _load_subtitle_text(self, video_id: str) -> str:
        if self.subtitle_dir is None:
            return ""
        p = self.subtitle_dir / f"{video_id}.txt"
        if not p.exists():
            return ""
        try:
            return p.read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    def _build_text(self, row: dict) -> str:
        parts: list[str] = []
        for f in self.text_fields:
            v = str(row.get(f, "") or "").strip()
            if v:
                parts.append(v)

        subtitle = self._load_subtitle_text(str(row.get("video_id", "") or ""))
        if subtitle:
            parts.append(subtitle)

        return " ".join(parts).strip()

    def __getitem__(self, idx: int) -> ScientificBranchTextSample:
        row = self.df.iloc[int(idx)].to_dict()
        text = self._build_text(row)

        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # squeeze batch dim (1, L) -> (L,)
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        token_type_ids = tokens.get("token_type_ids")
        token_type_ids = token_type_ids.squeeze(0) if token_type_ids is not None else None

        meta_feat = torch.tensor(self.meta_builder.build(row), dtype=torch.float32)
        label = torch.tensor(int(row.get("label", 0)), dtype=torch.float32)

        return ScientificBranchTextSample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            meta_feat=meta_feat,
            label=label,
        )


def collate_scientificbranch_text(
    batch: list[ScientificBranchTextSample],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    # Pad dynamically using tokenizer.pad.
    features = []
    for x in batch:
        item = {
            "input_ids": x.input_ids,
            "attention_mask": x.attention_mask,
        }
        if x.token_type_ids is not None:
            item["token_type_ids"] = x.token_type_ids
        features.append(item)

    padded = tokenizer.pad(features, padding=True, return_tensors="pt")

    meta_feat = torch.stack([x.meta_feat for x in batch], dim=0)
    labels = torch.stack([x.label for x in batch], dim=0)

    out: dict[str, torch.Tensor] = {
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"],
        "meta_feat": meta_feat,
        "labels": labels,
    }
    if "token_type_ids" in padded:
        out["token_type_ids"] = padded["token_type_ids"]
    return out
