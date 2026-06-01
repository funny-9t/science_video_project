import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from pipeline.utils_io import ensure_dir
from training.model_mvp import ScientificBranch


class SciBERTScientificBranchModel(nn.Module):
    """SciBERT encoder + existing ScientificBranch MLP.

    Goal: train a standalone 'scientificbranch' scorer using text + meta.

    Inputs:
      - tokenized text (input_ids/attention_mask[/token_type_ids])
      - meta_feat (B, meta_dim)

    Outputs:
      - logits (B,) for BCEWithLogitsLoss
      - prob (B,) sigmoid(logits)
    """

    def __init__(
        self,
        encoder_name_or_path: str,
        meta_dim: int,
        hidden_dim: int = 128,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.encoder_name_or_path = str(encoder_name_or_path)
        self.local_files_only = bool(local_files_only)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.encoder_name_or_path,
            local_files_only=self.local_files_only,
        )
        self.encoder = AutoModel.from_pretrained(
            self.encoder_name_or_path,
            local_files_only=self.local_files_only,
        )

        text_dim = int(getattr(self.encoder.config, "hidden_size", 768))
        self.scientific_branch = ScientificBranch(text_dim=text_dim, meta_dim=int(meta_dim), hidden_dim=int(hidden_dim))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        meta_feat: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        enc_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            enc_kwargs["token_type_ids"] = token_type_ids

        out = self.encoder(**enc_kwargs)
        cls_feat = out.last_hidden_state[:, 0, :]
        _, sci_score = self.scientific_branch(cls_feat, meta_feat)

        logits = sci_score.squeeze(-1)
        prob = torch.sigmoid(logits)
        return {
            "logits": logits,
            "prob": prob,
        }

    @torch.no_grad()
    def encode_text_cls(
        self,
        text: str,
        meta_feat: torch.Tensor,
        device: torch.device,
        max_length: int = 256,
    ) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=int(max_length),
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        meta_feat = meta_feat.to(device)
        return self.forward(meta_feat=meta_feat, **tokens)

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        ensure_dir(out_dir)

        # Save encoder + tokenizer in HuggingFace format.
        self.encoder.save_pretrained(out_dir / "encoder")
        self.tokenizer.save_pretrained(out_dir / "encoder")

        # Save MLP branch weights.
        torch.save(self.scientific_branch.state_dict(), out_dir / "scientific_branch.pt")

        meta = {
            "encoder_name_or_path": self.encoder_name_or_path,
            "local_files_only": self.local_files_only,
            "text_hidden_size": int(getattr(self.encoder.config, "hidden_size", 768)),
            "meta": {
                "meta_dim": int(self.scientific_branch.meta_proj.net[0].in_features),
                "hidden_dim": int(self.scientific_branch.text_proj.net[0].out_features),
            },
        }
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, model_dir: str | Path, device: str | torch.device = "cpu") -> "SciBERTScientificBranchModel":
        model_dir = Path(model_dir)
        with open(model_dir / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        model = cls(
            encoder_name_or_path=str(model_dir / "encoder"),
            meta_dim=int(meta["meta"]["meta_dim"]),
            hidden_dim=int(meta["meta"]["hidden_dim"]),
            local_files_only=True,
        )
        state = torch.load(model_dir / "scientific_branch.pt", map_location="cpu", weights_only=False)
        model.scientific_branch.load_state_dict(state)
        model.to(torch.device(device))
        return model
