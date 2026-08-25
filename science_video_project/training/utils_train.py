import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def build_model_config_from_cfg(cfg: Any) -> dict[str, Any]:
    """Return the MVP model arguments that must stay in sync across train/infer."""
    return {
        "text_dim": int(cfg.text_dim),
        "video_dim": int(cfg.video_dim),
        "audio_dim": int(cfg.audio_dim),
        "meta_dim": int(cfg.meta_dim),
        "aes_dim": int(cfg.aes_dim),
        "dnsmos_dim": int(cfg.dnsmos_dim),
        "wpm_dim": int(cfg.wpm_dim),
        "rhythm_dim": int(cfg.speech_rhythm_dim),
        "llm_knowledge_dim": int(cfg.llm_knowledge_dim),
        "llm_analysis_dim": int(cfg.llm_analysis_dim),
        "sci_hand_dim": int(cfg.sci_hand_dim),
        "cover_dim": int(cfg.cover_dim),
        "use_cover_features": bool(cfg.use_cover_features),
        "fusion_mode": str(cfg.fusion_mode),
        "branch_weight_floor": float(getattr(cfg, "branch_weight_floor", 0.0)),
        "technical_feature_mode": str(getattr(cfg, "technical_feature_mode", "full")),
        "use_knowledge_gate": bool(cfg.use_knowledge_gate),
        "science_fusion_mode": str(getattr(cfg, "science_fusion_mode", "concat")),
        "hidden_dim": int(cfg.hidden_dim),
        "use_cross_gating": bool(cfg.use_cross_gating),
        "cross_gating_dropout": float(cfg.cross_gating_dropout),
    }


def merge_checkpoint_model_config(ckpt: Any, default_config: dict[str, Any]) -> dict[str, Any]:
    """Use saved config when present, falling back to current CFG for old checkpoints."""
    merged = dict(default_config)
    saved_config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if isinstance(saved_config, dict):
        merged.update(saved_config)
        if "science_fusion_mode" not in saved_config:
            merged["science_fusion_mode"] = (
                "ifg" if saved_config.get("use_knowledge_gate", True) else "concat"
            )
        if "technical_feature_mode" not in saved_config:
            training_config = ckpt.get("training_config", {})
            merged["technical_feature_mode"] = str(
                training_config.get("technical_feature_mode", "full")
            )
    return merged


def adapt_checkpoint_state_dict(
    state_dict: dict[str, torch.Tensor],
    model_state_dict: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Adapt older MVP checkpoints without adding research-only parameters."""
    adapted = dict(state_dict)

    hidden_dim = 128
    for key in (
        "aesthetic_branch.video_proj.net.0.weight",
        "aesthetic_branch.text_proj.net.0.weight",
        "fusion.0.weight",
    ):
        if key in adapted:
            hidden_dim = adapted[key].shape[0]
            break

    has_gate = any(k.startswith("gate.") for k in state_dict)
    legacy_fusion_key = "fusion.0.weight"
    if (
        not has_gate
        and legacy_fusion_key in adapted
        and adapted[legacy_fusion_key].ndim == 2
        and adapted[legacy_fusion_key].shape[1] == hidden_dim * 3 + 3
    ):
        print("  [adapt] Detected legacy checkpoint without gate; converting fusion weights.")
        old_w = adapted[legacy_fusion_key]
        old_b = adapted["fusion.0.bias"]

        w_hidden = old_w[:, : hidden_dim * 3]
        w_hidden_avg = w_hidden.view(hidden_dim, 3, -1).mean(dim=1)
        w_scores = old_w[:, hidden_dim * 3 :]
        adapted[legacy_fusion_key] = torch.cat([w_hidden_avg, w_scores], dim=1)
        adapted["fusion.0.bias"] = old_b

        gate_0_w = torch.zeros(hidden_dim, hidden_dim * 3)
        gate_0_w[:, :hidden_dim] = 0.1
        gate_0_w[:, hidden_dim : hidden_dim * 2] = 0.1
        gate_0_w[:, hidden_dim * 2 :] = 0.1
        adapted["gate.0.weight"] = gate_0_w
        adapted["gate.0.bias"] = torch.zeros(hidden_dim)
        adapted["gate.3.weight"] = torch.zeros(3, hidden_dim)
        adapted["gate.3.bias"] = torch.zeros(3)
        print("  [adapt] Gate module converted.")

    has_aes_proj = any(k.startswith("aesthetic_branch.aes_proj.") for k in state_dict)
    if not has_aes_proj and "aesthetic_branch.fuse.net.0.weight" in adapted:
        print("  [adapt] Initializing missing aes_proj weights.")
        adapted["aesthetic_branch.aes_proj.net.0.weight"] = torch.randn(hidden_dim, 7) * 0.02
        adapted["aesthetic_branch.aes_proj.net.0.bias"] = torch.zeros(hidden_dim)
        adapted["aesthetic_branch.aes_proj.net.3.weight"] = torch.randn(hidden_dim, hidden_dim) * 0.02
        adapted["aesthetic_branch.aes_proj.net.3.bias"] = torch.zeros(hidden_dim)

        old_fuse_w = adapted["aesthetic_branch.fuse.net.0.weight"]
        new_fuse_w = torch.zeros(hidden_dim, hidden_dim * 4)
        new_fuse_w[:, : old_fuse_w.shape[1]] = old_fuse_w
        adapted["aesthetic_branch.fuse.net.0.weight"] = new_fuse_w
        print("  [adapt] aes_proj initialized.")

    if model_state_dict is not None:
        adapted = {
            k: v
            for k, v in adapted.items()
            if k in model_state_dict and tuple(v.shape) == tuple(model_state_dict[k].shape)
        }

    print("  [adapt] Checkpoint adaptation complete.")
    return adapted


def build_logger(log_file: Path, name: str = "train") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
