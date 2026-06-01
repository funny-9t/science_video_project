import logging
import random
from pathlib import Path

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
    return {k: v.to(device) for k, v in batch.items()}


def adapt_checkpoint_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """将旧版 checkpoint 适配到当前模型架构。

    旧版 checkpoint 可能缺少:
      1. gate 模块 (gate.0.weight/bias, gate.2.weight/bias, gate.3.weight/bias)
      2. aesthetic_branch.aes_proj 模块 (此前未使用 CLIP prompt scoring)
      3. fusion.0 输入维度: 387 → 131

    Returns:
        适配后的 state_dict。
    """
    adapted = dict(state_dict)

    # --- 推断 hidden_dim ---
    # 从 aesthetic_branch.video_proj.net.0.weight 或 fusion.0.weight 推断
    for key in ("aesthetic_branch.video_proj.net.0.weight",
                "aesthetic_branch.text_proj.net.0.weight",
                "fusion.0.weight"):
        if key in adapted:
            hidden_dim = adapted[key].shape[0]
            break
    else:
        hidden_dim = 128

    # --- 1) 适配 gate 模块 ---
    has_gate = any(k.startswith("gate.") for k in state_dict)
    if not has_gate:
        print("  [adapt] Detected legacy checkpoint (no gate) — converting...")
        old_w = adapted["fusion.0.weight"]
        old_b = adapted["fusion.0.bias"]

        w_hidden = old_w[:, :384]                         # [H, 384]
        w_hidden_avg = w_hidden.view(hidden_dim, 3, -1).mean(dim=1)  # [H, H]
        w_scores = old_w[:, 384:]                         # [H, 3]
        adapted["fusion.0.weight"] = torch.cat([w_hidden_avg, w_scores], dim=1)  # [H, H+3]
        adapted["fusion.0.bias"] = old_b

        gate_0_w = torch.zeros(hidden_dim, 384)
        gate_0_w[:, 32:96] = 0.1
        gate_0_w[:, 160:224] = 0.1
        gate_0_w[:, 288:352] = 0.1
        adapted["gate.0.weight"] = gate_0_w
        adapted["gate.0.bias"] = torch.zeros(hidden_dim)
        adapted["gate.3.weight"] = torch.zeros(3, hidden_dim)
        adapted["gate.3.bias"] = torch.zeros(3)
        print("  [adapt] ✓ Gate module converted successfully")

    # --- 2) 适配 aesthetic_branch.aes_proj —— 若 checkpoint 中缺失，随机初始化 ---
    has_aes_proj = any(k.startswith("aesthetic_branch.aes_proj.") for k in state_dict)
    if not has_aes_proj:
        print("  [adapt] Initializing missing aes_proj weights...")
        # aes_proj = MLPBlock(aes_dim=7, hidden_dim, hidden_dim)
        # net.0: Linear(7, hidden_dim)
        adapted["aesthetic_branch.aes_proj.net.0.weight"] = torch.randn(hidden_dim, 7) * 0.02
        adapted["aesthetic_branch.aes_proj.net.0.bias"] = torch.zeros(hidden_dim)
        # net.3: Linear(hidden_dim, hidden_dim)  (net.2 is Dropout, no weights)
        adapted["aesthetic_branch.aes_proj.net.3.weight"] = torch.randn(hidden_dim, hidden_dim) * 0.02
        adapted["aesthetic_branch.aes_proj.net.3.bias"] = torch.zeros(hidden_dim)

        # 扩展 aesthetic_branch.fuse.net.0.weight: [H, 384] → [H, 512]
        # 新增的 aes_proj 输出占最后 128 维，用零初始化
        old_fuse_w = adapted["aesthetic_branch.fuse.net.0.weight"]  # [128, 384]
        new_fuse_w = torch.zeros(hidden_dim, hidden_dim * 4)         # [128, 512]
        new_fuse_w[:, :384] = old_fuse_w
        adapted["aesthetic_branch.fuse.net.0.weight"] = new_fuse_w
        print("  [adapt] ✓ aes_proj initialized, fuse weight expanded 384→512")

    missing_keys = [k for k in adapted.keys() if k not in state_dict]
    if missing_keys:
        print(f"  [adapt] Added {len(missing_keys)} new keys: {missing_keys[:5]}...")

    print("  [adapt] ✓ Checkpoint adaptation complete")
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
