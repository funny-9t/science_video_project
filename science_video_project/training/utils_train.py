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

    适配内容:
      1. gate 模块缺失 → 自动初始化
      2. aesthetic_branch.aes_proj 缺失 → 随机初始化
      3. fusion.0 输入维度扩展 (387 → 131)
      4. 研究版新增模块缺失 → 零/随机初始化 (quality_head, cross_modal_attn,
         sci_feat_proj, aesthetic_mlp, temporal_encoder, engagement_branch)

    Returns:
        适配后的 state_dict。
    """
    adapted = dict(state_dict)

    # --- 推断 hidden_dim ---
    hidden_dim = 128
    for key in ("aesthetic_branch.video_proj.net.0.weight",
                "aesthetic_branch.text_proj.net.0.weight",
                "fusion.0.weight"):
        if key in adapted:
            hidden_dim = adapted[key].shape[0]
            break

    # --- 1) 适配 gate 模块 ---
    has_gate = any(k.startswith("gate.") for k in state_dict)
    if not has_gate:
        print("  [adapt] Detected legacy checkpoint (no gate) — converting...")
        old_w = adapted["fusion.0.weight"]
        old_b = adapted["fusion.0.bias"]

        w_hidden = old_w[:, :384]
        w_hidden_avg = w_hidden.view(hidden_dim, 3, -1).mean(dim=1)
        w_scores = old_w[:, 384:]
        adapted["fusion.0.weight"] = torch.cat([w_hidden_avg, w_scores], dim=1)
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

    # --- 2) 适配 aesthetic_branch.aes_proj ---
    has_aes_proj = any(k.startswith("aesthetic_branch.aes_proj.") for k in state_dict)
    if not has_aes_proj:
        print("  [adapt] Initializing missing aes_proj weights...")
        adapted["aesthetic_branch.aes_proj.net.0.weight"] = torch.randn(hidden_dim, 7) * 0.02
        adapted["aesthetic_branch.aes_proj.net.0.bias"] = torch.zeros(hidden_dim)
        adapted["aesthetic_branch.aes_proj.net.3.weight"] = torch.randn(hidden_dim, hidden_dim) * 0.02
        adapted["aesthetic_branch.aes_proj.net.3.bias"] = torch.zeros(hidden_dim)

        old_fuse_w = adapted["aesthetic_branch.fuse.net.0.weight"]
        new_fuse_w = torch.zeros(hidden_dim, hidden_dim * 4)
        new_fuse_w[:, :384] = old_fuse_w
        adapted["aesthetic_branch.fuse.net.0.weight"] = new_fuse_w
        print("  [adapt] ✓ aes_proj initialized, fuse weight expanded 384→512")

    # --- 3) 研究版新模块自适应初始化 (ResearchModel 新增) ---

    # quality_head
    has_qh = any(k.startswith("quality_head.") for k in state_dict)
    if not has_qh:
        print("  [adapt] Initializing quality_head (3→32→16→1)...")
        adapted["quality_head.net.0.weight"] = torch.randn(32, 3) * 0.02
        adapted["quality_head.net.0.bias"] = torch.zeros(32)
        adapted["quality_head.net.3.weight"] = torch.randn(16, 32) * 0.02
        adapted["quality_head.net.3.bias"] = torch.zeros(16)
        adapted["quality_head.net.6.weight"] = torch.randn(1, 16) * 0.02
        adapted["quality_head.net.6.bias"] = torch.zeros(1)

    # cross_modal_attn
    has_cm = any(k.startswith("cross_modal_attn.") for k in state_dict)
    if not has_cm:
        print("  [adapt] Initializing cross_modal_attn (random init)...")

    # sci_feat_proj
    has_sf = any(k.startswith("sci_feat_proj.") for k in state_dict)
    if not has_sf:
        print("  [adapt] Initializing sci_feat_proj (768+5→768)...")
        adapted["sci_feat_proj.fuse.0.weight"] = torch.randn(hidden_dim, 773) * 0.02
        adapted["sci_feat_proj.fuse.0.bias"] = torch.zeros(hidden_dim)
        adapted["sci_feat_proj.fuse.3.weight"] = torch.randn(768, hidden_dim) * 0.02
        adapted["sci_feat_proj.fuse.3.bias"] = torch.zeros(768)

    # aesthetic_mlp
    has_am = any(k.startswith("aesthetic_mlp.") for k in state_dict)
    if not has_am:
        print("  [adapt] Initializing aesthetic_mlp (7→128→64→1)...")
        adapted["aesthetic_mlp.net.0.weight"] = torch.randn(hidden_dim, 7) * 0.02
        adapted["aesthetic_mlp.net.0.bias"] = torch.zeros(hidden_dim)
        adapted["aesthetic_mlp.net.3.weight"] = torch.randn(hidden_dim // 2, hidden_dim) * 0.02
        adapted["aesthetic_mlp.net.3.bias"] = torch.zeros(hidden_dim // 2)
        adapted["aesthetic_mlp.net.6.weight"] = torch.randn(1, hidden_dim // 2) * 0.02
        adapted["aesthetic_mlp.net.6.bias"] = torch.zeros(1)

    # engagement_branch
    has_eg = any(k.startswith("engagement_branch.") for k in state_dict)
    if not has_eg:
        print("  [adapt] Initializing engagement_branch...")
        adapted["engagement_branch.mlp.net.0.weight"] = torch.randn(hidden_dim, 16) * 0.02
        adapted["engagement_branch.mlp.net.0.bias"] = torch.zeros(hidden_dim)
        adapted["engagement_branch.mlp.net.3.weight"] = torch.randn(hidden_dim, hidden_dim) * 0.02
        adapted["engagement_branch.mlp.net.3.bias"] = torch.zeros(hidden_dim)
        adapted["engagement_branch.score_head.0.weight"] = torch.randn(64, hidden_dim) * 0.02
        adapted["engagement_branch.score_head.0.bias"] = torch.zeros(64)
        adapted["engagement_branch.score_head.3.weight"] = torch.randn(1, 64) * 0.02
        adapted["engagement_branch.score_head.3.bias"] = torch.zeros(1)

    # temporal_encoder
    has_te = any(k.startswith("temporal_encoder.") for k in state_dict)
    if not has_te:
        print("  [adapt] Temporal encoder weights will be randomly initialized on first use")

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
