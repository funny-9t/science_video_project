"""
§13 自动消融实验系统

按 new_prompt.md 要求生成 7 组实验:

  Exp1 Baseline          — 原始 MVP (use_mvp_loss)
  Exp2 +QualityHead      — 启用 QualityHead
  Exp3 +Consistency Loss — 启用一致性损失
  Exp4 +Temporal Encoder — 时序编码器
  Exp5 +CrossModal Attn  — 跨模态注意力
  Exp6 +Science Features — 手工科学性特征
  Exp7 Full Model        — 全模块启用

输出:
  - experiments/results.csv (各实验指标汇总)
  - experiments/exp_N/    (各实验 checkpoint)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_CONFIGS: list[dict] = [
    # (name, args_dict)
    {
        "name": "Exp1_Baseline",
        "args": {
            "--use_mvp_loss": "",
            "--quality_head": "0", "--consistency_loss": "0",
            "--engagement_branch": "0", "--science_features": "0",
            "--aesthetic_mlp": "0", "--temporal_encoder": "0",
            "--cross_modal_attn": "0", "--diversity_loss": "0",
        },
    },
    {
        "name": "Exp2_QualityHead",
        "args": {
            "--quality_head": "1", "--consistency_loss": "0",
            "--engagement_branch": "0", "--science_features": "0",
            "--aesthetic_mlp": "0", "--temporal_encoder": "0",
            "--cross_modal_attn": "0", "--diversity_loss": "0",
        },
    },
    {
        "name": "Exp3_Consistency",
        "args": {
            "--quality_head": "1", "--consistency_loss": "1",
            "--engagement_branch": "0", "--science_features": "0",
            "--aesthetic_mlp": "0", "--temporal_encoder": "0",
            "--cross_modal_attn": "0", "--diversity_loss": "0",
        },
    },
    {
        "name": "Exp4_Temporal",
        "args": {
            "--quality_head": "1", "--consistency_loss": "1",
            "--engagement_branch": "0", "--science_features": "0",
            "--aesthetic_mlp": "0", "--temporal_encoder": "1",
            "--cross_modal_attn": "0", "--diversity_loss": "0",
        },
    },
    {
        "name": "Exp5_CrossModal",
        "args": {
            "--quality_head": "1", "--consistency_loss": "1",
            "--engagement_branch": "0", "--science_features": "0",
            "--aesthetic_mlp": "0", "--temporal_encoder": "1",
            "--cross_modal_attn": "1", "--diversity_loss": "0",
        },
    },
    {
        "name": "Exp6_ScienceFeat",
        "args": {
            "--quality_head": "1", "--consistency_loss": "1",
            "--engagement_branch": "0", "--science_features": "1",
            "--aesthetic_mlp": "0", "--temporal_encoder": "1",
            "--cross_modal_attn": "1", "--diversity_loss": "0",
        },
    },
    {
        "name": "Exp7_Full",
        "args": {
            "--quality_head": "1", "--consistency_loss": "1",
            "--engagement_branch": "1", "--science_features": "1",
            "--aesthetic_mlp": "1", "--temporal_encoder": "1",
            "--cross_modal_attn": "1", "--diversity_loss": "1",
        },
    },
]


def run_experiments(
    metadata: str | None = None,
    feature_dir: str | None = None,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """运行全部消融实验并汇总结果。

    Returns:
        DataFrame with columns: experiment, rank_acc, acc, f1, auc
    """

    from pipeline.config import CFG

    train_script = str(PROJECT_ROOT / "training" / "train_research.py")
    results = []

    for exp in EXPERIMENT_CONFIGS:
        exp_name = exp["name"]
        exp_checkpoint = str(CFG.checkpoint_dir / f"{exp_name}.pt")
        exp_log = str(CFG.log_dir / f"{exp_name}.log")

        if skip_existing and Path(exp_checkpoint).exists():
            print(f"[SKIP] {exp_name} — checkpoint already exists")
            continue

        print(f"\n{'='*60}")
        print(f"[RUNNING] {exp_name}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, train_script,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--lr", str(lr),
            "--checkpoint", exp_checkpoint,
        ]
        if metadata:
            cmd += ["--metadata", metadata]
        if feature_dir:
            cmd += ["--feature_dir", feature_dir]

        for flag, value in exp["args"].items():
            if value == "1":
                cmd.append(flag)
            elif value == "0":
                pass  # 默认已关闭
            elif value:
                cmd += [flag, value]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            if result.returncode != 0:
                print(f"[ERROR] {exp_name} failed:\n{result.stderr[-500:]}")
                results.append({"experiment": exp_name, "rank_acc": None, "acc": None, "f1": None, "auc": None, "status": "FAILED"})
            else:
                results.append({"experiment": exp_name, "rank_acc": None, "acc": None, "f1": None, "auc": None, "status": "DONE"})
        except Exception as e:
            print(f"[ERROR] {exp_name}: {e}")
            results.append({"experiment": exp_name, "rank_acc": None, "acc": None, "f1": None, "auc": None, "status": f"ERROR: {e}"})

    df = pd.DataFrame(results)
    out_path = CFG.output_dir / "experiment_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nExperiment results saved to {out_path}")
    return df


def print_experiment_summary():
    """打印实验设计概览。"""
    print("\n" + "=" * 60)
    print("  Ablation Experiment Matrix (§13 of new_prompt.md)")
    print("=" * 60)
    components = ["QualityHead", "Consistency", "Temporal", "CrossModal", "ScienceFeat", "Engagement", "Diversity"]
    header = f"{'Experiment':<20} " + " ".join(f"{c:<12}" for c in components)
    print(header)
    print("-" * len(header))

    for i, exp in enumerate(EXPERIMENT_CONFIGS):
        args = exp["args"]
        row = f"{exp['name']:<20} "
        flags = [
            args.get("--quality_head", "0") == "1",
            args.get("--consistency_loss", "0") == "1",
            args.get("--temporal_encoder", "0") == "1",
            args.get("--cross_modal_attn", "0") == "1",
            args.get("--science_features", "0") == "1",
            args.get("--engagement_branch", "0") == "1",
            args.get("--diversity_loss", "0") == "1",
        ]
        row += " ".join(f"{'✓':<12}" if f else f"{'—':<12}" for f in flags)
        print(row)
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--print_only", action="store_true", help="Just print experiment matrix")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no_skip", action="store_true", help="Don't skip existing checkpoints")
    args = parser.parse_args()

    print_experiment_summary()

    if not args.print_only:
        run_experiments(epochs=args.epochs, batch_size=args.batch_size,
                       lr=args.lr, skip_existing=not args.no_skip)
