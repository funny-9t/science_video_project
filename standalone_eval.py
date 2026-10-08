"""
================================================================================
 科普短视频字幕 — 独立无参考科学性评价脚本
================================================================================

基于 DeepSeek V4 Pro 两阶段结构化 Prompt，对科普字幕文本进行
"事实一致性 (Factual Consistency)" 和 "逻辑连贯性 (Logical Coherence)"
的独立无参考评价。

使用方法:
  1. 批量评价 (从 CSV/JSON 文件):
     python standalone_eval.py --input subtitles.csv --output results.json

  2. 单条评价:
     python standalone_eval.py --text "地球绕太阳公转一周需要365天..."

  3. 交互模式:
     python standalone_eval.py --interactive

环境变量:
  DEEPSEEK_API_KEY: DeepSeek API 密钥（必须设置，否则使用 Dummy 模式）

输出格式 (JSON):
  {
    "text": "...",
    "factual_consistency": 0.92,
    "logical_coherence": 0.87,
    "overall_science_quality": 0.895,
    "quality_label": "高科学性",
    "evidence": "...",
    "analysis": { "claims": [...], "logic_issues": [...], "knowledge_gaps": [...] }
  }

作者: 基于 bert_llm/LLMKnowledgeEncoder 的科普场景适配版
================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# 添加 pipeline 路径以复用 LLMKnowledgeExtractor
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from pipeline.step_llm_knowledge import (
    LLMKnowledgeExtractor,
    DummyLLMKnowledgeExtractor,
    SYSTEM_PROMPT,
    STAGE1_PROMPT,
    STAGE2_PROMPT,
    DEEPSEEK_API_BASE,
    DEEPSEEK_MODEL,
)


# ============================================================
# 质量等级映射
# ============================================================

def classify_quality(factual: float, logical: float) -> str:
    """基于事实与逻辑得分判定科学性质量等级。"""
    avg = (factual + logical) / 2.0
    if avg >= 0.80:
        return "高科学性"
    elif avg >= 0.60:
        return "中等科学性"
    elif avg >= 0.40:
        return "科学性偏低"
    else:
        return "科学性严重不足"


def classify_detail(score: float) -> str:
    if score >= 0.80:
        return "优秀"
    elif score >= 0.60:
        return "良好"
    elif score >= 0.40:
        return "一般"
    else:
        return "较差"


# ============================================================
# 独立评价器
# ============================================================

class ScienceSubtitleEvaluator:
    """
    科普字幕独立科学性评价器。

    基于 DeepSeek V4 Pro 对单条字幕进行无参考评价，
    输出事实一致性 + 逻辑连贯性两个核心指标。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEEPSEEK_MODEL,
        api_base: str = DEEPSEEK_API_BASE,
        cache_dir: str = "./cache/llm_knowledge",
        temperature: float = 0.1,
    ):
        self.api_key = api_key
        if api_key:
            self.extractor = LLMKnowledgeExtractor(
                api_key=api_key,
                model=model,
                api_base=api_base,
                cache_dir=cache_dir,
                temperature=temperature,
            )
        else:
            print("[Evaluator] ⚠ 未设置 API Key，使用启发式伪评分 (Dummy 模式)")
            self.extractor = DummyLLMKnowledgeExtractor()

    def evaluate(self, text: str, verbose: bool = True) -> Dict:
        """
        评价单条科普字幕。

        Args:
            text: 字幕文本
            verbose: 是否打印详细日志

        Returns:
            dict 包含评价结果
        """
        if not text or not text.strip():
            return {
                "text": text,
                "factual_consistency": 0.0,
                "logical_coherence": 0.0,
                "overall_science_quality": 0.0,
                "quality_label": "无文本",
                "factual_label": "无文本",
                "logical_label": "无文本",
                "error": "输入文本为空",
            }

        scores = self.extractor.extract(text, verbose=verbose)
        factual = float(scores[0])
        logical = float(scores[1])
        overall = (factual + logical) / 2.0

        result = {
            "text": text[:200] + ("..." if len(text) > 200 else ""),
            "text_length": len(text),
            "factual_consistency": round(factual, 4),
            "logical_coherence": round(logical, 4),
            "overall_science_quality": round(overall, 4),
            "quality_label": classify_quality(factual, logical),
            "factual_label": classify_detail(factual),
            "logical_label": classify_detail(logical),
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"  事实一致性:  {factual:.4f}  ({classify_detail(factual)})")
            print(f"  逻辑连贯性:  {logical:.4f}  ({classify_detail(logical)})")
            print(f"  综合科学质量: {overall:.4f}  ({classify_quality(factual, logical)})")
            print(f"{'='*60}")

        return result

    def evaluate_batch(
        self,
        texts: List[Dict[str, str]],
        verbose: bool = True,
    ) -> List[Dict]:
        """
        批量评价多条字幕。

        Args:
            texts: [{"id": "v001", "text": "..."}, ...]
            verbose: 是否打印进度

        Returns:
            评价结果列表
        """
        results = []
        for item in texts:
            vid = item.get("id", item.get("video_id", str(len(results))))
            text = item.get("text", item.get("subtitle", ""))
            if verbose:
                print(f"\n[{vid}] 评价中...")
            result = self.evaluate(text, verbose=verbose)
            result["id"] = vid
            results.append(result)
        return results


# ============================================================
# 文件 I/O
# ============================================================

def load_input_file(path: str) -> List[Dict[str, str]]:
    """从 CSV 或 JSON 加载字幕数据。"""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        import csv
        items = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                items.append(dict(row))
        return items
    elif path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError(f"无法解析 JSON 文件: {path}")
    else:
        # 尝试按纯文本文件处理（每行一条字幕）
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    items.append({"id": f"line_{i+1}", "text": line})
        return items


# ============================================================
# 主入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="科普字幕无参考科学性评价 (基于 DeepSeek V4 Pro)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python standalone_eval.py --text "地球绕太阳公转一周需要365天..."
  python standalone_eval.py --input subtitles.json --output results.json
  python standalone_eval.py --interactive
  python standalone_eval.py --input subtitles.csv --threshold 0.5 --filter
        """,
    )

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--text", type=str, help="直接评价一段字幕文本")
    input_group.add_argument("--input", type=str, help="输入文件路径 (CSV/JSON/TXT)")
    input_group.add_argument("--interactive", action="store_true", help="交互式评价模式")

    parser.add_argument("--output", type=str, default="", help="结果输出 JSON 路径")
    parser.add_argument("--threshold", type=float, default=0.5, help="科学性阈值 (低于此值标记为低质量)")
    parser.add_argument("--filter", action="store_true", help="仅输出低于阈值的低质量字幕")
    parser.add_argument("--api-key", type=str, default="", help="DeepSeek API Key (也可通过 DEEPSEEK_API_KEY 环境变量设置)")
    parser.add_argument("--cache-dir", type=str, default="./cache/llm_knowledge", help="LLM 特征缓存目录")
    return parser.parse_args()


def main():
    args = parse_args()

    # API Key
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("⚠ 未检测到 DEEPSEEK_API_KEY，将使用启发式 Dummy 评分。")
        print("  设置方法: export DEEPSEEK_API_KEY=sk-xxxx  或  --api-key sk-xxxx")

    evaluator = ScienceSubtitleEvaluator(
        api_key=api_key,
        cache_dir=args.cache_dir,
    )

    # ── 单条文本评价 ──
    if args.text:
        result = evaluator.evaluate(args.text)
        results = [result]

    # ── 交互模式 ──
    elif args.interactive:
        print("\n🎬 科普字幕科学性评价器 (输入 'quit' 退出)\n")
        results = []
        while True:
            try:
                text = input("请输入科普字幕文本: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            result = evaluator.evaluate(text)
            results.append(result)
        if not args.output and results:
            args.output = "interactive_results.json"

    # ── 文件批量评价 ──
    elif args.input:
        items = load_input_file(args.input)
        print(f"📂 加载 {len(items)} 条字幕")

        results = []
        for item in items:
            vid = item.get("id", item.get("video_id", str(len(results))))
            text = item.get("text", item.get("subtitle", ""))
            result = evaluator.evaluate(text, verbose=False)
            result["id"] = vid
            # 保留原始元数据
            for k, v in item.items():
                if k not in result and k not in ("text", "subtitle"):
                    result[f"meta_{k}"] = v
            results.append(result)

            # 实时打印概览
            print(f"  [{vid}] fact={result['factual_consistency']:.3f}  "
                  f"logic={result['logical_coherence']:.3f}  "
                  f"→ {result['quality_label']}")

    else:
        print("请指定 --text、--input 或 --interactive。使用 --help 查看帮助。")
        return

    # ── 筛选低质量 ──
    if args.filter:
        before = len(results)
        results = [r for r in results
                   if r["overall_science_quality"] < args.threshold]
        print(f"\n🔍 阈值过滤: {before} → {len(results)} 条 (threshold={args.threshold})")

    # ── 汇总统计 ──
    if results:
        factuals = [r["factual_consistency"] for r in results]
        logicals = [r["logical_coherence"] for r in results]
        overalls = [r["overall_science_quality"] for r in results]
        labels = [r["quality_label"] for r in results]

        print(f"\n{'='*60}")
        print(f"📊 汇总统计 (N={len(results)})")
        print(f"  事实一致性:  μ={np.mean(factuals):.4f}  σ={np.std(factuals):.4f}  "
              f"[{np.min(factuals):.4f}, {np.max(factuals):.4f}]")
        print(f"  逻辑连贯性:  μ={np.mean(logicals):.4f}  σ={np.std(logicals):.4f}  "
              f"[{np.min(logicals):.4f}, {np.max(logicals):.4f}]")
        print(f"  综合质量:    μ={np.mean(overalls):.4f}  σ={np.std(overalls):.4f}  "
              f"[{np.min(overalls):.4f}, {np.max(overalls):.4f}]")
        for label in ["高科学性", "中等科学性", "科学性偏低", "科学性严重不足"]:
            cnt = labels.count(label)
            if cnt:
                print(f"  {label}: {cnt} 条 ({cnt/len(results)*100:.1f}%)")
        print(f"{'='*60}")

    # ── 保存结果 ──
    output_path = args.output or "science_eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
