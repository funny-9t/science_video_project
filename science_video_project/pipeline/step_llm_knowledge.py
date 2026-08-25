"""DeepSeek V4-Pro scientific-audit features for science videos."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-pro"
PROMPT_VERSION = "science-audit-v2"
EMPTY_CONTENT_PLACEHOLDER = "【视频文本为空：未检测到标题、标签或可识别语音】"

SYSTEM_PROMPT = """你是一名严谨的科普内容审核专家。请基于公开、稳定的科学知识审视输入内容，区分可核验事实、合理推断和缺乏证据的主张。不要根据表达风格、流量或账号身份判断内容是否科学。输出必须是 JSON。"""

AUDIT_PROMPT = """请审核下面的科普短视频文本。先识别核心科学主张，再分析事实一致性、逻辑链条、证据充分性以及对不确定性的表达。不要把无法核验等同于错误，也不要虚构参考文献。

【视频文本】
{text}

仅输出一个 JSON 对象，格式如下：
{{
  "claims": ["核心主张1", "核心主张2"],
  "factual_analysis": "对主张与公认科学知识一致性的简洁分析",
  "logical_analysis": "对前提、推理和结论是否自洽的简洁分析",
  "evidence_analysis": "对证据、数据来源和可核验性的简洁分析",
  "uncertainty_analysis": "对边界条件、限定语和不确定性表达的简洁分析",
  "factual_consistency": 0.0,
  "logical_coherence": 0.0,
  "evidence_sufficiency": 0.0,
  "uncertainty_awareness": 0.0
}}

四个分数必须位于 0 到 1 之间，越高表示该维度越好。"""


@dataclass
class LLMKnowledgeResult:
    scores: np.ndarray
    analysis_text: str
    reasoning_text: str
    is_valid: bool
    model: str

    def feature_text(self, include_reasoning: bool = True, max_chars: int = 6000) -> str:
        parts = []
        if include_reasoning and self.reasoning_text:
            parts.append(self.reasoning_text)
        if self.analysis_text:
            parts.append(self.analysis_text)
        return "\n".join(parts)[:max_chars]


class LLMKnowledgeExtractor:
    """Generate four audit scores plus analysis/reasoning text with disk caching."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEEPSEEK_MODEL,
        api_base: str = DEEPSEEK_API_BASE,
        cache_dir: str | Path = "./cache/llm_knowledge",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        max_input_chars: int = 12000,
        reasoning_effort: str = "high",
        request_timeout: float = 90.0,
        cache_policy: str = "prefer",
    ) -> None:
        del temperature  # Thinking mode ignores sampling controls.
        self.api_key = (api_key or "").strip()
        self.model = model
        self.api_base = api_base
        self.cache_dir = Path(cache_dir)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_tokens = max_tokens
        self.max_input_chars = max_input_chars
        self.reasoning_effort = reasoning_effort
        self.request_timeout = request_timeout
        if cache_policy not in {"prefer", "require"}:
            raise ValueError("cache_policy must be 'prefer' or 'require'.")
        self.cache_policy = cache_policy

        self.client = None
        if self.api_key:
            if not self.api_key.startswith("sk-"):
                raise ValueError("DEEPSEEK_API_KEY format is invalid; expected a value starting with 'sk-'.")
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.request_timeout,
                max_retries=0,
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / "llm_knowledge_v2.jsonl"
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_lock = threading.Lock()
        self._load_cache()

    def _cache_key(self, text: str) -> str:
        value = f"{PROMPT_VERSION}\0{self.model}\0{text}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        with self.cache_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                    if item.get("cache_key"):
                        self._cache[str(item["cache_key"])] = item
                except (json.JSONDecodeError, TypeError):
                    continue

    def _append_cache(self, item: dict[str, Any]) -> None:
        with self._cache_lock:
            with self.cache_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            self._cache[str(item["cache_key"])] = item

    def _call_api(self, text: str) -> tuple[str, str]:
        if self.client is None:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured.")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": AUDIT_PROMPT.format(text=text[: self.max_input_chars])},
        ]
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    reasoning_effort=self.reasoning_effort,
                    extra_body={"thinking": {"type": "enabled"}},
                )
                message = response.choices[0].message
                return message.content or "", getattr(message, "reasoning_content", "") or ""
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError(f"DeepSeek API failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def _score(value: Any) -> float:
        return float(np.clip(float(value), 0.0, 1.0))

    @staticmethod
    def _analysis_text(payload: dict[str, Any]) -> str:
        claims = payload.get("claims", [])
        if not isinstance(claims, list):
            claims = [str(claims)]
        lines = ["核心主张：" + "；".join(str(item) for item in claims if item)]
        mapping = [
            ("事实分析", "factual_analysis"),
            ("逻辑分析", "logical_analysis"),
            ("证据分析", "evidence_analysis"),
            ("不确定性分析", "uncertainty_analysis"),
        ]
        lines.extend(f"{label}：{payload.get(key, '')}" for label, key in mapping)
        return "\n".join(line for line in lines if not line.endswith("："))

    def _parse_response(self, content: str) -> tuple[np.ndarray, str, bool]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
        try:
            payload = json.loads(cleaned)
            scores = np.asarray(
                [
                    self._score(payload["factual_consistency"]),
                    self._score(payload["logical_coherence"]),
                    self._score(payload["evidence_sufficiency"]),
                    self._score(payload["uncertainty_awareness"]),
                ],
                dtype=np.float32,
            )
            return scores, self._analysis_text(payload), True
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return np.full(4, 0.5, dtype=np.float32), cleaned, False

    @staticmethod
    def _from_cache(item: dict[str, Any]) -> LLMKnowledgeResult:
        return LLMKnowledgeResult(
            scores=np.asarray(item["scores"], dtype=np.float32),
            analysis_text=str(item.get("analysis_text", "")),
            reasoning_text=str(item.get("reasoning_text", "")),
            is_valid=bool(item.get("is_valid", False)),
            model=str(item.get("model", DEEPSEEK_MODEL)),
        )

    def extract_details(self, text: str, verbose: bool = False) -> LLMKnowledgeResult:
        content_text = (text or "").strip() or EMPTY_CONTENT_PLACEHOLDER

        cache_key = self._cache_key(content_text)
        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            result = self._from_cache(cached)
            if result.is_valid:
                if verbose:
                    print(f"[LLMKnowledge] cache hit: {result.scores.tolist()}")
                return result
            if self.cache_policy == "require":
                raise ValueError(f"Cached DeepSeek response is invalid: {cache_key}")
            if verbose:
                print("[LLMKnowledge] retrying an invalid cached response")

        if self.cache_policy == "require":
            raise KeyError(
                f"Required DeepSeek cache entry is missing: {cache_key}. "
                "Run tools/prefetch_llm_cache.py before feature extraction."
            )

        content, reasoning = self._call_api(content_text)
        scores, analysis_text, is_valid = self._parse_response(content)
        result = LLMKnowledgeResult(
            scores=scores,
            analysis_text=analysis_text,
            reasoning_text=reasoning,
            is_valid=is_valid,
            model=self.model,
        )
        self._append_cache(
            {
                "cache_key": cache_key,
                "prompt_version": PROMPT_VERSION,
                "model": self.model,
                "scores": scores.tolist(),
                "analysis_text": analysis_text,
                "reasoning_text": reasoning,
                "raw_content": content,
                "is_valid": is_valid,
            }
        )
        if verbose:
            print(f"[LLMKnowledge] valid={is_valid} scores={scores.tolist()}")
        return result

    def extract(self, text: str, verbose: bool = False) -> np.ndarray:
        return self.extract_details(text, verbose=verbose).scores

    def extract_batch(self, texts: Iterable[str], verbose: bool = True) -> np.ndarray:
        text_list = list(texts)
        iterator = text_list
        if verbose and len(text_list) > 1:
            from tqdm import tqdm

            iterator = tqdm(text_list, desc="DeepSeek V4 scientific audit")
        return np.stack([self.extract(text) for text in iterator], axis=0).astype(np.float32)

    def get_cached(self, text: str) -> Optional[np.ndarray]:
        item = self._cache.get(self._cache_key(text))
        return None if item is None else np.asarray(item["scores"], dtype=np.float32)


class RobertaAnalysisEncoder:
    """Encode LLM audit text with the project's local Chinese RoBERTa."""

    def __init__(self, model_path: str | Path, device: str = "cuda", max_length: int = 512) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True).to(self.device)
        self.model.eval()

    def encode(self, text: str) -> np.ndarray:
        if not (text or "").strip():
            return np.zeros(int(self.model.config.hidden_size), dtype=np.float32)
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        with self.torch.no_grad():
            output = self.model(**tokens).last_hidden_state[:, 0, :].squeeze(0)
        return output.detach().cpu().float().numpy()


class DummyLLMKnowledgeExtractor:
    """Deterministic offline placeholder; never use it for final feature generation."""

    def extract_details(self, text: str, verbose: bool = False) -> LLMKnowledgeResult:
        length_score = min(len(text or "") / 1000.0, 1.0)
        scores = np.asarray([0.4 + 0.2 * length_score] * 4, dtype=np.float32)
        if verbose:
            print("[DummyLLM] offline placeholder")
        return LLMKnowledgeResult(scores, "", "", False, "dummy")

    def extract(self, text: str, verbose: bool = False) -> np.ndarray:
        return self.extract_details(text, verbose).scores

    def extract_batch(self, texts: Iterable[str], verbose: bool = True) -> np.ndarray:
        del verbose
        return np.stack([self.extract(text) for text in texts], axis=0)
