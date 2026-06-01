"""
§7 Scientific Branch 升级 — 手工科学特征提取器

从文本中提取与科学性相关的语言学/统计学特征，与 BERT embedding 拼接，
形成更丰富的科学性表示。

特征维度 (5):
  - term_density:      科学术语密度 (关键词出现次数/总词数)
  - entity_count:      实体数量 (限词比例)
  - number_density:    数字密度 (含数字词/总词数)
  - avg_sentence_length: 平均句长
  - keyword_coverage:  科普关键词覆盖率
"""

from __future__ import annotations

import re
from typing import Dict, List


# 科普科学语料关键词表
SCIENCE_KEYWORDS: List[str] = [
    # 物理
    "量子", "原子", "分子", "光子", "电子", "质子", "中子", "核", "磁场", "电场",
    "引力", "重力", "速度", "加速度", "能量", "动量", "熵", "波", "频率", "振幅",
    # 化学
    "化学", "元素", "化合物", "反应", "催化", "氧化", "还原", "电解", "溶液", "pH",
    "离子", "分子式", "共价键", "离子键", "晶体", "聚合物",
    # 生物
    "细胞", "DNA", "RNA", "基因", "蛋白质", "酶", "染色体", "突变", "进化", "物种",
    "光合", "呼吸", "代谢", "免疫", "抗体", "病毒", "细菌", "真菌",
    # 天文地理
    "行星", "恒星", "星系", "宇宙", "黑洞", "暗物质", "暗能量", "轨道", "光年",
    "地球", "板块", "火山", "地震", "大气", "气候", "洋流",
    # 数学
    "函数", "方程", "概率", "统计", "微积分", "矩阵", "向量", "几何", "拓扑",
    "算法", "复杂度", "数论", "集合",
    # 工程
    "电路", "电压", "电流", "半导体", "芯片", "纳米", "材料", "合金", "陶瓷",
    "机器人", "传感器", "自动化", "人工智能", "神经网络", "深度学习",
    # 医学
    "病理", "诊断", "治疗", "药物", "疫苗", "手术", "器官", "血液", "神经",
    "激素", "CT", "MRI", "X光",
]


class ScienceFeatureExtractor:
    """手工科学性特征提取器：从文本中计算 5 维语言学+术语特征。"""

    def __init__(self, science_keywords: List[str] | None = None):
        self.keywords: List[str] = science_keywords or SCIENCE_KEYWORDS
        self._kw_lower = set(k.lower() for k in self.keywords)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """中文兼容分词：按 + 空白 + 标点切分。"""
        text = re.sub(r"[，。！？；：""''（）—…\-,.!?;:()\"'\[\]{}<>]", " ", text)
        tokens = text.strip().split()
        return [t for t in tokens if t]

    def extract(self, text: str) -> Dict[str, float]:
        """
        Args:
            text: 拼接后的文本（标题 + 标签 + ASR 字幕）

        Returns:
            Dict with: term_density, entity_count, number_density,
                       avg_sentence_length, keyword_coverage
        """
        text_clean = (text or "").strip()
        if not text_clean:
            return {
                "term_density": 0.0,
                "entity_count": 0.0,
                "number_density": 0.0,
                "avg_sentence_length": 0.0,
                "keyword_coverage": 0.0,
            }

        tokens = self._tokenize(text_clean)
        n_tokens = max(len(tokens), 1)

        # 1) term_density: 科学关键词命中率
        text_lower = text_clean.lower()
        kw_hits = sum(1 for kw in self._kw_lower if kw in text_lower)
        term_density = min(kw_hits / n_tokens, 5.0)  # 截断

        # 2) entity_count: 专有名词/大写词比例 (中文:用长词比例替代)
        long_words = sum(1 for t in tokens if len(t) >= 3)
        entity_count = min(long_words / n_tokens, 1.0)

        # 3) number_density: 数字/数值密度
        num_tokens = sum(1 for t in tokens if re.search(r"\d", t))
        number_density = num_tokens / n_tokens

        # 4) avg_sentence_length: 平均句长
        sentences = re.split(r"[。！？.!?]", text_clean)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            avg_len = min(sum(len(s) for s in sentences) / len(sentences) / 200.0, 1.0)
        else:
            avg_len = 0.0

        # 5) keyword_coverage: 科普关键词覆盖相对密度
        kw_count = sum(1 for t in tokens if t.lower() in self._kw_lower)
        keyword_coverage = min(kw_count / max(n_tokens * 0.1, 1.0), 1.0)

        return {
            "term_density": float(term_density),
            "entity_count": float(entity_count),
            "number_density": float(number_density),
            "avg_sentence_length": float(avg_len),
            "keyword_coverage": float(keyword_coverage),
        }

    def extract_vector(self, text: str) -> "torch.Tensor":
        """返回 (sci_hand_dim,) 的 float32 张量。"""
        import torch
        feats = self.extract(text)
        vec = torch.tensor(
            [
                feats["term_density"],
                feats["entity_count"],
                feats["number_density"],
                feats["avg_sentence_length"],
                feats["keyword_coverage"],
            ],
            dtype=torch.float32,
        )
        return vec
