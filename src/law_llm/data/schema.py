# -*- coding: utf-8 -*-
"""Data schemas and type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

import json


class DataCategory(str, Enum):
    """数据大类。"""

    LEGAL_CONSULTATION = "legal_consultation"  # 法律咨询
    LEGAL_EXAM = "legal_exam"  # 法考题目
    GENERAL_INSTRUCTION = "general_instruction"  # 通用指令
    UNKNOWN = "unknown"


class CleanReason(str, Enum):
    """样本被过滤的原因。"""

    MISSING_FIELD = "missing_field"
    TOO_SHORT = "too_short"
    INVALID_ARTICLE_REF = "invalid_article_ref"
    TEMPLATE_REPLY = "template_reply"
    ENTERTAINMENT = "entertainment"
    FACT_MEMORIZATION = "fact_memorization"
    LOW_QUALITY = "low_quality"
    LENGTH_ABNORMAL = "length_abnormal"
    DUPLICATE_CONTENT = "duplicate_content"
    FORMAT_ERROR = "format_error"


class DedupMethod(str, Enum):
    """去重方法。"""

    EXACT = "exact"
    MINHASH = "minhash"
    SEMANTIC = "semantic"


class SplitType(str, Enum):
    """数据集划分类型。"""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass
class SFTSample:
    """一条 SFT 训练样本（ShareGPT / Alpaca 格式）。"""

    instruction: str
    input: str = ""
    output: str = ""
    category: str = DataCategory.UNKNOWN.value
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_alpaca_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }

    def to_sharegpt_dict(self) -> dict[str, Any]:
        return {
            "conversations": [
                {"from": "human", "value": self.instruction},
                {"from": "gpt", "value": self.output},
            ],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SFTSample":
        """从任意格式的 dict 构建，兼容多种字段名。"""
        instruction = (
            raw.get("instruction")
            or raw.get("query")
            or raw.get("question")
            or ""
        )
        output = (
            raw.get("output")
            or raw.get("response")
            or raw.get("answer")
            or ""
        )
        inp = raw.get("input", "")
        return cls(
            instruction=instruction.strip(),
            input=inp.strip() if isinstance(inp, str) else "",
            output=output.strip(),
            category=raw.get("category", DataCategory.UNKNOWN.value),
            source=raw.get("source", ""),
            metadata=raw.get("metadata", {}),
        )


@dataclass
class CleanRecord:
    """记录被过滤的样本及其原因。"""

    item: dict[str, Any]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"item": self.item, "reason": self.reason}


@dataclass
class ProcessingReport:
    """数据处理流程报告。"""

    original_count: int = 0
    format_error_count: int = 0
    rule_filtered_count: int = 0
    semantic_duplicate_count: int = 0
    final_count: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    category_distribution: dict[str, int] = field(default_factory=dict)
    clean_reason_distribution: dict[str, int] = field(default_factory=dict)
    seed: int = 42
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def summary(self) -> str:
        lines = [
            "===== 数据处理报告 =====",
            f"原始数据数量:      {self.original_count}",
            f"格式错误数量:      {self.format_error_count}",
            f"规则过滤数量:      {self.rule_filtered_count}",
            f"语义重复数量:      {self.semantic_duplicate_count}",
            f"最终保留数量:      {self.final_count}",
            f"训练/验证/测试:    {self.train_count} / {self.val_count} / {self.test_count}",
            f"随机种子:          {self.seed}",
            "",
            "类别分布:",
        ]
        for cat, cnt in sorted(self.category_distribution.items()):
            lines.append(f"  {cat}: {cnt}")
        if self.clean_reason_distribution:
            lines.append("")
            lines.append("过滤原因分布:")
            for reason, cnt in sorted(
                self.clean_reason_distribution.items(), key=lambda x: -x[1]
            ):
                lines.append(f"  {reason}: {cnt}")
        return "\n".join(lines)


@dataclass
class LawArticle:
    """一条法律条文，用于 RAG 索引。"""

    law_name: str
    article: str  # e.g. "第二十八条"
    chapter: str  # e.g. "第二章"
    section: str  # e.g. "第一节"
    effective_date: str
    source: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LawArticle":
        return cls(
            law_name=d.get("law_name", ""),
            article=d.get("article", ""),
            chapter=d.get("chapter", ""),
            section=d.get("section", ""),
            effective_date=d.get("effective_date", ""),
            source=d.get("source", ""),
            content=d.get("content", ""),
        )
