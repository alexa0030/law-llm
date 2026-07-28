# -*- coding: utf-8 -*-
"""API 请求/响应模型。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class QueryRequest:
    """法律问答请求。"""

    question: str
    use_rag: bool = True
    top_k: int = 5
    temperature: float = 0.3
    max_tokens: int = 2048

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QueryRequest":
        return cls(
            question=d.get("question", ""),
            use_rag=d.get("use_rag", True),
            top_k=d.get("top_k", 5),
            temperature=d.get("temperature", 0.3),
            max_tokens=d.get("max_tokens", 2048),
        )


@dataclass
class Citation:
    """法条引用。"""

    law_name: str
    article: str
    chapter: str = ""
    in_retrieved: bool = True
    content: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueryResponse:
    """法律问答回复。"""

    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    retrieved_docs: list[dict[str, Any]] = field(default_factory=list)
    has_evidence: bool = True
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "retrieved_docs": self.retrieved_docs,
            "has_evidence": self.has_evidence,
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass
class HealthResponse:
    """健康检查响应。"""

    status: str = "ok"
    model_loaded: bool = False
    rag_ready: bool = False
    num_law_articles: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
