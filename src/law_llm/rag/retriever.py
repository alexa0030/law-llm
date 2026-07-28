# -*- coding: utf-8 -*-
"""向量检索器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from langchain_core.documents import Document

from .indexer import FaissIndexer


@dataclass
class RetrievalResult:
    """单条检索结果。"""

    document: Document
    score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": round(self.score, 4),
            "law_name": self.document.metadata.get("law_name", ""),
            "article": self.document.metadata.get("article", ""),
            "chapter": self.document.metadata.get("chapter", ""),
            "content": self.document.page_content,
        }


class LawRetriever:
    """法律知识库检索器。

    功能：
        - 向量检索（FAISS）
        - Top-k 召回
        - 可选元数据过滤（按法律名称、章节等）
    """

    def __init__(self, indexer: FaissIndexer) -> None:
        self.indexer = indexer

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_law_name: str | None = None,
    ) -> list[RetrievalResult]:
        """检索最相关的法条。

        Args:
            query:          查询文本
            top_k:          返回数量
            filter_law_name: 只在指定法律中检索（可选）

        Returns:
            检索结果列表
        """
        # 生成查询向量
        query_embedding = self.indexer.embed_texts([query])
        query_embedding = query_embedding.astype(np.float32)

        # 搜索
        k = min(top_k * 3, len(self.indexer.documents))  # 多检索一些用于过滤
        scores, indices = self.indexer.index.search(query_embedding, k)

        results: list[RetrievalResult] = []
        rank = 0
        for i, idx in enumerate(indices[0]):
            if idx < 0:
                continue

            doc = self.indexer.documents[idx]
            score = float(scores[0][i])

            # 元数据过滤
            if filter_law_name and doc.metadata.get("law_name") != filter_law_name:
                continue

            rank += 1
            results.append(RetrievalResult(document=doc, score=score, rank=rank))

            if len(results) >= top_k:
                break

        return results

    def batch_search(
        self,
        queries: list[str],
        top_k: int = 5,
    ) -> list[list[RetrievalResult]]:
        """批量检索。"""
        return [self.search(q, top_k) for q in queries]
