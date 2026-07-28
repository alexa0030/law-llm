# -*- coding: utf-8 -*-
"""重排序模块（可选）。

使用 Cross-Encoder 对初步检索结果进行重排序，提升精度。
如果未安装相关依赖，则退化为不重排序。
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from .retriever import RetrievalResult


class LawReranker:
    """法律文档重排序器。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        """延迟加载 Cross-Encoder。"""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, device=self.device)
            except ImportError:
                print("  ⚠ sentence-transformers 未安装，跳过重排序")
                return None
        return self._model

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """对检索结果进行重排序。

        Args:
            query:   查询文本
            results: 初步检索结果
            top_k:   重排序后保留的数量

        Returns:
            重排序后的结果列表
        """
        if self.model is None or len(results) == 0:
            return results[:top_k] if top_k else results

        # 使用 Cross-Encoder 计算查询-文档对分数
        pairs = [(query, r.document.page_content) for r in results]
        scores = self.model.predict(pairs)

        # 按新分数排序
        scored = list(zip(results, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for rank, (result, new_score) in enumerate(scored, 1):
            result.score = float(new_score)
            result.rank = rank
            reranked.append(result)
            if top_k and rank >= top_k:
                break

        return reranked
