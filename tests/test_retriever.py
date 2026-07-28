# -*- coding: utf-8 -*-
"""检索器测试（使用模拟数据，不需要实际模型）。"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_core.documents import Document
from law_llm.rag.retriever import LawRetriever, RetrievalResult


class TestLawRetriever:
    """测试法律检索器。"""

    def test_search_with_mock_index(self):
        """使用 mock 索引测试检索。"""
        # 创建 mock indexer
        mock_indexer = MagicMock()
        mock_indexer.documents = [
            Document(
                page_content="第一条 为了保护个人信息权益。",
                metadata={"law_name": "个人信息保护法", "article": "第一条"},
            ),
            Document(
                page_content="第二条 自然人的个人信息受法律保护。",
                metadata={"law_name": "个人信息保护法", "article": "第二条"},
            ),
        ]
        mock_indexer.embed_texts.return_value = np.array([[0.1, 0.2]], dtype=np.float32)

        # mock FAISS search
        mock_indexer.index = MagicMock()
        mock_indexer.index.search.return_value = (
            np.array([[0.9, 0.8]]),
            np.array([[0, 1]]),
        )

        retriever = LawRetriever(mock_indexer)
        results = retriever.search("个人信息保护", top_k=2)

        assert len(results) <= 2
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_retrieval_result_to_dict(self):
        """测试检索结果序列化。"""
        doc = Document(
            page_content="测试内容",
            metadata={"law_name": "测试法", "article": "第一条", "chapter": "第一章"},
        )
        result = RetrievalResult(document=doc, score=0.95, rank=1)
        d = result.to_dict()

        assert d["rank"] == 1
        assert d["score"] == 0.95
        assert d["law_name"] == "测试法"
        assert d["article"] == "第一条"
        assert d["content"] == "测试内容"

    def test_filter_by_law_name(self):
        """测试按法律名称过滤。"""
        mock_indexer = MagicMock()
        mock_indexer.documents = [
            Document(
                page_content="刑法第一条",
                metadata={"law_name": "刑法", "article": "第一条"},
            ),
            Document(
                page_content="民法第一条",
                metadata={"law_name": "民法", "article": "第一条"},
            ),
        ]
        mock_indexer.embed_texts.return_value = np.array([[0.1, 0.2]], dtype=np.float32)
        mock_indexer.index = MagicMock()
        mock_indexer.index.search.return_value = (
            np.array([[0.9, 0.8]]),
            np.array([[0, 1]]),
        )

        retriever = LawRetriever(mock_indexer)
        results = retriever.search("测试", top_k=5, filter_law_name="刑法")

        # 只有刑法的文档应该返回
        for r in results:
            assert r.document.metadata["law_name"] == "刑法"
