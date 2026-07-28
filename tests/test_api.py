# -*- coding: utf-8 -*-
"""API 测试。"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestSchemas:
    """测试 API 数据模型。"""

    def test_query_request_from_dict(self):
        from law_llm.service.schemas import QueryRequest

        req = QueryRequest.from_dict({
            "question": "什么是故意犯罪？",
            "use_rag": True,
            "top_k": 3,
        })
        assert req.question == "什么是故意犯罪？"
        assert req.use_rag is True
        assert req.top_k == 3

    def test_query_request_defaults(self):
        from law_llm.service.schemas import QueryRequest

        req = QueryRequest.from_dict({"question": "测试"})
        assert req.use_rag is True
        assert req.top_k == 5
        assert req.temperature == 0.3

    def test_query_response_to_dict(self):
        from law_llm.service.schemas import QueryResponse, Citation

        resp = QueryResponse(
            question="测试问题",
            answer="测试回答",
            citations=[
                Citation(law_name="刑法", article="第十四条"),
            ],
            has_evidence=True,
            latency_ms=123.45,
        )
        d = resp.to_dict()
        assert d["question"] == "测试问题"
        assert d["answer"] == "测试回答"
        assert len(d["citations"]) == 1
        assert d["citations"][0]["law_name"] == "刑法"
        assert d["has_evidence"] is True


class TestFlaskApp:
    """测试 Flask API（轻量测试，不加载模型）。"""

    def test_health_endpoint(self):
        from law_llm.service.api import create_app

        app = create_app(index_dir="nonexistent", model_mode="api")
        client = app.test_client()

        response = client.get("/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"

    def test_query_missing_question(self):
        from law_llm.service.api import create_app

        app = create_app(index_dir="nonexistent", model_mode="api")
        client = app.test_client()

        response = client.post("/query", json={})
        assert response.status_code == 400

    def test_query_without_rag_ready(self):
        from law_llm.service.api import create_app

        app = create_app(index_dir="nonexistent", model_mode="api")
        client = app.test_client()

        response = client.post("/query", json={"question": "测试"})
        assert response.status_code == 503
