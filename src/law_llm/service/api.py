# -*- coding: utf-8 -*-
"""Flask API 服务。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

from ..rag.indexer import FaissIndexer
from ..rag.pipeline import RAGPipeline
from ..rag.retriever import RetrievalResult
from .model_client import ModelClient
from .schemas import HealthResponse, QueryRequest, QueryResponse, Citation


def create_app(
    index_dir: str = "data/faiss_index",
    embedding_model: str = "BAAI/bge-small-zh-v1.5",
    model_mode: str = "api",
    api_base: str = "http://localhost:8000/v1",
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    local_model_path: str = "",
    top_k: int = 5,
    use_reranker: bool = False,
    device: str = "cpu",
) -> Flask:
    """创建并配置 Flask 应用。

    Args:
        index_dir:       FAISS 索引目录
        embedding_model: embedding 模型名
        model_mode:      "api" (vLLM) 或 "local" (HF)
        api_base:        vLLM API 地址
        model_name:      模型名称
        local_model_path: 本地模型路径
        top_k:           检索数量
        use_reranker:    是否启用重排序
        device:          计算设备

    Returns:
        Flask 应用实例
    """
    app = Flask(__name__)

    # 初始化组件
    indexer = FaissIndexer(embedding_model=embedding_model, device=device)
    rag_ready = False
    if Path(index_dir).exists():
        try:
            indexer.load(index_dir)
            rag_ready = True
        except Exception as e:
            app.logger.warning(f"FAISS 索引加载失败: {e}")

    # 初始化模型客户端
    model_client = ModelClient(
        mode=model_mode,
        api_base=api_base,
        model_name=model_name,
        local_model_path=local_model_path,
        device=device,
    )

    # 初始化 RAG 流水线
    pipeline = RAGPipeline(
        indexer=indexer,
        model_client=model_client,
        top_k=top_k,
        use_reranker=use_reranker,
        device=device,
    ) if rag_ready else None

    # -----------------------------------------------------------------------
    # 路由
    # -----------------------------------------------------------------------

    @app.route("/health", methods=["GET"])
    def health():
        """健康检查。"""
        resp = HealthResponse(
            status="ok",
            model_loaded=model_client.is_ready() if model_mode == "api" else True,
            rag_ready=rag_ready,
            num_law_articles=len(indexer.documents) if rag_ready else 0,
        )
        return jsonify(resp.to_dict())

    @app.route("/query", methods=["POST"])
    def query():
        """法律问答接口。"""
        req_data = request.get_json()
        if not req_data or "question" not in req_data:
            return jsonify({"error": "缺少 'question' 字段"}), 400

        req = QueryRequest.from_dict(req_data)
        start_time = time.time()

        if pipeline is None:
            return jsonify({"error": "RAG 系统未就绪，请先构建索引"}), 503

        # 执行 RAG
        rag_response = pipeline.answer(req.question)
        latency = (time.time() - start_time) * 1000

        # 转换为 API 响应
        citations = [
            Citation(
                law_name=c.get("law_name", ""),
                article=c.get("article", ""),
                chapter=c.get("chapter", ""),
                in_retrieved=c.get("in_retrieved", True),
                content=c.get("content", ""),
            )
            for c in rag_response.citations
        ]

        response = QueryResponse(
            question=rag_response.question,
            answer=rag_response.answer,
            citations=citations,
            retrieved_docs=rag_response.retrieved_docs,
            has_evidence=rag_response.has_evidence,
            latency_ms=latency,
        )

        return jsonify(response.to_dict())

    @app.route("/retrieve", methods=["POST"])
    def retrieve():
        """仅检索法条（不生成回答）。"""
        req_data = request.get_json()
        if not req_data or "question" not in req_data:
            return jsonify({"error": "缺少 'question' 字段"}), 400

        question = req_data["question"]
        top_k = req_data.get("top_k", 5)

        if pipeline is None:
            return jsonify({"error": "RAG 系统未就绪"}), 503

        results = pipeline.retrieve(question, top_k=top_k)
        return jsonify({"results": [r.to_dict() for r in results]})

    @app.route("/laws", methods=["GET"])
    def list_laws():
        """列出知识库中的所有法律。"""
        if not rag_ready:
            return jsonify({"error": "RAG 系统未就绪"}), 503

        law_names = set()
        for doc in indexer.documents:
            name = doc.metadata.get("law_name", "")
            if name:
                law_names.add(name)

        return jsonify({"laws": sorted(law_names)})

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 5000,
    **kwargs: Any,
) -> None:
    """启动 Flask 服务器。"""
    app = create_app(**kwargs)
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    run_server()
