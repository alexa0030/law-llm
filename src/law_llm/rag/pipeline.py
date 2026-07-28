# -*- coding: utf-8 -*-
"""RAG 完整流水线 —— 从检索到生成的端到端链路。

流程：
    用户问题
      ↓
    向量检索 Top-k 法条
      ↓
    可选重排序
      ↓
    构建带法条引用的提示词
      ↓
    LLM 生成回答
      ↓
    提取引用并格式化输出
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .indexer import FaissIndexer
from .retriever import LawRetriever, RetrievalResult
from .reranker import LawReranker


# ---------------------------------------------------------------------------
# 提示词模板
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一个专业的法律咨询助手。请根据以下检索到的法律条文回答用户的问题。

要求：
1. 只根据提供的法律条文回答，不要编造不存在的法条。
2. 如果提供的法律条文不足以回答问题，请明确说明"根据现有法律条文，无法完整回答该问题"。
3. 回答中引用法律条文时，请使用格式：《法律名称》第X条。
4. 回答结构清晰，先给出结论，再列出法律依据。
5. 如果问题涉及的法律领域不在提供的条文中，请说明并建议咨询专业律师。

注意：本回答仅供参考，不构成正式法律意见。"""

USER_PROMPT_TEMPLATE = """用户问题：{question}

检索到的相关法律条文：

{context}

请根据以上法律条文回答用户的问题。回答时请引用具体法律条文作为依据。"""


@dataclass
class RAGResponse:
    """RAG 系统的完整响应。"""

    question: str
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    retrieved_docs: list[dict[str, Any]] = field(default_factory=list)
    has_evidence: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "retrieved_docs": self.retrieved_docs,
            "has_evidence": self.has_evidence,
        }


class RAGPipeline:
    """RAG 完整流水线。"""

    def __init__(
        self,
        indexer: FaissIndexer,
        model_client: Any | None = None,
        top_k: int = 5,
        use_reranker: bool = False,
        reranker_model: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
    ) -> None:
        self.retriever = LawRetriever(indexer)
        self.model_client = model_client
        self.top_k = top_k
        self.use_reranker = use_reranker
        self.reranker = LawReranker(reranker_model, device) if use_reranker else None

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievalResult]:
        """检索相关法条。"""
        k = top_k or self.top_k
        results = self.retriever.search(question, top_k=k)

        if self.reranker:
            results = self.reranker.rerank(question, results, top_k=k)

        return results

    def build_context(self, results: list[RetrievalResult]) -> str:
        """构建提示词中的上下文。"""
        context_parts = []
        for i, result in enumerate(results, 1):
            meta = result.document.metadata
            law_name = meta.get("law_name", "未知法律")
            article = meta.get("article", "")
            chapter = meta.get("chapter", "")

            header = f"[{i}] {law_name}"
            if chapter:
                header += f" {chapter}"
            if article:
                header += f" {article}"

            context_parts.append(f"{header}\n{result.document.page_content}")

        return "\n\n---\n\n".join(context_parts)

    def build_prompt(self, question: str, context: str) -> list[dict[str, str]]:
        """构建完整的消息列表。"""
        user_content = USER_PROMPT_TEMPLATE.format(question=question, context=context)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def extract_citations(
        self, answer: str, results: list[RetrievalResult]
    ) -> list[dict[str, Any]]:
        """从回答中提取引用的法条。"""
        citations = []

        # 匹配 《法律名》第X条 格式
        citation_pattern = re.compile(r"《([^》]+)》第([一二三四五六七八九十百千万零\d]+条)")
        found_citations = citation_pattern.findall(answer)

        # 去重并关联检索到的文档
        seen = set()
        for law_name, article in found_citations:
            key = f"{law_name}_{article}"
            if key in seen:
                continue
            seen.add(key)

            # 在检索结果中查找对应文档
            source_doc = None
            for result in results:
                meta = result.document.metadata
                if (
                    meta.get("law_name", "") in law_name
                    or law_name in meta.get("law_name", "")
                ) and meta.get("article", "") == article:
                    source_doc = result.document
                    break

            citation = {
                "law_name": law_name,
                "article": article,
                "in_retrieved": source_doc is not None,
            }
            if source_doc:
                citation["chapter"] = source_doc.metadata.get("chapter", "")
                citation["content"] = source_doc.page_content[:200]

            citations.append(citation)

        return citations

    def check_evidence(self, results: list[RetrievalResult], min_score: float = 0.3) -> bool:
        """检查是否有足够的证据回答问题。"""
        if not results:
            return False
        # 如果最高分低于阈值，认为没有足够证据
        return results[0].score >= min_score

    def answer(self, question: str) -> RAGResponse:
        """端到端回答。"""
        # 1. 检索
        results = self.retrieve(question)

        # 2. 检查是否有证据
        has_evidence = self.check_evidence(results)

        if not has_evidence:
            return RAGResponse(
                question=question,
                answer="抱歉，根据现有法律知识库，未检索到与您问题相关的法律条文。"
                "建议您咨询专业律师获取针对性的法律意见。",
                citations=[],
                retrieved_docs=[],
                has_evidence=False,
            )

        # 3. 构建上下文和提示词
        context = self.build_context(results)
        messages = self.build_prompt(question, context)

        # 4. 生成回答
        if self.model_client is not None:
            answer_text = self.model_client.chat(messages)
        else:
            # 无 LLM 时返回检索结果作为参考
            answer_text = self._format_retrieval_answer(question, results)

        # 5. 提取引用
        citations = self.extract_citations(answer_text, results)

        # 6. 构建检索文档记录
        retrieved_docs = [r.to_dict() for r in results]

        return RAGResponse(
            question=question,
            answer=answer_text,
            citations=citations,
            retrieved_docs=retrieved_docs,
            has_evidence=True,
        )

    def _format_retrieval_answer(
        self, question: str, results: list[RetrievalResult]
    ) -> str:
        """无 LLM 时的检索结果格式化输出。"""
        lines = [f"关于您的问题「{question}」，检索到以下相关法律条文：\n"]
        for result in results:
            meta = result.document.metadata
            law_name = meta.get("law_name", "未知法律")
            article = meta.get("article", "")
            chapter = meta.get("chapter", "")

            ref = f"《{law_name}》"
            if chapter:
                ref += f" {chapter}"
            if article:
                ref += f" {article}"

            lines.append(f"【{ref}】\n{result.document.page_content}\n")

        lines.append("\n⚠ 本回答仅展示检索到的法律条文，如需综合分析请配置 LLM 模型。")
        return "\n".join(lines)
