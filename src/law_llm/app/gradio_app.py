# -*- coding: utf-8 -*-
"""Gradio 交互界面。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..rag.indexer import FaissIndexer
from ..rag.pipeline import RAGPipeline
from ..service.model_client import ModelClient


def create_gradio_app(
    index_dir: str = "data/faiss_index",
    embedding_model: str = "BAAI/bge-small-zh-v1.5",
    model_mode: str = "api",
    api_base: str = "http://localhost:8000/v1",
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    local_model_path: str = "",
    top_k: int = 5,
    use_reranker: bool = False,
    device: str = "cpu",
) -> Any:
    """创建 Gradio 应用。

    Returns:
        gradio.Blocks 实例
    """
    import gradio as gr

    # ��始化组件
    indexer = FaissIndexer(embedding_model=embedding_model, device=device)
    if Path(index_dir).exists():
        try:
            indexer.load(index_dir)
        except Exception as e:
            print(f"⚠ FAISS 索引加载失败: {e}")

    model_client = ModelClient(
        mode=model_mode,
        api_base=api_base,
        model_name=model_name,
        local_model_path=local_model_path,
        device=device,
    )

    pipeline = RAGPipeline(
        indexer=indexer,
        model_client=model_client,
        top_k=top_k,
        use_reranker=use_reranker,
        device=device,
    )

    def answer_question(question: str, use_rag: bool, top_k: int) -> tuple[str, str, str]:
        """处理用户问题。"""
        if not question.strip():
            return "请输入您的问题。", "", ""

        start = time.time()

        if use_rag:
            response = pipeline.answer(question)
            latency = time.time() - start

            # 格式化回答
            answer_text = response.answer
            if response.citations:
                answer_text += "\n\n---\n\n**引用依据：**\n"
                for i, cite in enumerate(response.citations, 1):
                    ref = f"[{i}]《{cite['law_name']}》{cite['article']}"
                    if cite.get("chapter"):
                        ref += f" ({cite['chapter']})"
                    answer_text += f"{ref}\n"

            # 格式化检索结果
            retrieved_text = ""
            for doc in response.retrieved_docs:
                law_name = doc.get("law_name", "未知")
                article = doc.get("article", "")
                score = doc.get("score", 0)
                retrieved_text += f"**{law_name} {article}** (相似度: {score})\n"
                retrieved_text += f"{doc.get('content', '')[:200]}...\n\n"

            latency_text = f"耗时: {latency:.2f}s | 检索到 {len(response.retrieved_docs)} 条法条"

            if not response.has_evidence:
                latency_text += " | ⚠ 未检索到相关法条"

            return answer_text, retrieved_text, latency_text
        else:
            # 不使用 RAG，直接调用模型
            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的法律咨询助手。请根据你的知识回答用户的问题。"
                    "如果不确定，请说明。本回答仅供参考，不构成正式法律意见。",
                },
                {"role": "user", "content": question},
            ]
            answer = model_client.chat(messages)
            latency = time.time() - start
            return answer, "（未使用 RAG）", f"耗时: {latency:.2f}s | 纯模型回答"

    # 构建界面
    with gr.Blocks(
        title="法律智能问答系统",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# ⚖️ 法律智能问答系统\n"
            "基于 DeepSeek-R1-Distill + LoRA 微调 + RAG 检索增强的法律咨询系统。\n\n"
            "⚠ **免责声明**：本系统生成的内容仅供参考，不构成正式法律意见。"
        )

        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="请输入您的法律问题",
                    placeholder="例如：生物识别信息属于什么类型的个人信息？",
                    lines=3,
                )

                with gr.Row():
                    use_rag_checkbox = gr.Checkbox(
                        label="启用 RAG 检索",
                        value=True,
                    )
                    top_k_slider = gr.Slider(
                        label="检索数量 (Top-K)",
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                    )

                submit_btn = gr.Button("提交问题", variant="primary")
                clear_btn = gr.Button("清空")

            with gr.Column(scale=2):
                gr.Markdown("### 示例问题")
                example_questions = [
                    "生物识别信息属于什么类型的个人信息？",
                    "处理个人信息需要遵循哪些原则？",
                    "什么是故意犯罪？",
                    "个人信息的处理包括哪些活动？",
                    "中国去年的GDP增长率是多少？",
                ]
                for q in example_questions:
                    gr.Button(q, size="sm").click(
                        lambda x=q: x, outputs=question_input
                    )

        with gr.Row():
            answer_output = gr.Markdown(label="回答")

        with gr.Accordion("检索到的法律条文", open=False):
            retrieved_output = gr.Markdown(label="检索结果")

        latency_output = gr.Markdown(label="状态")

        # 事件绑定
        submit_btn.click(
            answer_question,
            inputs=[question_input, use_rag_checkbox, top_k_slider],
            outputs=[answer_output, retrieved_output, latency_output],
        )

        clear_btn.click(
            lambda: ("", "", ""),
            outputs=[question_input, answer_output, retrieved_output],
        )

    return demo


def launch(
    host: str = "0.0.0.0",
    port: int = 7860,
    **kwargs: Any,
) -> None:
    """启动 Gradio 应用。"""
    demo = create_gradio_app(**kwargs)
    demo.launch(server_name=host, server_port=port)


if __name__ == "__main__":
    launch()
