# -*- coding: utf-8 -*-
"""评测脚本 —— 对 RAG 系统进行多维度评测。

支持四组对比实验:
    1. base:       基础模型（无 RAG）
    2. sft:        SFT 微调模型（无 RAG）
    3. base+rag:   基础模型 + RAG
    4. sft+rag:    SFT 微调模型 + RAG

用法:
    python scripts/evaluate.py \
        --eval-dataset data/eval/eval_dataset.json \
        --index-dir data/faiss_index \
        --experiment sft+rag \
        --output-dir outputs/eval_results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from law_llm.evaluation.dataset import load_eval_dataset, create_sample_eval_dataset
from law_llm.evaluation.metrics import (
    evaluate_retrieval,
    evaluate_citations,
    evaluate_answer_correctness,
    evaluate_refusal,
)
from law_llm.evaluation.report import EvaluationReport, generate_comparison_table
from law_llm.rag.indexer import FaissIndexer
from law_llm.rag.pipeline import RAGPipeline
from law_llm.service.model_client import ModelClient


def parse_args():
    parser = argparse.ArgumentParser(description="法律 RAG 系统评测")
    parser.add_argument(
        "--eval-dataset", type=str, default="",
        help="评测数据集路径（为空时使用内置示例集）",
    )
    parser.add_argument(
        "--index-dir", type=str, default="data/faiss_index",
        help="FAISS 索引目录",
    )
    parser.add_argument(
        "--experiment", type=str, default="sft+rag",
        choices=["base", "sft", "base+rag", "sft+rag"],
        help="实验类型",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/eval_results",
        help="评测结果输出目录",
    )
    parser.add_argument(
        "--embedding-model", type=str, default="BAAI/bge-small-zh-v1.5",
        help="embedding 模型",
    )
    parser.add_argument(
        "--model-mode", type=str, default="api",
        choices=["api", "local"],
        help="模型推理模式",
    )
    parser.add_argument(
        "--api-base", type=str, default="http://localhost:8000/v1",
        help="vLLM API 地址",
    )
    parser.add_argument(
        "--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="模型名称",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="检索 Top-K",
    )
    parser.add_argument(
        "--use-llm-judge", action="store_true",
        help="启用 LLM Judge 评测",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="计算设备",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载评测集
    if args.eval_dataset and Path(args.eval_dataset).exists():
        samples = load_eval_dataset(args.eval_dataset)
    else:
        print("⚠ 未指定评测集，使用内置 10 条示例集")
        samples = create_sample_eval_dataset()

    print(f"\n评测集: {len(samples)} 条")
    print(f"实验类型: {args.experiment}")

    use_rag = "rag" in args.experiment
    is_sft = "sft" in args.experiment

    # 初始化组件
    model_client = None
    pipeline = None

    if use_rag:
        indexer = FaissIndexer(embedding_model=args.embedding_model, device=args.device)
        if Path(args.index_dir).exists():
            indexer.load(args.index_dir)
        else:
            print(f"⚠ 索引目录不存在: {args.index_dir}")
            return

    if args.model_mode == "api" or args.model_mode == "local":
        model_client = ModelClient(
            mode=args.model_mode,
            api_base=args.api_base,
            model_name=args.model_name,
            device=args.device,
        )

    if use_rag:
        pipeline = RAGPipeline(
            indexer=indexer,
            model_client=model_client,
            top_k=args.top_k,
            device=args.device,
        )

    # 执行评测
    print("\n开始评测...")
    predicted_answers = []
    predicted_citations = []
    retrieved_results = []
    errors = []

    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {sample.question[:50]}...")
        start = time.time()

        try:
            if use_rag and pipeline:
                response = pipeline.answer(sample.question)
                predicted_answers.append(response.answer)
                predicted_citations.append(response.citations)
                retrieved_results.append(response.retrieved_docs)
            elif model_client:
                messages = [
                    {
                        "role": "system",
                        "content": "你是一个专业的法律咨询助手。请根据你的知识回答问题。"
                        "本回答仅供参考，不构成正式法律意见。",
                    },
                    {"role": "user", "content": sample.question},
                ]
                answer = model_client.chat(messages)
                predicted_answers.append(answer)
                predicted_citations.append([])
                retrieved_results.append([])
            else:
                predicted_answers.append("(无模型可用)")
                predicted_citations.append([])
                retrieved_results.append([])

            latency = time.time() - start
            print(f"    耗时: {latency:.2f}s")

        except Exception as e:
            print(f"    ✗ 错误: {e}")
            predicted_answers.append(f"(错误: {e})")
            predicted_citations.append([])
            retrieved_results.append([])
            errors.append({"question": sample.question, "error": str(e)})

    # 计算指标
    print("\n计算评测指标...")

    report = EvaluationReport(
        experiment_name=args.experiment,
        config={
            "use_rag": use_rag,
            "is_sft": is_sft,
            "model_name": args.model_name,
            "top_k": args.top_k,
            "num_samples": len(samples),
        },
        num_samples=len(samples),
        errors=errors,
    )

    if use_rag:
        report.retrieval = evaluate_retrieval(samples, retrieved_results)
        report.citation = evaluate_citations(samples, predicted_citations)

    report.answer = evaluate_answer_correctness(
        samples,
        predicted_answers,
        use_llm_judge=args.use_llm_judge,
        llm_client=model_client if args.use_llm_judge else None,
    )

    report.refusal = evaluate_refusal(samples, predicted_answers)

    # 输出结果
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = str(out_dir / f"{args.experiment}_report.json")
    report.to_json(report_path)

    # 打印摘要
    print("\n" + "=" * 60)
    print(report.summary())
    print(f"\n报告已保存至 {report_path}")

    # 保存详细预测结果
    predictions_path = str(out_dir / f"{args.experiment}_predictions.json")
    predictions = []
    for sample, answer, citations, retrieved in zip(
        samples, predicted_answers, predicted_citations, retrieved_results
    ):
        predictions.append({
            "question": sample.question,
            "ground_truth": sample.ground_truth,
            "predicted_answer": answer,
            "citations": citations,
            "retrieved_docs": retrieved,
            "category": sample.category,
        })
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"预测结果已保存至 {predictions_path}")


if __name__ == "__main__":
    main()
