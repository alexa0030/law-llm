# -*- coding: utf-8 -*-
"""评测指标计算模块。

实现多维度评测：
- 检索效果：Recall@k, MRR
- 法条引用：Citation Precision, Citation Recall
- 答案正确性：LLM Judge, 字符串匹配准确率
- 拒答能力：不可回答问题的拒答准确率
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any

from .dataset import EvalSample


# ---------------------------------------------------------------------------
# 检索评测
# ---------------------------------------------------------------------------

@dataclass
class RetrievalMetrics:
    """检索效果指标。"""
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    num_queries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_retrieval(
    samples: list[EvalSample],
    retrieved_results: list[list[dict[str, Any]]],
) -> RetrievalMetrics:
    """计算检索指标。

    Args:
        samples:           评测样本
        retrieved_results: 每个样本的检索结果列表

    Returns:
        RetrievalMetrics
    """
    total = len(samples)
    if total == 0:
        return RetrievalMetrics()

    recall_1 = 0
    recall_3 = 0
    recall_5 = 0
    reciprocal_ranks = []

    for sample, results in zip(samples, retrieved_results):
        if not sample.relevant_laws:
            continue  # 跳过没有标注的样本

        # 将标注的法律条文转为集合
        relevant_set = {
            (law["law_name"], law["article"])
            for law in sample.relevant_laws
        }

        # 检查检索结果中是否包含标注的法律条文
        found_ranks = []
        for rank, result in enumerate(results, 1):
            law_name = result.get("law_name", "")
            article = result.get("article", "")
            # 模糊匹配法律名称
            for ref_law, ref_article in relevant_set:
                if (ref_law in law_name or law_name in ref_law) and ref_article == article:
                    found_ranks.append(rank)
                    break

        if found_ranks:
            min_rank = min(found_ranks)
            if min_rank <= 1:
                recall_1 += 1
            if min_rank <= 3:
                recall_3 += 1
            if min_rank <= 5:
                recall_5 += 1
            reciprocal_ranks.append(1.0 / min_rank)
        else:
            reciprocal_ranks.append(0.0)

    return RetrievalMetrics(
        recall_at_1=recall_1 / total,
        recall_at_3=recall_3 / total,
        recall_at_5=recall_5 / total,
        mrr=sum(reciprocal_ranks) / total,
        num_queries=total,
    )


# ---------------------------------------------------------------------------
# 法条引用评测
# ---------------------------------------------------------------------------

@dataclass
class CitationMetrics:
    """法条引用指标。"""
    precision: float = 0.0  # 引用的法条中有多少是正确的
    recall: float = 0.0  # 应该引用的法条中有多少被引用了
    f1: float = 0.0
    hallucination_rate: float = 0.0  # 引用了不在检索结果中的法条比例
    num_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_citations(
    samples: list[EvalSample],
    predicted_citations: list[list[dict[str, Any]]],
) -> CitationMetrics:
    """计算法条引用指标。

    Args:
        samples:             评测样本
        predicted_citations: 每个样本的回答中提取的引用列表

    Returns:
        CitationMetrics
    """
    total = len(samples)
    if total == 0:
        return CitationMetrics()

    total_precision = 0.0
    total_recall = 0.0
    total_hallucination = 0.0
    valid_samples = 0

    for sample, citations in zip(samples, predicted_citations):
        if not sample.relevant_laws and not citations:
            valid_samples += 1
            continue

        if not sample.relevant_laws:
            # 不应该有引用但有引用 → 全是幻觉
            if citations:
                total_hallucination += 1.0
                total_precision += 0.0
            valid_samples += 1
            continue

        relevant_set = {
            (law["law_name"], law["article"])
            for law in sample.relevant_laws
        }
        predicted_set = {
            (c["law_name"], c["article"])
            for c in citations
        }

        # 精确率：引用中有多少是正确的
        if predicted_set:
            correct = 0
            for pred_law, pred_article in predicted_set:
                for ref_law, ref_article in relevant_set:
                    if (ref_law in pred_law or pred_law in ref_law) and ref_article == pred_article:
                        correct += 1
                        break
            precision = correct / len(predicted_set)
        else:
            precision = 0.0

        # 召回率：应该引用的有多少被引用了
        if relevant_set:
            found = 0
            for ref_law, ref_article in relevant_set:
                for pred_law, pred_article in predicted_set:
                    if (ref_law in pred_law or pred_law in ref_law) and ref_article == pred_article:
                        found += 1
                        break
            recall = found / len(relevant_set)
        else:
            recall = 1.0 if not predicted_set else 0.0

        # 幻觉率：引用中不在检索结果中的比例
        if citations:
            in_retrieved = sum(1 for c in citations if c.get("in_retrieved", False))
            hallucination = 1 - in_retrieved / len(citations)
        else:
            hallucination = 0.0

        total_precision += precision
        total_recall += recall
        total_hallucination += hallucination
        valid_samples += 1

    avg_precision = total_precision / valid_samples if valid_samples else 0
    avg_recall = total_recall / valid_samples if valid_samples else 0
    f1 = (
        2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        if (avg_precision + avg_recall) > 0
        else 0.0
    )

    return CitationMetrics(
        precision=avg_precision,
        recall=avg_recall,
        f1=f1,
        hallucination_rate=total_hallucination / valid_samples if valid_samples else 0,
        num_samples=valid_samples,
    )


# ---------------------------------------------------------------------------
# 答案正确性评测
# ---------------------------------------------------------------------------

@dataclass
class AnswerCorrectnessMetrics:
    """答案正确性指标。"""
    exact_match_rate: float = 0.0
    keyword_match_rate: float = 0.0
    llm_judge_score: float = 0.0  # 0-1
    conclusion_accuracy: float = 0.0  # 结论判定正确率
    num_samples: int = 0
    category_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_answer_correctness(
    samples: list[EvalSample],
    predicted_answers: list[str],
    use_llm_judge: bool = False,
    llm_client: Any | None = None,
) -> AnswerCorrectnessMetrics:
    """计算答案正确性指标。

    使用关键词匹配作为基础评测，可选 LLM Judge。
    """
    total = len(samples)
    if total == 0:
        return AnswerCorrectnessMetrics()

    exact_matches = 0
    keyword_matches = 0
    llm_scores = []
    category_correct: dict[str, list[bool]] = {}

    for sample, pred_answer in zip(samples, predicted_answers):
        gt = sample.ground_truth
        category = sample.category

        # 关键词匹配：检查 ground_truth 中的关键信息是否出现在预测答案中
        keywords = _extract_keywords(gt)
        matched = sum(1 for kw in keywords if kw in pred_answer)
        keyword_score = matched / len(keywords) if keywords else 1.0

        if keyword_score >= 0.5:
            keyword_matches += 1
            category_correct.setdefault(category, []).append(True)
        else:
            category_correct.setdefault(category, []).append(False)

        # 精确匹配（归一化后比较）
        if _normalize(gt) == _normalize(pred_answer):
            exact_matches += 1

        # LLM Judge
        if use_llm_judge and llm_client is not None:
            score = _llm_judge(llm_client, sample.question, gt, pred_answer)
            llm_scores.append(score)

    keyword_rate = keyword_matches / total
    exact_rate = exact_matches / total
    llm_avg = sum(llm_scores) / len(llm_scores) if llm_scores else 0.0

    # 按类别统计正确率
    cat_breakdown = {}
    for cat, results in category_correct.items():
        cat_breakdown[cat] = sum(results) / len(results) if results else 0.0

    return AnswerCorrectnessMetrics(
        exact_match_rate=exact_rate,
        keyword_match_rate=keyword_rate,
        llm_judge_score=llm_avg,
        conclusion_accuracy=keyword_rate,  # 以关键词匹配率作为结论准确率
        num_samples=total,
        category_breakdown=cat_breakdown,
    )


# ---------------------------------------------------------------------------
# 拒答能力评测
# ---------------------------------------------------------------------------

@dataclass
class RefusalMetrics:
    """拒答能力指标。"""
    refusal_accuracy: float = 0.0  # 不可回答问题的拒答准确率
    num_unanswerable: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_refusal(
    samples: list[EvalSample],
    predicted_answers: list[str],
) -> RefusalMetrics:
    """计算拒答能力指标。"""
    refusal_keywords = [
        "无法", "不清楚", "不在", "未检索到", "不构成法律意见",
        "建议咨询", "无法完整回答", "不在法律知识库",
    ]

    unanswerable = [
        (s, a) for s, a in zip(samples, predicted_answers)
        if s.category == "unanswerable"
    ]

    if not unanswerable:
        return RefusalMetrics(refusal_accuracy=0.0, num_unanswerable=0)

    correct_refusals = 0
    for sample, answer in unanswerable:
        if any(kw in answer for kw in refusal_keywords):
            correct_refusals += 1

    return RefusalMetrics(
        refusal_accuracy=correct_refusals / len(unanswerable),
        num_unanswerable=len(unanswerable),
    )


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """文本归一化用于比较。"""
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。！？；：""''（）【】《》、,.!?;:]", "", text)
    return text.lower().strip()


def _extract_keywords(text: str) -> list[str]:
    """从 ground truth 中提取关键词用于匹配评测。"""
    # 去除标点和停用词，保留有意义的内容
    cleaned = re.sub(r"[，。！？；：""''（）【】《》、,.!?;:\s]+", " ", text)
    words = cleaned.split()

    # 过滤过短的词
    keywords = [w for w in words if len(w) >= 2]

    # 如果文本不长，也加入完整文本作为关键词
    if len(text) < 100:
        keywords.append(_normalize(text))

    return keywords if keywords else [text]


def _llm_judge(
    client: Any,
    question: str,
    ground_truth: str,
    predicted: str,
) -> float:
    """使用 LLM 作为评判者打分（0-1）。

    评判标准：
    - 1.0: 完全正确，包含所有关键信息
    - 0.5: 部分正确，缺少部分信息
    - 0.0: 完全错误
    """
    prompt = f"""请评判以下回答的质量。

问题：{question}
标准答案：{ground_truth}
待评判回答：{predicted}

请只输出一个 0 到 1 之间的数字，表示回答的正确程度：
- 1.0: 完全正确
- 0.5: 部分正确
- 0.0: 完全错误

只输出数字，不要其他内容。"""

    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat(messages)
        score = float(response.strip())
        return max(0.0, min(1.0, score))
    except (ValueError, Exception):
        return 0.5  # 默认中等分数
