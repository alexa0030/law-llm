# -*- coding: utf-8 -*-
"""法条引用评测专用模块。"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class CitationEvaluation:
    """单条样本的法条引用评测结果。"""

    question: str
    predicted_citations: list[dict[str, Any]]
    ground_truth_citations: list[dict[str, Any]]
    correct: list[bool]  # 每条预测引用是否正确
    missed: list[dict[str, Any]]  # 漏掉的标注引用
    hallucinated: list[dict[str, Any]]  # 不在检索结果中的引用

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_citations_from_text(text: str) -> list[dict[str, str]]:
    """从文本中提取法条引用。

    支持格式：
        《法律名》第X条
        《法律名》第X章
    """
    citations = []
    pattern = re.compile(r"《([^》]+)》第([一二三四五六七八九十百千万零\d]+[条章节])")
    for match in pattern.finditer(text):
        citations.append({
            "law_name": match.group(1),
            "article": match.group(2),
        })
    return citations


def evaluate_single_citation(
    predicted: list[dict[str, Any]],
    ground_truth: list[dict[str, str]],
    retrieved_docs: list[dict[str, Any]] | None = None,
) -> CitationEvaluation:
    """评测单条样本的法条引用。

    Args:
        predicted:      预测引用列表
        ground_truth:   标注引用列表
        retrieved_docs: 检索到的文档（用于判断幻觉）

    Returns:
        CitationEvaluation
    """
    # 构建 retrieved 集合用于幻觉检测
    retrieved_set = set()
    if retrieved_docs:
        for doc in retrieved_docs:
            law_name = doc.get("law_name", "")
            article = doc.get("article", "")
            if law_name and article:
                retrieved_set.add((law_name, article))

    # 判断每条预测引用是否正确
    correct_flags = []
    hallucinated = []
    for pred in predicted:
        pred_law = pred.get("law_name", "")
        pred_article = pred.get("article", "")

        # 与标注对比
        is_correct = False
        for gt in ground_truth:
            gt_law = gt.get("law_name", "")
            gt_article = gt.get("article", "")
            if (gt_law in pred_law or pred_law in gt_law) and gt_article == pred_article:
                is_correct = True
                break
        correct_flags.append(is_correct)

        # 幻觉检测：引用不在检索结果中
        in_retrieved = pred.get("in_retrieved", False)
        if not in_retrieved and retrieved_set:
            # 进一步检查
            found = False
            for ret_law, ret_article in retrieved_set:
                if (ret_law in pred_law or pred_law in ret_law) and ret_article == pred_article:
                    found = True
                    break
            if not found:
                hallucinated.append(pred)

    # 找出漏掉的标注引用
    missed = []
    for gt in ground_truth:
        gt_law = gt.get("law_name", "")
        gt_article = gt.get("article", "")
        found = False
        for pred in predicted:
            pred_law = pred.get("law_name", "")
            pred_article = pred.get("article", "")
            if (gt_law in pred_law or pred_law in gt_law) and gt_article == pred_article:
                found = True
                break
        if not found:
            missed.append(gt)

    return CitationEvaluation(
        question="",  # 由调用方填充
        predicted_citations=predicted,
        ground_truth_citations=ground_truth,
        correct=correct_flags,
        missed=missed,
        hallucinated=hallucinated,
    )
