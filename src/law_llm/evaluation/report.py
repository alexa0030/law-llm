# -*- coding: utf-8 -*-
"""评测报告生成模块。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .metrics import (
    AnswerCorrectnessMetrics,
    CitationMetrics,
    RefusalMetrics,
    RetrievalMetrics,
)


@dataclass
class EvaluationReport:
    """完整评测报告。"""

    experiment_name: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    retrieval: RetrievalMetrics | None = None
    citation: CitationMetrics | None = None
    answer: AnswerCorrectnessMetrics | None = None
    refusal: RefusalMetrics | None = None
    num_samples: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "config": self.config,
            "num_samples": self.num_samples,
            "retrieval": self.retrieval.to_dict() if self.retrieval else None,
            "citation": self.citation.to_dict() if self.citation else None,
            "answer": self.answer.to_dict() if self.answer else None,
            "refusal": self.refusal.to_dict() if self.refusal else None,
            "errors": self.errors,
        }

    def to_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def summary(self) -> str:
        lines = [
            "===== 评测报告 =====",
            f"实验名称: {self.experiment_name}",
            f"样本数量: {self.num_samples}",
            "",
        ]

        if self.retrieval:
            lines.extend([
                "--- 检索效果 ---",
                f"  Recall@1:  {self.retrieval.recall_at_1:.4f}",
                f"  Recall@3:  {self.retrieval.recall_at_3:.4f}",
                f"  Recall@5:  {self.retrieval.recall_at_5:.4f}",
                f"  MRR:       {self.retrieval.mrr:.4f}",
                "",
            ])

        if self.citation:
            lines.extend([
                "--- 法条引用 ---",
                f"  Precision:         {self.citation.precision:.4f}",
                f"  Recall:            {self.citation.recall:.4f}",
                f"  F1:                {self.citation.f1:.4f}",
                f"  幻觉率:             {self.citation.hallucination_rate:.4f}",
                "",
            ])

        if self.answer:
            lines.extend([
                "--- 答案正确性 ---",
                f"  精确匹配率:         {self.answer.exact_match_rate:.4f}",
                f"  关键词匹配率:       {self.answer.keyword_match_rate:.4f}",
                f"  结论判定正确率:     {self.answer.conclusion_accuracy:.4f}",
                f"  LLM Judge 分数:    {self.answer.llm_judge_score:.4f}",
                "",
            ])
            if self.answer.category_breakdown:
                lines.append("  按类别:")
                for cat, acc in self.answer.category_breakdown.items():
                    lines.append(f"    {cat}: {acc:.4f}")
                lines.append("")

        if self.refusal and self.refusal.num_unanswerable > 0:
            lines.extend([
                "--- 拒答能力 ---",
                f"  拒答准确率:         {self.refusal.refusal_accuracy:.4f}",
                f"  不可回答问题数:     {self.refusal.num_unanswerable}",
                "",
            ])

        if self.errors:
            lines.append(f"--- 错误案例 ({len(self.errors)} 条) ---")

        lines.append("注：本评测结果仅供参考，不构成法律意见。")
        return "\n".join(lines)


def generate_comparison_table(
    reports: list[EvaluationReport],
) -> str:
    """生成多组实验的对比表格。"""
    headers = ["实验", "R@1", "R@3", "MRR", "Cite-P", "Cite-R", "Cite-F1", "准确率", "拒答率"]
    rows = []

    for report in reports:
        row = [report.experiment_name]
        if report.retrieval:
            row.extend([
                f"{report.retrieval.recall_at_1:.3f}",
                f"{report.retrieval.recall_at_3:.3f}",
                f"{report.retrieval.mrr:.3f}",
            ])
        else:
            row.extend(["-", "-", "-"])

        if report.citation:
            row.extend([
                f"{report.citation.precision:.3f}",
                f"{report.citation.recall:.3f}",
                f"{report.citation.f1:.3f}",
            ])
        else:
            row.extend(["-", "-", "-"])

        if report.answer:
            row.append(f"{report.answer.conclusion_accuracy:.3f}")
        else:
            row.append("-")

        if report.refusal and report.refusal.num_unanswerable > 0:
            row.append(f"{report.refusal.refusal_accuracy:.3f}")
        else:
            row.append("-")

        rows.append(row)

    # 格式化表格
    col_widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    lines = []
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep_line = "-+-".join("-" * w for w in col_widths)
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        lines.append(" | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)))

    return "\n".join(lines)
