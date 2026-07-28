# -*- coding: utf-8 -*-
"""数据处理报告生成模块。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import ProcessingReport


def build_report(
    original_count: int,
    format_error_count: int,
    rule_filtered_count: int,
    semantic_duplicate_count: int,
    final_count: int,
    train_count: int = 0,
    val_count: int = 0,
    test_count: int = 0,
    category_distribution: dict[str, int] | None = None,
    clean_reason_distribution: dict[str, int] | None = None,
    seed: int = 42,
    config: dict[str, Any] | None = None,
) -> ProcessingReport:
    """构建处理报告对象。"""
    return ProcessingReport(
        original_count=original_count,
        format_error_count=format_error_count,
        rule_filtered_count=rule_filtered_count,
        semantic_duplicate_count=semantic_duplicate_count,
        final_count=final_count,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        category_distribution=category_distribution or {},
        clean_reason_distribution=clean_reason_distribution or {},
        seed=seed,
        config=config or {},
    )


def save_report(report: ProcessingReport, path: str) -> None:
    """保存报告为 JSON 文件。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    report.to_json(path)


def print_report(report: ProcessingReport) -> None:
    """打印报告摘要。"""
    print(report.summary())
