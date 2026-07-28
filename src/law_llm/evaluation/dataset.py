# -*- coding: utf-8 -*-
"""评测数据集管理模块。

定义评测集格式、加载和保存功能。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class EvalSample:
    """一条评测样本。"""

    question: str
    ground_truth: str
    category: str = "legal_consultation"  # legal_consultation / legal_exam / unanswerable
    relevant_laws: list[dict[str, str]] = field(default_factory=list)
    # relevant_laws: [{"law_name": "...", "article": "第二十八条"}]
    expected_answer_type: str = "with_citation"  # with_citation / factual / refuse
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalSample":
        return cls(
            question=d.get("question", ""),
            ground_truth=d.get("ground_truth", ""),
            category=d.get("category", "legal_consultation"),
            relevant_laws=d.get("relevant_laws", []),
            expected_answer_type=d.get("expected_answer_type", "with_citation"),
            metadata=d.get("metadata", {}),
        )


def load_eval_dataset(path: str) -> list[EvalSample]:
    """加载评测数据集。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [EvalSample.from_dict(d) for d in data]


def save_eval_dataset(samples: list[EvalSample], path: str) -> None:
    """保存评测数据集。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in samples], f, ensure_ascii=False, indent=2)


def create_sample_eval_dataset() -> list[EvalSample]:
    """创建示例评测集（10 条）。"""
    return [
        EvalSample(
            question="生物识别信息属于什么类型的个人信息？",
            ground_truth="生物识别信息属于敏感个人信息。",
            relevant_laws=[
                {"law_name": "中华人民共和国个人信息保护法", "article": "第二十八条"}
            ],
            expected_answer_type="with_citation",
        ),
        EvalSample(
            question="处理个人信息需要遵循哪些原则？",
            ground_truth="处理个人信息应遵循合法、正当、必要和诚信原则。",
            relevant_laws=[
                {"law_name": "中华人民共和国个人信息保护法", "article": "第五条"}
            ],
            expected_answer_type="with_citation",
        ),
        EvalSample(
            question="什么是故意犯罪？",
            ground_truth="明知自己的行为会发生危害社会的结果，并且希望或者放任这种结果发生，因而构成犯罪的，是故意犯罪。",
            relevant_laws=[
                {"law_name": "中华人民共和国刑法", "article": "第十四条"}
            ],
            expected_answer_type="with_citation",
        ),
        EvalSample(
            question="个人信息的处理包括哪些活动？",
            ground_truth="个人信息的处理包括个人信息的收集、存储、使用、加工、传输、提供、公开、删除等。",
            relevant_laws=[
                {"law_name": "中华人民共和国个人信息保护法", "article": "第四条"}
            ],
            expected_answer_type="with_citation",
        ),
        EvalSample(
            question="刑法的任务是什么？",
            ground_truth="刑法的任务是用刑罚同一切犯罪行为作斗争，以保卫国家安全，保卫人民民主专政的政权和社会主义制度，保护国有财产和劳动群众集体所有的财产，保护公民私人所有的财产，保护公民的人身权利、民主权利和其他权利，维护社会秩序、经济秩序，保障社会主义建设事业的顺利进行。",
            relevant_laws=[
                {"law_name": "中华人民共和国刑法", "article": "第二条"}
            ],
            expected_answer_type="with_citation",
        ),
        EvalSample(
            question="什么是法律面前人人平等？",
            ground_truth="对任何人犯罪，在适用法律上一律平等。不允许任何人有超越法律的特权。",
            relevant_laws=[
                {"law_name": "中华人民共和国刑法", "article": "第四条"}
            ],
            expected_answer_type="with_citation",
        ),
        EvalSample(
            question="个人信息处理者在处理个人信息前应当告知哪些事项？",
            ground_truth="个人信息处理者在处理个人信息前，应当以显著方式、清晰易懂的语言真实、准确、完整地向个人告知个人信息处理者的名称或者姓名和联系方式等信息。",
            relevant_laws=[
                {"law_name": "中华人民共和国个人信息保护法", "article": "第十七条"}
            ],
            expected_answer_type="with_citation",
        ),
        EvalSample(
            question="什么是犯罪未完成形态？",
            ground_truth="犯罪未完成形态包括犯罪预备、犯罪未遂和犯罪中止。",
            relevant_laws=[
                {"law_name": "中华人民共和国刑法", "article": "第五章"}
            ],
            expected_answer_type="with_citation",
        ),
        EvalSample(
            question="中国去年的GDP增长率是多少？",
            ground_truth="该问题不在法律知识库覆盖范围内。",
            category="unanswerable",
            relevant_laws=[],
            expected_answer_type="refuse",
        ),
        EvalSample(
            question="个人撤回同意后，之前的处理活动是否还有效？",
            ground_truth="个人撤回同意，不影响撤回前基于个人同意已进行的个人信息处理活动的效力。",
            relevant_laws=[
                {"law_name": "中华人民共和国个人信息保护法", "article": "第十五条"}
            ],
            expected_answer_type="with_citation",
        ),
    ]
