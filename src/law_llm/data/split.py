# -*- coding: utf-8 -*-
"""数据集划分模块 —— 训练/验证/测试集划分。

支持：
- 按比例随机划分（固定种子）
- 按类别分层划分
- 混合多个数据源后划分
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm


def split_dataset(
    data: list[dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify_key: str | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """划分数据集。

    Args:
        data:         原始数据列表
        train_ratio:  训练集比例
        val_ratio:    验证集比例
        test_ratio:   测试集比例
        seed:         随机种子
        stratify_key: 分层所依据的 key（如 "category"）；None 表示不分层

    Returns:
        (train, val, test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "比例之和必须为 1"
    )

    rng = random.Random(seed)

    if stratify_key is None:
        shuffled = data[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train = shuffled[:n_train]
        val = shuffled[n_train : n_train + n_val]
        test = shuffled[n_train + n_val :]
    else:
        # 分层划分
        groups: dict[str, list[dict]] = defaultdict(list)
        for item in data:
            key = item.get(stratify_key, "unknown")
            groups[key].append(item)

        train, val, test = [], [], []
        for key, items in groups.items():
            shuffled = items[:]
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train.extend(shuffled[:n_train])
            val.extend(shuffled[n_train : n_train + n_val])
            test.extend(shuffled[n_train + n_val :])

        # 打乱各组合并后的顺序
        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)

    return train, val, test


def mix_and_split(
    sources: list[tuple[str, str, float]],
    output_dir: str,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    mix_ratio: dict[str, float] | None = None,
) -> dict[str, Any]:
    """混合多个数据源后划分。

    Args:
        sources:    [(名称, 文件路径, 混合权重), ...]
        output_dir: 输出目录
        train_ratio / val_ratio / test_ratio: 划分比例
        seed:       随机种子
        mix_ratio:  可选的混合比例覆盖，如 {"legal": 0.8, "general": 0.2}

    Returns:
        统计字典
    """
    rng = random.Random(seed)
    all_data: list[dict[str, Any]] = []

    for name, path, weight in sources:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 标记来源
        for item in data:
            item["source"] = name
            item.setdefault("category", name)
        all_data.extend(data)
        print(f"  加载 {name}: {len(data)} 条")

    # 如果指定了混合比例，按比例采样
    if mix_ratio:
        total = len(all_data)
        mixed: list[dict[str, Any]] = []
        for name, _, _ in sources:
            subset = [d for d in all_data if d.get("source") == name]
            target = int(total * mix_ratio.get(name, 0))
            target = min(target, len(subset))
            rng.shuffle(subset)
            mixed.extend(subset[:target])
        all_data = mixed

    # 划分
    train, val, test = split_dataset(
        all_data, train_ratio, val_ratio, test_ratio, seed=seed, stratify_key="source"
    )

    # 保存
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = out / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)

    # 同时生成 LLaMA-Factory 格式（alpaca → 已有格式）
    for name, split in [("train", train), ("val", val)]:
        path = out / f"sft_{name}.json"
        alpaca = [
            {
                "instruction": d.get("instruction", ""),
                "input": d.get("input", ""),
                "output": d.get("output", ""),
            }
            for d in split
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(alpaca, f, ensure_ascii=False, indent=2)

    # dataset_info.json（LLaMA-Factory 需要）
    dataset_info = {
        "law_sft_train": {
            "file_name": "sft_train.json",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
        "law_sft_val": {
            "file_name": "sft_val.json",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
    }
    with open(out / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    return {
        "total": len(all_data),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "seed": seed,
    }
