# -*- coding: utf-8 -*-
"""代表性采样模块 —— K-Center-Greedy 聚类采样。

K-Center-Greedy 算法：
1. 随机选择一个初始中心
2. 每次选择距离已选中心集合最远的样本作为新中心
3. 重复直到达到目标数量

这样选出的子集在向量空间中分布均匀，具有代表性。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


def k_center_greedy(
    embeddings: np.ndarray,
    n_samples: int,
    seed: int = 42,
) -> list[int]:
    """K-Center-Greedy 聚类采样。

    Args:
        embeddings: (N, dim) 向量矩阵
        n_samples:  要采样的数量
        seed:       随机种子

    Returns:
        被选中的样本索引列表
    """
    rng = np.random.RandomState(seed)
    n = embeddings.shape[0]
    n_samples = min(n_samples, n)

    # 归一化用于余弦距离
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms

    # 随机选择初始中心
    first = rng.randint(n)
    selected = [first]

    # min_distances[i] = 样本 i 到已选中心集合的最小距离
    min_distances = 1 - normalized @ normalized[first]

    for _ in tqdm(range(1, n_samples), desc="K-Center-Greedy 采样"):
        # 选择距离已选中心最远的样本
        next_idx = int(np.argmax(min_distances))
        selected.append(next_idx)

        # 更新最小距离
        new_distances = 1 - normalized @ normalized[next_idx]
        min_distances = np.minimum(min_distances, new_distances)

    return sorted(selected)


def ability_based_sample(
    data: list[dict[str, Any]],
    embeddings: np.ndarray,
    quotas: dict[str, int],
    ability_keywords: dict[str, list[str]] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """按能力分类 + K-Center-Greedy 采样。

    先用关键词对数据分类，再在每个类别内做 K-Center-Greedy 采样。

    Args:
        data:      SFT 样本列表
        embeddings: 对应的向量矩阵
        quotas:    {能力类别: 采样数量}
        ability_keywords: {能力类别: [关键词列表]}
        seed:      随机种子

    Returns:
        采样后的数据列表
    """
    if ability_keywords is None:
        ability_keywords = {
            "长文本理解": ["背景", "事件", "过程", "多轮对话", "上下文", "篇章",
                         "段落", "文章", "文档", "报告", "论文", "故事", "叙述",
                         "描述", "详细说明", "全面分析", "整体理解", "全文",
                         "摘要", "总结", "概述", "梗概", "大意", "主旨",
                         "中心思想", "结构分析", "逻辑关系", "因果关系",
                         "时间顺序", "空间顺序", "论证过程", "推理链条",
                         "证据支持", "论点论据", "文本解读", "语义分析",
                         "情感分析", "意图识别", "信息抽取", "关键信息",
                         "核心内容", "重点部分"],
            "逻辑推理": ["为什么", "因为", "所以", "导致", "根据规则", "推理",
                         "推断", "推论", "演绎", "归纳", "类比", "因果",
                         "前提", "结论", "假设", "证明", "论证", "理由",
                         "依据", "证据", "逻辑", "必然", "可能", "必然性",
                         "可能性", "充分条件", "必要条件", "因果关系",
                         "相关关系", "前提条件", "结论推导", "逻辑链条",
                         "推理过程", "思维过程", "分析判断", "评估判断",
                         "决策过程", "问题解决", "策略制定", "方案选择",
                         "最优解", "最优化", "权衡利弊"],
            "结构化表达": ["1.", "2.", "首先", "其次", "根据", "第一", "第二",
                           "第三", "第四", "第五", "最后", "综上所述",
                           "总而言之", "总的来说", "一方面", "另一方面",
                           "此外", "另外", "同时", "而且", "并且", "然而",
                           "但是", "虽然", "尽管", "因此", "于是", "结果",
                           "导致", "造成", "影响", "步骤", "流程", "方法",
                           "方式", "途径", "手段", "措施", "方案", "计划",
                           "安排", "顺序", "层次", "级别", "分类", "分组",
                           "列表", "表格", "图表", "图示", "框架", "结构",
                           "体系", "系统", "模块", "组件", "部分", "要素",
                           "因素"],
            "歧义消解": ["什么意思", "具体", "哪方面", "澄清", "解释", "说明",
                         "定义", "含义", "概念", "术语", "专业名词", "缩写",
                         "简称", "全称", "同义词", "近义词", "反义词", "多义词",
                         "歧义", "模糊", "不明确", "不清楚", "不确定", "困惑",
                         "疑问", "质疑", "询问", "求证", "确认", "核实", "验证",
                         "辨别", "区分", "分辨", "识别", "判断", "确定", "明确",
                         "具体化", "细化", "详细说明", "举例说明", "实例说明",
                         "案例说明", "比喻说明", "类比说明", "对比说明"],
        }

    # 分类
    ability_data: dict[str, list[int]] = {k: [] for k in ability_keywords}
    unclassified: list[int] = []

    for i, item in enumerate(data):
        text = f"{item.get('instruction', '')} {item.get('output', '')}"
        matched = False
        for ability, keywords in ability_keywords.items():
            if any(k in text for k in keywords):
                ability_data[ability].append(i)
                matched = True
                break
        if not matched:
            unclassified.append(i)

    # 每个类别内做 K-Center-Greedy
    selected_indices: list[int] = []
    for ability, quota in quotas.items():
        candidates = ability_data[ability]
        actual_quota = min(quota, len(candidates))
        if actual_quota == 0:
            print(f"  ⚠ 类别 '{ability}' 无可用样本，跳过")
            continue

        candidate_embeddings = embeddings[candidates]
        chosen = k_center_greedy(candidate_embeddings, actual_quota, seed=seed)
        selected_indices.extend([candidates[j] for j in chosen])
        print(f"  {ability}: 请求 {quota}, 可用 {len(candidates)}, 采样 {actual_quota}")

    # 未分类样本也保留一部分
    if unclassified:
        remaining_quota = sum(quotas.values()) - len(selected_indices)
        if remaining_quota > 0:
            unc_embeddings = embeddings[unclassified]
            actual = min(remaining_quota, len(unclassified))
            chosen = k_center_greedy(unc_embeddings, actual, seed=seed)
            selected_indices.extend([unclassified[j] for j in chosen])

    selected_indices.sort()
    return [data[i] for i in selected_indices]


def sample_dataset(
    input_path: str,
    output_path: str,
    *,
    embedding_model: str = "BAAI/bge-small-zh-v1.5",
    n_samples: int = 3000,
    quotas: dict[str, int] | None = None,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    """对数据集执行代表性采样。

    Args:
        input_path:  输入 JSON 路径
        output_path: 输出 JSON 路径
        embedding_model: Sentence-BERT 模型
        n_samples:   目标采样数量
        quotas:      各能力类别的配额；为 None 时使用默认比例
        seed:        随机种子
        device:      计算设备

    Returns:
        统计字典
    """
    from .deduplicate import generate_embeddings

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if quotas is None:
        quotas = {
            "长文本理解": int(n_samples * 0.30),
            "逻辑推理": int(n_samples * 0.30),
            "结构化表达": int(n_samples * 0.20),
            "歧义消解": int(n_samples * 0.20),
        }

    embeddings = generate_embeddings(data, embedding_model, device=device)
    sampled = ability_based_sample(data, embeddings, quotas, seed=seed)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    return {
        "input_count": len(data),
        "sampled_count": len(sampled),
        "quotas": quotas,
        "seed": seed,
    }
