# -*- coding: utf-8 -*-
"""语义去重模块 —— 基于向量相似度的 FAISS 近重复检测。

实现流程：
1. 文本归一化（去空白、统一标点）
2. 精确重复去除（hash）
3. Sentence-BERT 向量生成
4. FAISS 相似邻居检索
5. 阈值去重
6. 记录保留/删除映射
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


def normalize_text(text: str) -> str:
    """文本归一化：去多余空白、统一标点。"""
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。！？；：""''（）【】《》]", lambda m: m.group(0), text)
    return text.strip()


def text_hash(text: str) -> str:
    """计算归一化后文本的 MD5 哈希。"""
    return hashlib.md5(normalize_text(text).encode("utf-8")).hexdigest()


def remove_exact_duplicates(
    data: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    """精确重复去除（基于归一化文本哈希）。

    Returns:
        (去重后数据, 被删除的原始索引列表)
    """
    seen: dict[str, int] = {}
    unique: list[dict[str, Any]] = []
    removed_indices: list[int] = []

    for i, item in enumerate(data):
        key = text_hash(f"{item.get('instruction', '')} {item.get('output', '')}")
        if key in seen:
            removed_indices.append(i)
        else:
            seen[key] = i
            unique.append(item)

    return unique, removed_indices


def generate_embeddings(
    data: list[dict[str, Any]],
    model_name: str = "BAAI/bge-small-zh-v1.5",
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    """为数据列表生成 Sentence-BERT 向量。

    Args:
        data:          SFT 样本列表
        model_name:    HuggingFace 模型名或本地路径
        batch_size:    批大小
        device:        cpu / cuda

    Returns:
        (N, dim) numpy 数组
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    texts = [f"{item.get('instruction', '')} {item.get('output', '')}" for item in data]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


def faiss_semantic_dedup(
    embeddings: np.ndarray,
    threshold: float = 0.80,
) -> tuple[list[int], list[tuple[int, int, float]]]:
    """使用 FAISS 进行语义去重。

    通过 L2 距离找到每条样本的近重复邻居，按相似度阈值去重。

    Args:
        embeddings: (N, dim) 归一化后的向量
        threshold:  余弦相似度阈值（>= threshold 视为重复）

    Returns:
        (保留的索引列表, 删除记录 [(removed_idx, kept_idx, similarity)])
    """
    import faiss

    n, dim = embeddings.shape
    if n == 0:
        return [], []

    # 归一化向量用于内积搜索（等价于余弦相似度）
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    # 对每条样本搜索最近的 k 个邻居（含自身）
    k = min(n, 20)
    similarities, indices = index.search(embeddings.astype(np.float32), k)

    # 并查集：将相似度 >= threshold 的样本合并
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    removed_pairs: list[tuple[int, int, float]] = []

    for i in range(n):
        for j_pos in range(1, k):  # 跳过自身（位置 0）
            neighbor = int(indices[i][j_pos])
            sim = float(similarities[i][j_pos])
            if neighbor < 0:
                break
            if sim >= threshold:
                if find(i) != find(neighbor):
                    # 保留较小的索引，删除较大的
                    kept, removed = sorted([find(i), find(neighbor)])
                    union(kept, removed)
                    removed_pairs.append((removed, kept, sim))

    # 每组保留代表（最小索引）
    keep_set: set[int] = set()
    for i in range(n):
        keep_set.add(find(i))

    keep_indices = sorted(keep_set)
    return keep_indices, removed_pairs


def deduplicate_dataset(
    input_path: str,
    output_path: str,
    log_path: str | None = None,
    *,
    embedding_model: str = "BAAI/bge-small-zh-v1.5",
    similarity_threshold: float = 0.80,
    device: str = "cpu",
    batch_size: int = 64,
) -> dict[str, Any]:
    """完整的去重流程：精确去重 → 语义去重。

    Returns:
        统计字典
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_count = len(data)

    # Step 1: 精确去重
    data, exact_removed = remove_exact_duplicates(data)
    exact_count = len(data)

    # Step 2: 语义去重
    embeddings = generate_embeddings(data, embedding_model, batch_size, device)
    keep_indices, removed_pairs = faiss_semantic_dedup(embeddings, similarity_threshold)

    deduped = [data[i] for i in keep_indices]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    if log_path:
        log = {
            "original_count": original_count,
            "exact_duplicate_removed": len(exact_removed),
            "after_exact": exact_count,
            "semantic_duplicate_removed": exact_count - len(deduped),
            "final_count": len(deduped),
            "threshold": similarity_threshold,
            "removed_pairs_sample": [
                {"removed": r, "kept": k, "similarity": round(s, 4)}
                for r, k, s in removed_pairs[:100]
            ],
        }
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)

    return {
        "original": original_count,
        "exact_removed": len(exact_removed),
        "semantic_removed": exact_count - len(deduped),
        "final": len(deduped),
        "threshold": similarity_threshold,
    }
