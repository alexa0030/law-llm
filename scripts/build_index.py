# -*- coding: utf-8 -*-
"""构建 FAISS 法律知识库索引。

用法:
    python scripts/build_index.py \
        --law-dir data/laws \
        --output-dir data/faiss_index \
        --embedding-model BAAI/bge-small-zh-v1.5 \
        --chunk-size 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from law_llm.rag.loader import LawLoader
from law_llm.rag.splitter import LawSplitter
from law_llm.rag.indexer import FaissIndexer


def parse_args():
    parser = argparse.ArgumentParser(description="构建 FAISS 法律知识库索引")
    parser.add_argument(
        "--law-dir", type=str, default="data/laws",
        help="法律文档目录（Markdown 文件）",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/faiss_index",
        help="索引输出目录",
    )
    parser.add_argument(
        "--embedding-model", type=str, default="BAAI/bge-small-zh-v1.5",
        help="Sentence-BERT 模型",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=512,
        help="文本块最大长度",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=50,
        help="文本块重叠长度",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="计算设备 (cpu/cuda)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("构建 FAISS 法律知识库索引")
    print("=" * 60)

    # Step 1: 加载法律文档
    print(f"\n[1/3] 加载法律文档: {args.law_dir}")
    loader = LawLoader(args.law_dir)
    documents = loader.load()
    print(f"  加载了 {len(documents)} 个法律文档")
    for doc in documents:
        print(f"    - {doc.metadata.get('law_name', '未知')} ({doc.metadata.get('source', '')})")

    # Step 2: 切分法条
    print(f"\n[2/3] 切分法条 (chunk_size={args.chunk_size})")
    splitter = LawSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"  切分为 {len(chunks)} 个法条块")

    # 统计切分结果
    law_stats = {}
    for chunk in chunks:
        law_name = chunk.metadata.get("law_name", "未知")
        law_stats[law_name] = law_stats.get(law_name, 0) + 1
    for law, count in sorted(law_stats.items()):
        print(f"    {law}: {count} 条")

    # Step 3: 构建 FAISS 索引
    print(f"\n[3/3] 构建 FAISS 索引 (model={args.embedding_model})")
    indexer = FaissIndexer(
        embedding_model=args.embedding_model,
        device=args.device,
    )
    indexer.build_index(chunks)

    # 保存
    indexer.save(args.output_dir)
    print(f"\n✓ 索引构建完成，保存至 {args.output_dir}")
    print(f"  共 {len(chunks)} 条法条，可使用 retriever 进行检索")


if __name__ == "__main__":
    main()
