# -*- coding: utf-8 -*-
"""FAISS 索引构建与管理模块。"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from langchain_core.documents import Document


class FaissIndexer:
    """基于 FAISS 的法律知识库索引器。

    功能：
        - 为法条文档生成 embedding
        - 构建 FAISS 索引
        - 保存/加载索引和文档
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        device: str = "cpu",
    ) -> None:
        self.embedding_model_name = embedding_model
        self.device = device
        self._encoder = None
        self._index = None
        self._documents: list[Document] = []

    @property
    def encoder(self):
        """延迟加载 SentenceTransformer。"""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(
                self.embedding_model_name, device=self.device
            )
        return self._encoder

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """生成文本向量。"""
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def build_index(self, documents: list[Document]) -> None:
        """构建 FAISS 索引。

        Args:
            documents: 切分后的法条 Document 列表
        """
        import faiss

        self._documents = documents
        texts = [doc.page_content for doc in documents]
        embeddings = self.embed_texts(texts)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

        print(f"  FAISS 索引构建完成: {len(documents)} 条法条, 维度 {dim}")

    def save(self, output_dir: str) -> None:
        """保存索引和文档到目录。"""
        import faiss

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 保存 FAISS 索引
        if self._index is not None:
            faiss.write_index(self._index, str(out / "faiss.index"))

        # 保存文档（含元数据）
        docs_data = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in self._documents
        ]
        with open(out / "documents.json", "w", encoding="utf-8") as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)

        # 保存配置
        config = {
            "embedding_model": self.embedding_model_name,
            "device": self.device,
            "num_documents": len(self._documents),
        }
        with open(out / "index_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        print(f"  索引已保存至 {output_dir}")

    def load(self, input_dir: str) -> None:
        """从目录加载索引和文档。"""
        import faiss

        inp = Path(input_dir)
        self._index = faiss.read_index(str(inp / "faiss.index"))

        with open(inp / "documents.json", "r", encoding="utf-8") as f:
            docs_data = json.load(f)
        self._documents = [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in docs_data
        ]

        with open(inp / "index_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        self.embedding_model_name = config["embedding_model"]

        print(f"  索引已加载: {len(self._documents)} 条法条")

    @property
    def documents(self) -> list[Document]:
        return self._documents

    @property
    def index(self):
        return self._index
