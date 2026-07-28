# -*- coding: utf-8 -*-
"""法律文档加载器。

修复原 loader.py 的问题：
- `form` → `from` 拼写错误
- 正确使用 langchain 的 DirectoryLoader
- 支持多种文档格式
- 提取法律元数据（法律名称、生效日期）
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain_core.documents import Document


def extract_law_metadata(text: str, source: str = "") -> dict[str, str]:
    """从法律文本头部提取元数据。

    支持格式：
        # 中华人民共和国个人信息保护法
        2021年8月20日 第十三届全国人民代表大会常务委员会第三十次会议通过
    """
    metadata: dict[str, str] = {"source": source}

    # 法律名称：第一个 # 标题
    title_match = re.search(r"^#\s*(.+)$", text, re.MULTILINE)
    if title_match:
        metadata["law_name"] = title_match.group(1).strip()

    # 生效日期：从头部文本中提取
    date_patterns = [
        r"(\d{4}年\d{1,2}月\d{1,2}日)",
        r"(\d{4}-\d{2}-\d{2})",
    ]
    for pattern in date_patterns:
        date_match = re.search(pattern, text[:500])
        if date_match:
            metadata["effective_date"] = date_match.group(1)
            break

    return metadata


class LawLoader:
    """法律文档加载器，支持递归加载目录下的 Markdown/TXT 文件。"""

    def __init__(
        self,
        path: str,
        glob_pattern: str = "**/*.md",
        encoding: str = "utf-8",
    ) -> None:
        self.path = Path(path)
        self.glob_pattern = glob_pattern
        self.encoding = encoding

    def load(self) -> list[Document]:
        """加载所有匹配的文档，返回 LangChain Document 列表。"""
        if self.path.is_file():
            return self._load_single_file(self.path)
        return self._load_directory()

    def _load_single_file(self, filepath: Path) -> list[Document]:
        """加载单个文件。"""
        loader = TextLoader(str(filepath), encoding=self.encoding)
        docs = loader.load()
        result = []
        for doc in docs:
            metadata = extract_law_metadata(doc.page_content, source=str(filepath))
            doc.metadata.update(metadata)
            result.append(doc)
        return result

    def _load_directory(self) -> list[Document]:
        """递归加载目录。"""
        loader = DirectoryLoader(
            str(self.path),
            glob=self.glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": self.encoding},
            show_progress=True,
        )
        docs = loader.load()
        result = []
        for doc in docs:
            source = doc.metadata.get("source", "")
            metadata = extract_law_metadata(doc.page_content, source=source)
            doc.metadata.update(metadata)
            result.append(doc)
        return result
