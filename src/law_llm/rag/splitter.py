# -*- coding: utf-8 -*-
r"""法律文档切分器。

修复原 splitter.py 的问题：
- 正则 `第\S*条 ` 依赖"条"后恰好有空格，大部分法律文本会切分失败
- 改为更鲁棒的正则，支持中文数字和多种格式
- 增加法律层级切分：编 → 章 → 节 → 条
- 保留完整元数据
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


# 匹配法律条文编号，支持：
#   第一条 / 第二条 / 第一百零一条
#   第1条 / 第2条
#   第一编 / 第一章 / 第一节
ARTICLE_PATTERN = re.compile(
    r"(第[一二三四五六七八九十百千万零\d]+条)"  # 条
)
CHAPTER_PATTERN = re.compile(r"(第[一二三四五六七八九十百千万零\d]+编)")
SECTION_PATTERN = re.compile(r"(第[一二三四五六七八九十百千万零\d]+章)")
SUBSECTION_PATTERN = re.compile(r"(第[一二三四五六七八九十百千万零\d]+节)")


class LawSplitter:
    """法律文档切分器。

    切分层级：
        1. Markdown 标题切分（# → 法律名, ## → 编/章, ### → 节）
        2. 法条正则切分（第X条）
        3. 长文本递归切分（超过 chunk_size 时）
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "law_name"),
                ("##", "chapter"),
                ("###", "section"),
                ("####", "subsection"),
            ]
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """切分文档列表，返回带完整元数据的法条块。

        每个输出 Document 的 metadata 包含：
            - law_name:     法律名称
            - chapter:      章（如 "第一编 总则" / "第一章 总则"）
            - section:      节
            - article:      条（如 "第一条"）
            - effective_date: 生效日期
            - source:       文件来源
        """
        result: list[Document] = []

        for doc in documents:
            # Step 1: Markdown 标题切分
            try:
                md_docs = self.md_splitter.split_text(doc.page_content)
            except Exception:
                # 如果 Markdown 解析失败，直接用法条正则切分
                md_docs = [Document(page_content=doc.page_content, metadata={})]

            for md_doc in md_docs:
                # 合并元数据
                base_metadata = {**doc.metadata, **md_doc.metadata}

                # Step 2: 法条正则切分
                article_chunks = self._split_by_article(md_doc.page_content)

                for article_text, article_num in article_chunks:
                    metadata = {
                        **base_metadata,
                        "article": article_num,
                    }

                    # Step 3: 如果法条文本过长，再递归切分
                    if len(article_text) > self.chunk_size:
                        sub_docs = self.text_splitter.split_text(article_text)
                        for sub_text in sub_docs:
                            result.append(
                                Document(page_content=sub_text, metadata=metadata.copy())
                            )
                    else:
                        result.append(
                            Document(page_content=article_text, metadata=metadata.copy())
                        )

        return result

    def _split_by_article(self, text: str) -> list[tuple[str, str]]:
        """按法条编号切分文本。

        Returns:
            [(条文内容, 条文编号), ...]
            条文编号如 "第一条"、"第二十八条"；无法切分时返回 [(全文, "")]
        """
        # 找到所有法条编号的位置
        matches = list(ARTICLE_PATTERN.finditer(text))

        if not matches:
            return [(text.strip(), "")]

        chunks: list[tuple[str, str]] = []

        # 如果第一个法条之前有内容（通常是章标题），附加到第一个法条
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                # 尝试提取章节信息
                chapter = CHAPTER_PATTERN.search(preamble)
                section = SECTION_PATTERN.search(preamble)
                chunks.append((preamble, ""))

        for i, match in enumerate(matches):
            article_num = match.group(1)
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            article_text = text[start:end].strip()
            if article_text:
                chunks.append((article_text, article_num))

        return chunks
