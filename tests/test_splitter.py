# -*- coding: utf-8 -*-
"""法条切分器测试。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_core.documents import Document
from law_llm.rag.splitter import LawSplitter, ARTICLE_PATTERN


class TestArticlePattern:
    """测试法条正则匹配。"""

    def test_chinese_number_article(self):
        text = "第一条 为了保护个人信息权益"
        matches = ARTICLE_PATTERN.findall(text)
        assert "第一条" in matches

    def test_multi_digit_chinese_number(self):
        text = "第二十八条 生物识别信息属于敏感个人信息"
        matches = ARTICLE_PATTERN.findall(text)
        assert "第二十八条" in matches

    def test_arabic_number_article(self):
        text = "第1条 测试条文"
        matches = ARTICLE_PATTERN.findall(text)
        assert "第1条" in matches

    def test_no_space_after_article(self):
        """关键测试：原脚本要求'条'后有空格，修复后不需要。"""
        text = "第一条为了保护个人信息权益"
        matches = ARTICLE_PATTERN.findall(text)
        assert "第一条" in matches

    def test_multiple_articles(self):
        text = """
        第一条 为了保护个人信息权益
        第二条 自然人的个人信息受法律保护
        第三条 在中华人民共和国境内处理自然人个人信息的活动
        """
        matches = ARTICLE_PATTERN.findall(text)
        assert "第一条" in matches
        assert "第二条" in matches
        assert "第三条" in matches


class TestLawSplitter:
    """测试法律文档切分器。"""

    def test_split_simple_law(self):
        """测试简单法律文档切分。"""
        content = """# 中华人民共和国测试法

2020年1月1日 通过

## 第一章 总则

第一条 这是第一条的内容，用于测试切分功能是否正常工作。

第二条 这是第二条的内容，同样用于测试切分功能。

## 第二章 具体规定

第三条 这是第三条的内容，属于第二章的范畴。
"""
        doc = Document(page_content=content, metadata={"source": "test.md"})
        splitter = LawSplitter(chunk_size=512, chunk_overlap=50)
        chunks = splitter.split_documents([doc])

        # 应该至少切分出 3 个法条
        articles = [c.metadata.get("article", "") for c in chunks]
        assert "第一条" in articles
        assert "第二条" in articles
        assert "第三条" in articles

    def test_split_with_metadata(self):
        """测试元数据保留。"""
        content = """# 中华人民共和国个人信息保护法

2021年8月20日 通过

## 第一章 总则

第一条 为了保护个人信息权益，规范个人信息处理活动。
"""
        doc = Document(page_content=content, metadata={"source": "test.md"})
        splitter = LawSplitter(chunk_size=512, chunk_overlap=50)
        chunks = splitter.split_documents([doc])

        assert len(chunks) > 0
        first_chunk = chunks[0]
        assert first_chunk.metadata.get("law_name") == "中华人民共和国个人信息保护法"

    def test_split_long_article(self):
        """测试长法条递归切分。"""
        long_content = "第一条 " + "这是一段很长的法律条文内容。" * 100
        content = f"""# 测试法

## 第一章 总则

{long_content}
"""
        doc = Document(page_content=content, metadata={"source": "test.md"})
        splitter = LawSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_documents([doc])

        # 长文本应该被切成多个块
        assert len(chunks) > 1
