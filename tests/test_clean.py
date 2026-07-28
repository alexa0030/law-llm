# -*- coding: utf-8 -*-
"""数据清洗模块测试。"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from law_llm.data.clean import (
    clean_sample,
    clean_dataset,
    check_too_short,
    check_invalid_article_ref,
    check_entertainment,
    check_low_quality,
    check_length_abnormal,
)
from law_llm.data.schema import SFTSample, CleanReason


class TestCleanSample:
    """测试单条样本清洗。"""

    def test_valid_sample_passes(self):
        raw = {
            "instruction": "什么是故意犯罪？",
            "output": "明知自己的行为会发生危害社会的结果，并且希望或者放任这种结果发生，因而构成犯罪的，是故意犯罪。",
        }
        sample, record = clean_sample(raw)
        assert sample is not None
        assert record is None
        assert sample.instruction == "什么是故意犯罪？"

    def test_missing_field_filtered(self):
        raw = {"instruction": "只有 instruction"}
        sample, record = clean_sample(raw)
        assert sample is None
        assert record is not None
        assert record.reason == CleanReason.MISSING_FIELD.value

    def test_too_short_filtered(self):
        raw = {"instruction": "ab", "output": "短"}
        sample, record = clean_sample(raw)
        assert sample is None
        assert record.reason == CleanReason.TOO_SHORT.value

    def test_invalid_article_ref_filtered(self):
        raw = {
            "instruction": "某法律条文是什么？",
            "output": "根据第0条规定，这是一个测试回答，长度足够通过基本检查。",
        }
        sample, record = clean_sample(raw)
        assert sample is None
        assert record.reason == CleanReason.INVALID_ARTICLE_REF.value

    def test_entertainment_filtered(self):
        raw = {
            "instruction": "推荐一部好看的电影",
            "output": "我推荐你看看这个电影，非常好看，剧情引人入胜，值得一看。",
        }
        sample, record = clean_sample(raw)
        assert sample is None

    def test_low_quality_filtered(self):
        raw = {
            "instruction": "这个问题怎么回答？",
            "output": "我不知道，也许可能大概是这样吧，非常抱歉无法确定。",
        }
        sample, record = clean_sample(raw)
        assert sample is None

    def test_too_long_filtered(self):
        raw = {
            "instruction": "请详细解释" * 10,
            "output": "x" * 3000,
        }
        sample, record = clean_sample(raw)
        assert sample is None
        assert record.reason == CleanReason.LENGTH_ABNORMAL.value


class TestCleanDataset:
    """测试数据集清洗。"""

    def test_clean_dataset(self, tmp_path):
        data = [
            {"instruction": "合法问题一", "output": "这是一个合法的回答，长度足够。" * 2},
            {"instruction": "ab", "output": "短"},  # 过滤
            {"instruction": "合法问题二", "output": "另一个合法的回答，内容充分。" * 2},
        ]
        input_path = tmp_path / "input.json"
        output_path = tmp_path / "output.json"
        log_path = tmp_path / "log.json"

        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        stats = clean_dataset(
            input_path=str(input_path),
            output_path=str(output_path),
            log_path=str(log_path),
        )

        assert stats["original"] == 3
        assert stats["cleaned"] == 2  # 可能因模板化回复过滤更多
        assert stats["filtered"] >= 1
        assert output_path.exists()
        assert log_path.exists()
