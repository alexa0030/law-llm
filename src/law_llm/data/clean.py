# -*- coding: utf-8 -*-
"""规则清洗模块 —— 对 SFT 数据进行格式校验和质量过滤。

修复原脚本的问题：
- 消除硬编码路径，所有路径通过参数传入
- 输出文件名通过参数指定
- 清洗逻辑模块化，可单独测试
- 记录每条样本的过滤原因
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .schema import CleanReason, CleanRecord, SFTSample

# ---------------------------------------------------------------------------
# 关键词库
# ---------------------------------------------------------------------------

ENTERTAINMENT_KEYWORDS = [
    "电影", "音乐", "游戏", "天气", "美食", "旅游", "明星", "综艺",
    "电视剧", "歌曲", "演唱会", "综艺节目", "娱乐", "八卦", "热搜",
    "影视", "动漫", "动画", "小说", "漫画", "体育", "运动", "健身",
    "美食推荐", "餐厅", "食谱", "烹饪", "旅行攻略", "景点", "酒店",
    "节假日", "假期", "周末", "聚会", "约会", "聊天", "闲聊", "开玩笑",
    "段子", "笑话", "幽默", "搞笑", "表情包", "梗", "流行语",
]

FACT_KEYWORDS = [
    "首都", "人口", "面积", "历史年代", "公式", "定理", "周期", "日期",
    "地理位置", "坐标", "经纬度", "海拔", "气候", "温度", "降水量",
    "历史事件", "年份", "朝代", "皇帝", "总统", "领导人", "政治制度",
    "化学元素", "物理常数", "数学常数", "单位换算", "汇率", "时区",
    "节日日期", "纪念日", "生日", "星座", "生肖", "农历", "公历",
    "国家代码", "区号", "邮政编码", "行政区划", "GDP", "经济数据",
    "统计数字", "百分比", "比率", "平均值", "最大值", "最小值",
]

LOW_QUALITY_KEYWORDS = [
    "我不知道", "我不清楚", "无法回答", "没有信息", "资料不足",
    "请提供更多", "需要更多信息",
    "很抱歉", "对不起", "无法确定", "可能", "也许", "大概", "或许",
    "欢迎咨询", "请咨询", "请联系", "谢谢提问", "感谢咨询",
    "祝您", "再见", "拜拜",
]

TEMPLATE_PATTERNS = [
    "这是一个很好的问题", "您问得很好", "这个问题很有趣",
    "根据相关资料", "研究表明", "专家认为",
    "建议您", "您可以", "请尝试", "请注意",
]

LAW_TERMS = ["法律", "法条", "法规", "宪法", "刑法", "民法", "行政法"]


# ---------------------------------------------------------------------------
# 校验与过滤
# ---------------------------------------------------------------------------

def validate_format(raw: dict[str, Any]) -> bool:
    """检查必要字段是否存在。"""
    return all(key in raw for key in ("instruction", "output"))


def check_too_short(sample: SFTSample, min_instruction: int = 5, min_output: int = 20) -> bool:
    """内容过短。"""
    return len(sample.instruction) < min_instruction or len(sample.output) < min_output


def check_invalid_article_ref(sample: SFTSample) -> bool:
    """明显错误的法条引用。"""
    output = sample.output
    if re.search(r"第\s*0\s*条", output):
        return True
    if re.search(r"第\s*[a-zA-Z]+\s*条", output):
        return True
    return False


def check_template_reply(sample: SFTSample) -> bool:
    """无实质内容的模板化回答。"""
    if "根据相关法律规定" in sample.output and len(
        re.findall(r"《.*?》第\d+条", sample.output)
    ) == 0:
        return True
    return False


def check_entertainment(sample: SFTSample) -> bool:
    """娱乐/闲聊类内容。"""
    text = f"{sample.instruction} {sample.output}".lower()
    return any(kw in text for kw in ENTERTAINMENT_KEYWORDS)


def check_fact_memorization(sample: SFTSample) -> bool:
    """事实性记忆类内容（排除法律相关）。"""
    text = f"{sample.instruction} {sample.output}".lower()
    if any(kw in text for kw in FACT_KEYWORDS):
        if not any(term in text for term in LAW_TERMS):
            return True
    return False


def check_low_quality(sample: SFTSample) -> bool:
    """低质量回复。"""
    output = sample.output.lower()
    return any(kw in output for kw in LOW_QUALITY_KEYWORDS)


def check_template_pattern(sample: SFTSample) -> bool:
    """模板化回复。"""
    output = sample.output
    return any(p in output for p in TEMPLATE_PATTERNS)


def check_length_abnormal(sample: SFTSample, min_len: int = 20, max_len: int = 2000) -> bool:
    """输出长度异常。"""
    length = len(sample.output.strip())
    return length < min_len or length > max_len


def check_format_error(sample: SFTSample) -> bool:
    """格式混乱。"""
    text = f"{sample.instruction} {sample.output}"
    if re.search(r"答：\s*\.\.\.|无内容|error|undefined|\[.*?\]|<.*?>", text):
        return True
    return False


def check_duplicate_content(sample: SFTSample) -> bool:
    """输入与输出相同。"""
    return sample.instruction.strip() == sample.output.strip()


# ---------------------------------------------------------------------------
# 主清洗函数
# ---------------------------------------------------------------------------

def clean_sample(
    raw: dict[str, Any],
    *,
    min_instruction: int = 5,
    min_output: int = 20,
    max_output: int = 2000,
) -> tuple[SFTSample | None, CleanRecord | None]:
    """对单条数据进行清洗，返回 (通过样本, 过滤记录)。

    通过返回 (sample, None)；过滤返回 (None, record)。
    """
    if not validate_format(raw):
        return None, CleanRecord(item=raw, reason=CleanReason.MISSING_FIELD.value)

    sample = SFTSample.from_dict(raw)

    if check_too_short(sample, min_instruction, min_output):
        return None, CleanRecord(item=raw, reason=CleanReason.TOO_SHORT.value)

    if check_invalid_article_ref(sample):
        return None, CleanRecord(item=raw, reason=CleanReason.INVALID_ARTICLE_REF.value)

    if check_template_reply(sample):
        return None, CleanRecord(item=raw, reason=CleanReason.TEMPLATE_REPLY.value)

    if check_format_error(sample):
        return None, CleanRecord(item=raw, reason=CleanReason.FORMAT_ERROR.value)

    if check_entertainment(sample):
        return None, CleanRecord(item=raw, reason=CleanReason.ENTERTAINMENT.value)

    if check_fact_memorization(sample):
        return None, CleanRecord(item=raw, reason=CleanReason.FACT_MEMORIZATION.value)

    if check_low_quality(sample):
        return None, CleanRecord(item=raw, reason=CleanReason.LOW_QUALITY.value)

    if check_template_pattern(sample):
        return None, CleanRecord(item=raw, reason=CleanReason.TEMPLATE_REPLY.value)

    if check_length_abnormal(sample, min_output, max_output):
        return None, CleanRecord(item=raw, reason=CleanReason.LENGTH_ABNORMAL.value)

    if check_duplicate_content(sample):
        return None, CleanRecord(item=raw, reason=CleanReason.DUPLICATE_CONTENT.value)

    return sample, None


def clean_dataset(
    input_path: str,
    output_path: str,
    log_path: str | None = None,
    *,
    min_instruction: int = 5,
    min_output: int = 20,
    max_output: int = 2000,
) -> dict[str, int]:
    """对整个数据集执行规则清洗。

    Args:
        input_path:  输入 JSON 文件路径
        output_path: 清洗后输出路径
        log_path:    过滤记录日志路径（可选）
        min_instruction: instruction 最短字符数
        min_output:      output 最短字符数
        max_output:      output 最大字符数

    Returns:
        统计字典 {"original": N, "cleaned": M, "filtered": N-M}
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned: list[dict[str, Any]] = []
    deleted: list[dict[str, Any]] = []
    reason_count: dict[str, int] = defaultdict(int)

    for item in tqdm(data, desc="规则清洗"):
        sample, record = clean_sample(
            item,
            min_instruction=min_instruction,
            min_output=min_output,
            max_output=max_output,
        )
        if sample is not None:
            cleaned.append(sample.to_alpaca_dict())
        else:
            assert record is not None
            deleted.append(record.to_dict())
            reason_count[record.reason] += 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(deleted, f, ensure_ascii=False, indent=2)

    stats = {
        "original": len(data),
        "cleaned": len(cleaned),
        "filtered": len(data) - len(cleaned),
        "reason_distribution": dict(reason_count),
    }
    return stats
