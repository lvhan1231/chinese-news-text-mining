"""Human-readable labels for the TNEWS dataset."""

from __future__ import annotations


TNEWS_LABEL_NAMES = {
    "100": "故事",
    "101": "文化",
    "102": "娱乐",
    "103": "体育",
    "104": "财经",
    "106": "房产",
    "107": "汽车",
    "108": "教育",
    "109": "科技",
    "110": "军事",
    "112": "旅游",
    "113": "国际",
    "114": "股票",
    "115": "农业",
    "116": "游戏",
}


def label_to_display(label: object) -> str:
    """Return a Chinese display name for numeric TNEWS labels."""

    value = "" if label is None else str(label)
    return TNEWS_LABEL_NAMES.get(value, value)
