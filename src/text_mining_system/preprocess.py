"""Chinese text cleaning and tokenization helpers."""

from __future__ import annotations

import re
from functools import lru_cache

import jieba


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_SPACE_RE = re.compile(r"\s+")
_KEEP_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")


STOPWORDS = {
    "一个",
    "一些",
    "以及",
    "已经",
    "进行",
    "没有",
    "记者",
    "表示",
    "今日",
    "今年",
    "可能",
    "可以",
    "相关",
    "成为",
    "发布",
    "推出",
    "获得",
    "提升",
    "持续",
    "明显",
    "多地",
    "多家",
    "多部",
    "本周",
    "新高",
}


def clean_text(text: object) -> str:
    """Normalize noisy Chinese news text while keeping letters and digits."""

    value = "" if text is None else str(text)
    value = _URL_RE.sub(" ", value)
    value = _KEEP_RE.sub(" ", value)
    value = _SPACE_RE.sub(" ", value).strip()
    return value


@lru_cache(maxsize=200_000)
def _cached_cut(text: str) -> tuple[str, ...]:
    return tuple(jieba.lcut(text))


def tokenize_zh(text: object) -> list[str]:
    """Tokenize Chinese text and filter low-information tokens."""

    cleaned = clean_text(text)
    tokens: list[str] = []
    for token in _cached_cut(cleaned):
        token = token.strip().lower()
        if len(token) < 2:
            continue
        if token in STOPWORDS:
            continue
        if token.isdigit():
            continue
        tokens.append(token)
    return tokens
