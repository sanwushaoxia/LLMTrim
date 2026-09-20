"""Bilingual stop word lists, bundled as package data."""
from __future__ import annotations

from pathlib import Path
from typing import FrozenSet, Set

_DATA_DIR = Path(__file__).parent / "data"

_CACHE: dict = {}


def _load(name: str) -> FrozenSet[str]:
    if name not in _CACHE:
        path = _DATA_DIR / name
        words = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                words.add(line)
        _CACHE[name] = frozenset(words)
    return _CACHE[name]


def english_stopwords() -> FrozenSet[str]:
    return _load("stopwords_en.txt")


def chinese_stopwords() -> FrozenSet[str]:
    return _load("stopwords_zh.txt")


def stopwords_for(language: str) -> FrozenSet[str]:
    """Combined or single-language stop word set.

    ``zh`` -> Chinese, ``en`` -> English, anything else -> both merged.
    """
    if language == "zh":
        return chinese_stopwords()
    if language == "en":
        return english_stopwords()
    return english_stopwords() | chinese_stopwords()


def add_extra(words: Set[str]) -> None:
    """Extend both lists' union at runtime (used by config extensions)."""
    en = _CACHE.get("stopwords_en.txt")
    zh = _CACHE.get("stopwords_zh.txt")
    if en is not None:
        _CACHE["stopwords_en.txt"] = frozenset(en | words)
    if zh is not None:
        _CACHE["stopwords_zh.txt"] = frozenset(zh | words)
