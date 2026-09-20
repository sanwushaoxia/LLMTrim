"""Token counters: tiktoken adapter with a deterministic heuristic fallback.

The counter is the yardstick for every compression decision, so it must be
stable within a process. ``get_counter("auto")`` returns tiktoken's
``o200k_base`` encoder when the package is installed, otherwise
:class:`HeuristicCounter`, which approximates BPE behaviour: CJK chars cost
~1 token each, words cost roughly 0.75 tokens per 4 chars, punctuation counts.
"""
from __future__ import annotations

import re
from typing import List, Protocol

_CJK_RANGES = (
    (0x4E00, 0x9FFF),   # CJK unified
    (0x3400, 0x4DBF),   # CJK ext A
    (0x20000, 0x2A6DF), # CJK ext B
    (0x3040, 0x30FF),   # hiragana / katakana
    (0xAC00, 0xD7AF),   # hangul
)

_CJK_RE = re.compile(
    "[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in _CJK_RANGES) + "]"
)
# Words: runs of non-space, non-CJK visible chars.
_WORD_RE = re.compile(r"[^\s]+")


def cjk_char_count(text: str) -> int:
    return len(_CJK_RE.findall(text))


def has_cjk(text: str) -> bool:
    return _CJK_RE.search(text) is not None


def cjk_share(text: str) -> float:
    """Share of non-space characters that are CJK."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return cjk_char_count(text) / len(chars)


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class HeuristicCounter:
    """Dependency-free token estimator.

    Rules of thumb calibrated against cl100k/o200k behaviour:
    - each CJK char ~= 1 token
    - common words up to ~8 chars are usually a single token
    - longer words (URLs, hashes, identifiers) split into ~4-char chunks
    """

    name = "heuristic"

    def count(self, text: str) -> int:
        if not text:
            return 0
        tokens = cjk_char_count(text)
        remainder = _CJK_RE.sub(" ", text)
        for word in _WORD_RE.findall(remainder):
            if len(word) >= 9:
                tokens += max(1, (len(word) + 3) // 4)
            else:
                tokens += 1
        return tokens


class TiktokenCounter:
    """Wrapper around tiktoken's o200k_base encoding."""

    def __init__(self, encoding_name: str = "o200k_base") -> None:
        import tiktoken  # optional dependency

        self._encoding = tiktoken.get_encoding(encoding_name)
        self.name = f"tiktoken:{encoding_name}"

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))


def count_many(counter: TokenCounter, texts: List[str]) -> List[int]:
    return [counter.count(t) for t in texts]


def get_counter(mode: str = "auto") -> TokenCounter:
    """Return a counter by mode: ``auto``, ``heuristic`` or ``tiktoken``."""
    if mode == "heuristic":
        return HeuristicCounter()
    if mode in ("auto", "tiktoken"):
        try:
            return TiktokenCounter()
        except ImportError:
            if mode == "tiktoken":
                raise
            return HeuristicCounter()
    raise ValueError(f"unknown counter mode: {mode!r}")
