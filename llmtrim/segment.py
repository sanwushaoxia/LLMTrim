"""Mixed zh/en segmentation for scoring and reporting.

Chinese text is segmented with jieba (loaded lazily and quietly); other
scripts fall back to regex word extraction. A single :func:`tokenize` call
works on mixed-language lines and yields ``(token, is_protected)`` pairs
elsewhere in the package; here we just produce plain tokens.
"""
from __future__ import annotations

import re
from typing import List

from .counters import cjk_char_count

_JIEBA = None


def _jieba():
    global _JIEBA
    if _JIEBA is None:
        import jieba

        jieba.setLogLevel(60)  # silence "Building prefix dict" on first cut
        _JIEBA = jieba
    return _JIEBA


# ASCII / latin words, numbers (with . , % kept), and code-ish symbols.
_LATIN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-./%]*")
_PUNCT_RE = re.compile(r"[^\sA-Za-z0-9_]")

_TOKEN_CACHE: dict = {}


def tokenize(text: str) -> List[str]:
    """Segment ``text`` into tokens (jieba for CJK, regex for the rest)."""
    if not text:
        return []
    cached = _TOKEN_CACHE.get(text)
    if cached is not None:
        return cached

    tokens: List[str] = []
    if cjk_char_count(text):
        for piece in _jieba().cut(text):
            if not piece.strip():
                continue
            if cjk_char_count(piece):
                tokens.append(piece)
            else:
                tokens.extend(_LATIN_RE.findall(piece) or [piece])
    else:
        tokens.extend(_LATIN_RE.findall(text))
        if not tokens:
            tokens = _PUNCT_RE.findall(text)

    if len(_TOKEN_CACHE) > 10_000:
        _TOKEN_CACHE.clear()
    _TOKEN_CACHE[text] = tokens
    return tokens


def is_number_token(token: str) -> bool:
    """True for pure numbers/versions/metrics: 42, 3.14, 50%, v2, 2024-01."""
    t = token.strip().lower()
    if not t:
        return False
    if t.isdigit():
        return True
    return bool(re.fullmatch(r"v?\d+(\.\d+)+%?", t))
