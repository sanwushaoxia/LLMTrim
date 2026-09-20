"""Stage 4 - statistical pruning by token importance.

Architecture: mask-then-prune-then-unmask.

1. Protected spans (fenced code, inline code, user patterns — possibly
   multi-line) are replaced with ``\\x02K\\x02`` placeholders so pruning can
   never touch them and no offset bookkeeping is needed.
2. Tokens are scored over line-level pseudo-documents:
   ``(1 + ln tf) * idf`` plus bumps — stop words far negative, numbers and
   headings positive.
3. Lowest-scoring (line, token) groups are deleted until the free-text
   token budget (``target_ratio`` minus what protected content costs) is
   met. Spans are re-derived from a fresh tokenisation of the current line
   at every edit, so no stale offsets can corrupt the output.
4. Placeholders are substituted back.

Deletion only removes tokens — wording is never rewritten, so results are
deterministic.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Pattern, Sequence, Set, Tuple

from .config import TrimConfig
from .counters import TokenCounter, has_cjk
from .segment import is_number_token
from .stopwords import stopwords_for

_PLACEHOLDER_FMT = "\x02{}\x02"
_PLACEHOLDER_RE = re.compile("\x02(\\d+)\x02")

_WORD_OFF_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-./%]*")


def prune(text: str, config: TrimConfig, counter: TokenCounter,
          budget_tokens: int) -> str:
    """Best-effort prune ``text`` down to ``budget_tokens``."""
    if counter.count(text) <= budget_tokens:
        return text

    spans = _find_protected_spans(text, config.compiled_protected_patterns())
    masked, protected_parts = _mask(text, spans)
    protected_tokens = sum(counter.count(p) for p in protected_parts)
    free_budget = budget_tokens - protected_tokens

    if free_budget > 0:
        masked = _prune_masked(masked, config, counter, free_budget)

    return _unmask(masked, protected_parts)


# --- masking ---------------------------------------------------------------

def _find_protected_spans(text: str,
                          patterns: Sequence[Pattern[str]]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for pat in patterns:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))
    spans.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _mask(text: str, spans: Sequence[Tuple[int, int]]
          ) -> Tuple[str, List[str]]:
    out: List[str] = []
    protected: List[str] = []
    pos = 0
    for i, (s, e) in enumerate(spans):
        out.append(text[pos:s])
        out.append(_PLACEHOLDER_FMT.format(i))
        protected.append(text[s:e])
        pos = e
    out.append(text[pos:])
    return "".join(out), protected


def _unmask(masked: str, protected: Sequence[str]) -> str:
    return _PLACEHOLDER_RE.sub(
        lambda m: protected[int(m.group(1))], masked
    )


# --- tokenisation with offsets ----------------------------------------------

def _line_tokens(line: str) -> List[Tuple[str, int, int]]:
    """(token, start, end) with jieba for CJK, regex for the rest."""
    if not line.strip():
        return []
    out: List[Tuple[str, int, int]] = []
    if has_cjk(line):
        from .segment import _jieba

        pos = 0
        for piece in _jieba().cut(line):
            start = line.find(piece, pos)
            if start < 0:
                start, end = pos, pos + len(piece)
            else:
                end = start + len(piece)
            pos = max(end, start)
            if not piece.strip():
                continue
            if has_cjk(piece):
                out.append((piece, start, end))
            else:
                for m in _WORD_OFF_RE.finditer(piece):
                    out.append((m.group(0), start + m.start(), start + m.end()))
    else:
        for m in _WORD_OFF_RE.finditer(line):
            out.append((m.group(0), m.start(), m.end()))
    return out


def _line_tokens_safe(line: str) -> List[Tuple[str, int, int]]:
    """Tokens that do not overlap any placeholder (placeholders are taboo)."""
    ph_spans = [(m.start(), m.end()) for m in _PLACEHOLDER_RE.finditer(line)]
    out: List[Tuple[str, int, int]] = []
    for tok, s, e in _line_tokens(line):
        if any(s < pe and e > ps for ps, pe in ph_spans):
            continue
        out.append((tok, s, e))
    return out


# --- scoring -----------------------------------------------------------------

def _score_groups(lines: List[str], config: TrimConfig) -> Dict[Tuple[int, str], float]:
    n_docs = max(1, sum(1 for l in lines if l.strip()))
    stop_words = stopwords_for(config.language)
    heading = [bool(re.match(r"^\s*#{1,6}\s+\S", l)) for l in lines]

    # document frequency = number of distinct lines containing the token
    line_sets: List[Set[str]] = []
    df: Dict[str, int] = {}
    for i, line in enumerate(lines):
        toks = {t for t, _, _ in _line_tokens_safe(line)}
        line_sets.append(toks)
        for t in toks:
            df[t] = df.get(t, 0) + 1

    scores: Dict[Tuple[int, str], float] = {}
    for i, line in enumerate(lines):
        counts: Dict[str, int] = {}
        for t, _, _ in _line_tokens_safe(line):
            counts[t] = counts.get(t, 0) + 1
        for t, n in counts.items():
            tf = 1.0 + math.log(n)
            idf = math.log(1.0 + n_docs / (1.0 + min(df[t], n_docs)))
            score = tf * idf
            if t.lower() in stop_words:
                score -= 3.0
            if is_number_token(t):
                score += 2.0
            if len(t) <= 1:
                score -= 0.5
            if t[0].isupper() or t.startswith("_"):
                score += 0.3
            if heading[i]:
                score += 1.5
            scores[(i, t)] = score
    return scores


# --- deletion ----------------------------------------------------------------

def _prune_masked(masked: str, config: TrimConfig, counter: TokenCounter,
                  free_budget: int) -> str:
    lines = masked.split("\n")

    def count_free(s: str) -> int:
        return sum(counter.count(p) for p in _PLACEHOLDER_RE.split(s) if p)

    current = sum(count_free(l) for l in lines)
    if current <= free_budget:
        return masked

    scores = _score_groups(lines, config)
    # Stable sort: equal scores keep first-appearance (text) order, so the
    # earliest filler goes first and later key terms survive ties.
    order = sorted(scores.items(), key=lambda kv: kv[1])
    # First pass honours the aggressiveness rank cap (0.0 -> only negative-
    # scoring stop-word groups, 1.0 -> bottom half of the ranking); the
    # second pass extends through the rest so the budget stays a hard goal.
    max_rank = int(config.aggressiveness * len(order))

    for allow_all in (False, True):
        if current <= free_budget:
            break
        for rank, ((li, tok), score) in enumerate(order):
            if current <= free_budget:
                break
            if not allow_all and score >= 0 and rank >= max_rank:
                continue
            line = lines[li]
            spans = [(s, e) for t, s, e in _line_tokens_safe(line) if t == tok]
            if not spans:
                continue
            new_line = _remove_spans(line, spans)
            delta = count_free(line) - count_free(new_line)
            if delta <= 0:
                continue
            lines[li] = new_line
            current -= delta

    return "\n".join(lines)


def _remove_spans(line: str, spans: Sequence[Tuple[int, int]]) -> str:
    for s, e in sorted(spans, reverse=True):
        line = line[:s] + line[e:]
    line = re.sub(r"  +", " ", line)
    return line.strip()


def _looks_like_heading(line: str) -> bool:
    return bool(re.match(r"^\s*#{1,6}\s+\S", line))
