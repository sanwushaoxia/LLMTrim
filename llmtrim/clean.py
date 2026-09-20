"""Stage 1 - rule-based cleaning. Deterministic, meaning-preserving.

Operations (all idempotent):
- strip zero-width / bidi control characters
- normalise tabs and runs of spaces inside lines to single spaces
- collapse 3+ consecutive blank lines to one blank line
- drop exact duplicate consecutive lines
- shrink decorative separator lines (---, ===, ***, ~~~ ...)
- cap runs of a single repeated character (e.g. "!!!!!!!!") to 3
"""
from __future__ import annotations

import re

# Zero-width and invisible control chars, incl. bidi overrides.
_INVISIBLE_RE = re.compile(
    "[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\u2060\ufeff\u00ad]"
)
_TAB_RE = re.compile(r"\t+")
_SPACE_RUN_RE = re.compile(r" {2,}")
_TRAIL_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_SEPARATOR_RE = re.compile(
    r"^ {0,3}([-_*~=])\1{2,}[ \t\1]*$"
)
_CHAR_RUN_RE = re.compile(r"(.)\1{3,}")


def clean(text: str) -> str:
    if not text:
        return text

    # 1. invisible characters (safe everywhere, incl. code)
    text = _INVISIBLE_RE.sub("", text)

    # 2. per-line whitespace normalisation, but fenced code blocks keep
    #    their exact spacing (indentation is semantics there)
    lines = text.split("\n")
    out: list = []
    in_fence = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line.rstrip())
            continue
        out.append(line if in_fence else _normalize_line(line))

    # 3. collapse blank-line runs (keep at most one empty line)
    collapsed: list = []
    blank_run = 0
    for line in out:
        if line == "":
            blank_run += 1
            if blank_run > 1:
                continue
        else:
            blank_run = 0
        collapsed.append(line)

    # 4. drop exact duplicate consecutive lines (but never inside blank runs
    #    which are already handled; a duplicated content line is noise)
    deduped: list = []
    for line in collapsed:
        if line and deduped and deduped[-1] == line:
            continue
        deduped.append(line)

    return "\n".join(deduped)


def _normalize_line(line: str) -> str:
    if not line.strip():
        return ""
    line = _SEPARATOR_RE.sub(lambda m: m.group(1) * 3, line)
    line = _CHAR_RUN_RE.sub(r"\1\1\1", line)
    line = _TAB_RE.sub(" ", line)
    line = _SPACE_RUN_RE.sub(" ", line)
    return line.rstrip()
