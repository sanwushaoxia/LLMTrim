"""Stage 2 - structural compression of common machine-generated shapes.

All operations are deterministic and idempotent:
- long URLs collapse to ``domain…`` (path/query dropped)
- log timestamps and repeated log prefixes fold away on continuation lines
- single-line JSON re-indented to one compact line (spaces after ,: removed)
- markdown emphasis/backtick decoration stripped outside code spans
"""
from __future__ import annotations

import json
import re
from typing import List, Tuple

_URL_RE = re.compile(
    r"https?://[^\s<>\"')\]]+", re.IGNORECASE
)
# Log-ish line: 2024-01-02 03:04:05,123 [INFO] / 2024-01-02T03:04:05Z INFO /
# 2024-01-02 03:04:05 INFO
_TS_RE = re.compile(
    r"^\s*\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?"
    r"(?:\s+\[(?:DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL)\]|"
    r"\s+(?:DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL)\b)?",
    re.IGNORECASE,
)
_MD_EMPH_RE = re.compile(
    r"(?<![\w\\])(\*\*\*|\*\*|\*|___|__|_)(?![\s_])((?:[^*_\n]|\*(?!\*))+?)(?<![\s\\])\1(?![\w])"
)
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)[^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)[^)]*\)")


def structure(text: str) -> str:
    if not text:
        return text
    text = _compact_json_block(text)
    lines = text.split("\n")
    lines = [_compress_urls_in_line(l) for l in lines]
    lines = _fold_log_timestamps(lines)
    lines = [_compact_json_line(l) for l in lines]
    out = "\n".join(lines)
    out = _strip_markdown(out)
    return out


# --- URLs ---------------------------------------------------------------

def _shrink_url(match: re.Match) -> str:
    url = match.group(0)
    if len(url) <= 40:
        return url
    m = re.match(r"(https?://[^/?#]+)", url, re.IGNORECASE)
    if not m:
        return url
    return m.group(1) + "/\u2026"


def _compress_urls_in_line(line: str) -> str:
    return _URL_RE.sub(_shrink_url, line)


# --- logs ---------------------------------------------------------------

def _fold_log_timestamps(lines: List[str]) -> List[str]:
    """Drop the timestamp+level prefix when consecutive lines share it."""
    out: List[str] = []
    last_key = None
    for line in lines:
        m = _TS_RE.match(line)
        if m:
            key = re.sub(r"[.,]\d+", "", m.group(0))  # ignore millis
            body = line[m.end():].lstrip()
            if key == last_key and body:
                out.append("… " + body)
                continue
            last_key = key
            out.append(line)
        else:
            last_key = None
            out.append(line)
    return out


# --- JSON ---------------------------------------------------------------

def _compact_json_block(text: str) -> str:
    """Collapse a multi-line JSON document (the whole text) to one line.

    Only fires when the entire text parses as JSON and spans 2+ lines;
    mixed prose with embedded JSON is left to the per-line pass.
    """
    s = text.strip()
    if "\n" not in text or not (
        (s.startswith("{") and s.endswith("}"))
        or (s.startswith("[") and s.endswith("]"))
    ):
        return text
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _compact_json_line(line: str) -> str:
    s = line.strip()
    if not ((s.startswith("{") and s.endswith("}"))
            or (s.startswith("[") and s.endswith("]"))):
        return line
    if len(s) < 60 and "\n" not in line:
        return line
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return line
    try:
        compact = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return line
    # preserve original leading indentation
    indent = line[: len(line) - len(line.lstrip())]
    return indent + compact


# --- markdown -----------------------------------------------------------

def _strip_markdown(text: str) -> str:
    # protect fenced code blocks
    parts = text.split("```")
    for i in range(0, len(parts), 2):  # even indexes are outside fences
        part = parts[i]
        part = _MD_IMAGE_RE.sub(r"\1", part)
        part = _MD_LINK_RE.sub(r"\1 \2", part)
        part = _MD_EMPH_RE.sub(r"\2", part)
        parts[i] = part
    return "```".join(parts)
