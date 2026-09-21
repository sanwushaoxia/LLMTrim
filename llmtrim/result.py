"""TrimResult: outcome of a trim run, with per-stage token accounting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StageReport:
    """Token accounting for one pipeline stage."""

    name: str
    tokens_before: int
    tokens_after: int
    applied: bool = True
    note: str = ""
    elapsed_ms: float = 0.0
    reverted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after

    def as_dict(self) -> Dict:
        return {
            "name": self.name,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": self.tokens_saved,
            "applied": self.applied,
            "note": self.note,
            "elapsed_ms": self.elapsed_ms,
            "reverted": self.reverted,
            "metadata": self.metadata,
        }


@dataclass
class TrimResult:
    """Result of :func:`llmtrim.trim`.

    Attributes
    ----------
    text:
        The compressed text.
    original_text:
        The input as received (before any stage).
    original_tokens / final_tokens:
        Token counts measured with the configured counter.
    stages:
        One :class:`StageReport` per executed stage, in execution order.
    translator_used:
        Name of the translator used, when the translate stage ran.
    """

    text: str
    original_text: str
    original_tokens: int
    final_tokens: int
    stages: List[StageReport] = field(default_factory=list)
    translator_used: Optional[str] = None
    counter_name: str = ""
    translation: Dict[str, int] = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        if self.original_tokens <= 0:
            return 1.0
        return self.final_tokens / self.original_tokens

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.final_tokens

    def summary(self) -> str:
        lines = [
            f"original: {self.original_tokens} tokens",
            f"final:    {self.final_tokens} tokens "
            f"({self.ratio:.1%} of original, saved {self.tokens_saved})",
        ]
        for s in self.stages:
            status = "ok" if s.applied else "skipped"
            change = f"-{s.tokens_saved}" if s.tokens_saved >= 0 else f"+{-s.tokens_saved}"
            lines.append(
                f"  [{s.name:<9}] {s.tokens_before:>6} -> {s.tokens_after:>6} "
                f"({change}) {status}"
                + (f" | {s.note}" if s.note else "")
            )
        return "\n".join(lines)

    def as_dict(self) -> Dict:
        return {
            "original_tokens": self.original_tokens,
            "final_tokens": self.final_tokens,
            "ratio": self.ratio,
            "tokens_saved": self.tokens_saved,
            "translator_used": self.translator_used,
            "counter_name": self.counter_name,
            "translation": self.translation,
            "stages": [s.as_dict() for s in self.stages],
        }
