"""LLMTrim: token-aware input compression for LLM prompts.

Quick start::

    from llmtrim import trim, analyze, TrimConfig

    result = trim(text, TrimConfig(target_ratio=0.6))
    print(result.text)
    print(result.summary())
"""
from .config import TrimConfig
from .counters import get_counter
from .pipeline import run_pipeline
from .result import StageReport, TrimResult

__version__ = "0.1.0"

__all__ = [
    "trim",
    "analyze",
    "TrimConfig",
    "TrimResult",
    "StageReport",
    "get_counter",
    "__version__",
]


def trim(text: str, config: TrimConfig | None = None) -> TrimResult:
    """Compress ``text`` according to ``config`` and return a TrimResult."""
    return run_pipeline(text, config, dry_run=False)


def analyze(text: str, config: TrimConfig | None = None) -> TrimResult:
    """Report what each stage would save without modifying the text.

    The returned result's ``text`` is the untouched input; ``stages`` hold
    the would-be savings per stage.
    """
    return run_pipeline(text, config, dry_run=True)
