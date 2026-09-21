"""TrimConfig: all knobs for the LLMTrim pipeline."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Pattern, Sequence

# Default protected regions: fenced code blocks and inline code.
DEFAULT_PROTECTED_PATTERNS: Sequence[str] = (
    r"```.*?```",      # fenced code blocks (DOTALL applied at compile time)
    r"`[^`\n]+`",      # inline code
)


@dataclass
class TrimConfig:
    """Configuration for :func:`llmtrim.trim`.

    Parameters
    ----------
    target_ratio:
        Desired ``final_tokens / original_tokens``. The prune budget is
        calculated from the original input token count, even when earlier
        stages have already reduced the text. ``1.0`` disables pruning.
    aggressiveness:
        0.0-1.0. Higher values delete content words more aggressively
        (lower score floor) and allow shorter fragments to remain.
    enabled_stages:
        Which stages to run, in execution order. Defaults to all stages.
    clean / structure / prune:
        Per-stage switches (kept as explicit booleans for convenience).
    translate_enabled:
        Run the zh->en translation stage before pruning. Default off.
    force_translate:
        Accept a validated translation even when it is not cheaper in tokens.
        Default off; implies no stage by itself when used through Python.
    translator:
        Registered translator name: ``"argos"`` (default) or ``"openai"``.
    translator_kwargs:
        Extra options for the chosen translator (e.g. base_url, api_key,
        model for the OpenAI-compatible one).
    protected_patterns:
        Regex source strings whose matches are shielded from pruning and
        translation. Defaults to fenced and inline code.
    language:
        ``"auto"`` detects CJK share per document; ``"zh"``/``"en"`` force.
    counter:
        ``"auto"`` uses tiktoken when installed, else the heuristic counter.
        ``"heuristic"`` forces the built-in counter.
    """

    target_ratio: float = 0.6
    aggressiveness: float = 0.5

    clean: bool = True
    structure: bool = True
    prune: bool = True
    translate_enabled: bool = False
    force_translate: bool = False

    enabled_stages: Optional[Sequence[str]] = None

    protected_patterns: Sequence[str] = field(
        default_factory=lambda: list(DEFAULT_PROTECTED_PATTERNS)
    )
    extra_protected_patterns: Sequence[str] = field(default_factory=list)

    language: str = "auto"
    counter: str = "auto"

    translate_target_lang: str = "en"
    translator: str = "argos"
    translator_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.target_ratio <= 1.0:
            raise ValueError(
                f"target_ratio must be in (0, 1], got {self.target_ratio}"
            )
        if not 0.0 <= self.aggressiveness <= 1.0:
            raise ValueError(
                f"aggressiveness must be in [0, 1], got {self.aggressiveness}"
            )

    def stage_enabled(self, name: str) -> bool:
        if self.enabled_stages is not None:
            return name in self.enabled_stages
        if name == "translate":
            return self.translate_enabled
        return bool(getattr(self, name, True))

    def compiled_protected_patterns(self) -> Sequence[Pattern[str]]:
        pats = list(self.protected_patterns) + list(self.extra_protected_patterns)
        return [re.compile(p, re.DOTALL) for p in pats]
