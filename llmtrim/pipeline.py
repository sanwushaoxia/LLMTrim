"""Stage orchestration: clean -> structure -> translate -> prune."""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Pattern, Sequence, Tuple

from .config import TrimConfig
from .counters import TokenCounter, cjk_char_count, get_counter
from .result import StageReport, TrimResult

STAGE_ORDER = ("clean", "structure", "translate", "prune")


def run_pipeline(text: str, config: Optional[TrimConfig] = None,
                 dry_run: bool = False) -> TrimResult:
    config = config or TrimConfig()
    counter = get_counter(config.counter)

    original_tokens = counter.count(text)
    # The public ratio is final/original, so pruning uses the original budget
    # even when an earlier stage has already removed tokens.
    prune_budget = max(1, int(config.target_ratio * original_tokens))
    reports: List[StageReport] = []
    translator_used: Optional[str] = None

    from .translate import TranslationStats

    translation_stats = TranslationStats()
    current = text
    for name in STAGE_ORDER:
        started = time.perf_counter()
        if not config.stage_enabled(name):
            reports.append(StageReport(
                name, 0, 0, applied=False, note="disabled",
                elapsed_ms=_elapsed_ms(started),
            ))
            continue

        before = counter.count(current)
        new_text, note, tr_name, metadata = _apply_stage(
            name, current, config, counter, prune_budget, translation_stats,
        )
        after = counter.count(new_text)
        reverted = False
        if name in ("clean", "structure") and after > before:
            # Safety net: tidy-up stages must never grow the text.
            new_text, after = current, before
            note = "reverted (would grow)"
            reverted = True

        reports.append(StageReport(
            name, before, after, applied=True, note=note,
            elapsed_ms=_elapsed_ms(started), reverted=reverted,
            metadata=metadata,
        ))
        if tr_name:
            translator_used = tr_name
        current = new_text

    return TrimResult(
        text=text if dry_run else current,
        original_text=text,
        original_tokens=original_tokens,
        final_tokens=counter.count(current),
        stages=reports,
        translator_used=translator_used,
        counter_name=getattr(counter, "name", counter.__class__.__name__),
        translation=translation_stats.__dict__.copy(),
    )


def _apply_stage(name: str, text: str, config: TrimConfig,
                 counter: TokenCounter, prune_budget: int,
                 translation_stats) -> Tuple[str, str, Optional[str], Dict]:
    note, tr_name = "", None
    if name == "clean":
        from .clean import clean

        return _preserve_protected(
            text, config.compiled_protected_patterns(), clean,
        ), note, None, {}
    if name == "structure":
        from .structure import structure

        return _preserve_protected(
            text, config.compiled_protected_patterns(), structure,
        ), note, None, {}
    if name == "translate":
        if not cjk_char_count(text):
            return text, "no CJK content", None, {}
        from .translate import get_translator, translate_stage

        translator = get_translator(config.translator, **config.translator_kwargs)
        result = translate_stage(
            text, translator, counter,
            target_lang=config.translate_target_lang,
            protected_patterns=tuple(config.compiled_protected_patterns()),
            stats=translation_stats,
            force=config.force_translate,
        )
        note = f"translator={translator.name}"
        return result, note, translator.name, {
            "attempted": translation_stats.attempted,
            "accepted": translation_stats.accepted,
            "rejected": translation_stats.rejected,
            "failures": translation_stats.failures,
            "force": config.force_translate,
        }
    if name == "prune":
        from .prune import prune

        result = prune(text, config, counter, prune_budget)
        return result, f"budget={prune_budget}", None, {
            "budget": prune_budget,
            "budget_basis": "original_tokens",
        }
    raise ValueError(f"unknown stage: {name}")


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def _preserve_protected(text: str, patterns: Sequence[Pattern[str]],
                        transform: Callable[[str], str]) -> str:
    """Run a stage while keeping configured protected spans byte-for-byte."""
    spans = _find_spans(text, patterns)
    if not spans:
        return transform(text)

    masked, protected = _mask_spans(text, spans)
    transformed = transform(masked)
    if any(token not in transformed for token, _ in protected):
        return text
    return _unmask_spans(transformed, protected)


def _find_spans(text: str, patterns: Sequence[Pattern[str]]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for pattern in patterns:
        spans.extend(
            (match.start(), match.end()) for match in pattern.finditer(text)
        )
    spans.sort()
    merged: List[Tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _mask_spans(text: str, spans: Sequence[Tuple[int, int]]):
    output: List[str] = []
    protected: List[Tuple[str, str]] = []
    position = 0
    for index, (start, end) in enumerate(spans):
        token = f"\x02LTM{index}\x03"
        output.extend((text[position:start], token))
        protected.append((token, text[start:end]))
        position = end
    output.append(text[position:])
    return "".join(output), protected


def _unmask_spans(text: str, protected: Sequence[Tuple[str, str]]) -> str:
    for token, original in protected:
        text = text.replace(token, original)
    return text
