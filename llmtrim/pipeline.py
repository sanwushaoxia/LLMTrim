"""Stage orchestration: clean -> structure -> translate -> prune."""
from __future__ import annotations

from typing import List, Optional

from .config import TrimConfig
from .counters import TokenCounter, cjk_char_count, get_counter
from .result import StageReport, TrimResult

STAGE_ORDER = ("clean", "structure", "translate", "prune")


def run_pipeline(text: str, config: Optional[TrimConfig] = None,
                 dry_run: bool = False) -> TrimResult:
    config = config or TrimConfig()
    counter = get_counter(config.counter)

    original_tokens = counter.count(text)
    reports: List[StageReport] = []
    translator_used: Optional[str] = None

    if dry_run:
        # analyse only: simulate each stage, report would-be savings,
        # but return the untouched input as ``text``.
        current = text
        for name in STAGE_ORDER:
            if not config.stage_enabled(name):
                reports.append(StageReport(name, 0, 0, applied=False,
                                           note="disabled"))
                continue
            before = counter.count(current)
            new_text, note, tr_name = _apply_stage(name, current, config, counter)
            reports.append(StageReport(
                name, before, counter.count(new_text),
                applied=True, note=note,
            ))
            if tr_name:
                translator_used = tr_name
            current = new_text
        return TrimResult(
            text=text,
            original_text=text,
            original_tokens=original_tokens,
            final_tokens=counter.count(current),
            stages=reports,
            translator_used=translator_used,
        )

    current = text
    for name in STAGE_ORDER:
        if not config.stage_enabled(name):
            reports.append(StageReport(name, 0, 0, applied=False,
                                       note="disabled"))
            continue
        before = counter.count(current)
        new_text, note, tr_name = _apply_stage(name, current, config, counter)
        after = counter.count(new_text)
        if name in ("clean", "structure") and after > before:
            # safety net: tidy-up stages must never grow the text
            new_text, after = current, before
            note = "reverted (would grow)"
        reports.append(StageReport(name, before, after, applied=True, note=note))
        if tr_name:
            translator_used = tr_name
        current = new_text

    return TrimResult(
        text=current,
        original_text=text,
        original_tokens=original_tokens,
        final_tokens=counter.count(current),
        stages=reports,
        translator_used=translator_used,
    )


def _apply_stage(name: str, text: str, config: TrimConfig,
                 counter: TokenCounter):
    note, tr_name = "", None
    if name == "clean":
        from .clean import clean

        return clean(text), note, None
    if name == "structure":
        from .structure import structure

        return structure(text), note, None
    if name == "translate":
        if not cjk_char_count(text):
            return text, "no CJK content", None
        from .translate import get_translator, translate_stage

        translator = get_translator(config.translator, **config.translator_kwargs)
        result = translate_stage(
            text, translator, counter,
            target_lang=config.translate_target_lang,
            protected_patterns=tuple(config.compiled_protected_patterns()),
        )
        note = f"translator={translator.name}"
        return result, note, translator.name
    if name == "prune":
        from .prune import prune

        budget = max(1, int(config.target_ratio * counter.count(text)))
        result = prune(text, config, counter, budget)
        return result, f"budget={budget}", None
    raise ValueError(f"unknown stage: {name}")
