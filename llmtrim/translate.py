"""Stage 3 (optional) - zh->en translation with token-aware acceptance.

Design:
- ``Translator`` is a pluggable interface; implementations register by name.
- ``ArgosTranslator`` runs argostranslate fully offline (model downloaded
  explicitly via ``llmtrim setup-translate`` — never implicitly).
- ``OpenAICompatTranslator`` calls any OpenAI-compatible chat API.
- The stage processes line blocks (blank-line separated), masks protected
  spans (code, inline code, URLs) with stable placeholders, and translates
  only blocks that actually contain CJK.
- **Token-aware acceptance**: each translated block is adopted only when the
  English version counts fewer tokens than the original with the same
  counter. The stage can therefore never grow the input.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Protocol

from .counters import TokenCounter, cjk_char_count

_ZH_MIN_CJK = 2  # blocks need >=2 CJK chars before we bother translating


class Translator(Protocol):
    name: str

    def translate(self, text: str, source: str, target: str) -> str: ...


class ArgosTranslator:
    """Offline argostranslate engine (zh -> en)."""

    name = "argos"

    def translate(self, text: str, source: str = "zh", target: str = "en") -> str:
        import argostranslate.translate as at

        return at.translate(text, source, target)

    def is_ready(self, source: str = "zh", target: str = "en") -> bool:
        try:
            import argostranslate.translate as at
        except ImportError:
            return False
        langs = at.get_installed_languages()
        try:
            src = next(l for l in langs if l.code == source)
            dst = next(l for l in langs if l.code == target)
        except StopIteration:
            return False
        return src.get_translation_to(dst) is not None


class OpenAICompatTranslator:
    """Translation via any OpenAI-compatible chat completions endpoint."""

    name = "openai"

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 model: str = "gpt-4o-mini", timeout: float = 60.0) -> None:
        import os

        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.timeout = timeout
        if not self.api_key:
            raise ValueError(
                "OpenAICompatTranslator needs an api_key (or OPENAI_API_KEY env var)"
            )

    def translate(self, text: str, source: str = "zh", target: str = "en") -> str:
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=self.api_key,
                        timeout=self.timeout)
        prompt = (
            f"Translate the following text from {source} to {target}. "
            f"Output only the translation, preserving line structure.\n\n{text}"
        )
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return (resp.choices[0].message.content or text).strip()


_REGISTRY: Dict[str, type] = {
    ArgosTranslator.name: ArgosTranslator,
    OpenAICompatTranslator.name: OpenAICompatTranslator,
}


def register_translator(translator_cls: type) -> None:
    """Register a custom Translator class under its ``name`` attribute."""
    _REGISTRY[translator_cls.name] = translator_cls


def get_translator(name: str, **kwargs) -> Translator:
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown translator {name!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


# --- the stage itself -----------------------------------------------------

_URL_TOKEN_RE = re.compile(r"https?://\S+")


class _Placeholder:
    """Masks protected spans so the translator never sees them."""

    def __init__(self) -> None:
        self._store: Dict[str, str] = {}

    def mask(self, text: str) -> str:
        def repl(m: re.Match) -> str:
            key = f"@@T{len(self._store)}@@"
            self._store[key] = m.group(0)
            return key

        masked = _URL_TOKEN_RE.sub(repl, text)
        return masked

    def unmask(self, text: str) -> str:
        for key, original in self._store.items():
            text = text.replace(key, original)
        return text


def translate_stage(text: str, translator: Translator, counter: TokenCounter,
                    target_lang: str = "en",
                    protected_patterns: tuple = ()) -> str:
    """Translate CJK blocks to ``target_lang`` when it saves tokens."""
    if not cjk_char_count(text):
        return text

    blocks = re.split(r"\n{2,}", text)  # blank-line separated blocks
    out_blocks: List[str] = []
    for block in blocks:
        out_blocks.append(
            _translate_block(block, translator, counter, target_lang,
                             protected_patterns)
        )
    return "\n\n".join(out_blocks)


_NOTRANS = "\x00"  # line-prefix marker: never translate this line


def _translate_block(block: str, translator: Translator, counter: TokenCounter,
                     target_lang: str, protected_patterns: tuple) -> str:
    if cjk_char_count(block) < _ZH_MIN_CJK:
        return block

    # line-level protection: fenced code and protected-pattern lines stay put
    lines = block.split("\n")
    in_fence = False
    for i, line in enumerate(lines):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            lines[i] = _NOTRANS + line
            continue
        if in_fence or any(p.search(line) for p in protected_patterns):
            lines[i] = _NOTRANS + line
            continue
        if not cjk_char_count(line):
            lines[i] = _NOTRANS + line

    translatable = [i for i, l in enumerate(lines) if not l.startswith(_NOTRANS)]
    if not translatable:
        return block

    # mask protected spans (URLs) inside translatable lines
    ph = _Placeholder()
    joined = ph.mask("\n".join(lines[i] for i in translatable))

    try:
        translated = translator.translate(joined, "zh", target_lang)
    except Exception:
        return block  # translation is best-effort; never fail the pipeline

    if not translated or not translated.strip():
        return block

    translated = ph.unmask(translated)
    new_lines = translated.split("\n")
    if len(new_lines) != len(translatable):
        return block  # translator broke line structure; keep the original

    candidate_lines = list(lines)
    for i, new in zip(translatable, new_lines):
        candidate_lines[i] = new
    candidate = "\n".join(candidate_lines)

    # token-aware acceptance: adopt only when strictly smaller
    if counter.count(candidate) < counter.count(block):
        return candidate
    return block


def install_argos_model(source: str = "zh", target: str = "en") -> bool:
    """Explicitly download the argos model. Called by ``llmtrim setup-translate``."""
    try:
        import argostranslate.package as argopkg
    except ImportError:
        raise RuntimeError(
            "argostranslate is not installed; run: pip install llmtrim[translate]"
        )
    argopkg.update_package_index()
    available = argopkg.get_available_packages()
    pkg = next(
        (p for p in available if p.from_code == source and p.to_code == target),
        None,
    )
    if pkg is None:
        raise RuntimeError(f"no argos package for {source}->{target}")
    argopkg.install_from_path(pkg.download())
    return True
