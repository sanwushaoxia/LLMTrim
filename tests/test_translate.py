"""Translate stage tests use a FakeTranslator — no models, no network."""
import pytest

from llmtrim.config import TrimConfig
from llmtrim.counters import HeuristicCounter
from llmtrim.translate import (
    translate_stage,
    get_translator,
    register_translator,
    Translator,
)

COUNTER = HeuristicCounter()


class FakeTranslator:
    """Translates by replacing CJK runs with a terse English gloss."""

    name = "fake"

    def translate(self, text: str, source: str, target: str) -> str:
        out = []
        for line in text.split("\n"):
            if any("\u4e00" <= ch <= "\u9fff" for ch in line):
                out.append("terse english gloss")
            else:
                out.append(line)
        return "\n".join(out)


class ExplodingTranslator:
    name = "exploding"

    def translate(self, text: str, source: str, target: str) -> str:
        raise RuntimeError("boom")


@pytest.fixture(autouse=True)
def _register_fake():
    register_translator(FakeTranslator)
    yield


def test_adopted_when_smaller():
    long_zh = "这是一段非常长的中文句子它包含很多字符应该被翻译成更短的英文"
    out = translate_stage(long_zh, FakeTranslator(), COUNTER)
    assert out == "terse english gloss"


def test_rejected_when_larger():
    class InflatingTranslator(FakeTranslator):
        name = "inflating"

        def translate(self, text, source, target):
            return FakeTranslator.translate(self, text, source, target) + " padding " * 30

    register_translator(InflatingTranslator)
    zh = "这是一段中文"
    out = translate_stage(zh, InflatingTranslator(), COUNTER)
    assert out == zh  # kept original


def test_never_grows_the_input():
    text = "中文行\n英文 line stays\n另一中文行"
    out = translate_stage(text, FakeTranslator(), COUNTER)
    assert COUNTER.count(out) <= COUNTER.count(text)


def test_code_fence_not_translated():
    text = "中文说明\n```\n中文在代码块里\n```\n更多中文"
    out = translate_stage(text, FakeTranslator(), COUNTER)
    assert "中文在代码块里" in out  # untouched


def test_inline_code_protected():
    text = "运行 `npm install zh-utils` 然后重新启动服务进程"
    out = translate_stage(text, FakeTranslator(), COUNTER, "en",
                          (TrimConfig().compiled_protected_patterns()[1],))
    assert "`npm install zh-utils`" in out


def test_translator_failure_keeps_original():
    zh = "这段中文不会因为翻译器崩溃而丢失"
    out = translate_stage(zh, ExplodingTranslator(), COUNTER)
    assert out == zh


def test_no_cjk_short_circuits():
    text = "plain english only"
    assert translate_stage(text, ExplodingTranslator(), COUNTER) == text


def test_url_masked_from_translator():
    seen = {}

    class RecordingTranslator(FakeTranslator):
        name = "recording"

        def translate(self, text, source, target):
            seen["text"] = text
            return super().translate(text, source, target)

    register_translator(RecordingTranslator)
    translate_stage("访问 https://example.com/a?b=c 页面获取信息", RecordingTranslator(), COUNTER)
    assert "example.com" not in seen["text"] or seen["text"].count("@@T") > 0


def test_get_translator_unknown_raises():
    with pytest.raises(ValueError):
        get_translator("does-not-exist")
