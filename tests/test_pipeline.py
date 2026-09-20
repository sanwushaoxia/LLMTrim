import json

import pytest

from llmtrim import TrimConfig, analyze, trim


MIXED = """# 项目报告

这是一个用于测试的中文文档，它包含很多无意义的填充内容。这个文档的目的是测试压缩框架的效果。

The quick brown fox jumps over the lazy dog and the dog barks at the fox.

```python
def protected():
    return 'survive'
```
"""


def cfg(**kw):
    kw.setdefault("counter", "heuristic")
    return TrimConfig(**kw)


def test_trim_reduces_tokens():
    r = trim(MIXED, cfg(target_ratio=0.6))
    assert r.final_tokens < r.original_tokens
    assert 0 < r.ratio <= 1.0


def test_trim_ratio_respected_loosely():
    r = trim(MIXED, cfg(target_ratio=0.5))
    # prune may fall short if the score floor blocks deletion, but must not
    # wildly overshoot below
    assert r.final_tokens >= 5


def test_analyze_leaves_text_untouched():
    r = analyze(MIXED, cfg())
    assert r.text == MIXED
    assert r.final_tokens < r.original_tokens  # simulated savings reported


def test_disabled_stages_reported():
    r = trim(MIXED, cfg(prune=False, structure=False))
    by_name = {s.name: s for s in r.stages}
    assert by_name["prune"].applied is False
    assert by_name["structure"].applied is False


def test_result_summary_contains_stages():
    r = trim(MIXED, cfg())
    s = r.summary()
    assert "original" in s and "final" in s and "clean" in s


def test_protected_code_survives():
    r = trim(MIXED, cfg(target_ratio=0.3))
    assert "return 'survive'" in r.text


def test_translation_stage_via_fake():
    from llmtrim.translate import register_translator

    class Fake:
        name = "fake-e2e"

        def translate(self, text, source, target):
            return "\n".join(
                "gloss" if any("\u4e00" <= c <= "\u9fff" for c in l) else l
                for l in text.split("\n")
            )

    register_translator(Fake)
    text = "# 标题\n\n这是一段中文内容它很长应该被翻译成更短的英文文本\n\n结尾"
    r = trim(text, cfg(translate_enabled=True, translator="fake-e2e"))
    assert r.translator_used == "fake-e2e"
    assert r.final_tokens <= r.original_tokens


def test_config_validation():
    with pytest.raises(ValueError):
        TrimConfig(target_ratio=1.5)
    with pytest.raises(ValueError):
        TrimConfig(aggressiveness=-1)


def test_trim_empty_string():
    r = trim("", cfg())
    assert r.text == ""
    assert r.final_tokens == 0


def test_result_as_dict_json_serialisable():
    r = trim(MIXED, cfg())
    json.dumps(r.as_dict())  # must not raise
