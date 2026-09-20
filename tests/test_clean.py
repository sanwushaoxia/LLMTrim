from llmtrim.clean import clean


def test_strips_invisible_chars():
    text = "hello\u200b world\ufeff"
    assert clean(text) == "hello world"


def test_collapses_blank_lines():
    assert clean("a\n\n\n\nb") == "a\n\nb"


def test_drops_duplicate_consecutive_lines():
    assert clean("x\nx\nx\ny") == "x\ny"


def test_shrinks_separators():
    assert clean("-----") == "---"
    assert clean("====================") == "==="


def test_caps_char_runs():
    assert clean("wow!!!!!!!") == "wow!!!"


def test_normalises_spaces_but_keeps_single():
    assert clean("a    b\tc") == "a b c"


def test_preserves_code_fence_indentation():
    text = "```python\ndef f():\n    return 1\n```"
    assert clean(text) == text


def test_idempotent():
    text = "a  b\n\n\n\nc!!!\n-----\n"
    once = clean(text)
    assert clean(once) == once
