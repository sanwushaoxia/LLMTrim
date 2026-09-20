from llmtrim.counters import HeuristicCounter, get_counter


def test_cjk_counts_per_char():
    c = HeuristicCounter()
    n = c.count("你好世界")
    assert 3 <= n <= 6  # ~1 token per CJK char


def test_english_counts_words():
    c = HeuristicCounter()
    assert c.count("hello world") < c.count("你好世界") * 2


def test_empty():
    assert HeuristicCounter().count("") == 0


def test_get_counter_auto_returns_something():
    c = get_counter("auto")
    assert c.count("test 测试") > 0


def test_more_text_more_tokens():
    c = HeuristicCounter()
    assert c.count("one two three") < c.count("one two three four five six")


def test_long_words_split():
    c = HeuristicCounter()
    # a 40-char URL-ish token should cost more than 4 tokens
    assert c.count("a" * 40) >= 8
