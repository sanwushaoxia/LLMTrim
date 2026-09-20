from llmtrim.config import TrimConfig
from llmtrim.counters import HeuristicCounter, get_counter
from llmtrim.prune import prune

COUNTER = HeuristicCounter()


def cfg(**kw):
    kw.setdefault("counter", "heuristic")
    return TrimConfig(**kw)


def test_noop_when_under_budget():
    text = "hello world foo bar"
    out = prune(text, cfg(), COUNTER, 1000)
    assert out == text


def test_prunes_stopwords_first():
    text = ("the quick brown fox jumps over the lazy dog " * 8).strip()
    budget = COUNTER.count(text) * 2 // 3
    out = prune(text, cfg(), COUNTER, budget)
    assert COUNTER.count(out) <= budget
    # content words must survive longer than "the"/"over"
    assert "quick" in out and "dog" in out
    assert COUNTER.count(out) < COUNTER.count(text)


def test_fenced_code_protected():
    text = (
        "filler words are here and they should go away soon enough\n"
        "```python\nresult = calculate(x, y, z)\n```\n"
        "more filler words repeated again and again to force pruning now\n"
        "even more filler text because we really need the pruning to bite\n"
    )
    out = prune(text, cfg(), COUNTER, COUNTER.count(text) // 2)
    assert "calculate(x, y, z)" in out


def test_inline_code_protected():
    text = "use `keep_me_intact()` please with lots of filler around it " * 6
    out = prune(text, cfg(), COUNTER, COUNTER.count(text) // 2)
    assert "`keep_me_intact()`" in out


def test_numbers_protected():
    text = ("the model scored 98.6 accuracy on 2024 benchmark results " * 8).strip()
    out = prune(text, cfg(), COUNTER, COUNTER.count(text) // 2)
    assert "98.6" in out


def test_never_grows():
    text = "short text"
    out = prune(text, cfg(), COUNTER, 1)
    assert COUNTER.count(out) <= COUNTER.count(text)


def test_chinese_pruning():
    text = (
        "这个文档包含很多无意义的填充内容需要被压缩掉。压缩框架应该删除停用词和无意义的内容。"
        "我们测试压缩功能是否正常工作并且保留重要的关键词信息比如机器学习和深度学习。"
    )
    budget = COUNTER.count(text) // 2
    out = prune(text, cfg(), COUNTER, budget)
    assert COUNTER.count(out) <= budget
    assert "机器学习" in out or "深度学习" in out


def test_deterministic():
    text = "alpha beta gamma delta epsilon zeta eta theta " * 5
    a = prune(text, cfg(), COUNTER, 20)
    b = prune(text, cfg(), COUNTER, 20)
    assert a == b
