from llmtrim.structure import structure


def test_long_url_shrinks():
    url = "https://example.com/very/long/path?q=abc&x=1" * 3
    out = structure(f"see {url} here")
    assert out.startswith("see https://example.com/")
    assert "…" in out
    assert "very/long/path" not in out


def test_short_url_kept():
    url = "https://example.com/a"
    assert structure(f"see {url} here") == f"see {url} here"


def test_log_timestamp_folding():
    text = (
        "2024-01-02 03:04:05 INFO User 42 logged in\n"
        "2024-01-02 03:04:05 INFO User 43 logged in"
    )
    out = structure(text)
    assert out.count("2024-01-02 03:04:05") == 1
    assert "… User 43 logged in" in out


def test_json_compaction():
    text = '{\n  "a": 1,\n  "b": [1, 2, 3]\n}'
    out = structure(text)
    assert "\n" not in out
    assert out == '{"a":1,"b":[1,2,3]}'


def test_markdown_decoration_stripped():
    assert structure("this is **bold** and *italic*") == "this is bold and italic"


def test_markdown_inside_fence_untouched():
    text = "```\n**not markdown**\n```"
    assert structure(text) == text


def test_idempotent():
    text = "see https://example.com/a/b/c/d/e/f/g here\n**bold** text"
    once = structure(text)
    assert structure(once) == once
