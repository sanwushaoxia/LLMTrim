# LLMTrim

Token-aware input compression for LLM prompts — cut prompt tokens before
they hit the model, without changing what the prompt means.

```
$ echo "很多很多填充内容 ..." | llmtrim -r 0.5 --report
original: 223 tokens
final:    134 tokens (60.1% of original, saved 89)
```

## Why

LLM input tokens cost money and context window. A lot of real-world prompt
bulk is *not* signal: duplicated lines, repeated log timestamps, long URLs,
decorative markdown, boilerplate filler words. LLMTrim removes that bulk
with a deterministic pipeline and reports exactly what each stage saved.

## Install

```bash
pip install -e .                 # core (jieba included)
pip install -e ".[tiktoken]"     # exact token counting with tiktoken
pip install -e ".[translate]"    # offline zh->en translation (argos)
```

## Quick start (Python)

```python
from llmtrim import trim, analyze, TrimConfig

result = trim(text, TrimConfig(target_ratio=0.6, aggressiveness=0.5))

print(result.text)             # compressed prompt
print(result.original_tokens)  # measured before
print(result.final_tokens)     # measured after
print(result.summary())        # per-stage token report

# dry run: what would each stage save?
report = analyze(text)
```

## Quick start (CLI)

```bash
llmtrim trim input.txt -r 0.5 --report      # compress + report to stderr
cat prompt.txt | llmtrim -r 0.6             # stdin -> stdout
llmtrim trim input.txt -o out.txt --json    # metadata as JSON on stderr
llmtrim report input.txt                    # analysis only, text untouched
llmtrim setup-translate                     # download argos zh->en model
```

## The pipeline

Stages run in order; every stage's token delta is recorded in
`TrimResult.stages`.

1. **clean** — zero-width/bidi chars out, whitespace normalised, blank-line
   runs collapsed, consecutive duplicate lines dropped, decorative
   separators (`----`, `====`) and `!!!!!` runs capped. Fenced code keeps
   its exact indentation.
2. **structure** — long URLs collapse to `https://domain/…`, repeated log
   timestamps fold (`… User 43 logged in`), JSON documents compact to one
   line, markdown emphasis stripped outside code.
3. **translate** *(optional, off by default)* — Chinese blocks become
   English, which is usually much cheaper under BPE tokenisation. See
   below for the safety rules.
4. **prune** — tokens scored TF-IDF-style over line pseudo-documents:
   stop words score far below zero, numbers/headings get bumps. The
   lowest-scoring tokens are deleted until the `target_ratio` budget is
   met. Deletion-only (no rewriting), deterministic, and protected spans
   (fenced code, inline code, custom patterns) are untouchable.

## zh -> en translation

Same meaning, fewer tokens: English text is typically 30-50% cheaper than
Chinese under BPE tokenisers. Enable with:

```python
TrimConfig(translate_enabled=True)                    # argos, offline
TrimConfig(translate_enabled=True, translator="openai",
           translator_kwargs={"base_url": "...", "api_key": "...",
                              "model": "gpt-4o-mini"})
```

Safety rules baked in:

- **Token-aware acceptance** — a block is adopted only if the translation
  counts *fewer* tokens with the same counter. The stage can never grow
  the input.
- Code fences, inline code and URLs never reach the translator.
- Any translator failure keeps the original text (best-effort by design).
- Custom engines: subclass, give it a `name`, call
  `llmtrim.translate.register_translator(MyCls)`.

## Configuration reference

| Option | Default | Meaning |
| --- | --- | --- |
| `target_ratio` | `0.6` | target `final/original` token ratio (0–1] |
| `aggressiveness` | `0.5` | 0–1; how deep into the score ranking pruning may reach |
| `clean` / `structure` / `prune` | `True` | per-stage switches |
| `translate_enabled` | `False` | run the zh->en stage |
| `translator` | `"argos"` | engine name (`argos`, `openai`, or registered) |
| `translator_kwargs` | `{}` | engine options (base_url, api_key, model) |
| `protected_patterns` | code fences, inline code | extra regexes are appended via `extra_protected_patterns` |
| `language` | `"auto"` | `"zh"` / `"en"` force stop-word set |
| `counter` | `"auto"` | `"heuristic"` forces the built-in estimator |

## Extending

```python
from llmtrim.translate import register_translator

class MyTranslator:
    name = "my-engine"
    def translate(self, text, source, target): ...

register_translator(MyTranslator)
trim(text, TrimConfig(translate_enabled=True, translator="my-engine"))
```

Token counters are swappable too — anything with `.count(text) -> int`
works; pass `"heuristic"` to avoid the tiktoken dependency.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Scope & guarantees

- Deletion-only compression: wording is never rewritten (translation is
  the single opt-in exception, and it is accepted only when smaller).
- Deterministic: same input + config ⇒ same output.
- The clean/structure stages revert automatically if they would ever grow
  the text.
- Protected regions (code fences, inline code, custom regexes) are never
  pruned, cleaned away, or translated.
