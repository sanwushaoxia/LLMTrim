# LLMTrim

[English](README.md) | 简体中文

面向 LLM 提示词的 token 感知输入压缩框架——在请求模型之前削减提示词 token，而不改变提示词的含义。

```
$ echo "很多很多填充内容 ..." | llmtrim -r 0.5 --report
original: 223 tokens
final:    134 tokens (60.1% of original, saved 89)
```

## 为什么需要它

LLM 的输入 token 既花钱又占上下文窗口。而真实场景的提示词里有大量内容*并非*有效信息：重复的行、重复的日志时间戳、超长 URL、装饰性 markdown、样板填充词。LLMTrim 用一条确定性管线剔除这些冗余，并精确报告每个阶段节省了多少。

## 安装

推荐使用项目内的虚拟环境，让 token 计数和可选的翻译依赖与系统 Python
隔离。

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[tiktoken,translate]"
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[tiktoken,translate]"
```

### Windows 命令提示符

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -e ".[tiktoken,translate]"
```

如果只需要精确 token 计数，可以只安装较小的 extra：

```bash
python -m pip install -e ".[tiktoken]"
```

也可以继续使用全局安装方式：

```bash
pip install -e .                 # 核心功能（已包含 jieba）
pip install -e ".[tiktoken]"     # 用 tiktoken 精确计数 token
pip install -e ".[translate]"    # 离线中→英翻译（argos）
```

## 快速上手（Python）

```python
from llmtrim import trim, analyze, TrimConfig

result = trim(text, TrimConfig(target_ratio=0.6, aggressiveness=0.5))

print(result.text)             # 压缩后的提示词
print(result.original_tokens)  # 压缩前的 token 数
print(result.final_tokens)     # 压缩后的 token 数
print(result.summary())        # 分阶段 token 报告

# 试运行：每个阶段分别能省多少？
report = analyze(text)
```

## 快速上手（CLI）

```bash
llmtrim trim input.txt -r 0.5 --report      # 压缩 + 报告输出到 stderr
cat prompt.txt | llmtrim -r 0.6             # stdin -> stdout
llmtrim trim input.txt -o out.txt --json    # 元数据以 JSON 输出到 stderr
llmtrim report input.txt                    # 仅分析，不修改文本
llmtrim setup-translate                     # 下载 argos 中→英模型
```

安装 `.[tiktoken]` 后，可以使用 `tiktoken` 进行精确的
`o200k_base` token 计数：

```bash
llmtrim trim input.txt -r 0.5 --counter tiktoken --report
```

## 压缩管线

各阶段按顺序执行；每个阶段的 token 变化都记录在 `TrimResult.stages` 里。

1. **clean（清理）** — 去除零宽/双向控制字符，规范化空白，合并连续空行，删除连续重复行，收缩装饰性分隔线（`----`、`====`）和 `!!!!!` 重复字符。围栏代码块内的缩进保持原样。
2. **structure（结构）** — 长 URL 收缩为 `https://domain/…`，重复的日志时间戳折叠（`… User 43 logged in`），JSON 文档压成单行，去除代码外的 markdown 强调标记。
3. **translate（翻译）** *（可选，默认关闭）* — 中文块转为英文，在 BPE 分词下英文通常便宜得多。安全规则见下文。
4. **prune（剪枝）** — 以行为伪文档按 TF-IDF 风格给词元打分：停用词得分远低于零，数字/标题获得加分。从得分最低的词元开始删除，直到达到 `target_ratio` 预算。只删除不改写（确定性结果），保护区（围栏代码、行内代码、自定义正则）不可触碰。

## 中文 → 英文翻译

同样的语义，更少的 token：在 BPE 分词器下，英文文本通常比中文便宜 30–50%。

### 可执行命令

一次性准备——下载离线 argos 模型（约 100–200 MB，无需 API key）：

```bash
pip install -e ".[translate]"
llmtrim setup-translate
```

对文件执行"先翻译、后压缩"（翻译在剪枝之前运行；最终预算仍按原始输入 token 数乘以 `target_ratio` 计算）：

```bash
llmtrim trim prompt_zh.txt --translate --counter tiktoken \
  -r 0.5 --report -o prompt_en.txt
```

即使译文消耗更多 token，也强制采用通过校验的英文译文：

```bash
llmtrim trim prompt_zh.txt --force-translate --counter tiktoken \
  -r 0.5 --report -o prompt_en.txt
```

普通 `--translate` 只有在所选计数器下译文更省 token 时才会采纳。
`--force-translate` 只绕过这一项成本检查；空结果、格式错误、翻译失败或破坏保护内容的译文仍会被拒绝。强制模式可能增加 token 用量并改变措辞。

也可以用管道从 stdin 读入：

```bash
cat prompt_zh.txt | llmtrim trim --translate -r 0.6
```

仅分析——查看翻译 + 剪枝各阶段能省多少，不修改文本：

```bash
llmtrim report prompt_zh.txt --translate
```

等价的 Python API：

```python
from llmtrim import trim, TrimConfig

result = trim(text, TrimConfig(
    translate_enabled=True,
    force_translate=True,
    target_ratio=0.5,
))
print(result.summary())
```

API 翻译替代方案（OpenAI 兼容接口，替代本地 argos）：

```python
TrimConfig(translate_enabled=True, translator="openai",
           translator_kwargs={"base_url": "...", "api_key": "...",
                              "model": "gpt-4o-mini"})
```

内置的安全规则：

- **Token 感知采纳** — 普通翻译只有在同一计数器下 token *更少* 时才会被采纳。`--force-translate` 可以绕过这项成本检查，但可能增加 token 用量或改变措辞。
- 代码围栏、行内代码和 URL 永远不会送进翻译器。
- 空结果、格式错误、翻译失败或破坏保护内容的译文始终会被拒绝。
- 自定义引擎：实现类并赋予 `name` 属性，然后调用 `llmtrim.translate.register_translator(MyCls)` 注册。

## 配置参考

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `target_ratio` | `0.6` | 目标 `最终/原始` token 比例（0–1] |
| `aggressiveness` | `0.5` | 0–1；剪枝可以深入打分排名的程度 |
| `clean` / `structure` / `prune` | `True` | 各阶段独立开关 |
| `translate_enabled` | `False` | 是否运行中→英翻译阶段 |
| `force_translate` | `False` | 即使译文 token 更多，也采纳通过校验的译文 |
| `translator` | `"argos"` | 翻译引擎名称（`argos`、`openai` 或已注册引擎） |
| `translator_kwargs` | `{}` | 引擎选项（base_url、api_key、model） |
| `protected_patterns` | 代码围栏、行内代码 | 通过 `extra_protected_patterns` 追加更多正则 |
| `language` | `"auto"` | `"zh"` / `"en"` 强制指定停用词表 |
| `counter` | `"auto"` | `"heuristic"` 强制使用内置估算器 |

## 扩展

```python
from llmtrim.translate import register_translator

class MyTranslator:
    name = "my-engine"
    def translate(self, text, source, target): ...

register_translator(MyTranslator)
trim(text, TrimConfig(translate_enabled=True, translator="my-engine"))
```

token 计数器同样可替换——任何具有 `.count(text) -> int` 方法的对象都可以；传 `"heuristic"` 可避免 tiktoken 依赖。

## 开发

```bash
pip install -e ".[dev]"
pytest
```

## 范围与保证

- 只删不改：从不改写措辞（翻译是唯一的可选例外；强制模式可能采纳更长的译文）。
- 确定性：相同输入 + 相同配置 ⇒ 相同输出。
- clean/structure 阶段如果会让文本变大，会自动回退。
- 保护区（代码围栏、行内代码、自定义正则）在 clean、structure、翻译和剪枝阶段都会按字节原样保留。URL 不会送入翻译器；结构阶段可以按设计收缩过长 URL。
- 翻译是可选的语义改写；即使有保护区和 token 增长检查，敏感或高风险提示词仍应人工复核。
