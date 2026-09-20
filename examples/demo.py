"""End-to-end demo: compress a realistic RAG-style prompt.

Run:  python examples/demo.py
"""
from llmtrim import TrimConfig, analyze, trim

PROMPT = """# 客户支持知识库检索结果

以下是与用户问题相关的知识库文档，请基于这些内容回答。

---

## 文档一：退款政策

我们公司的退款政策规定，用户在购买产品之后的三十天之内可以申请全额退款。三十天之后申请退款的话，将根据具体的使用情况进行审批。退款申请需要提供订单号码和购买凭证。退款会在审批通过之后的五个工作日之内原路退回。

## 文档二：发票开具

用户可以在个人中心的订单详情页面申请开具电子发票。电子发票会在申请之后的二十四小时之内发送到用户的注册邮箱。如果需要修改发票抬头，请在申请之前修改个人信息。增值税专用发票需要提供公司的税号信息。

## Document 3: Account Security

If you suspect unauthorized access to your account, change your password
immediately and enable two-factor authentication. Contact support with
your account ID and the approximate time of the suspicious activity.

---

2024-01-15 08:00:01 INFO  cache refreshed for 1,024 knowledge entries
2024-01-15 08:00:01 INFO  embedding index rebuilt in 3.2s

参考链接: https://help.example.com/zh-CN/articles/2043/refund-policy-details?utm_source=kb&utm_medium=search
"""


def main() -> None:
    print("=" * 60)
    print("1) analyze() — dry run, text untouched")
    print("=" * 60)
    report = analyze(PROMPT, TrimConfig(target_ratio=0.5, counter="heuristic"))
    print(report.summary())

    print()
    print("=" * 60)
    print("2) trim() — actual compression")
    print("=" * 60)
    result = trim(PROMPT, TrimConfig(target_ratio=0.5, counter="heuristic"))
    print(result.summary())
    print()
    print("---- compressed prompt ----")
    print(result.text)


if __name__ == "__main__":
    main()
