#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 CodeChecker viewer（如 ThorAWR）的 reports 页面 URL 提取代码告警。

数据来源说明：http://<host>/<product>/reports?run=xxx&is-unique=off 是一个 Vue SPA，
页面 HTML 中没有告警内容，前端通过 Thrift JSON-RPC 从
POST /<product>/<api-version>/CodeCheckerService 拉取数据。本脚本直接调用该 RPC 接口。

服务地址、产品路径、run 名与 is-unique 全部从 reports 页面 URL 解析，
因此唯一必需的输入就是这个 URL。

用法示例:
    # 表格输出（默认）
    codechecker_alerts.py "http://10.88.110.7:8080/ThorAWR/reports?run=thor-awr-17840&is-unique=off"
    # JSON / CSV 输出
    codechecker_alerts.py "http://10.88.110.7:8080/ThorAWR/reports?run=thor-awr-17840" --format json
    codechecker_alerts.py "http://10.88.110.7:8080/ThorAWR/reports?run=thor-awr-17840" --format csv > alerts.csv
    # 简化输出：文件:行:列 [checker] 消息
    codechecker_alerts.py "http://10.88.110.7:8080/ThorAWR/reports?run=thor-awr-17840" --brief
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import requests

API_VERSION = "v6.71"
PAGE_LIMIT = 500  # getRunResults 单页上限，实测超过 500 会被服务端截断
REQUEST_TIMEOUT = 60

# CodeChecker 前端 main.js 中的枚举映射（已在 JS bundle 中确认）
SEVERITY_NAMES = {
    0: "UNSPECIFIED",
    10: "STYLE",
    20: "LOW",
    30: "MEDIUM",
    40: "HIGH",
    50: "CRITICAL",
}
# 严重级别排序权重：越严重越靠前
SEVERITY_ORDER = {name: idx for idx, name in enumerate(
    ["CRITICAL", "HIGH", "MEDIUM", "LOW", "STYLE", "UNSPECIFIED"])}

DETECTION_STATUS_NAMES = {
    0: "New",
    1: "Resolved",
    2: "Unresolved",
    3: "Reopened",
    4: "Off",
    5: "Unavailable",
}

# ReportData (Thrift) 字段号 -> 名称
# 1=runId 2=checkerId 3=reportHash 4=file 5=checkerMsg 6=reportId 7=fileId
# 8=line 9=column 10=severity 11=reviewData 12=detectionStatus
# 13=detectionDate 15=reportCount 17=analyzerName
REPORT_FIELD = {
    "run_id": "1",
    "checker": "2",
    "report_hash": "3",
    "file": "4",
    "message": "5",
    "report_id": "6",
    "line": "8",
    "column": "9",
    "severity": "10",
    "detection_status": "12",
    "detection_date": "13",
    "report_count": "15",
    "analyzer": "17",
}

# RunData (Thrift) 字段号 -> 名称
RUN_FIELD = {
    "run_id": "1",
    "date": "2",
    "name": "3",
    "result_count": "5",
    "status_counts": "7",  # map<i32,i32> detectionStatus -> 数量
    "tag": "8",  # 运行标签，实测为 "CI"
}

# CSV 全字段输出顺序
CSV_FIELDS = ["severity", "file", "line", "column", "checker", "message",
              "analyzer", "detection_status", "detection_date",
              "report_id", "report_hash", "run_id"]
# 简化输出（--brief）：定位 + 解决告警所需字段
# （checker 指明触发的规则；report_hash 用于 CodeChecker 标记误报/已审阅）
BRIEF_FIELDS = ["file", "line", "column", "checker", "message", "report_hash"]


class RpcError(Exception):
    """CodeChecker RPC 调用失败。"""


class ThriftJson:
    """CodeChecker 服务端使用的 Thrift JSON 协议的极简编解码。

    请求包络: [1, "方法名", seqId, 0, {字段号: 值}]
    i64/i32 字段须写成 {"i64": 数字}（数字不带引号），字符串 {"str": "..."}，
    结构体 {"rec": {...}}，列表 {"lst": ["rec", 数量, 元素...]}，
    布尔 {"tf": 0/1}（服务端不接受 JSON 的 false/true）。
    响应包络 [1, "方法名", 2, 0, {字段号: 值}]，成功时返回值在字段 "0"。
    """

    def __init__(self, endpoint: str, token: Optional[str] = None) -> None:
        self.endpoint = endpoint
        self.session = requests.Session()
        self.session.headers["Content-Type"] = "application/json"
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self._seq = 0

    def call(self, method: str, args: Dict[str, Any], retries: int = 3) -> Any:
        self._seq += 1
        # 必须用紧凑分隔符：服务端的 Thrift JSON 解析器不接受
        # json.dumps 默认输出中逗号/冒号后的空格（会导致后端断连，nginx 502）
        payload = json.dumps([1, method, self._seq, 0, args],
                             separators=(",", ":"))
        last_error: Optional[RpcError] = None
        for attempt in range(retries):
            try:
                resp = self.session.post(self.endpoint, data=payload,
                                         timeout=REQUEST_TIMEOUT)
                envelope = resp.json()
            except requests.RequestException as exc:
                last_error = RpcError(f"请求 {self.endpoint} 失败: {exc}")
            except ValueError as exc:
                last_error = RpcError(
                    f"接口返回非 JSON 内容 (HTTP {resp.status_code})，"
                    f"前 200 字节: {resp.text[:200]!r}")
            else:
                break
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
        else:
            raise last_error
        if not (isinstance(envelope, list) and len(envelope) >= 5):
            raise RpcError(f"接口响应格式异常: {envelope!r}")
        success, result = envelope[2], envelope[4]
        if success != 2:
            msg = result.get("1", {}).get("str", "") if isinstance(result, dict) else str(result)
            code = result.get("2", {}).get("i32", "") if isinstance(result, dict) else ""
            raise RpcError(f"{method} 调用失败 (代码 {code}): {msg}")
        return result.get("0")

    @staticmethod
    def unpack_list(value: Any) -> List[Any]:
        """解包 {"lst": ["类型", 数量, 元素...]} 为 python 列表。"""
        if not value:
            return []
        items = value.get("lst", [])
        if not items:
            return []
        return list(items[2:2 + int(items[1])])

    @staticmethod
    def scalar(value: Any) -> Any:
        """解包 {"str": x} / {"i64": x} / {"i32": x} / {"tf": x} 为裸值。"""
        if isinstance(value, dict) and len(value) == 1:
            (kind, val), = value.items()
            if kind in ("str", "i64", "i32", "tf", "dbl"):
                return val
        return value


def build_report_args(run_id: int, limit: int, offset: int,
                      is_unique: bool) -> Dict[str, Any]:
    """构造 getRunResults 的请求参数 (QueryReportDataParams)。"""
    return {
        # 1: runIds, 2: limit, 3: offset, 4: sorting, 5: filters
        # 6: CompareData (字段 5: isUnique) 对应页面 is-unique 参数
        "1": {"lst": ["i64", 1, run_id]},
        "2": {"i64": limit},
        "3": {"i64": offset},
        "4": {"lst": ["rec", 0]},
        "5": {"rec": {}},
        "6": {"rec": {"5": {"tf": int(is_unique)}}},
        "7": {"rec": {}},
    }


def parse_reports_url(url: str) -> Tuple[str, str, bool]:
    """解析 reports 页面 URL。

    返回 (RPC base 地址如 http://host:8080/ThorAWR, run 名, is-unique)。
    URL 中必须带 run= 查询参数。
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise SystemExit(f"错误: 不是有效的 http(s) reports 页面 URL: {url}")
    query = parse_qs(parsed.query)
    run_names = query.get("run", [])
    if not run_names:
        raise SystemExit(f"错误: URL 中未找到 run= 参数: {url}")

    path = parsed.path
    marker = "/reports"
    idx = path.rfind(marker)
    if idx < 0:
        raise SystemExit(
            f"错误: URL 路径中未找到 {marker} 页面: {url}\n"
            "请传入浏览器中 reports 页面的完整地址。")
    base = f"{parsed.scheme}://{parsed.netloc}{path[:idx]}".rstrip("/")

    is_unique_vals = query.get("is-unique", [])
    is_unique = False
    if is_unique_vals:
        is_unique = is_unique_vals[0].strip().lower() in ("on", "true", "1")
    return base, run_names[0], is_unique


class ThorAwrClient:
    """CodeChecker 服务的只读客户端（base_url 形如 http://host/<product>）。"""

    def __init__(self, base_url: str, token: Optional[str] = None) -> None:
        self.rpc = ThriftJson(
            f"{base_url.rstrip('/')}/{API_VERSION}/CodeCheckerService", token)

    def get_run(self, run_name: str) -> Dict[str, Any]:
        """按名称查询 run，找不到时列出可用 run 并退出。"""
        args = {"1": {"rec": {"2": {"lst": ["str", 1, run_name]}}}}
        runs = ThriftJson.unpack_list(self.rpc.call("getRunData", args))
        if runs:
            return parse_run(runs[0])
        print(f"错误: 未找到 run '{run_name}'。", file=sys.stderr)
        print("可用的 run 如下（前 20 个）:", file=sys.stderr)
        print_runs(self.list_runs(), 20, sys.stderr)
        raise SystemExit(1)

    def list_runs(self) -> List[Dict[str, Any]]:
        args: Dict[str, Any] = {}
        return [parse_run(r) for r in
                ThriftJson.unpack_list(self.rpc.call("getRunData", args))]

    def fetch_alerts(self, run_id: int, is_unique: bool) -> List[Dict[str, Any]]:
        """分页拉取指定 run 的全部告警。"""
        alerts: List[Dict[str, Any]] = []
        offset = 0
        while True:
            args = build_report_args(run_id, PAGE_LIMIT, offset, is_unique)
            page = ThriftJson.unpack_list(self.rpc.call("getRunResults", args))
            alerts.extend(parse_report(rec) for rec in page)
            if len(page) < PAGE_LIMIT:
                return alerts
            offset += PAGE_LIMIT


def parse_run(rec: Dict[str, Any]) -> Dict[str, Any]:
    """把 RunData 原始字段 dict 转成语义化 dict。"""
    def field(name: str) -> Any:
        return ThriftJson.scalar(rec.get(RUN_FIELD[name]))

    return {
        "run_id": field("run_id"),
        "name": field("name"),
        "date": field("date"),
        "result_count": field("result_count"),
        "tag": field("tag"),
    }


def parse_report(rec: Dict[str, Any]) -> Dict[str, Any]:
    """把 ReportData 原始字段 dict 转成语义化告警 dict。"""
    def field(name: str) -> Any:
        raw = rec.get(REPORT_FIELD[name])
        return ThriftJson.scalar(raw)

    severity_code = int(field("severity") or 0)
    status_code = int(field("detection_status") or 0)
    return {
        "severity": SEVERITY_NAMES.get(severity_code, str(severity_code)),
        "checker": field("checker"),
        "file": field("file"),
        "line": field("line"),
        "column": field("column"),
        "message": field("message"),
        "analyzer": field("analyzer"),
        "detection_status": DETECTION_STATUS_NAMES.get(status_code, str(status_code)),
        "detection_date": field("detection_date"),
        "report_id": field("report_id"),
        "report_hash": field("report_hash"),
        "run_id": field("run_id"),
    }


def sort_alerts(alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按严重级别（严重在前）、文件路径、行号排序。"""
    return sorted(alerts, key=lambda a: (
        SEVERITY_ORDER.get(a["severity"], len(SEVERITY_ORDER)),
        a["file"] or "", a["line"] or 0))


def shorten_path(path: str, parts: int) -> str:
    """保留路径末尾 parts 段，前部以 ... 代替。"""
    tokens = [t for t in path.split("/") if t]
    if len(tokens) <= parts:
        return path
    return ".../" + "/".join(tokens[-parts:])


def print_table(alerts: List[Dict[str, Any]], run_name: str) -> None:
    if not alerts:
        print(f"run '{run_name}' 无代码告警。")
        return
    severities = [a["severity"] for a in alerts]
    sev_width = max(len(s) for s in severities)
    sev_width = max(sev_width, len("级别"))
    locs = [f"{shorten_path(a['file'], 3)}:{a['line']}:{a['column']}"
            for a in alerts]
    loc_width = max(max(len(x) for x in locs), len("位置"))
    checkers = [a["checker"] for a in alerts]
    chk_width = max(max(len(x) for x in checkers), len("checker"))

    print(f"run '{run_name}' 共 {len(alerts)} 条代码告警\n")
    header = f"{'级别':<{sev_width}}  {'位置':<{loc_width}}  {'checker':<{chk_width}}  消息"
    print(header)
    print("-" * len(header))
    for alert, loc in zip(alerts, locs):
        print(f"{alert['severity']:<{sev_width}}  {loc:<{loc_width}}  "
              f"{alert['checker']:<{chk_width}}  {alert['message']}")

    counts: Dict[str, int] = {}
    for sev in severities:
        counts[sev] = counts.get(sev, 0) + 1
    summary = ", ".join(f"{sev} {counts[sev]}"
                        for sev in SEVERITY_ORDER if sev in counts)
    print(f"\n汇总: {summary}")


def print_brief(alerts: List[Dict[str, Any]], run_name: str) -> None:
    """编译器风格的简化输出：文件:行:列: [checker] 消息。"""
    if not alerts:
        print(f"run '{run_name}' 无代码告警。")
        return
    print(f"run '{run_name}' 共 {len(alerts)} 条代码告警")
    for a in alerts:
        print(f"{a['file']}:{a['line']}:{a['column']}: "
              f"[{a['checker']}] {a['message']}")


def print_csv(alerts: List[Dict[str, Any]], file: Any,
              fields: List[str]) -> None:
    writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(alerts)


def print_runs(runs: List[Dict[str, Any]], limit: int, file: Any) -> None:
    shown = runs[:limit]
    name_w = max((len(r.get("name", "")) for r in shown), default=0)
    name_w = max(name_w, len("run 名"))
    date_w = max((len(r.get("date", "")) for r in shown), default=0)
    date_w = max(date_w, len("时间"))
    print(f"{'run 名':<{name_w}}  {'时间':<{date_w}}  告警数", file=file)
    for run in shown:
        print(f"{run.get('name', ''):<{name_w}}  {run.get('date', ''):<{date_w}}  "
              f"{run.get('result_count', 0)}", file=file)
    if len(runs) > limit:
        print(f"... 共 {len(runs)} 个 run，仅显示前 {limit} 个", file=file)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从 CodeChecker viewer（如 ThorAWR）的 reports 页面 URL 提取代码告警")
    parser.add_argument(
        "reports_url",
        help="reports 页面完整 URL（服务地址、run 名与 is-unique 均从中解析）")
    parser.add_argument(
        "--format", choices=["table", "json", "csv"], default="table",
        help="输出格式（默认 table）")
    parser.add_argument(
        "--brief", action="store_true",
        help="简化输出：仅文件/行/列/消息及解决告警相关字段（checker、report_hash）")
    parser.add_argument(
        "--token", default=None,
        help="可选的 Authorization Bearer token（当前服务匿名可读）")
    args = parser.parse_args()

    base_url, run_name, is_unique = parse_reports_url(args.reports_url)
    client = ThorAwrClient(base_url, args.token)
    run = client.get_run(run_name)
    alerts = sort_alerts(client.fetch_alerts(run["run_id"], is_unique))

    if args.format == "json":
        output_alerts = [{k: a[k] for k in BRIEF_FIELDS}
                         for a in alerts] if args.brief else alerts
        print(json.dumps({
            "run": run_name,
            "run_id": run["run_id"],
            "run_date": run["date"],
            "is_unique": is_unique,
            "alert_count": len(alerts),
            "alerts": output_alerts,
        }, ensure_ascii=False, indent=2))
    elif args.format == "csv":
        print_csv(alerts, sys.stdout,
                  BRIEF_FIELDS if args.brief else CSV_FIELDS)
    elif args.brief:
        print_brief(alerts, run_name)
    else:
        print_table(alerts, run_name)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RpcError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(130)
