"""Command-line interface: llmtrim trim|report|setup-translate|counters."""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__
from .config import TrimConfig
from .pipeline import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmtrim",
        description="Token-aware input compression for LLM prompts.",
    )
    parser.add_argument("--version", action="version",
                        version=f"llmtrim {__version__}")
    sub = parser.add_subparsers(dest="command")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("input", nargs="?",
                       help="input file (default: stdin)")
        p.add_argument("-o", "--output",
                       help="write compressed text to this file "
                            "(default: stdout)")
        p.add_argument("-r", "--target-ratio", type=float, default=0.6,
                       help="target final/original token ratio (0-1, default 0.6)")
        p.add_argument("-a", "--aggressiveness", type=float, default=0.5,
                       help="content-word pruning aggressiveness 0-1 (default 0.5)")
        p.add_argument("--no-clean", action="store_true", help="disable clean stage")
        p.add_argument("--no-structure", action="store_true",
                       help="disable structure stage")
        p.add_argument("--no-prune", action="store_true", help="disable prune stage")
        p.add_argument("--counter", choices=["auto", "heuristic", "tiktoken"],
                       default="auto")
        p.add_argument("--report", action="store_true",
                       help="print a per-stage token report to stderr")

    p_trim = sub.add_parser("trim", help="compress text (default command)")
    add_common(p_trim)
    p_trim.add_argument("--translate", action="store_true",
                        help="enable zh->en translation stage")
    p_trim.add_argument("--translator", default="argos",
                        choices=["argos", "openai"],
                        help="translator engine for --translate")
    p_trim.add_argument("--json", action="store_true",
                        help="emit result metadata as JSON to stderr")

    p_report = sub.add_parser("report",
                              help="analyse savings per stage; do not modify")
    add_common(p_report)
    p_report.add_argument("--translate", action="store_true",
                          help="include the translate stage in the analysis")
    p_report.add_argument("--translator", default="argos",
                          choices=["argos", "openai"])

    p_setup = sub.add_parser("setup-translate",
                             help="download the argos zh->en model (explicit)")
    p_setup.add_argument("--source", default="zh")
    p_setup.add_argument("--target", default="en")

    return parser


def _read_input(args: argparse.Namespace) -> str:
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            return f.read()
    return sys.stdin.read()


def _write_output(args: argparse.Namespace, text: str) -> None:
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


def _config_from_args(args: argparse.Namespace, translate: bool) -> TrimConfig:
    return TrimConfig(
        target_ratio=args.target_ratio,
        aggressiveness=args.aggressiveness,
        clean=not args.no_clean,
        structure=not args.no_structure,
        prune=not args.no_prune,
        counter=args.counter,
        translate_enabled=translate,
        translator=args.translator,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "trim"):
        translate = getattr(args, "translate", False)
        config = _config_from_args(args, translate)
        text = _read_input(args)
        result = run_pipeline(text, config)
        _write_output(args, result.text)
        if getattr(args, "json", False):
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
                  file=sys.stderr)
        elif args.report:
            print(result.summary(), file=sys.stderr)
        return 0

    if args.command == "report":
        config = _config_from_args(args, getattr(args, "translate", False))
        text = _read_input(args)
        result = run_pipeline(text, config, dry_run=True)
        print(result.summary())
        return 0

    if args.command == "setup-translate":
        from .translate import install_argos_model

        print(f"downloading argos model {args.source}->{args.target} ...")
        install_argos_model(args.source, args.target)
        print("done")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
