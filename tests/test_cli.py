import json
import subprocess
import sys

from llmtrim.cli import main  # noqa: F401


def run_cli(args, stdin=None):
    proc = subprocess.run(
        [sys.executable, "-m", "llmtrim.cli"] + args,
        input=stdin, capture_output=True, text=True,
    )
    return proc


def test_trim_stdin_pipe():
    proc = run_cli(["trim", "-r", "0.6", "--counter", "heuristic"],
                   stdin="The quick brown fox jumps over the lazy dog. " * 10)
    assert proc.returncode == 0
    assert len(proc.stdout) < len("The quick brown fox jumps over the lazy dog. " * 10)


def test_report_command():
    proc = run_cli(["report", "--counter", "heuristic"],
                   stdin="hello world " * 20)
    assert proc.returncode == 0
    assert "original" in proc.stdout
    assert "clean" in proc.stdout


def test_trim_report_flag_goes_to_stderr():
    proc = run_cli(["trim", "--report", "--counter", "heuristic"],
                   stdin="content " * 30)
    assert proc.returncode == 0
    assert "original" in proc.stderr


def test_json_metadata():
    proc = run_cli(["trim", "--json", "--counter", "heuristic"],
                   stdin="some content here " * 10)
    assert proc.returncode == 0
    data = json.loads(proc.stderr)
    assert "original_tokens" in data and "stages" in data


def test_trim_file(tmp_path):
    f = tmp_path / "in.txt"
    f.write_text("填充内容 " * 50, encoding="utf-8")
    out = tmp_path / "out.txt"
    proc = run_cli(["trim", str(f), "-o", str(out), "--counter", "heuristic"])
    assert proc.returncode == 0
    assert out.exists()


def test_no_translate_by_default_no_error():
    proc = run_cli(["trim", "--counter", "heuristic"], stdin="hello")
    assert proc.returncode == 0
