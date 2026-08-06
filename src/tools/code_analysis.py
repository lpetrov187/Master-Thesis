"""Code analysis tool: static findings via `ast` (syntax) and `ruff`
(style/lint), without executing the code under analysis.
"""
import ast
import json
import os
import subprocess
import sys
import tempfile


def _check_syntax(code: str) -> dict | None:
    """Return a syntax-error finding, or None if `code` parses cleanly."""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return {
            "rule": "syntax-error",
            "message": exc.msg,
            "line": exc.lineno or 0,
            "column": (exc.offset or 1) - 1,
        }
    return None


def _run_ruff(code: str) -> list[dict]:
    """Lint `code` with ruff and return its findings."""
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)

        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--output-format=json", "--no-cache", path],
            capture_output=True,
            text=True,
            check=False,
        )
        raw_findings = json.loads(result.stdout or "[]")
    finally:
        os.unlink(path)

    return [
        {
            "rule": item["code"],
            "message": item["message"],
            "line": item["location"]["row"],
            "column": item["location"]["column"],
        }
        for item in raw_findings
    ]


def analyze(code: str) -> dict:
    """Statically analyze `code`. Returns {"syntax_valid": bool, "findings": [...]}."""
    syntax_error = _check_syntax(code)
    if syntax_error is not None:
        return {"syntax_valid": False, "findings": [syntax_error]}

    return {"syntax_valid": True, "findings": _run_ruff(code)}
