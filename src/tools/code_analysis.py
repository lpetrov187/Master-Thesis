"""Code analysis tool: static findings via `ast` (syntax) and `ruff`
(style/lint) for Python, or `gcc -fsyntax-only` for C, without executing
the code under analysis.
"""
import ast
import json
import os
import re
import subprocess
import sys
import tempfile

_GCC_DIAG_RE = re.compile(
    r"^.+?:(?P<line>\d+):(?P<column>\d+): (?P<severity>error|warning): "
    r"(?P<message>.+?)(?:\s*\[(?P<rule>-W[\w=-]+)\])?$",
    re.MULTILINE,
)


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


def _run_gcc_analysis(code: str) -> dict:
    """Check C syntax and style with gcc's own diagnostics. `-fsyntax-only`
    stops after parsing/type-checking (no codegen); `-Wall -Wextra` gives
    warnings in the same pass, so - unlike Python's separate ast/ruff split
    - one gcc invocation covers both syntax and lint findings.

    gcc has no machine-readable output format (ruff's JSON isn't available
    here), so diagnostic lines are parsed with a best-effort regex - a rare
    diagnostic that doesn't match the usual "file:line:col: severity: msg"
    shape is simply not reported as a finding, rather than crashing.
    """
    fd, path = tempfile.mkstemp(suffix=".c")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)

        result = subprocess.run(
            ["gcc", "-fsyntax-only", "-Wall", "-Wextra", path],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        os.unlink(path)

    diagnostics = list(_GCC_DIAG_RE.finditer(result.stderr))
    findings = [
        {
            "rule": match.group("rule") or match.group("severity"),
            "message": match.group("message"),
            "line": int(match.group("line")),
            "column": int(match.group("column")),
        }
        for match in diagnostics
    ]
    syntax_valid = not any(match.group("severity") == "error" for match in diagnostics)
    return {"syntax_valid": syntax_valid, "findings": findings}


def analyze(code: str, language: str = "python") -> dict:
    """Statically analyze `code`. Returns {"syntax_valid": bool, "findings": [...]}."""
    if language == "c":
        return _run_gcc_analysis(code)

    syntax_error = _check_syntax(code)
    if syntax_error is not None:
        return {"syntax_valid": False, "findings": [syntax_error]}

    return {"syntax_valid": True, "findings": _run_ruff(code)}
