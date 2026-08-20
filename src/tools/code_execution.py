"""Code execution tool: run a Python or C snippet in a sandboxed subprocess
with a timeout, capturing stdout, stderr, and whether it crashed or timed
out.
"""
import os
import subprocess
import sys
import tempfile

_DEFAULT_TIMEOUT_SECONDS = 5


def _run_python(code: str, stdin: str, timeout: int) -> dict:
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "exit_code": None,
            "timed_out": True,
        }

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.returncode,
        "timed_out": False,
    }


def _run_c(code: str, stdin: str, timeout: int) -> dict:
    """Compile `code` with gcc, then run the resulting binary. A compile
    failure isn't a distinct return shape - it's surfaced through the same
    stdout/stderr/exit_code fields a runtime failure would use (compiler's
    stderr as `stderr`, its return code as `exit_code`), so callers that
    only ever handled Python don't need to special-case C at all."""
    source_fd, source_path = tempfile.mkstemp(suffix=".c")
    binary_fd, binary_path = tempfile.mkstemp(suffix=".exe")
    os.close(binary_fd)  # gcc writes here directly; just needed a free path

    try:
        with os.fdopen(source_fd, "w", encoding="utf-8") as f:
            f.write(code)

        compile_result = subprocess.run(
            ["gcc", source_path, "-o", binary_path],
            capture_output=True,
            text=True,
            check=False,
        )
        if compile_result.returncode != 0:
            return {
                "stdout": "",
                "stderr": compile_result.stderr,
                "exit_code": compile_result.returncode,
                "timed_out": False,
            }

        try:
            result = subprocess.run(
                [binary_path],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "exit_code": None,
                "timed_out": True,
            }

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "timed_out": False,
        }
    finally:
        os.unlink(source_path)
        os.unlink(binary_path)


def run(code: str, stdin: str = "", timeout: int = _DEFAULT_TIMEOUT_SECONDS, language: str = "python") -> dict:
    """Execute `code` in an isolated subprocess.

    Returns {"stdout": str, "stderr": str, "exit_code": int | None,
    "timed_out": bool}. `exit_code` is None only when `timed_out` is True.
    """
    if language == "c":
        return _run_c(code, stdin, timeout)

    return _run_python(code, stdin, timeout)
