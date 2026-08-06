"""Code execution tool: run a Python snippet in a sandboxed subprocess with
a timeout, capturing stdout, stderr, and whether it crashed or timed out.
"""
import subprocess
import sys

_DEFAULT_TIMEOUT_SECONDS = 5


def run(code: str, stdin: str = "", timeout: int = _DEFAULT_TIMEOUT_SECONDS) -> dict:
    """Execute `code` in an isolated subprocess.

    Returns {"stdout": str, "stderr": str, "exit_code": int | None,
    "timed_out": bool}. `exit_code` is None only when `timed_out` is True.
    """
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
