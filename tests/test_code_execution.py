"""Day-4 tests: code_execution runs a snippet in a sandboxed subprocess and
captures stdout, stderr, exit code, and timeouts."""
from src.tools.code_execution import run


def test_captures_stdout():
    result = run("print(sum(range(5)))")

    assert result["stdout"].strip() == "10"
    assert result["exit_code"] == 0
    assert result["timed_out"] is False


def test_captures_exception_in_stderr():
    result = run("raise ValueError('boom')")

    assert result["exit_code"] != 0
    assert "ValueError" in result["stderr"]


def test_times_out_long_running_code():
    result = run("import time; time.sleep(5)", timeout=1)

    assert result["timed_out"] is True
