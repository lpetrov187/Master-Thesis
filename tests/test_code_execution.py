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


# --- C support (compile with gcc, then run), added alongside the code-generation loop ---

_C_HELLO = '#include <stdio.h>\nint main() { printf("%d\\n", 2 + 3); return 0; }'


def test_c_captures_stdout():
    result = run(_C_HELLO, language="c")

    assert result["stdout"].strip() == "5"
    assert result["exit_code"] == 0
    assert result["timed_out"] is False


def test_c_compile_error_surfaces_via_exit_code_and_stderr():
    result = run("int main() { return 0 }", language="c")

    assert result["exit_code"] != 0
    assert result["stderr"]
    assert result["timed_out"] is False


def test_c_runtime_crash_is_nonzero_exit():
    result = run("int main() { int *p = 0; return *p; }", language="c")

    assert result["exit_code"] != 0


def test_c_times_out_long_running_code():
    result = run("int main() { for (;;) {} }", language="c", timeout=1)

    assert result["timed_out"] is True
