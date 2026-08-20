"""Tests for the pure-logic pieces of code_generation.py: extract_code(),
needs_revision(), _describe_problem(), and generate_and_verify_code()'s
iteration bookkeeping (with generate_code/revise_code/execute_tool mocked
out, so the control flow is tested without spending real model calls)."""
from unittest.mock import patch

from src.agent.code_generation import (
    _describe_problem,
    extract_code,
    generate_and_verify_code,
    needs_revision,
)


def test_extract_code_pulls_python_fenced_block():
    draft = "Here's the code:\n\n```python\ndef add(a, b):\n    return a + b\n```\n\nDone."

    assert extract_code(draft) == (["def add(a, b):", "    return a + b"], "python")


def test_extract_code_handles_bare_fence_with_no_language_tag():
    draft = "```\nprint('hi')\n```"

    assert extract_code(draft) == (["print('hi')"], "python")


def test_extract_code_returns_none_when_no_fenced_block():
    assert extract_code("just some prose, no code here") is None


def test_extract_code_returns_none_for_empty_fenced_block():
    assert extract_code("```python\n\n```") is None


def test_extract_code_takes_the_first_of_multiple_blocks():
    draft = "```python\nfirst_block()\n```\nsome text\n```python\nsecond_block()\n```"

    assert extract_code(draft) == (["first_block()"], "python")


def test_extract_code_detects_c_fence():
    draft = "```c\nint main() { return 0; }\n```"

    assert extract_code(draft) == (["int main() { return 0; }"], "c")


def test_extract_code_c_tag_is_case_insensitive():
    draft = "```C\nint main() { return 0; }\n```"

    assert extract_code(draft) == (["int main() { return 0; }"], "c")


def test_extract_code_unrecognized_tag_defaults_to_python():
    draft = "```javascript\nconsole.log('hi')\n```"

    assert extract_code(draft) == (["console.log('hi')"], "python")


def _analysis(syntax_valid=True, findings=None, error=None):
    return {"result": {"syntax_valid": syntax_valid, "findings": findings or []}, "error": error}


def _execution(exit_code=0, timed_out=False, error=None):
    return {"result": {"exit_code": exit_code, "timed_out": timed_out}, "error": error}


def test_needs_revision_false_when_everything_clean():
    assert needs_revision(_analysis(), _execution()) is False


def test_needs_revision_true_on_syntax_error():
    assert needs_revision(_analysis(syntax_valid=False), _execution()) is True


def test_needs_revision_true_on_lint_findings():
    assert needs_revision(_analysis(findings=[{"rule": "F401"}]), _execution()) is True


def test_needs_revision_true_on_nonzero_exit_code():
    assert needs_revision(_analysis(), _execution(exit_code=1)) is True


def test_needs_revision_true_on_timeout():
    assert needs_revision(_analysis(), _execution(exit_code=None, timed_out=True)) is True


def test_needs_revision_true_when_analysis_tool_errored():
    assert needs_revision(_analysis(error="boom"), _execution()) is True


def test_needs_revision_true_when_execution_tool_errored():
    assert needs_revision(_analysis(), _execution(error="boom")) is True


def test_describe_problem_none_when_everything_clean():
    assert _describe_problem(_analysis(), _execution()) is None


def test_describe_problem_reports_syntax_error_message():
    analysis = _analysis(syntax_valid=False, findings=[{"message": "invalid syntax"}])
    assert _describe_problem(analysis, _execution()) == "syntax error: invalid syntax"


def test_describe_problem_reports_lint_finding_count():
    analysis = _analysis(findings=[{"rule": "F401"}, {"rule": "E501"}])
    assert _describe_problem(analysis, _execution()) == "2 lint finding(s)"


def test_describe_problem_reports_nonzero_exit_code():
    assert _describe_problem(_analysis(), _execution(exit_code=1)) == "execution exited with code 1"


def test_describe_problem_reports_timeout():
    result = _describe_problem(_analysis(), _execution(exit_code=None, timed_out=True))
    assert result == "execution timed out"


def test_describe_problem_combines_multiple_problems():
    analysis = _analysis(findings=[{"rule": "F401"}])
    result = _describe_problem(analysis, _execution(exit_code=1))
    assert result == "1 lint finding(s); execution exited with code 1"


# --- generate_and_verify_code(): control flow only, model calls mocked out ---


_CLEAN_ANALYSIS = {"tool": "code_analysis", "args": {}, "result": {"syntax_valid": True, "findings": []}, "error": None}
_CLEAN_EXECUTION = {
    "tool": "code_execution",
    "args": {},
    "result": {"stdout": "ok\n", "stderr": "", "exit_code": 0, "timed_out": False},
    "error": None,
}
_BROKEN_ANALYSIS = {
    "tool": "code_analysis",
    "args": {},
    "result": {"syntax_valid": False, "findings": [{"message": "invalid syntax"}]},
    "error": None,
}


@patch("src.agent.code_generation.execute_tool")
@patch("src.agent.code_generation.revise_code")
@patch("src.agent.code_generation.generate_code")
def test_reports_one_iteration_when_first_pass_is_clean(mock_generate, mock_revise, mock_execute):
    mock_generate.return_value = "```python\ngood_code()\n```"
    mock_execute.side_effect = [_CLEAN_ANALYSIS, _CLEAN_EXECUTION]

    evidence, _, revised = generate_and_verify_code("some task")

    assert evidence["result"]["iterations"] == 1
    assert evidence["result"]["revised"] is False
    assert evidence["result"]["revision_reason"] is None
    assert evidence["result"]["language"] == "python"
    assert revised is False
    mock_revise.assert_not_called()
    mock_execute.assert_any_call("code_analysis", {"code": ["good_code()"], "language": "python"})
    mock_execute.assert_any_call("code_execution", {"code": ["good_code()"], "language": "python"})


@patch("src.agent.code_generation.execute_tool")
@patch("src.agent.code_generation.revise_code")
@patch("src.agent.code_generation.generate_code")
def test_reports_two_iterations_and_the_reason_when_a_revision_fires(mock_generate, mock_revise, mock_execute):
    mock_generate.return_value = "```python\nbroken code\n```"
    mock_revise.return_value = "```python\nfixed_code()\n```"
    mock_execute.side_effect = [_BROKEN_ANALYSIS, _CLEAN_EXECUTION, _CLEAN_ANALYSIS, _CLEAN_EXECUTION]

    evidence, _, revised = generate_and_verify_code("some task")

    assert evidence["result"]["iterations"] == 2
    assert evidence["result"]["revised"] is True
    assert evidence["result"]["revision_reason"] == "syntax error: invalid syntax"
    assert evidence["result"]["code"] == "fixed_code()"
    assert evidence["result"]["language"] == "python"
    assert revised is True


@patch("src.agent.code_generation.execute_tool")
@patch("src.agent.code_generation.generate_code")
def test_detects_c_language_from_the_fence_tag(mock_generate, mock_execute):
    mock_generate.return_value = "```c\nint main() { return 0; }\n```"
    mock_execute.side_effect = [_CLEAN_ANALYSIS, _CLEAN_EXECUTION]

    evidence, _, _ = generate_and_verify_code("some task")

    assert evidence["result"]["language"] == "c"
    mock_execute.assert_any_call(
        "code_analysis", {"code": ["int main() { return 0; }"], "language": "c"}
    )
    mock_execute.assert_any_call(
        "code_execution", {"code": ["int main() { return 0; }"], "language": "c"}
    )
