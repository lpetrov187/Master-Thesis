"""Tests for the pure-logic pieces of code_generation.py: extract_code()
and needs_revision(). No model calls - both are plain functions."""
from src.agent.code_generation import extract_code, needs_revision


def test_extract_code_pulls_python_fenced_block():
    draft = "Here's the code:\n\n```python\ndef add(a, b):\n    return a + b\n```\n\nDone."

    assert extract_code(draft) == ["def add(a, b):", "    return a + b"]


def test_extract_code_handles_bare_fence_with_no_language_tag():
    draft = "```\nprint('hi')\n```"

    assert extract_code(draft) == ["print('hi')"]


def test_extract_code_returns_none_when_no_fenced_block():
    assert extract_code("just some prose, no code here") is None


def test_extract_code_returns_none_for_empty_fenced_block():
    assert extract_code("```python\n\n```") is None


def test_extract_code_takes_the_first_of_multiple_blocks():
    draft = "```python\nfirst_block()\n```\nsome text\n```python\nsecond_block()\n```"

    assert extract_code(draft) == ["first_block()"]


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
