"""Day-4 tests: code_analysis flags syntax errors and lint issues without
executing the code under analysis."""
from src.tools.code_analysis import analyze


def test_flags_syntax_error():
    result = analyze("def broken(:\n    pass\n")

    assert result["syntax_valid"] is False
    assert result["findings"][0]["rule"] == "syntax-error"


def test_flags_lint_issue_in_valid_code():
    result = analyze("import os\n\ndef add(a, b):\n    return a + b\n")

    assert result["syntax_valid"] is True
    assert any(f["rule"] == "F401" for f in result["findings"])


def test_clean_code_has_no_findings():
    result = analyze("def add(a, b):\n    return a + b\n")

    assert result["syntax_valid"] is True
    assert result["findings"] == []
