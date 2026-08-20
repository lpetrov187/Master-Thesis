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


# --- C support (gcc -fsyntax-only), added alongside the code-generation loop ---


def test_c_flags_syntax_error():
    result = analyze("int main() { return 0 }", language="c")

    assert result["syntax_valid"] is False
    assert result["findings"]
    assert result["findings"][0]["rule"] == "error"


def test_c_flags_warning_in_valid_code():
    result = analyze("int main() { int unused; return 0; }", language="c")

    assert result["syntax_valid"] is True
    assert any(f["rule"] == "-Wunused-variable" for f in result["findings"])


def test_c_clean_code_has_no_findings():
    result = analyze("int main() { return 0; }", language="c")

    assert result["syntax_valid"] is True
    assert result["findings"] == []
