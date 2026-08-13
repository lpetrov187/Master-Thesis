"""Day-8 test: Tool Executor turns a tool's own exception into structured
evidence instead of letting it crash the pipeline."""
from src.agent import tool_executor


def test_execute_tool_catches_exceptions_from_the_tool():
    def broken_tool(**kwargs):
        raise ValueError("boom")

    original = tool_executor._TOOL_FUNCTIONS["code_analysis"]
    tool_executor._TOOL_FUNCTIONS["code_analysis"] = broken_tool
    try:
        evidence = tool_executor.execute_tool("code_analysis", {"code": "x = 1"})
    finally:
        tool_executor._TOOL_FUNCTIONS["code_analysis"] = original

    assert evidence["result"] is None
    assert evidence["error"] == "ValueError: boom"


def test_execute_tool_succeeds_normally():
    evidence = tool_executor.execute_tool("code_analysis", {"code": "x = 1\n"})

    assert evidence["error"] is None
    assert evidence["result"]["syntax_valid"] is True


def test_execute_tool_joins_line_array_into_source(monkeypatch):
    """Day-12 fix: tool_registry's code schema is now an array of lines
    (see PLAN.md's newline-escaping bug); execute_tool must join it back
    into a real multi-line string before the tool function sees it."""
    received = {}

    def spy_tool(**kwargs):
        received.update(kwargs)
        return {"ok": True}

    monkeypatch.setitem(tool_executor._TOOL_FUNCTIONS, "code_analysis", spy_tool)

    evidence = tool_executor.execute_tool(
        "code_analysis", {"code": ["def add(a, b):", "    return a + b"]}
    )

    assert evidence["error"] is None
    assert received["code"] == "def add(a, b):\n    return a + b"


def test_execute_tool_leaves_plain_string_code_untouched():
    evidence = tool_executor.execute_tool("code_analysis", {"code": "def f():\n    pass\n"})

    assert evidence["error"] is None
