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
