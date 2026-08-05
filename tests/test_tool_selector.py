"""Day-2 tests: Tool Selector picks the right tool and produces args that
validate against each of the 3 registered tool schemas."""
import pytest
from src.agent.tool_registry import TOOL_REGISTRY
from src.agent.tool_selector import select_tool

QUERIES = {
    "doc_rag": "How do I configure connection pooling in SQLAlchemy?",
    "code_analysis": (
        "Can you review this function for style issues?\n\n"
        "def add(a,b):\n    return a+b\n"
    ),
    "code_execution": "Run this and tell me what it prints:\n\nprint(sum(range(5)))\n",
}


@pytest.mark.parametrize("expected_tool,query", QUERIES.items())
def test_selects_expected_tool_with_valid_args(expected_tool, query):
    result = select_tool(query)
    assert result["tool"] == expected_tool

    spec = TOOL_REGISTRY[result["tool"]]
    for required_key in spec.parameters["required"]:
        assert required_key in result["args"]
