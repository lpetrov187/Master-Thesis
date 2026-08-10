"""Tool Executor: dispatches a Tool Selector's choice to the actual tool
function and packages the result as structured evidence.
"""
from src.tools import code_analysis, code_execution, doc_rag

_TOOL_FUNCTIONS = {
    "doc_rag": doc_rag.retrieve,
    "code_analysis": code_analysis.analyze,
    "code_execution": code_execution.run,
}


def execute_tool(tool_name: str, args: dict) -> dict:
    """Run the named tool with `args`. Returns {"tool", "args", "result"}."""
    fn = _TOOL_FUNCTIONS[tool_name]
    result = fn(**args)
    return {"tool": tool_name, "args": args, "result": result}
