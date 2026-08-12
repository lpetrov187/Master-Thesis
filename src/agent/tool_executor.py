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
    """Run the named tool with `args`.

    Returns {"tool", "args", "result", "error"}. If the tool itself raises,
    `result` is None and `error` describes what went wrong instead of the
    exception propagating - a failed tool run is still structured evidence
    for the pipeline to react to, not a crash.
    """
    fn = _TOOL_FUNCTIONS[tool_name]
    try:
        result = fn(**args)
        error = None
    except Exception as exc:  # noqa: BLE001 - tools are an external-call boundary
        result = None
        error = f"{type(exc).__name__}: {exc}"

    return {"tool": tool_name, "args": args, "result": result, "error": error}
