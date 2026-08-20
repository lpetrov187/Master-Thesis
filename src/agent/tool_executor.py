"""Tool Executor: dispatches a Tool Selector's choice to the actual tool
function and packages the result as structured evidence.
"""
from src.tools import code_analysis, code_execution, doc_rag, web_fetch

_TOOL_FUNCTIONS = {
    "doc_rag": doc_rag.retrieve,
    "code_analysis": code_analysis.analyze,
    "code_execution": code_execution.run,
    "web_fetch": web_fetch.fetch,
}


def _denormalize_args(args: dict) -> dict:
    """Undo the tool_registry "code" array-of-lines encoding (see Day 12
    fix) back into the plain string the tool functions themselves expect."""
    if isinstance(args.get("code"), list):
        return {**args, "code": "\n".join(args["code"])}
    return args


def execute_tool(tool_name: str, args: dict) -> dict:
    """Run the named tool with `args`.

    Returns {"tool", "args", "result", "error"}. If the tool itself raises,
    `result` is None and `error` describes what went wrong instead of the
    exception propagating - a failed tool run is still structured evidence
    for the pipeline to react to, not a crash.
    """
    fn = _TOOL_FUNCTIONS[tool_name]
    args = _denormalize_args(args)
    try:
        result = fn(**args)
        error = None
    except Exception as exc:  # noqa: BLE001 - tools are an external-call boundary
        result = None
        error = f"{type(exc).__name__}: {exc}"

    return {"tool": tool_name, "args": args, "result": result, "error": error}
