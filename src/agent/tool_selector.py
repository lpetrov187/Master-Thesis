"""Schema-constrained Tool Selector.

Given a user query, asks the LLM to pick one tool from TOOL_REGISTRY and
produce arguments for it, using Ollama's JSON-schema-constrained decoding so
the output is always valid JSON shaped like
{"tool": ..., "args": ..., "reasoning": ...}. The "tool" field is
schema-constrained to the registry's names; "args" is validated in Python
against the selected tool's own parameter schema, since polymorphic
per-tool arg shapes aren't expressible in a single constrained-decoding
schema on a small local model.
"""
import json

import ollama
from jsonschema import ValidationError, validate

from src.agent.history import format_history
from src.agent.tool_registry import TOOL_REGISTRY
from src.config import OLLAMA_HOST, PRIMARY_MODEL

_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "tool": {"type": "string", "enum": list(TOOL_REGISTRY)},
        "args": {"type": "object"},
    },
    "required": ["reasoning", "tool", "args"],
}


class ToolSelectionError(Exception):
    """Raised when the LLM's tool/args choice fails validation."""


def _build_prompt(query: str, history: list[dict] | None = None) -> str:
    tool_lines = "\n".join(
        f"- {spec.name}: {spec.description} Args schema: {json.dumps(spec.parameters)}"
        for spec in TOOL_REGISTRY.values()
    )
    return (
        "You are the tool-selection component of an agent system. Given the "
        "user request below, choose exactly one tool from the list and "
        "produce arguments that satisfy its JSON schema.\n\n"
        f"Available tools:\n{tool_lines}\n\n"
        f"{format_history(history)}"
        f"Current user request:\n{query}"
    )


def select_tool(query: str, client: ollama.Client | None = None, history: list[dict] | None = None) -> dict:
    """Select a tool + args for `query`. Returns {"tool", "args", "reasoning"}.

    `history` (optional) is prior {"query", "answer"} turns from the same
    conversation - e.g. so "run it with different input" can pull the code
    it refers to from an earlier turn rather than the current message.

    Raises ToolSelectionError if the model picks an unknown tool or produces
    args that don't validate against that tool's parameter schema.
    """
    client = client or ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=PRIMARY_MODEL,
        messages=[{"role": "user", "content": _build_prompt(query, history)}],
        format=_SELECTION_SCHEMA,
        options={"temperature": 0.1},
    )
    result = json.loads(response["message"]["content"])

    tool_name = result["tool"]
    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None:
        raise ToolSelectionError(f"Unknown tool selected: {tool_name!r}")

    try:
        validate(instance=result["args"], schema=spec.parameters)
    except ValidationError as exc:
        raise ToolSelectionError(
            f"Args for tool {tool_name!r} failed schema validation: {exc.message}"
        ) from exc

    return result
