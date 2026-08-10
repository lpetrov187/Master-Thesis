"""Tool registry: name, description, and JSON-schema args for each tool.

Consumed by the Tool Selector to build schema-constrained selection prompts
and by the Tool Executor to validate LLM-produced arguments before dispatch.
"""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the tool's args


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "doc_rag": ToolSpec(
        name="doc_rag",
        description=(
            "Search the project documentation corpus and return relevant "
            "passages. Use for questions about API usage, library behavior, "
            "or any 'how do I / what does X do' documentation lookup."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to run against the documentation corpus.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    "code_analysis": ToolSpec(
        name="code_analysis",
        description=(
            "Statically analyze a Python code snippet with ast + ruff to find "
            "syntax errors, style violations, and structural issues. Use when "
            "the user asks to review, lint, or explain the structure of code "
            "without running it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python source code to analyze.",
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    ),
    "code_execution": ToolSpec(
        name="code_execution",
        description=(
            "Run a Python code snippet in a sandboxed subprocess and capture "
            "stdout, stderr, and exceptions. Use when the user asks to run "
            "code, check its output, or verify a solution to a programming "
            "problem."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python source code to execute.",
                },
                "stdin": {
                    "type": "string",
                    "description": "Optional input to feed the program via stdin.",
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    ),
}
