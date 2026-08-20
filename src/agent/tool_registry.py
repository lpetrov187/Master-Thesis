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
            "Statically analyze a Python or C code snippet to find syntax "
            "errors, style violations, and structural issues (ast + ruff for "
            "Python, gcc -fsyntax-only for C). Use when the user asks to "
            "review, lint, or explain the structure of code without running it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "The source code to analyze, split into one array "
                        "element per line (no embedded newlines) - avoids the "
                        "model mis-escaping literal newlines inside a single "
                        "JSON string."
                    ),
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "c"],
                    "description": "The language `code` is written in. Defaults to python if omitted.",
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    ),
    "code_execution": ToolSpec(
        name="code_execution",
        description=(
            "Run a Python or C code snippet in a sandboxed subprocess (C is "
            "compiled with gcc first) and capture stdout, stderr, and "
            "exceptions. Use when the user asks to run code, check its "
            "output, or verify a solution to a programming problem."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "The source code to execute, split into one array "
                        "element per line (no embedded newlines) - avoids the "
                        "model mis-escaping literal newlines inside a single "
                        "JSON string."
                    ),
                },
                "stdin": {
                    "type": "string",
                    "description": "Optional input to feed the program via stdin.",
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "c"],
                    "description": "The language `code` is written in. Defaults to python if omitted.",
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    ),
    "web_fetch": ToolSpec(
        name="web_fetch",
        description=(
            "Fetch a specific URL and return its readable text content. Use "
            "only when the request names or pastes an explicit URL to read, "
            "fetch, or summarize - not for general web search or topics "
            "without a given link."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The exact URL to fetch.",
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    ),
}
