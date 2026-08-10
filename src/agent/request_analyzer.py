"""Request Analyzer: decides whether the query needs a tool at all, before
the Tool Selector decides *which* one.
"""
import json

import ollama

from src.config import OLLAMA_HOST, PRIMARY_MODEL

_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_tool": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": ["needs_tool", "reasoning"],
}


def _build_prompt(query: str) -> str:
    return (
        "You are the request-analysis component of an agent system. Decide "
        "whether answering the user's request requires using an external "
        "tool: searching stored documentation, statically analyzing a code "
        "snippet, or running code.\n\n"
        "Examples that NEED a tool:\n"
        '- "How do I configure connection pooling in SQLAlchemy?" '
        "(needs a documentation lookup, not a guess)\n"
        '- "How do I reuse a requests Session?" (needs a documentation lookup)\n'
        '- "Can you review this function for style issues?\\n\\ndef add(a,b): return a+b" '
        "(needs to analyze the given code)\n"
        '- "Run this and tell me what it prints:\\n\\nprint(sum(range(5)))" '
        "(needs to execute the given code)\n\n"
        "Examples that do NOT need a tool:\n"
        '- "What\'s a friendly way to greet someone in an email?" (general knowledge)\n'
        '- "What is a for loop?" (general knowledge, no specific doc/code to check)\n\n'
        "A tool is needed whenever the request either includes an actual "
        "code snippet to review or run, or asks how to use/configure a "
        "specific library or API (an answer that should be grounded in real "
        "documentation rather than guessed from memory). A tool is NOT "
        "needed for greetings, opinions, or generic explanations that don't "
        "reference a specific library, API, or piece of code.\n\n"
        f"User request:\n{query}"
    )


def analyze_request(query: str, client: ollama.Client | None = None) -> dict:
    """Return {"needs_tool": bool, "reasoning": str} for `query`."""
    client = client or ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=PRIMARY_MODEL,
        messages=[{"role": "user", "content": _build_prompt(query)}],
        format=_ANALYSIS_SCHEMA,
    )
    return json.loads(response["message"]["content"])
