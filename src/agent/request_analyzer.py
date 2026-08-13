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
        '- "What is the recommended way to get a logger in a Python module?" '
        "(asks about a specific library's recommended usage - needs a documentation lookup)\n"
        '- "Can you review this function for style issues?\\n\\ndef add(a,b): return a+b" '
        "(contains a code snippet to analyze)\n"
        '- "Is this code clean?\\n\\ndef greet():\\n    return f\'hi\'" '
        "(contains a code snippet to analyze, even though the question is short)\n"
        '- "Run this and tell me what it prints:\\n\\nprint(sum(range(5)))" '
        "(contains a code snippet to execute)\n"
        '- "What does this code output?\\n\\nfor i in range(3):\\n    print(i)" '
        "(contains a code snippet to execute)\n\n"
        "Examples that do NOT need a tool:\n"
        '- "What\'s a friendly way to greet someone in an email?" (general knowledge)\n'
        '- "What is a for loop?" (general knowledge, no specific doc/code to check)\n\n'
        "Rule of thumb: if the request contains an actual code snippet (even "
        "a one-line one, even without a code block), it needs a tool - no "
        "exceptions. If it names a specific library/API and asks how to use "
        "or configure it, it needs a tool. A tool is NOT needed only for "
        "greetings, opinions, or generic explanations with no specific "
        "library, API, or code attached.\n\n"
        f"User request:\n{query}"
    )


def analyze_request(query: str, client: ollama.Client | None = None) -> dict:
    """Return {"needs_tool": bool, "reasoning": str} for `query`."""
    client = client or ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=PRIMARY_MODEL,
        messages=[{"role": "user", "content": _build_prompt(query)}],
        format=_ANALYSIS_SCHEMA,
        options={"temperature": 0.1},
    )
    return json.loads(response["message"]["content"])
