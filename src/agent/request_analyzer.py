"""Request Analyzer: decides what kind of action the query needs, before
the rest of the pipeline acts on it - a documentation/code lookup, code to
be generated from scratch, or no tool at all.
"""
import json

import ollama

from src.agent.history import format_history
from src.config import OLLAMA_HOST, PRIMARY_MODEL

_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["none", "lookup_or_inspect", "generate"]},
        "reasoning": {"type": "string"},
    },
    "required": ["action", "reasoning"],
}


def _build_prompt(query: str, history: list[dict] | None = None) -> str:
    return (
        "You are the request-analysis component of an agent system. "
        "Classify the user's request into exactly one action:\n\n"
        '- "lookup_or_inspect": the request needs a documentation lookup '
        "(names a specific library/API and asks how to use or configure "
        "it), OR the request already contains a code snippet to statically "
        "analyze or run.\n"
        '- "generate": the request describes a coding task to *implement* '
        "from scratch - no existing code is given, and no specific "
        "library/API is being looked up. The agent should write new code "
        "to solve it.\n"
        '- "none": general knowledge, greetings, opinions, or explanations '
        "with no specific library, API, or code attached.\n\n"
        'Examples of "lookup_or_inspect":\n'
        '- "How do I configure connection pooling in SQLAlchemy?" '
        "(documentation lookup)\n"
        '- "How do I reuse a requests Session?" (documentation lookup)\n'
        '- "What is the recommended way to get a logger in a Python module?" '
        "(asks about a specific library's recommended usage - documentation lookup)\n"
        '- "Can you review this function for style issues?\\n\\ndef add(a,b): return a+b" '
        "(code snippet given, to analyze)\n"
        '- "Is this code clean?\\n\\ndef greet():\\n    return f\'hi\'" '
        "(code snippet given, even though the question is short)\n"
        '- "Run this and tell me what it prints:\\n\\nprint(sum(range(5)))" '
        "(code snippet given, to execute)\n"
        '- "What does this code output?\\n\\nfor i in range(3):\\n    print(i)" '
        "(code snippet given, to execute)\n"
        '- "can you tell me the how does one configure timeouts in httpx" '
        "(redundant, ungrammatical phrasing - but still names a specific "
        "library and asks how to configure something, so it's still a "
        "documentation lookup)\n\n"
        'Examples of "generate":\n'
        '- "Write a function that sorts a list of integers." '
        "(task to implement, no existing code, no library named)\n"
        '- "Sort this array: [5, 2, 8, 1]" '
        "(task to implement - describes desired behavior, not existing code to inspect)\n"
        '- "Can you write me a script that counts how many vowels are in a string?" '
        "(task to implement)\n\n"
        'Examples of "none":\n'
        '- "What\'s a friendly way to greet someone in an email?" (general knowledge)\n'
        '- "What is a for loop?" (general knowledge, no specific doc/code to check)\n\n'
        "Rule of thumb: if the request contains an actual code snippet "
        "(even a one-line one, even without a code block), it's "
        '"lookup_or_inspect" - no exceptions. If it names a specific '
        "library/API and asks how to use or configure it, it's also "
        '"lookup_or_inspect". If it instead describes a task to build with '
        'no code and no library named, it\'s "generate". Judge the '
        "request's underlying intent only - ignore grammar, typos, "
        "redundant words, or awkward phrasing; a badly-worded request "
        "still gets classified the same as a cleanly-worded one.\n\n"
        f"{format_history(history)}"
        f"Current user request:\n{query}"
    )


def analyze_request(query: str, client: ollama.Client | None = None, history: list[dict] | None = None) -> dict:
    """Return {"action": "none" | "lookup_or_inspect" | "generate", "reasoning": str} for `query`.

    `history` (optional) is a list of prior {"query", "answer"} turns from
    the same conversation, so a follow-up like "and what about httpx
    instead?" still classifies correctly even though the current message
    alone doesn't name a tool-worthy topic.
    """
    client = client or ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=PRIMARY_MODEL,
        messages=[{"role": "user", "content": _build_prompt(query, history)}],
        format=_ANALYSIS_SCHEMA,
        options={"temperature": 0.1},
    )
    return json.loads(response["message"]["content"])
