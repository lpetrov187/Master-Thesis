"""Shared conversation-history formatting, used by every pipeline stage
that needs to be aware of prior turns (Request Analyzer, Tool Selector,
Answer Synthesizer). Kept dependency-free (no import of orchestrator.py or
session.py) to avoid a circular import: session.py -> orchestrator.py ->
those stage modules -> (would need) session.py.
"""


def format_history(history: list[dict] | None) -> str:
    """Render recent (query, answer) turns as a readable transcript block.

    Returns "" if there's no history yet - callers should place this
    directly before their "current request" section and rely on it being
    empty for a fresh conversation, so prompts stay byte-identical to a
    call with no history at all.
    """
    if not history:
        return ""

    turns = "\n\n".join(f"User: {turn['query']}\nAssistant: {turn['answer']}" for turn in history)
    return f"Conversation so far:\n{turns}\n\n"
