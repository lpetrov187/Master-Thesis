"""Answer Synthesizer: drafts a response to the user's query, given whatever
evidence the Tool Executor produced (or no evidence, if no tool was needed).

No grounding/hedging policy yet - the Controlled-Access layer that restricts
claims to sourced evidence is added on top of this in Day 6.
"""
import ollama

from src.config import OLLAMA_HOST, PRIMARY_MODEL


def _build_prompt(query: str, evidence: dict | None) -> str:
    if evidence is None:
        return f"Answer the user's question directly and concisely.\n\nQuestion: {query}"

    return (
        f"Answer the user's question using the evidence below from the "
        f"'{evidence['tool']}' tool.\n\n"
        f"Question: {query}\n\n"
        f"Evidence:\n{evidence['result']}"
    )


def synthesize(query: str, evidence: dict | None = None, client: ollama.Client | None = None) -> str:
    """Draft a plain-text answer to `query`, informed by `evidence` if given."""
    client = client or ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=PRIMARY_MODEL,
        messages=[{"role": "user", "content": _build_prompt(query, evidence)}],
    )
    return response["message"]["content"]
