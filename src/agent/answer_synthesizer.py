"""Answer Synthesizer: drafts a response to the user's query, given whatever
evidence the Tool Executor produced (or no evidence, if no tool was needed).

When evidence is present, the Controlled-Access policy below restricts the
synthesizer to facts the evidence actually supports - anything else must be
hedged rather than filled in from the model's own training knowledge.
"""
import ollama

from src.config import OLLAMA_HOST, PRIMARY_MODEL

_CONTROLLED_ACCESS_POLICY = (
    "Base your answer only on the evidence below - never add facts from "
    "outside knowledge, even if you believe them to be true. If the "
    "evidence fully answers the question, answer normally with no "
    "disclaimer. If part or all of the question isn't addressed by the "
    "evidence, say plainly which specific part is unsupported instead of "
    "guessing - don't add a blanket disclaimer to an answer that IS "
    "supported by the evidence."
)


def _format_evidence(evidence: dict) -> str:
    """Render evidence as readable text instead of a raw Python repr.

    doc_rag's result is a list of {"text", "source", "distance"} hits, whose
    default repr mashes multiple docs' text into one hard-to-parse blob -
    format those explicitly, one labeled passage per hit. The other tools'
    results are already small, flat dicts, so their default repr is fine.
    """
    result = evidence["result"]
    if evidence["tool"] == "doc_rag":
        return "\n\n".join(f"[source: {hit['source']}]\n{hit['text']}" for hit in result)
    return str(result)


def _build_prompt(query: str, evidence: dict | None) -> str:
    if evidence is None:
        return f"Answer the user's question directly and concisely.\n\nQuestion: {query}"

    return (
        f"{_CONTROLLED_ACCESS_POLICY}\n\n"
        f"Question: {query}\n\n"
        f"Evidence from the '{evidence['tool']}' tool:\n{_format_evidence(evidence)}"
    )


def synthesize(query: str, evidence: dict | None = None, client: ollama.Client | None = None) -> str:
    """Draft a plain-text answer to `query`, informed by `evidence` if given.

    If the tool that produced `evidence` failed, report that directly
    instead of calling the model on evidence that isn't actually there -
    we already know the fact ("the tool failed"), so there's nothing an
    LLM call would add.
    """
    if evidence is not None and evidence.get("error"):
        return f"I couldn't answer this because the '{evidence['tool']}' tool failed: {evidence['error']}"

    client = client or ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=PRIMARY_MODEL,
        messages=[{"role": "user", "content": _build_prompt(query, evidence)}],
    )
    return response["message"]["content"]
