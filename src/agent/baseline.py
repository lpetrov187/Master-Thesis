"""No-tool baseline: the same model, prompted directly with no tools, no
Controlled-Access policy, and no Claim Verifier. This is the control
condition the evaluation chapter measures the full agent system against.
"""
import ollama

from src.agent.trace_log import log_trace
from src.config import OLLAMA_HOST, PRIMARY_MODEL


def run(query: str, client: ollama.Client | None = None) -> dict:
    """Answer `query` directly, with no tools and no grounding mechanisms.

    Returns {"query", "answer", "error"} - `answer` is None if `error` is
    set, mirroring the agent pipeline's trace shape for easy comparison.
    """
    client = client or ollama.Client(host=OLLAMA_HOST)
    trace = {"query": query, "answer": None, "error": None}

    try:
        response = client.chat(model=PRIMARY_MODEL, messages=[{"role": "user", "content": query}])
        trace["answer"] = response["message"]["content"]
    except ConnectionError as exc:
        trace["error"] = f"could not reach Ollama: {exc}"

    log_trace(trace, condition="baseline")
    return trace
