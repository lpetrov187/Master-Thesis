"""Orchestration loop: Request Analyzer -> Tool Selector -> Tool Executor ->
Answer Synthesizer -> Claim Verifier.

Hardening (Day 8): a bad tool selection or a failed tool run no longer
crashes the whole pipeline - each is caught and turned into a degraded but
well-formed trace instead, since an eval run over dozens of tasks (Day 10)
shouldn't die on one bad query. Every run is also appended to the trace
log, whether it succeeded or not.
"""
import ollama

from src.agent.answer_synthesizer import synthesize
from src.agent.claim_verifier import hedge_unsupported_claims, verify
from src.agent.request_analyzer import analyze_request
from src.agent.tool_executor import execute_tool
from src.agent.tool_selector import ToolSelectionError, select_tool
from src.agent.trace_log import log_trace
from src.config import OLLAMA_HOST

_GROUNDEDNESS_RETRY_THRESHOLD = 1.0


def _select_and_execute(query: str, client: ollama.Client) -> tuple[dict | None, dict | None, str | None]:
    """Select a tool and run it. Returns (selection, evidence, error) - all
    three are None/None/None only when this succeeds without a selection
    error; `error` is set (selection and evidence stay None) if the model's
    tool choice fails schema validation."""
    try:
        selection = select_tool(query, client=client)
    except ToolSelectionError as exc:
        return None, None, f"tool selection failed, falling back to a no-tool answer: {exc}"

    evidence = execute_tool(selection["tool"], selection["args"])
    return selection, evidence, None


def run(query: str, client: ollama.Client | None = None) -> dict:
    """Run the full pipeline for `query`.

    Returns a trace dict: query, analysis, selection, evidence, answer,
    verification, retried, error. `error` is None on a clean run; it's set
    (with a best-effort partial trace) if the model was unreachable or the
    Tool Selector's output failed validation.
    """
    client = client or ollama.Client(host=OLLAMA_HOST)
    trace = {
        "query": query,
        "analysis": None,
        "selection": None,
        "evidence": None,
        "answer": None,
        "verification": None,
        "retried": False,
        "error": None,
    }

    try:
        trace["analysis"] = analyze_request(query, client=client)

        if trace["analysis"]["needs_tool"]:
            selection, evidence, selection_error = _select_and_execute(query, client)
            trace["selection"], trace["evidence"], trace["error"] = selection, evidence, selection_error

        trace["answer"] = synthesize(query, trace["evidence"], client=client)

        has_usable_evidence = trace["evidence"] is not None and not trace["evidence"].get("error")
        if has_usable_evidence:
            trace["verification"] = verify(trace["answer"], trace["evidence"], client=client)

            if trace["verification"]["groundedness_score"] < _GROUNDEDNESS_RETRY_THRESHOLD:
                selection, evidence, selection_error = _select_and_execute(query, client)
                if selection_error is None:
                    trace["selection"], trace["evidence"] = selection, evidence
                    trace["answer"] = synthesize(query, evidence, client=client)
                    trace["verification"] = verify(trace["answer"], evidence, client=client)
                    trace["retried"] = True

                if trace["verification"]["groundedness_score"] < _GROUNDEDNESS_RETRY_THRESHOLD:
                    trace["answer"] = hedge_unsupported_claims(trace["answer"], trace["verification"]["verdicts"])
    except ConnectionError as exc:
        trace["error"] = f"could not reach Ollama: {exc}"

    log_trace(trace, condition="agent")
    return trace
