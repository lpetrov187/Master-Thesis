"""Orchestration loop: Request Analyzer -> Tool Selector -> Tool Executor ->
Answer Synthesizer -> Claim Verifier.

Hardening (Day 8): a bad tool selection or a failed tool run no longer
crashes the whole pipeline - each is caught and turned into a degraded but
well-formed trace instead, since an eval run over dozens of tasks (Day 10)
shouldn't die on one bad query. Every run is also appended to the trace
log, whether it succeeded or not.

Timing: every stage is wall-clock timed and recorded in trace["timings"],
to find where latency actually goes (LLM calls vs. local tool execution).

The Claim Verifier stage is gated by config.ENABLE_CLAIM_VERIFIER (see
config.py for why) - when off, trace["verification"] stays None and
trace["retried"] stays False, same as a no-evidence run.
"""
import time

import ollama

from src.agent.answer_synthesizer import synthesize
from src.agent.claim_verifier import hedge_unsupported_claims, verify
from src.agent.request_analyzer import analyze_request
from src.agent.tool_executor import execute_tool
from src.agent.tool_selector import ToolSelectionError, select_tool
from src.agent.trace_log import log_trace
from src.config import ENABLE_CLAIM_VERIFIER, OLLAMA_HOST

_GROUNDEDNESS_RETRY_THRESHOLD = 1.0


def _timed(fn, *args, **kwargs):
    """Call `fn` and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def _select_and_execute(
    query: str, client: ollama.Client, history: list[dict] | None = None
) -> tuple[dict | None, dict | None, str | None, dict]:
    """Select a tool and run it. Returns (selection, evidence, error, timings).

    `error` is set (selection and evidence stay None) if the model's tool
    choice fails schema validation; `timings` always has a "selection" key,
    and an "execution" key too once a tool actually ran.
    """
    start = time.perf_counter()
    try:
        selection = select_tool(query, client=client, history=history)
    except ToolSelectionError as exc:
        error = f"tool selection failed, falling back to a no-tool answer: {exc}"
        return None, None, error, {"selection": time.perf_counter() - start}
    selection_time = time.perf_counter() - start

    evidence, execution_time = _timed(execute_tool, selection["tool"], selection["args"])
    return selection, evidence, None, {"selection": selection_time, "execution": execution_time}


def run(query: str, client: ollama.Client | None = None, history: list[dict] | None = None) -> dict:
    """Run the full pipeline for `query`.

    `history` (optional) is a list of prior {"query", "answer"} turns from
    the same conversation - see src/agent/session.py, which is what
    actually accumulates it across calls; this function itself stays a
    pure, stateless call each time. Threaded into the Request Analyzer,
    Tool Selector, and Answer Synthesizer; the Claim Verifier does not
    receive it, since it only checks the current answer against current
    evidence and has no conversational judgment to make.

    Returns a trace dict: query, analysis, selection, evidence, answer,
    verification, retried, error, timings. `error` is None on a clean run;
    it's set (with a best-effort partial trace) if the model was
    unreachable or the Tool Selector's output failed validation.
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
        "timings": {},
    }
    timings = trace["timings"]
    run_start = time.perf_counter()

    try:
        trace["analysis"], timings["analysis"] = _timed(analyze_request, query, client=client, history=history)

        if trace["analysis"]["needs_tool"]:
            selection, evidence, selection_error, select_timings = _select_and_execute(query, client, history)
            trace["selection"], trace["evidence"], trace["error"] = selection, evidence, selection_error
            timings.update(select_timings)

        trace["answer"], timings["synthesis"] = _timed(
            synthesize, query, trace["evidence"], client=client, history=history
        )

        has_usable_evidence = trace["evidence"] is not None and not trace["evidence"].get("error")
        if has_usable_evidence and ENABLE_CLAIM_VERIFIER:
            trace["verification"], timings["verification"] = _timed(
                verify, trace["answer"], trace["evidence"], client=client
            )

            if trace["verification"]["groundedness_score"] < _GROUNDEDNESS_RETRY_THRESHOLD:
                selection, evidence, selection_error, retry_timings = _select_and_execute(query, client, history)
                timings["retry_selection"] = retry_timings.get("selection")
                timings["retry_execution"] = retry_timings.get("execution")

                if selection_error is None:
                    trace["selection"], trace["evidence"] = selection, evidence
                    trace["answer"], timings["retry_synthesis"] = _timed(
                        synthesize, query, evidence, client=client, history=history
                    )
                    trace["verification"], timings["retry_verification"] = _timed(
                        verify, trace["answer"], evidence, client=client
                    )
                    trace["retried"] = True

                if trace["verification"]["groundedness_score"] < _GROUNDEDNESS_RETRY_THRESHOLD:
                    trace["answer"] = hedge_unsupported_claims(trace["answer"], trace["verification"]["verdicts"])
    except ConnectionError as exc:
        trace["error"] = f"could not reach Ollama: {exc}"

    timings["total"] = time.perf_counter() - run_start
    log_trace(trace, condition="agent")
    return trace
