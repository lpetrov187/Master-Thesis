"""Code generation loop: generate -> verify -> revise (bounded to one
revision), for requests that describe a coding task to implement rather
than code to inspect or run.

Mirrors claim_verifier.py's shape - a few focused, mostly-testable
functions. The interesting design choice is in orchestrator.py, not here:
the result of generate_and_verify_code() is packaged into the same
{"tool", "args", "result", "error"} evidence shape tool_executor.py
already produces for the 3 existing tools, so answer_synthesizer.py needs
no changes at all to consume it (it already falls back to str(result) for
any tool name it doesn't specifically recognize).
"""
import re
import time

import ollama

from src.agent.history import format_history
from src.agent.tool_executor import execute_tool
from src.config import OLLAMA_HOST, PRIMARY_MODEL

_FENCE_RE = re.compile(r"```[\w+-]*\n(.*?)```", re.DOTALL)


def _timed(fn, *args, **kwargs):
    """Call `fn` and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def extract_code(draft: str) -> list[str] | None:
    """Pull the first fenced code block out of `draft` and return it as a
    list of lines - matching code_analysis/code_execution's line-array arg
    schema (the Day 12 fix, avoiding the model mis-escaping embedded
    newlines in a single JSON string). Returns None if no fenced block is
    found, or the block is empty."""
    match = _FENCE_RE.search(draft)
    if match is None:
        return None

    code = match.group(1).rstrip("\n")
    if not code.strip():
        return None

    return code.split("\n")


def needs_revision(analysis_evidence: dict, execution_evidence: dict) -> bool:
    """True if either verification tool found a real problem: a syntax
    error or lint findings from code_analysis, a non-zero exit code or
    timeout from code_execution, or either tool itself failing to run.
    Pure logic, no LLM call needed to decide this."""
    if analysis_evidence.get("error") or execution_evidence.get("error"):
        return True

    analysis_result = analysis_evidence["result"]
    if not analysis_result["syntax_valid"] or analysis_result["findings"]:
        return True

    execution_result = execution_evidence["result"]
    return bool(execution_result["timed_out"] or execution_result["exit_code"] != 0)


def _describe_problem(analysis_evidence: dict, execution_evidence: dict) -> str | None:
    """Human-readable explanation of whatever needs_revision() flagged -
    used so the CLI/evidence can say *why* a revision fired, not just that
    one did. Mirrors needs_revision()'s own checks; returns None if it
    would return False."""
    if analysis_evidence.get("error"):
        return f"code_analysis tool failed: {analysis_evidence['error']}"
    if execution_evidence.get("error"):
        return f"code_execution tool failed: {execution_evidence['error']}"

    problems = []
    analysis_result = analysis_evidence["result"]
    if not analysis_result["syntax_valid"]:
        message = analysis_result["findings"][0]["message"] if analysis_result["findings"] else "invalid syntax"
        problems.append(f"syntax error: {message}")
    elif analysis_result["findings"]:
        problems.append(f"{len(analysis_result['findings'])} lint finding(s)")

    execution_result = execution_evidence["result"]
    if execution_result["timed_out"]:
        problems.append("execution timed out")
    elif execution_result["exit_code"] != 0:
        problems.append(f"execution exited with code {execution_result['exit_code']}")

    return "; ".join(problems) if problems else None


def _build_generation_prompt(query: str, history: list[dict] | None = None) -> str:
    return (
        "You are a Python programmer. Write clean, correct Python code "
        "that solves the user's request below. Return ONLY the code in a "
        "single fenced code block (```python ... ```) - no other tools or "
        "documentation are available, so the code must be self-contained "
        "and runnable as-is, using only the Python standard library. If "
        "the request includes example input, make the code actually run "
        "on that input and print the result.\n\n"
        f"{format_history(history)}"
        f"Request:\n{query}"
    )


def generate_code(query: str, client: ollama.Client | None = None, history: list[dict] | None = None) -> str:
    """Ask the model to write code solving `query`. Returns the raw draft
    text (expected to contain a fenced code block - see extract_code)."""
    client = client or ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=PRIMARY_MODEL,
        messages=[{"role": "user", "content": _build_generation_prompt(query, history)}],
    )
    return response["message"]["content"]


def _build_revision_prompt(
    query: str, code_lines: list[str], analysis_evidence: dict, execution_evidence: dict
) -> str:
    code = "\n".join(code_lines)
    return (
        "You previously wrote this Python code to solve a request, but it "
        "has a problem. Fix it and return ONLY the corrected code in a "
        "single fenced code block (```python ... ```).\n\n"
        f"Original request:\n{query}\n\n"
        f"Your code:\n```python\n{code}\n```\n\n"
        f"Static analysis result: {analysis_evidence['result']}\n\n"
        f"Execution result: {execution_evidence['result']}"
    )


def revise_code(
    query: str,
    code_lines: list[str],
    analysis_evidence: dict,
    execution_evidence: dict,
    client: ollama.Client | None = None,
) -> str:
    """Ask the model to fix `code_lines` given what verification found."""
    client = client or ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=PRIMARY_MODEL,
        messages=[
            {
                "role": "user",
                "content": _build_revision_prompt(query, code_lines, analysis_evidence, execution_evidence),
            }
        ],
    )
    return response["message"]["content"]


def generate_and_verify_code(
    query: str, client: ollama.Client | None = None, history: list[dict] | None = None
) -> tuple[dict, dict, bool]:
    """Generate code for `query`, verify it with code_analysis +
    code_execution, and - if either finds a real problem - make one
    bounded revision attempt before giving up.

    Returns (evidence, timings, revised). `evidence` is shaped like any
    other tool's evidence dict, so it can be handed straight to
    answer_synthesizer.synthesize() with no special-casing needed there.
    """
    client = client or ollama.Client(host=OLLAMA_HOST)
    timings: dict = {}

    draft, timings["generation"] = _timed(generate_code, query, client=client, history=history)
    code_lines = extract_code(draft)
    if code_lines is None:
        evidence = {
            "tool": "generate_and_verify_code",
            "args": {"task": query},
            "result": None,
            "error": "couldn't extract a code block from the generated draft",
        }
        return evidence, timings, False

    analysis_evidence, timings["code_analysis"] = _timed(execute_tool, "code_analysis", {"code": code_lines})
    execution_evidence, timings["code_execution"] = _timed(execute_tool, "code_execution", {"code": code_lines})
    revised = False
    revision_attempted = False
    revision_reason = None

    if needs_revision(analysis_evidence, execution_evidence):
        revision_attempted = True
        revision_reason = _describe_problem(analysis_evidence, execution_evidence)
        revision_draft, timings["revision"] = _timed(
            revise_code, query, code_lines, analysis_evidence, execution_evidence, client=client
        )
        revised_lines = extract_code(revision_draft)
        if revised_lines is not None:
            code_lines = revised_lines
            analysis_evidence, timings["retry_code_analysis"] = _timed(
                execute_tool, "code_analysis", {"code": code_lines}
            )
            execution_evidence, timings["retry_code_execution"] = _timed(
                execute_tool, "code_execution", {"code": code_lines}
            )
            revised = True

    evidence = {
        "tool": "generate_and_verify_code",
        "args": {"task": query},
        "result": {
            "code": "\n".join(code_lines),
            "analysis": analysis_evidence["result"],
            "execution": execution_evidence["result"],
            # "iterations" is 1 or 2: this loop is bounded to exactly one
            # revision pass, never open-ended. "revised" is false when a
            # revision was attempted but extract_code() couldn't pull a
            # code block out of it, so the original (still-broken) code
            # is what's actually being reported here.
            "iterations": 2 if revision_attempted else 1,
            "revised": revised,
            "revision_reason": revision_reason,
        },
        "error": None,
    }
    return evidence, timings, revised
