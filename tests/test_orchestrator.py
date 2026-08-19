"""Smoke tests for the full pipeline: analyzer -> selector -> executor ->
synthesizer -> verifier (Day 5 wired the first four stages end-to-end; Day 7
added the verifier and its retry/hedge outcome to the trace; Day 8 added the
hardening covered by the mocked tests below)."""
from unittest.mock import patch

from src.agent.orchestrator import run
from src.agent.tool_selector import ToolSelectionError
from src.config import ENABLE_CLAIM_VERIFIER
from src.tools.doc_rag import ingest


def _assert_verification_ran_iff_enabled(trace: dict) -> None:
    """The Claim Verifier is gated by config.ENABLE_CLAIM_VERIFIER (Day 12 -
    it's the pipeline's biggest latency cost, disabled for faster iteration
    without deleting the mechanism). Assert whichever behavior is currently
    configured, so this test stays correct in both states."""
    if ENABLE_CLAIM_VERIFIER:
        assert trace["verification"] is not None
        assert 0.0 <= trace["verification"]["groundedness_score"] <= 1.0
    else:
        assert trace["verification"] is None
        assert trace["retried"] is False


def test_tool_query_runs_end_to_end():
    ingest()  # make sure the doc corpus is populated, independent of test order

    trace = run("How do I configure connection pooling in SQLAlchemy?")

    assert trace["analysis"]["action"] == "lookup_or_inspect"
    assert trace["selection"]["tool"] == "doc_rag"
    assert trace["evidence"]["tool"] == "doc_rag"
    assert trace["evidence"]["result"]
    assert trace["answer"]

    _assert_verification_ran_iff_enabled(trace)
    assert isinstance(trace["retried"], bool)


def test_no_tool_query_skips_tool_stages():
    trace = run("What's a friendly way to greet someone in an email?")

    assert trace["analysis"]["action"] == "none"
    assert trace["selection"] is None
    assert trace["evidence"] is None
    assert trace["answer"]

    assert trace["verification"] is None
    assert trace["retried"] is False


def test_code_analysis_query_dispatches_correct_tool():
    trace = run("Can you review this function for style issues?\n\ndef add(a,b):\n    return a+b\n")

    assert trace["selection"]["tool"] == "code_analysis"
    assert trace["evidence"]["result"]["syntax_valid"] is True
    assert trace["answer"]
    _assert_verification_ran_iff_enabled(trace)


def test_generate_query_runs_the_code_generation_loop():
    trace = run("Write a function that sorts a list of integers, then show it working on [5, 2, 8, 1].")

    assert trace["analysis"]["action"] == "generate"
    assert trace["selection"]["tool"] == "generate_and_verify_code"
    assert trace["evidence"]["tool"] == "generate_and_verify_code"
    assert trace["answer"]
    # Claim Verifier is deliberately skipped for the "generate" branch (its
    # evidence already went through its own verify/revise cycle).
    assert trace["verification"] is None


# --- hardening: control-flow only, no real model calls ---


@patch("src.agent.orchestrator.synthesize")
@patch("src.agent.orchestrator.select_tool")
@patch("src.agent.orchestrator.analyze_request")
def test_falls_back_to_no_tool_answer_when_tool_selection_fails(mock_analyze, mock_select, mock_synth):
    mock_analyze.return_value = {"action": "lookup_or_inspect", "reasoning": "x"}
    mock_select.side_effect = ToolSelectionError("bad selection")
    mock_synth.return_value = "fallback answer"

    trace = run("some query")

    assert trace["selection"] is None
    assert trace["evidence"] is None
    assert trace["error"] is not None
    assert trace["answer"] == "fallback answer"
    assert trace["verification"] is None


@patch("src.agent.orchestrator.synthesize")
@patch("src.agent.orchestrator.generate_and_verify_code")
@patch("src.agent.orchestrator.analyze_request")
def test_generate_action_wires_selection_and_evidence_from_the_generation_loop(
    mock_analyze, mock_generate, mock_synth
):
    mock_analyze.return_value = {"action": "generate", "reasoning": "task to implement"}
    fake_evidence = {"tool": "generate_and_verify_code", "args": {"task": "x"}, "result": {}, "error": None}
    mock_generate.return_value = (fake_evidence, {"generation": 0.1}, True)
    mock_synth.return_value = "the answer"

    trace = run("write a function that sorts a list")

    assert trace["selection"]["tool"] == "generate_and_verify_code"
    assert trace["evidence"] == fake_evidence
    assert trace["retried"] is True
    assert trace["answer"] == "the answer"
    assert trace["verification"] is None
    assert trace["timings"]["generation"] == 0.1


@patch("src.agent.orchestrator.analyze_request")
def test_returns_a_clean_error_when_ollama_is_unreachable(mock_analyze):
    mock_analyze.side_effect = ConnectionError("no ollama")

    trace = run("some query")

    assert trace["error"] is not None
    assert "ollama" in trace["error"].lower()
    assert trace["answer"] is None
