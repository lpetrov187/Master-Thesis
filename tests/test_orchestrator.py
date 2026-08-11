"""Smoke tests for the full pipeline: analyzer -> selector -> executor ->
synthesizer -> verifier (Day 5 wired the first four stages end-to-end; Day 7
added the verifier and its retry/hedge outcome to the trace)."""
from src.agent.orchestrator import run
from src.tools.doc_rag import ingest


def test_tool_query_runs_end_to_end():
    ingest()  # make sure the doc corpus is populated, independent of test order

    trace = run("How do I configure connection pooling in SQLAlchemy?")

    assert trace["analysis"]["needs_tool"] is True
    assert trace["selection"]["tool"] == "doc_rag"
    assert trace["evidence"]["tool"] == "doc_rag"
    assert trace["evidence"]["result"]
    assert trace["answer"]

    assert trace["verification"] is not None
    assert 0.0 <= trace["verification"]["groundedness_score"] <= 1.0
    assert isinstance(trace["retried"], bool)


def test_no_tool_query_skips_tool_stages():
    trace = run("What's a friendly way to greet someone in an email?")

    assert trace["analysis"]["needs_tool"] is False
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
    assert trace["verification"] is not None
