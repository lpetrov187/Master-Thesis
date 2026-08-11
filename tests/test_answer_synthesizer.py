"""Day-6 tests: the Controlled-Access policy makes the Answer Synthesizer
hedge on claims the evidence doesn't support, without over-hedging on
claims it does support."""
from src.agent.answer_synthesizer import synthesize
from src.tools.doc_rag import ingest, retrieve

_HEDGE_PHRASES = [
    "doesn't cover", "does not cover", "don't know", "do not know",
    "not covered", "no information", "not mentioned", "not specified",
    "not provided", "cannot find", "can't find", "not available",
    "does not address", "does not contain", "unsupported",
]


def test_hedges_when_evidence_does_not_support_the_answer():
    ingest()
    query = "How do I set a custom retry policy with exponential backoff in the requests library?"
    hits = retrieve(query, top_k=2)

    answer = synthesize(query, {"tool": "doc_rag", "args": {}, "result": hits})

    assert any(phrase in answer.lower() for phrase in _HEDGE_PHRASES)
    assert "tenacity" not in answer.lower()  # outside-knowledge fabrication seen pre-Day-6


def test_still_answers_when_evidence_supports_the_answer():
    ingest()
    query = "How do I configure connection pooling in SQLAlchemy?"
    hits = retrieve(query, top_k=2)

    answer = synthesize(query, {"tool": "doc_rag", "args": {}, "result": hits})

    assert "pool_size" in answer


def test_answers_normally_with_no_evidence():
    answer = synthesize("What's a friendly way to greet someone in an email?")

    assert answer
