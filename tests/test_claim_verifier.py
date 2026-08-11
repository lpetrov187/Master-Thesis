"""Day-7 tests: the Claim Verifier extracts atomic claims, matches them
against evidence, scores groundedness, and hedges unsupported claims."""
from src.agent.claim_verifier import (
    extract_claims,
    hedge_unsupported_claims,
    match_claims,
    score_groundedness,
)
from src.tools.doc_rag import ingest, retrieve

# --- pure functions: no model calls, no network ---


def test_score_groundedness_all_supported():
    verdicts = [{"claim": "a", "supported": True}, {"claim": "b", "supported": True}]
    assert score_groundedness(verdicts) == 1.0


def test_score_groundedness_mixed():
    verdicts = [{"claim": "a", "supported": True}, {"claim": "b", "supported": False}]
    assert score_groundedness(verdicts) == 0.5


def test_score_groundedness_no_claims_is_trivially_grounded():
    assert score_groundedness([]) == 1.0


def test_hedge_appends_caveat_for_unsupported_claims():
    verdicts = [
        {"claim": "pool_size sets the pool's connection count.", "supported": True},
        {"claim": "SQLAlchemy retries automatically with backoff.", "supported": False},
    ]
    hedged = hedge_unsupported_claims("Some answer text.", verdicts)

    assert hedged.startswith("Some answer text.")
    assert "SQLAlchemy retries automatically with backoff." in hedged
    assert "pool_size sets the pool's connection count." not in hedged.split("Note:")[1]


def test_hedge_is_a_noop_when_everything_is_supported():
    verdicts = [{"claim": "pool_size sets the pool's connection count.", "supported": True}]
    answer = "Some answer text."

    assert hedge_unsupported_claims(answer, verdicts) == answer


# --- real model calls: verified manually first (see Day-7 changeset notes) ---


def test_extract_claims_splits_answer_into_atomic_claims():
    answer = (
        "To configure connection pooling in SQLAlchemy, create an Engine with "
        "pool_size and max_overflow parameters. pool_size sets the number of "
        "connections kept open, and max_overflow allows extra temporary "
        "connections under load."
    )

    claims = extract_claims(answer)

    assert len(claims) >= 2
    assert any("pool_size" in c for c in claims)


def test_match_claims_discriminates_supported_from_fabricated():
    ingest()
    hits = retrieve("How do I configure connection pooling in SQLAlchemy?", top_k=2)
    evidence = {"tool": "doc_rag", "args": {}, "result": hits}

    claims = [
        "pool_size sets the number of connections kept open in the pool.",
        "SQLAlchemy supports automatic retries with exponential backoff out of the box.",
    ]
    verdicts = match_claims(claims, evidence)
    by_claim = {v["claim"]: v["supported"] for v in verdicts}

    assert by_claim[claims[0]] is True
    assert by_claim[claims[1]] is False
