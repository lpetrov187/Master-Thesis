"""Claim Verifier: extract the atomic factual claims in a draft answer,
check each against the available evidence, and compute a groundedness
score. This is the verification mechanism from the thesis proposal - the
last line of defense against hallucination, checking the actual output of
the Answer Synthesizer rather than just instructing it to behave (Day 6's
Controlled-Access policy).
"""
import json

import ollama

from src.config import OLLAMA_HOST, PRIMARY_MODEL

_CLAIM_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["claims"],
}

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "supported": {"type": "boolean"},
                },
                "required": ["claim", "supported"],
            },
        },
    },
    "required": ["verdicts"],
}


def extract_claims(answer: str, client: ollama.Client | None = None) -> list[str]:
    """Break `answer` down into its individual atomic factual claims."""
    client = client or ollama.Client(host=OLLAMA_HOST)
    prompt = (
        "Break the following answer down into a list of atomic factual "
        "claims - short, single-fact statements. Skip greetings, hedges, "
        "and filler words; only extract things asserted as fact. If the "
        "answer makes no factual claims, return an empty list.\n\n"
        f"Answer:\n{answer}"
    )
    response = client.chat(
        model=PRIMARY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=_CLAIM_EXTRACTION_SCHEMA,
    )
    return json.loads(response["message"]["content"])["claims"]


def _format_evidence_text(evidence: dict) -> str:
    result = evidence["result"]
    if evidence["tool"] == "doc_rag":
        return "\n\n".join(f"[source: {hit['source']}]\n{hit['text']}" for hit in result)
    return str(result)


def match_claims(claims: list[str], evidence: dict, client: ollama.Client | None = None) -> list[dict]:
    """Check each claim against `evidence`. Returns [{"claim", "supported"}, ...]."""
    if not claims:
        return []

    client = client or ollama.Client(host=OLLAMA_HOST)
    claims_block = "\n".join(f"- {c}" for c in claims)
    prompt = (
        "For each claim below, decide whether it is explicitly supported by "
        "the evidence. A claim is supported only if the evidence directly "
        "states it - do not use outside knowledge to decide support, even "
        "if you know the claim to be true in general.\n\n"
        f"Evidence from the '{evidence['tool']}' tool:\n{_format_evidence_text(evidence)}\n\n"
        f"Claims:\n{claims_block}"
    )
    response = client.chat(
        model=PRIMARY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=_VERDICT_SCHEMA,
    )
    return json.loads(response["message"]["content"])["verdicts"]


def score_groundedness(verdicts: list[dict]) -> float:
    """Fraction of claims marked supported. 1.0 when there are no claims to
    check (an answer that asserts nothing can't hallucinate anything)."""
    if not verdicts:
        return 1.0
    return sum(v["supported"] for v in verdicts) / len(verdicts)


def hedge_unsupported_claims(answer: str, verdicts: list[dict]) -> str:
    """Append a caveat listing any claims the evidence didn't support.

    Deterministic and model-free by design: by this point we already know
    exactly which claims failed, so rewriting the answer with another LLM
    call would spend a call to reconstruct information we already have.
    """
    unsupported = [v["claim"] for v in verdicts if not v["supported"]]
    if not unsupported:
        return answer

    caveat_lines = "\n".join(f"- {c}" for c in unsupported)
    return (
        f"{answer}\n\n"
        f"Note: the following claim(s) above are not directly supported by "
        f"the retrieved evidence and should be verified independently:\n"
        f"{caveat_lines}"
    )


def verify(answer: str, evidence: dict, client: ollama.Client | None = None) -> dict:
    """Extract + match + score in one call.

    Returns {"claims": [...], "verdicts": [...], "groundedness_score": float}.
    """
    client = client or ollama.Client(host=OLLAMA_HOST)
    claims = extract_claims(answer, client=client)
    verdicts = match_claims(claims, evidence, client=client)
    return {
        "claims": claims,
        "verdicts": verdicts,
        "groundedness_score": score_groundedness(verdicts),
    }
