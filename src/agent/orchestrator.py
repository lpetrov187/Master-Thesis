"""Orchestration loop: Request Analyzer -> Tool Selector -> Tool Executor ->
Answer Synthesizer -> Claim Verifier.

Verification only runs when there's evidence to check claims against (the
no-tool path has nothing to verify against, same reasoning as Day 6's
Controlled-Access policy). An unsupported claim triggers one retry of
Tool Selector -> Tool Executor -> Answer Synthesizer; if the retried
answer still has unsupported claims, they get hedged instead of retried
again, per the architecture's "hedge or one retry" rule.
"""
import ollama

from src.agent.answer_synthesizer import synthesize
from src.agent.claim_verifier import hedge_unsupported_claims, verify
from src.agent.request_analyzer import analyze_request
from src.agent.tool_executor import execute_tool
from src.agent.tool_selector import select_tool
from src.config import OLLAMA_HOST

_GROUNDEDNESS_RETRY_THRESHOLD = 1.0


def run(query: str, client: ollama.Client | None = None) -> dict:
    """Run the full pipeline for `query`.

    Returns a trace dict with every intermediate step (query, analysis,
    selection, evidence, answer, verification, retried) - this shape is the
    seed of the full trace log the finished pipeline will produce.
    """
    client = client or ollama.Client(host=OLLAMA_HOST)

    analysis = analyze_request(query, client=client)

    selection = None
    evidence = None
    if analysis["needs_tool"]:
        selection = select_tool(query, client=client)
        evidence = execute_tool(selection["tool"], selection["args"])

    answer = synthesize(query, evidence, client=client)

    verification = None
    retried = False
    if evidence is not None:
        verification = verify(answer, evidence, client=client)

        if verification["groundedness_score"] < _GROUNDEDNESS_RETRY_THRESHOLD:
            selection = select_tool(query, client=client)
            evidence = execute_tool(selection["tool"], selection["args"])
            answer = synthesize(query, evidence, client=client)
            verification = verify(answer, evidence, client=client)
            retried = True

            if verification["groundedness_score"] < _GROUNDEDNESS_RETRY_THRESHOLD:
                answer = hedge_unsupported_claims(answer, verification["verdicts"])

    return {
        "query": query,
        "analysis": analysis,
        "selection": selection,
        "evidence": evidence,
        "answer": answer,
        "verification": verification,
        "retried": retried,
    }
