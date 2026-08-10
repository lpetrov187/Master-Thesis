"""Orchestration loop: Request Analyzer -> Tool Selector -> Tool Executor ->
Answer Synthesizer. No Claim Verifier yet - that's added on Day 7.
"""
import ollama

from src.agent.answer_synthesizer import synthesize
from src.agent.request_analyzer import analyze_request
from src.agent.tool_executor import execute_tool
from src.agent.tool_selector import select_tool
from src.config import OLLAMA_HOST


def run(query: str, client: ollama.Client | None = None) -> dict:
    """Run the full pipeline for `query`.

    Returns a trace dict with every intermediate step (query, analysis,
    selection, evidence, answer) - this shape is the seed of the full trace
    log the finished pipeline will produce.
    """
    client = client or ollama.Client(host=OLLAMA_HOST)

    analysis = analyze_request(query, client=client)

    selection = None
    evidence = None
    if analysis["needs_tool"]:
        selection = select_tool(query, client=client)
        evidence = execute_tool(selection["tool"], selection["args"])

    answer = synthesize(query, evidence, client=client)

    return {
        "query": query,
        "analysis": analysis,
        "selection": selection,
        "evidence": evidence,
        "answer": answer,
    }
