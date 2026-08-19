"""Interactive command-line demo for the agent pipeline - type a query and
see the agent's answer, without writing any Python.

Usage:
    python -m src.cli

Type a question and press Enter. The agent remembers recent turns in this
session, so follow-ups can reference earlier questions/answers. Prefix a
line with "baseline: " to run that one query through the no-tool baseline
instead - deliberately excluded from history, since it's meant to stay the
clean, memory-free control condition used throughout the eval work. Type
"exit" or press Ctrl+C to quit.
"""
from src.agent.baseline import run as run_baseline
from src.agent.session import Session
from src.tools.doc_rag import ingest


def _print_timings(timings: dict) -> None:
    print("[timings]")
    for stage, secs in timings.items():
        if secs is not None:
            print(f"  {stage:18} {secs:6.2f}s")


def _print_generation_details(evidence: dict) -> None:
    """Show whether the code-generation loop actually did anything: how
    many passes it took, and - if a revision fired - why, so it's clear
    the agent isn't just reporting the first draft as-is."""
    result = evidence.get("result")
    if not result:
        return

    print(f"[code generation] iterations: {result['iterations']}")
    if result["revised"]:
        print(f"  first attempt failed verification ({result['revision_reason']}) -> revised and re-verified")
    elif result["revision_reason"] is not None:
        print(f"  revision attempted but couldn't be applied ({result['revision_reason']})")
    else:
        print("  passed verification on the first attempt, no revision needed")


def _print_agent_result(trace: dict) -> None:
    tool = trace["selection"]["tool"] if trace["selection"] else "none"
    verification = trace["verification"]

    status = f"tool: {tool}"
    if verification is not None:
        status += f" | groundedness: {verification['groundedness_score']:.2f}"
    if trace["retried"]:
        status += " | retried"
    if trace["error"]:
        status += f" | ERROR: {trace['error']}"

    print(f"\n[{status}]")
    if tool == "generate_and_verify_code" and trace["evidence"]:
        _print_generation_details(trace["evidence"])
    print(trace["answer"])
    print()
    _print_timings(trace["timings"])


def _print_baseline_result(trace: dict) -> None:
    print("\n[baseline - no tools, no verification]")
    print(f"ERROR: {trace['error']}" if trace["error"] else trace["answer"])


def main() -> None:
    print("Loading documentation corpus...")
    ingest()
    session = Session()

    print("Agent CLI - type a question, or 'exit' to quit.")
    print("Prefix a line with 'baseline: ' to run it through the no-tool baseline instead.\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        print("(thinking...)")
        if query.lower().startswith("baseline:"):
            trace = run_baseline(query.split(":", 1)[1].strip())
            _print_baseline_result(trace)
        else:
            trace = session.ask(query)
            _print_agent_result(trace)
        print()


if __name__ == "__main__":
    main()
