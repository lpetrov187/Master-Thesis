"""Interactive command-line demo for the agent pipeline - type a query and
see the agent's answer, without writing any Python.

Usage:
    python -m src.cli

Type a question and press Enter. Prefix a line with "baseline: " to run
that one query through the no-tool baseline instead of the agent, for a
side-by-side comparison. Type "exit" or press Ctrl+C to quit.
"""
from src.agent.baseline import run as run_baseline
from src.agent.orchestrator import run as run_agent
from src.tools.doc_rag import ingest


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
    print(trace["answer"])


def _print_baseline_result(trace: dict) -> None:
    print("\n[baseline - no tools, no verification]")
    print(f"ERROR: {trace['error']}" if trace["error"] else trace["answer"])


def main() -> None:
    print("Loading documentation corpus...")
    ingest()

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
            trace = run_agent(query)
            _print_agent_result(trace)
        print()


if __name__ == "__main__":
    main()
