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

_BANNER = """\
============================================================
  Master's Thesis Agent - Tool-Using LLM Pipeline
============================================================
  Tools available: doc_rag, code_analysis, code_execution, web_fetch
  Type a question and press Enter.
  Prefix a line with 'baseline: ' to run the no-tool baseline instead.
  Type 'exit' or press Ctrl+C to quit.
============================================================
"""


def _print_agent_result(trace: dict) -> None:
    tool = trace["selection"]["tool"] if trace["selection"] else "none"
    print(f"\n[tool: {tool}]")
    print(trace["answer"] if trace["answer"] is not None else f"ERROR: {trace['error']}")


def _print_baseline_result(trace: dict) -> None:
    print("\n[tool: none (baseline)]")
    print(trace["answer"] if trace["answer"] is not None else f"ERROR: {trace['error']}")


def main() -> None:
    print("Loading documentation corpus...")
    ingest()
    session = Session()

    print(_BANNER)

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
