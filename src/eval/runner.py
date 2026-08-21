"""Runs the curated eval task set through both conditions - the full agent
pipeline and the no-tool baseline - and pairs each task with both results.

Individual runs are already persisted to logs/traces.jsonl by
orchestrator.run()/baseline.run() (Day 8), but that log only tags each
entry with its condition and timestamp, not which curated task it came
from. save_results() writes a second, task-indexed file specifically for
Day 11's hand-annotation, so grading doesn't require cross-referencing the
append-only trace log after the fact.
"""
import json
from pathlib import Path

from src.agent.baseline import run as run_baseline
from src.agent.orchestrator import run as run_agent
from src.config import EVAL_DIR
from src.eval.tasks import load_holdout_tasks, load_tasks

_RESULTS_PATH = EVAL_DIR / "results.json"
_HOLDOUT_RESULTS_PATH = EVAL_DIR / "results_holdout.json"


def run_eval(tasks: list[dict] | None = None, verbose: bool = True) -> list[dict]:
    """Run every task through both conditions.

    Returns a list of {"task_id", "category", "query", "expected_tool",
    "reference_answer", "agent_trace", "baseline_trace"}.
    """
    tasks = tasks if tasks is not None else load_tasks()
    results = []

    for i, task in enumerate(tasks, 1):
        if verbose:
            print(f"[{i}/{len(tasks)}] {task['id']} ({task['category']})")

        agent_trace = run_agent(task["query"])
        baseline_trace = run_baseline(task["query"])

        results.append({
            "task_id": task["id"],
            "category": task["category"],
            "query": task["query"],
            "expected_tool": task["expected_tool"],
            "reference_answer": task["reference_answer"],
            "agent_trace": agent_trace,
            "baseline_trace": baseline_trace,
        })

    return results


def save_results(results: list[dict], path: Path | None = None) -> None:
    """Write `results` to a single JSON file, indexed by task."""
    path = path or _RESULTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    from src.tools.doc_rag import ingest

    ingest()

    dev_results = run_eval(load_tasks())
    save_results(dev_results, path=_RESULTS_PATH)
    print(f"Saved {len(dev_results)} dev-set task results to {_RESULTS_PATH}")

    holdout_results = run_eval(load_holdout_tasks())
    save_results(holdout_results, path=_HOLDOUT_RESULTS_PATH)
    print(f"Saved {len(holdout_results)} held-out task results to {_HOLDOUT_RESULTS_PATH}")
