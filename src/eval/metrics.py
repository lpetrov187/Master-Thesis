"""Computes the numbers PLAN.md's "done" criteria call for - tool-selection
accuracy, groundedness score distribution, hallucination rate, and task
success rate, both overall and broken down per task category.

Pure function over already-collected data (results.json + annotations.json)
- no model calls, no I/O beyond what the caller already loaded.
"""


def compute_metrics(results: list[dict], annotations: list[dict]) -> dict:
    """Join `results` (from src.eval.runner) with `annotations` (hand-
    reviewed judgments) by task_id and compute summary metrics."""
    annotations_by_id = {a["task_id"]: a for a in annotations}
    rows = [{**r, **annotations_by_id[r["task_id"]]} for r in results]
    n = len(rows)

    def rate(key: str) -> float:
        return sum(row[key] for row in rows) / n

    hallucination_category_counts: dict[str, int] = {}
    for row in rows:
        cat = row["agent_hallucination_category"]
        hallucination_category_counts[cat] = hallucination_category_counts.get(cat, 0) + 1

    groundedness_scores = [
        r["agent_trace"]["verification"]["groundedness_score"]
        for r in results
        if r["agent_trace"]["verification"] is not None
    ]
    groundedness = {
        "count": len(groundedness_scores),
        "mean": sum(groundedness_scores) / len(groundedness_scores) if groundedness_scores else None,
        "min": min(groundedness_scores) if groundedness_scores else None,
        "max": max(groundedness_scores) if groundedness_scores else None,
    }

    by_category = {}
    for category in sorted({row["category"] for row in rows}):
        cat_rows = [row for row in rows if row["category"] == category]
        cn = len(cat_rows)
        by_category[category] = {
            "n": cn,
            "tool_selection_accuracy": sum(row["tool_selection_correct"] for row in cat_rows) / cn,
            "agent_hallucination_rate": sum(row["agent_hallucination"] for row in cat_rows) / cn,
            "baseline_hallucination_rate": sum(row["baseline_hallucination"] for row in cat_rows) / cn,
            "agent_task_success_rate": sum(row["agent_task_success"] for row in cat_rows) / cn,
            "baseline_task_success_rate": sum(row["baseline_task_success"] for row in cat_rows) / cn,
        }

    return {
        "n_tasks": n,
        "tool_selection_accuracy": rate("tool_selection_correct"),
        "agent_hallucination_rate": rate("agent_hallucination"),
        "baseline_hallucination_rate": rate("baseline_hallucination"),
        "agent_task_success_rate": rate("agent_task_success"),
        "baseline_task_success_rate": rate("baseline_task_success"),
        "hallucination_category_counts": hallucination_category_counts,
        "groundedness": groundedness,
        "by_category": by_category,
    }
