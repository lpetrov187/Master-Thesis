"""Day-11 test: compute_metrics computes correct rates from a small,
hand-built fixture with known expected numbers (no model calls)."""
from src.eval.metrics import compute_metrics


def _row(task_id, category, verification=None):
    return {
        "task_id": task_id,
        "category": category,
        "agent_trace": {"verification": verification},
    }


def _ann(task_id, tool_ok, agent_hall, hall_cat, agent_ok, baseline_hall, baseline_ok):
    return {
        "task_id": task_id,
        "tool_selection_correct": tool_ok,
        "agent_hallucination": agent_hall,
        "agent_hallucination_category": hall_cat,
        "agent_task_success": agent_ok,
        "baseline_hallucination": baseline_hall,
        "baseline_task_success": baseline_ok,
    }


def test_compute_metrics_overall_rates():
    results = [
        _row("t1", "doc_search", {"groundedness_score": 1.0}),
        _row("t2", "doc_search", {"groundedness_score": 0.5}),
        _row("t3", "code_analysis", None),
        _row("t4", "code_analysis", None),
    ]
    annotations = [
        _ann("t1", True, False, "none", True, False, True),
        _ann("t2", False, True, "synthesis", False, True, False),
        _ann("t3", True, False, "none", True, False, True),
        _ann("t4", True, True, "evidence_corruption", False, False, True),
    ]

    metrics = compute_metrics(results, annotations)

    assert metrics["n_tasks"] == 4
    assert metrics["tool_selection_accuracy"] == 0.75
    assert metrics["agent_hallucination_rate"] == 0.5
    assert metrics["baseline_hallucination_rate"] == 0.25
    assert metrics["agent_task_success_rate"] == 0.5
    assert metrics["baseline_task_success_rate"] == 0.75

    assert metrics["hallucination_category_counts"] == {
        "none": 2,
        "synthesis": 1,
        "evidence_corruption": 1,
    }

    assert metrics["groundedness"]["count"] == 2
    assert metrics["groundedness"]["mean"] == 0.75
    assert metrics["groundedness"]["min"] == 0.5
    assert metrics["groundedness"]["max"] == 1.0


def test_compute_metrics_by_category():
    results = [
        _row("t1", "doc_search"),
        _row("t2", "doc_search"),
        _row("t3", "code_analysis"),
    ]
    annotations = [
        _ann("t1", True, False, "none", True, False, True),
        _ann("t2", True, False, "none", True, False, True),
        _ann("t3", False, True, "synthesis", False, True, False),
    ]

    metrics = compute_metrics(results, annotations)

    assert metrics["by_category"]["doc_search"]["n"] == 2
    assert metrics["by_category"]["doc_search"]["tool_selection_accuracy"] == 1.0
    assert metrics["by_category"]["code_analysis"]["n"] == 1
    assert metrics["by_category"]["code_analysis"]["agent_task_success_rate"] == 0.0
