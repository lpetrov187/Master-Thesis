"""Day-11 tests: report building - markdown formatting and chart file
creation. No model calls; matplotlib runs headless (Agg backend)."""
from src.eval.report import build_comparison_chart, build_markdown_report

_METRICS = {
    "n_tasks": 4,
    "tool_selection_accuracy": 0.75,
    "agent_hallucination_rate": 0.25,
    "baseline_hallucination_rate": 0.5,
    "agent_task_success_rate": 0.75,
    "baseline_task_success_rate": 0.5,
    "hallucination_category_counts": {"none": 3, "synthesis": 1},
    "by_category": {
        "doc_search": {
            "n": 2,
            "tool_selection_accuracy": 1.0,
            "agent_hallucination_rate": 0.0,
            "baseline_hallucination_rate": 0.5,
            "agent_task_success_rate": 1.0,
            "baseline_task_success_rate": 0.5,
        },
    },
}


def test_build_markdown_report_contains_key_numbers():
    report = build_markdown_report(_METRICS)

    assert "75.0%" in report  # tool_selection_accuracy and agent_task_success_rate
    assert "25.0%" in report  # agent_hallucination_rate
    assert "doc_search" in report
    assert "synthesis" in report
    assert "claim-verifier" not in report.lower()
    assert "groundedness" not in report.lower()


def test_build_markdown_report_includes_label_and_caveat():
    report = build_markdown_report(_METRICS, label="(held-out set)", caveat="Cite these numbers.")

    assert "(held-out set)" in report
    assert "Cite these numbers." in report


def test_build_comparison_chart_creates_file(tmp_path):
    path = tmp_path / "figures" / "comparison_bar.png"

    build_comparison_chart(_METRICS, path)

    assert path.exists()
    assert path.stat().st_size > 0
