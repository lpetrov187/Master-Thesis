"""Builds the Day-11 deliverables: a markdown comparison table and two
charts, from an already-computed metrics dict (see src.eval.metrics).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless - no display available/needed
import matplotlib.pyplot as plt

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"


def build_markdown_report(metrics: dict) -> str:
    """Render `metrics` as a markdown report: overall numbers, the
    hallucination-category breakdown, and a per-category comparison table."""
    g = metrics["groundedness"]
    groundedness_line = (
        f"| Groundedness score (mean, n={g['count']}) | {g['mean']:.3f} (min {g['min']:.3f}, max {g['max']:.3f}) |"
        if g["mean"] is not None
        else "| Groundedness score | n/a - no verified tasks |"
    )

    lines = [
        "# Evaluation Report",
        "",
        f"{metrics['n_tasks']} tasks, agent pipeline vs. no-tool baseline.",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Tool-selection accuracy | {metrics['tool_selection_accuracy']:.1%} |",
        f"| Agent hallucination rate | {metrics['agent_hallucination_rate']:.1%} |",
        f"| Baseline hallucination rate | {metrics['baseline_hallucination_rate']:.1%} |",
        f"| Agent task success rate | {metrics['agent_task_success_rate']:.1%} |",
        f"| Baseline task success rate | {metrics['baseline_task_success_rate']:.1%} |",
        groundedness_line,
        "",
        "## Agent hallucination breakdown",
        "",
        (
            "`synthesis` = an ungrounded claim made at answer-drafting time. "
            "`evidence_corruption` = the claim was faithfully grounded in tool "
            "evidence, but that evidence was itself wrong due to an upstream "
            "bug (see PLAN.md) - a different failure mode than the Controlled-"
            "Access/Claim-Verifier mechanisms are designed to catch."
        ),
        "",
        "| Category | Count |",
        "|---|---|",
    ]
    for category, count in sorted(metrics["hallucination_category_counts"].items()):
        lines.append(f"| {category} | {count} |")

    lines += [
        "",
        "## By task category",
        "",
        "| Category | n | Tool acc. | Agent halluc. | Baseline halluc. | Agent success | Baseline success |",
        "|---|---|---|---|---|---|---|",
    ]
    for category, m in sorted(metrics["by_category"].items()):
        lines.append(
            f"| {category} | {m['n']} | {m['tool_selection_accuracy']:.1%} | "
            f"{m['agent_hallucination_rate']:.1%} | {m['baseline_hallucination_rate']:.1%} | "
            f"{m['agent_task_success_rate']:.1%} | {m['baseline_task_success_rate']:.1%} |"
        )

    return "\n".join(lines) + "\n"


def build_comparison_chart(metrics: dict, path: Path) -> None:
    """Grouped bar chart: agent vs. baseline on hallucination rate and task
    success rate."""
    labels = ["Hallucination rate", "Task success rate"]
    agent_vals = [metrics["agent_hallucination_rate"], metrics["agent_task_success_rate"]]
    baseline_vals = [metrics["baseline_hallucination_rate"], metrics["baseline_task_success_rate"]]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - width / 2 for i in x], agent_vals, width, label="Agent")
    ax.bar([i + width / 2 for i in x], baseline_vals, width, label="Baseline")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Agent vs. Baseline")
    ax.legend()
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def build_groundedness_chart(results: list[dict], path: Path) -> None:
    """Histogram of the agent's groundedness scores across verified tasks."""
    scores = [
        r["agent_trace"]["verification"]["groundedness_score"]
        for r in results
        if r["agent_trace"]["verification"] is not None
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.01], edgecolor="black")
    ax.set_xlabel("Groundedness score")
    ax.set_ylabel("Task count")
    ax.set_title(f"Groundedness score distribution (n={len(scores)})")
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    import json

    from src.config import EVAL_DIR
    from src.eval.annotations import load_annotations
    from src.eval.metrics import compute_metrics

    results = json.loads((EVAL_DIR / "results.json").read_text(encoding="utf-8"))
    annotations = load_annotations()
    metrics = compute_metrics(results, annotations)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "day11_eval_report.md").write_text(build_markdown_report(metrics), encoding="utf-8")
    build_comparison_chart(metrics, REPORTS_DIR / "figures" / "comparison_bar.png")
    build_groundedness_chart(results, REPORTS_DIR / "figures" / "groundedness_hist.png")

    print(f"Wrote report + charts to {REPORTS_DIR}")
