"""Builds the Day-11 deliverables: a markdown comparison table and two
charts, from an already-computed metrics dict (see src.eval.metrics).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless - no display available/needed
import matplotlib.pyplot as plt

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"


def build_markdown_report(metrics: dict, label: str = "", caveat: str = "") -> str:
    """Render `metrics` as a markdown report: overall numbers, the
    hallucination-category breakdown, and a per-category comparison table.

    `label` (optional) is appended to the title, e.g. "(held-out set)".
    `caveat` (optional) is a short note rendered right under the title -
    used to flag a set's leakage/generalization status.
    """
    lines = [
        f"# Evaluation Report{f' {label}' if label else ''}",
        "",
    ]
    if caveat:
        lines += [caveat, ""]
    lines += [
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
        "",
        "## Agent hallucination breakdown",
        "",
        (
            "`synthesis` = an ungrounded claim made at answer-drafting time. "
            "`evidence_corruption` = the claim was faithfully grounded in tool "
            "evidence, but that evidence was itself wrong due to an upstream "
            "bug (see PLAN.md) - a different failure mode than the Controlled-"
            "Access policy is designed to catch."
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


if __name__ == "__main__":
    import json

    from src.config import EVAL_DIR
    from src.eval.annotations import load_annotations, load_holdout_annotations
    from src.eval.metrics import compute_metrics

    def _build_one_report(results_filename: str, annotations_loader, label: str, caveat: str, out_prefix: str) -> None:
        results_path = EVAL_DIR / results_filename
        results = json.loads(results_path.read_text(encoding="utf-8"))
        annotations = annotations_loader()
        metrics = compute_metrics(results, annotations)

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_text = build_markdown_report(metrics, label=label, caveat=caveat)
        (REPORTS_DIR / f"{out_prefix}_eval_report.md").write_text(report_text, encoding="utf-8")
        build_comparison_chart(metrics, REPORTS_DIR / "figures" / f"{out_prefix}_comparison_bar.png")
        print(f"Wrote {out_prefix} report + chart to {REPORTS_DIR}")

    _build_one_report(
        "results.json",
        load_annotations,
        label="(dev set)",
        caveat=(
            "**Dev set - do not cite these numbers as evidence of generalization.** "
            "Used for iteration/debugging; see PLAN.md's eval-set-leakage section. "
            "Report the held-out set's numbers instead."
        ),
        out_prefix="dev",
    )
    _build_one_report(
        "results_holdout.json",
        load_holdout_annotations,
        label="(held-out set)",
        caveat="Held-out set - never referenced by any prompt or debugging session. These are the numbers to cite.",
        out_prefix="holdout",
    )
