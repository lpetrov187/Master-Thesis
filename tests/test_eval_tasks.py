"""Day-9 test: the curated eval set is well-formed - every task has the
required fields, a valid category, and a real registered expected_tool."""
from src.agent.tool_registry import TOOL_REGISTRY
from src.eval.tasks import load_tasks

_REQUIRED_KEYS = {"id", "category", "query", "expected_tool", "reference_answer"}
_VALID_CATEGORIES = {"doc_search", "code_analysis", "programming_problem"}


def test_eval_set_has_enough_tasks():
    tasks = load_tasks()
    assert 20 <= len(tasks) <= 25


def test_every_task_is_well_formed():
    tasks = load_tasks()
    seen_ids = set()

    for task in tasks:
        assert _REQUIRED_KEYS <= task.keys()
        assert task["category"] in _VALID_CATEGORIES
        assert task["expected_tool"] in TOOL_REGISTRY
        assert task["query"].strip()
        assert task["reference_answer"].strip()

        assert task["id"] not in seen_ids, f"duplicate id: {task['id']}"
        seen_ids.add(task["id"])


def test_categories_are_reasonably_balanced():
    tasks = load_tasks()
    counts = {}
    for task in tasks:
        counts[task["category"]] = counts.get(task["category"], 0) + 1

    assert set(counts) == _VALID_CATEGORIES
    for category in _VALID_CATEGORIES:
        assert counts[category] >= 5
