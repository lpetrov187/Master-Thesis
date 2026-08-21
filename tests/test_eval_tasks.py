"""Day-9 test: the curated eval set is well-formed - every task has the
required fields, a valid category, and a real registered expected_tool.

Categories changed post-FastAPI-corpus-swap: code_analysis/code_execution
are no longer eval'd standalone (per user decision), only as exercised by
the code_generation ("generate" action) loop - see PLAN.md/eval-framework
fix notes. `expected_tool` for code_generation tasks is
"generate_and_verify_code", the synthetic tool name the orchestrator uses
for that action, not a literal TOOL_REGISTRY entry."""
from src.agent.tool_registry import TOOL_REGISTRY
from src.eval.tasks import load_holdout_tasks, load_tasks

_REQUIRED_KEYS = {"id", "category", "query", "expected_tool", "reference_answer"}
_VALID_CATEGORIES = {"doc_search", "code_generation", "web_fetch"}
_VALID_EXPECTED_TOOLS = set(TOOL_REGISTRY) | {"generate_and_verify_code"}


def test_eval_set_has_enough_tasks():
    tasks = load_tasks()
    assert 18 <= len(tasks) <= 25


def test_every_task_is_well_formed():
    tasks = load_tasks()
    seen_ids = set()

    for task in tasks:
        assert _REQUIRED_KEYS <= task.keys()
        assert task["category"] in _VALID_CATEGORIES
        assert task["expected_tool"] in _VALID_EXPECTED_TOOLS
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
    # web_fetch is deliberately smaller - simpler, lower-risk capability
    # than doc_search/code_generation, doesn't need equal coverage.
    for category in _VALID_CATEGORIES:
        assert counts[category] >= 4


def test_holdout_set_is_well_formed_and_disjoint_from_dev_set():
    dev_tasks = load_tasks()
    holdout_tasks = load_holdout_tasks()

    dev_ids = {t["id"] for t in dev_tasks}
    dev_queries = {t["query"] for t in dev_tasks}
    seen_ids = set()

    assert 8 <= len(holdout_tasks) <= 15
    for task in holdout_tasks:
        assert _REQUIRED_KEYS <= task.keys()
        assert task["category"] in _VALID_CATEGORIES
        assert task["expected_tool"] in _VALID_EXPECTED_TOOLS
        assert task["query"].strip()
        assert task["reference_answer"].strip()

        assert task["id"] not in seen_ids, f"duplicate id: {task['id']}"
        seen_ids.add(task["id"])

        assert task["id"] not in dev_ids, f"held-out id also in dev set: {task['id']}"
        assert task["query"] not in dev_queries, f"held-out query duplicates a dev query: {task['query']}"
