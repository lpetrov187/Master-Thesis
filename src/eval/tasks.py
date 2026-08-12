"""Loads the curated evaluation task set from data/eval/tasks.json."""
import json

from src.config import EVAL_DIR

_TASKS_PATH = EVAL_DIR / "tasks.json"


def load_tasks() -> list[dict]:
    """Return the curated eval task set as a list of task dicts."""
    return json.loads(_TASKS_PATH.read_text(encoding="utf-8"))
