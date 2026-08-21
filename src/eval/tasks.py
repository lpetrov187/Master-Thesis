"""Loads the curated evaluation task sets from data/eval/."""
import json

from src.config import EVAL_DIR

_TASKS_PATH = EVAL_DIR / "tasks.json"
_HOLDOUT_TASKS_PATH = EVAL_DIR / "tasks_holdout.json"


def load_tasks() -> list[dict]:
    """Return the dev eval task set as a list of task dicts. Used for
    iteration/debugging - numbers from this set should not be cited as
    evidence of generalization (see PLAN.md's eval-set-leakage section)."""
    return json.loads(_TASKS_PATH.read_text(encoding="utf-8"))


def load_holdout_tasks() -> list[dict]:
    """Return the held-out eval task set - written fresh, never referenced
    by any prompt, few-shot example, or debugging session. This is the set
    whose numbers should actually be cited."""
    return json.loads(_HOLDOUT_TASKS_PATH.read_text(encoding="utf-8"))
