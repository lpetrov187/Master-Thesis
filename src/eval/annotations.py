"""Loads the hand-annotation sets from data/eval/."""
import json

from src.config import EVAL_DIR

_ANNOTATIONS_PATH = EVAL_DIR / "annotations.json"
_HOLDOUT_ANNOTATIONS_PATH = EVAL_DIR / "annotations_holdout.json"


def load_annotations() -> list[dict]:
    """Return the dev-set annotation list, keyed by task_id in the same
    order as data/eval/tasks.json. Does not include the file's "meta" block."""
    data = json.loads(_ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    return data["annotations"]


def load_holdout_annotations() -> list[dict]:
    """Return the held-out annotation list, matching data/eval/tasks_holdout.json."""
    data = json.loads(_HOLDOUT_ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    return data["annotations"]
