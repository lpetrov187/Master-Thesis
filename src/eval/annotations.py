"""Loads the hand-annotation set from data/eval/annotations.json."""
import json

from src.config import EVAL_DIR

_ANNOTATIONS_PATH = EVAL_DIR / "annotations.json"


def load_annotations() -> list[dict]:
    """Return the annotation list, keyed by task_id in the same order as
    data/eval/tasks.json. Does not include the file's "meta" block."""
    data = json.loads(_ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    return data["annotations"]
