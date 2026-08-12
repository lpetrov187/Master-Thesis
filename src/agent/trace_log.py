"""Append-only JSONL trace log - one line per pipeline run (agent or
baseline), the raw data the evaluation chapter's metrics get computed from.
"""
import json
from datetime import UTC, datetime
from pathlib import Path

from src.config import LOGS_DIR

_DEFAULT_LOG_PATH = LOGS_DIR / "traces.jsonl"


def log_trace(trace: dict, condition: str, path: Path | None = None) -> None:
    """Append `trace` to the trace log, tagged with which condition produced
    it ("agent" or "baseline") and when."""
    path = path or _DEFAULT_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {"condition": condition, "logged_at": datetime.now(UTC).isoformat(), **trace}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
