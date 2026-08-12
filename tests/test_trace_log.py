"""Day-8 test: log_trace appends valid JSONL records tagged with a
condition and a timestamp."""
import json

from src.agent.trace_log import log_trace


def test_log_trace_appends_json_lines(tmp_path):
    log_path = tmp_path / "traces.jsonl"

    log_trace({"query": "q1", "answer": "a1", "error": None}, condition="agent", path=log_path)
    log_trace({"query": "q2", "answer": "a2", "error": None}, condition="baseline", path=log_path)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["condition"] == "agent"
    assert first["query"] == "q1"
    assert "logged_at" in first

    second = json.loads(lines[1])
    assert second["condition"] == "baseline"
    assert second["query"] == "q2"
