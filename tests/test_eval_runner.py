"""Day-10 tests: the eval runner pairs each task with both conditions'
results and can save them to disk - wiring only, mocked to avoid spending
real model calls on logic that's already covered by orchestrator/baseline
tests."""
import json
from unittest.mock import patch

from src.eval.runner import run_eval, save_results

_TASK = {
    "id": "t1",
    "category": "doc_search",
    "query": "q1",
    "expected_tool": "doc_rag",
    "reference_answer": "ref1",
}


@patch("src.eval.runner.run_baseline")
@patch("src.eval.runner.run_agent")
def test_run_eval_pairs_each_task_with_both_conditions(mock_agent, mock_baseline):
    mock_agent.return_value = {"answer": "agent answer"}
    mock_baseline.return_value = {"answer": "baseline answer"}

    results = run_eval([_TASK], verbose=False)

    assert len(results) == 1
    result = results[0]
    assert result["task_id"] == "t1"
    assert result["category"] == "doc_search"
    assert result["expected_tool"] == "doc_rag"
    assert result["agent_trace"] == {"answer": "agent answer"}
    assert result["baseline_trace"] == {"answer": "baseline answer"}
    mock_agent.assert_called_once_with("q1")
    mock_baseline.assert_called_once_with("q1")


@patch("src.eval.runner.run_baseline")
@patch("src.eval.runner.run_agent")
def test_run_eval_runs_every_task(mock_agent, mock_baseline):
    mock_agent.return_value = {"answer": "a"}
    mock_baseline.return_value = {"answer": "b"}
    tasks = [_TASK, {**_TASK, "id": "t2"}, {**_TASK, "id": "t3"}]

    results = run_eval(tasks, verbose=False)

    assert [r["task_id"] for r in results] == ["t1", "t2", "t3"]
    assert mock_agent.call_count == 3
    assert mock_baseline.call_count == 3


def test_save_results_writes_valid_json(tmp_path):
    path = tmp_path / "results.json"
    results = [{"task_id": "t1", "agent_trace": {"answer": "a"}, "baseline_trace": {"answer": "b"}}]

    save_results(results, path=path)

    assert json.loads(path.read_text(encoding="utf-8")) == results
