"""Day-8 test: the no-tool baseline answers directly, with no tools, no
policy, and no verifier involved - the control condition for evaluation."""
from src.agent.baseline import run


def test_baseline_answers_without_any_tool():
    trace = run("What's a friendly way to greet someone in an email?")

    assert trace["error"] is None
    assert trace["answer"]
