"""Tests for Session: control-flow only, run_agent mocked so this doesn't
spend real model calls (same pattern as test_orchestrator.py's hardening
tests - Session's own logic is just list bookkeeping, not model behavior)."""
from unittest.mock import patch

from src.agent.session import Session


@patch("src.agent.session.run_agent")
def test_ask_appends_the_turn_to_history(mock_run_agent):
    mock_run_agent.return_value = {"answer": "the answer"}
    session = Session()

    session.ask("the question")

    assert session.history == [{"query": "the question", "answer": "the answer"}]


@patch("src.agent.session.run_agent")
def test_ask_passes_accumulated_history_to_run_agent(mock_run_agent):
    mock_run_agent.return_value = {"answer": "a2"}
    session = Session()
    session.history = [{"query": "q1", "answer": "a1"}]

    session.ask("q2")

    _, kwargs = mock_run_agent.call_args
    assert kwargs["history"] == [{"query": "q1", "answer": "a1"}]


@patch("src.agent.session.run_agent")
def test_history_is_trimmed_to_max_history(mock_run_agent):
    mock_run_agent.return_value = {"answer": "a"}
    session = Session(max_history=2)

    session.ask("q1")
    session.ask("q2")
    session.ask("q3")

    assert len(session.history) == 2
    assert [t["query"] for t in session.history] == ["q2", "q3"]


@patch("src.agent.session.run_agent")
def test_ask_returns_the_full_trace(mock_run_agent):
    mock_run_agent.return_value = {"answer": "the answer", "selection": None}
    session = Session()

    trace = session.ask("the question")

    assert trace == {"answer": "the answer", "selection": None}
