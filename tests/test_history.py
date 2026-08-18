"""Tests for format_history(): pure string formatting, no model calls."""
from src.agent.history import format_history


def test_empty_or_none_history_returns_empty_string():
    assert format_history(None) == ""
    assert format_history([]) == ""


def test_one_turn_is_formatted_as_a_transcript():
    history = [{"query": "How do I use Depends?", "answer": "Pass it as a default value."}]

    result = format_history(history)

    assert "Conversation so far:" in result
    assert "User: How do I use Depends?" in result
    assert "Assistant: Pass it as a default value." in result


def test_multiple_turns_are_all_included_in_order():
    history = [
        {"query": "first question", "answer": "first answer"},
        {"query": "second question", "answer": "second answer"},
    ]

    result = format_history(history)

    assert result.index("first question") < result.index("second question")
    assert "first answer" in result
    assert "second answer" in result
