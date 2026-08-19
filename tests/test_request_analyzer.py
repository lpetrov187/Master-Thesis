"""Tests for the Request Analyzer's 3-way action classification. Real
model calls - added after the needs_tool -> action schema change (Phase 2
of "make this a real agent") so each of the 3 categories has a direct,
isolated check instead of relying only on other stages' end-to-end tests."""
from src.agent.request_analyzer import analyze_request


def test_classifies_a_documentation_lookup():
    result = analyze_request("How do I configure connection pooling in SQLAlchemy?")
    assert result["action"] == "lookup_or_inspect"


def test_classifies_a_code_snippet_to_inspect():
    result = analyze_request("Can you review this function for style issues?\n\ndef add(a,b):\n    return a+b\n")
    assert result["action"] == "lookup_or_inspect"


def test_classifies_a_code_generation_task():
    result = analyze_request("Write a function that sorts a list of integers.")
    assert result["action"] == "generate"


def test_classifies_general_knowledge_as_none():
    result = analyze_request("What's a friendly way to greet someone in an email?")
    assert result["action"] == "none"
