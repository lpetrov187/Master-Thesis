"""Tests for web_fetch.fetch(): requests.get is mocked, so these exercise
the HTML-to-text extraction/truncation/error logic without making a real
network call."""
from unittest.mock import Mock, patch

import pytest
import requests
from src.tools.web_fetch import fetch


def _mock_response(text="", content_type="text/html; charset=utf-8", status_ok=True):
    response = Mock()
    response.text = text
    response.headers = {"Content-Type": content_type}
    if status_ok:
        response.raise_for_status.return_value = None
    else:
        response.raise_for_status.side_effect = requests.HTTPError("404")
    return response


@patch("src.tools.web_fetch.requests.get")
def test_extracts_title_and_visible_text(mock_get):
    html = "<html><head><title>My Page</title></head><body><p>Hello world</p></body></html>"
    mock_get.return_value = _mock_response(html)

    result = fetch("https://example.com")

    assert result["url"] == "https://example.com"
    assert result["title"] == "My Page"
    assert result["text"] == "Hello world"
    assert result["truncated"] is False


@patch("src.tools.web_fetch.requests.get")
def test_strips_script_and_style_tags(mock_get):
    html = "<html><body><script>evil()</script><style>.a{}</style><p>Visible</p></body></html>"
    mock_get.return_value = _mock_response(html)

    result = fetch("https://example.com")

    assert "evil" not in result["text"]
    assert result["text"] == "Visible"


@patch("src.tools.web_fetch.requests.get")
def test_missing_title_is_none(mock_get):
    mock_get.return_value = _mock_response("<html><body><p>No title here</p></body></html>")

    result = fetch("https://example.com")

    assert result["title"] is None


@patch("src.tools.web_fetch.requests.get")
def test_collapses_whitespace(mock_get):
    html = "<html><body><p>Line one</p>\n\n<p>   Line   two  </p></body></html>"
    mock_get.return_value = _mock_response(html)

    result = fetch("https://example.com")

    assert result["text"] == "Line one Line two"


@patch("src.tools.web_fetch.requests.get")
def test_truncates_long_text_and_sets_flag(mock_get):
    html = f"<html><body><p>{'a' * 100}</p></body></html>"
    mock_get.return_value = _mock_response(html)

    result = fetch("https://example.com", max_chars=10)

    assert result["text"] == "a" * 10
    assert result["truncated"] is True


@patch("src.tools.web_fetch.requests.get")
def test_rejects_non_html_content_type(mock_get):
    mock_get.return_value = _mock_response("%PDF-1.4 ...", content_type="application/pdf")

    with pytest.raises(ValueError, match="unsupported content type"):
        fetch("https://example.com/doc.pdf")


@patch("src.tools.web_fetch.requests.get")
def test_raises_on_bad_status(mock_get):
    mock_get.return_value = _mock_response("not found", status_ok=False)

    with pytest.raises(requests.HTTPError):
        fetch("https://example.com/missing")
