"""Web fetch tool: retrieve a specific, user-named URL and return its
readable text content. Unlike a general web-search tool, the agent never
picks which page to trust - the caller always names the exact URL, so the
source-vetting decision stays with the user, not the model.
"""
import re

import requests
from bs4 import BeautifulSoup

_DEFAULT_TIMEOUT_SECONDS = 10
_DEFAULT_MAX_CHARS = 8000
_USER_AGENT = "Mozilla/5.0 (compatible; MasterRadAgent/1.0)"
_WHITESPACE_RE = re.compile(r"\s+")


def fetch(url: str, timeout: int = _DEFAULT_TIMEOUT_SECONDS, max_chars: int = _DEFAULT_MAX_CHARS) -> dict:
    """Fetch `url` and extract its visible text.

    Returns {"url": str, "title": str | None, "text": str, "truncated": bool}.
    Raises requests.RequestException on network failure or a non-2xx
    status, and ValueError if the response isn't HTML - callers don't need
    to catch these themselves, since tool_executor.execute_tool() already
    wraps any tool exception into structured {"error": ...} evidence.
    """
    response = requests.get(url, timeout=timeout, headers={"User-Agent": _USER_AGENT})
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "html" not in content_type.lower():
        raise ValueError(f"unsupported content type for web_fetch: {content_type or 'unknown'}")

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else None
    body = soup.body or soup
    text = _WHITESPACE_RE.sub(" ", body.get_text(separator=" ")).strip()

    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]

    return {"url": url, "title": title, "text": text, "truncated": truncated}
