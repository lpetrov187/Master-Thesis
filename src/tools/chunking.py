"""Markdown-aware chunking: splits documentation into chunks that respect
heading and code-fence boundaries, instead of a blind word-count window.

The original chunker (Day 3) just slid a fixed-size window across raw
whitespace-split text. That was fine for 4 short, hand-written docs with no
nested headings and short code snippets - it falls apart on real
documentation: it can split a code block mid-function, or start a chunk
mid-sentence with no idea which section it's even in. This chunks *within*
sections, never merges text across a heading or a code fence, and tags
every chunk with the heading path it came from.
"""
import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE_RE = re.compile(r"^```")

_CHUNK_WORDS = 300
_CHUNK_OVERLAP_WORDS = 20

# Headings at or above this level (## and shallower, by default) are hard
# chunk boundaries; deeper headings (### and beyond) stay inline as visible
# text within their parent's chunk instead of forcing their own. Found by
# a real retrieval failure: a page's actual answer lived in a ### child
# 3 levels deep (Dependencies > First Steps > Declare the dependency),
# split away from the parent section whose intro phrase ("how the
# Dependency Injection system works") was what made the section rank for
# a "how does X work" query in the first place. Splitting every heading
# level fragments a single coherent explanation into pieces that don't
# individually look relevant, even though the whole is.
_HEADING_BOUNDARY_LEVEL = 2


@dataclass(frozen=True)
class Chunk:
    text: str
    heading: str  # e.g. "SQLAlchemy: Configuring Connection Pooling", "" if none
    is_code: bool


def _split_into_blocks(text: str, boundary_level: int = _HEADING_BOUNDARY_LEVEL) -> list[Chunk]:
    """Split markdown into blocks - one per paragraph or code fence - each
    tagged with the heading path active at that point in the document.

    Only headings at or above `boundary_level` start a new block and update
    the tracked heading path; deeper headings are left as plain text inside
    the current block, so a section's subsections stay together instead of
    each becoming their own disconnected fragment.
    """
    heading_stack: list[tuple[int, str]] = []
    blocks: list[Chunk] = []
    current_lines: list[str] = []
    in_code_fence = False

    def heading_path() -> str:
        return " > ".join(title for _, title in heading_stack)

    def flush_text() -> None:
        content = "\n".join(current_lines).strip()
        current_lines.clear()
        if content:
            blocks.append(Chunk(text=content, heading=heading_path(), is_code=False))

    for line in text.splitlines():
        if _FENCE_RE.match(line):
            if in_code_fence:
                current_lines.append(line)
                blocks.append(Chunk(text="\n".join(current_lines).strip(), heading=heading_path(), is_code=True))
                current_lines.clear()
            else:
                flush_text()
                current_lines.append(line)
            in_code_fence = not in_code_fence
            continue

        if in_code_fence:
            current_lines.append(line)
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            if level > boundary_level:
                current_lines.append(line)  # deeper heading: keep inline, don't split or retag
                continue
            flush_text()
            heading_stack = [h for h in heading_stack if h[0] < level]
            heading_stack.append((level, heading_match.group(2).strip()))
            continue

        current_lines.append(line)

    flush_text()
    return blocks


def _window_words(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    """Slide a word-count window over `text` (used within one block)."""
    words = text.split()
    if not words:
        return []

    windows = []
    start = 0
    while start < len(words):
        end = start + chunk_words
        windows.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap_words
    return windows


def chunk_markdown(
    text: str,
    chunk_words: int = _CHUNK_WORDS,
    overlap_words: int = _CHUNK_OVERLAP_WORDS,
    boundary_level: int = _HEADING_BOUNDARY_LEVEL,
) -> list[Chunk]:
    """Chunk `text` respecting heading and code-fence boundaries.

    Code blocks are kept atomic (never split) unless larger than
    `chunk_words`, in which case they're windowed like prose but stay
    tagged `is_code=True`. Consecutive prose blocks under the same heading
    (heading level <= `boundary_level`; deeper headings stay inline within
    their parent's chunk) are merged up to `chunk_words` before windowing,
    so short paragraphs - or whole subsections - don't each become their
    own tiny, context-free chunk.
    """
    blocks = _split_into_blocks(text, boundary_level)
    chunks: list[Chunk] = []
    buffer_text = ""
    buffer_heading = ""

    def flush_buffer() -> None:
        if buffer_text.strip():
            for window in _window_words(buffer_text, chunk_words, overlap_words):
                chunks.append(Chunk(text=window, heading=buffer_heading, is_code=False))

    for block in blocks:
        if block.is_code:
            flush_buffer()
            buffer_text = ""
            if len(block.text.split()) > chunk_words:
                chunks.extend(
                    Chunk(text=window, heading=block.heading, is_code=True)
                    for window in _window_words(block.text, chunk_words, overlap_words)
                )
            else:
                chunks.append(block)
            continue

        merged = f"{buffer_text} {block.text}".strip()
        if block.heading != buffer_heading or len(merged.split()) > chunk_words:
            flush_buffer()
            buffer_text, buffer_heading = block.text, block.heading
        else:
            buffer_text = merged

    flush_buffer()
    return chunks
