"""Tests for the markdown-aware chunker: heading boundaries, code-fence
atomicity, and word-count windowing within a section. No model calls."""
from src.tools.chunking import chunk_markdown

_DOC = """# Widget Library

## Installation

Run pip install widget to get started. This is a short paragraph that
should stay under the chunk word limit on its own.

## Usage

```python
import widget

w = widget.Widget()
w.run()
```

Here is a follow-up paragraph explaining the snippet above in more detail,
describing what each line does and why it matters for typical usage.
"""


def test_chunks_carry_the_correct_heading_path():
    chunks = chunk_markdown(_DOC)

    install_chunks = [c for c in chunks if c.heading == "Widget Library > Installation"]
    usage_chunks = [c for c in chunks if c.heading == "Widget Library > Usage"]

    assert install_chunks
    assert usage_chunks
    assert all("pip install widget" in c.text for c in install_chunks)


def test_code_block_is_kept_atomic_and_tagged():
    chunks = chunk_markdown(_DOC)
    code_chunks = [c for c in chunks if c.is_code]

    assert len(code_chunks) == 1
    assert "import widget" in code_chunks[0].text
    assert "w.run()" in code_chunks[0].text
    assert code_chunks[0].heading == "Widget Library > Usage"


def test_prose_never_merges_across_a_heading_boundary():
    chunks = chunk_markdown(_DOC)

    for chunk in chunks:
        if chunk.is_code:
            continue
        # every prose chunk's text should belong entirely to one section
        assert not ("pip install widget" in chunk.text and "import widget" in chunk.text)


def test_long_code_block_is_windowed_but_stays_tagged_as_code():
    long_code = "\n".join(f"line_{i} = {i}" for i in range(200))
    doc = f"# Big Module\n\n```python\n{long_code}\n```\n"

    chunks = chunk_markdown(doc, chunk_words=50, overlap_words=5)

    assert len(chunks) > 1
    assert all(c.is_code for c in chunks)
    assert all(c.heading == "Big Module" for c in chunks)


def test_long_prose_under_one_heading_splits_into_multiple_chunks():
    long_prose = " ".join(f"word{i}" for i in range(400))
    doc = f"# Section\n\n{long_prose}\n"

    chunks = chunk_markdown(doc, chunk_words=100, overlap_words=10)

    assert len(chunks) > 1
    assert all(c.heading == "Section" for c in chunks)
    assert all(not c.is_code for c in chunks)


def test_empty_document_produces_no_chunks():
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n  ") == []


def test_short_paragraphs_under_same_heading_get_merged():
    doc = "# Notes\n\nFirst short line.\n\nSecond short line.\n\nThird short line.\n"

    chunks = chunk_markdown(doc, chunk_words=100, overlap_words=10)

    assert len(chunks) == 1
    assert "First short line." in chunks[0].text
    assert "Third short line." in chunks[0].text


_NESTED_DOC = """# Dependencies

## First Steps

Intro sentence explaining how the system works overall.

### Create a dependency

Explanation of creating a dependency.

### Declare the dependency

Explanation of declaring the dependency, the actual mechanism detail.

## Other Section

Unrelated content that should not be pulled in.
"""


def test_subsections_below_boundary_level_merge_into_the_parent_chunk():
    chunks = chunk_markdown(_NESTED_DOC, chunk_words=200, overlap_words=10)

    first_steps_chunks = [c for c in chunks if c.heading == "Dependencies > First Steps"]
    assert len(first_steps_chunks) == 1
    text = first_steps_chunks[0].text
    assert "Intro sentence explaining how the system works overall." in text
    assert "Explanation of creating a dependency." in text
    assert "Explanation of declaring the dependency" in text


def test_subheading_lines_stay_visible_inline_in_the_merged_text():
    chunks = chunk_markdown(_NESTED_DOC, chunk_words=200, overlap_words=10)

    first_steps_chunks = [c for c in chunks if c.heading == "Dependencies > First Steps"]
    text = first_steps_chunks[0].text
    assert "### Create a dependency" in text
    assert "### Declare the dependency" in text


def test_sections_at_or_above_boundary_level_still_split():
    chunks = chunk_markdown(_NESTED_DOC, chunk_words=200, overlap_words=10)

    other_chunks = [c for c in chunks if c.heading == "Dependencies > Other Section"]
    assert len(other_chunks) == 1
    assert "Unrelated content" in other_chunks[0].text
    assert "Explanation of declaring the dependency" not in other_chunks[0].text


def test_boundary_level_is_configurable():
    # boundary_level=3 should keep ### headings as their own hard split too
    chunks = chunk_markdown(_NESTED_DOC, chunk_words=200, overlap_words=10, boundary_level=3)

    headings = {c.heading for c in chunks}
    assert "Dependencies > First Steps > Create a dependency" in headings
    assert "Dependencies > First Steps > Declare the dependency" in headings
