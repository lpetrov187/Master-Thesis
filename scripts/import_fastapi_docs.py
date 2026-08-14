"""One-time import: pull FastAPI's tutorial + advanced docs (Markdown
source) and clean them up for ingestion into the doc_rag corpus.

FastAPI's docs use MkDocs Material conventions that plain markdown parsing
doesn't understand:
  - `{* path/to/file.py *}` snippet-include directives - the actual code
    examples live in separate .py files under docs_src/, not inline in the
    .md file. Left unresolved, the most valuable content (working code)
    would be entirely missing from the corpus.
  - `///tip ... ///` admonition blocks (Material's callout syntax).
  - Raw styled HTML in simulated terminal output (`<div class="termy">`,
    `<font color=...>`, etc.) and `{ #anchor-id }` suffixes on headings.

This resolves includes by inlining the referenced .py file as a real code
fence, strips the admonition markers and heading-id suffixes, and strips
HTML tags (only from the original prose - never from injected code, since
tags are stripped before includes are resolved).

Usage:
    1. Sparse-clone FastAPI's docs + docs_src (Markdown source only):
       git clone --depth 1 --filter=blob:none --sparse \
           https://github.com/fastapi/fastapi.git fastapi-src
       cd fastapi-src
       git sparse-checkout set docs/en/docs/tutorial docs/en/docs/advanced docs_src

    2. python scripts/import_fastapi_docs.py <path-to-fastapi-src> data/docs/fastapi [--limit N]
"""
import argparse
import re
from pathlib import Path

# The path is the first token; anything after it (hl[...], ln[...], or any
# combination/future variant) is a display modifier we don't need, matched
# generically as extra whitespace-separated tokens rather than enumerated -
# a naive hl[...]-only pattern silently failed to match ln[...] instances.
_INCLUDE_RE = re.compile(r"\{\*\s*(\S+)(?:\s+\S+)*?\s*\*\}")
_ADMONITION_RE = re.compile(r"^///.*$\n?", re.MULTILINE)
_HEADING_ID_RE = re.compile(r"\s*\{\s*#[\w-]+\s*\}\s*$", re.MULTILINE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _resolve_includes(text: str, docs_en_root: Path) -> str:
    """Inline each {* path *} reference. MkDocs resolves these paths
    relative to docs/en/ (the language root), NOT relative to the
    individual .md file's own directory - e.g. a file at
    docs/en/docs/advanced/x.md referencing "../../docs_src/y.py" actually
    means docs/en/../../docs_src/y.py, not docs/en/docs/advanced/../../docs_src/y.py.
    Verified by checking where the referenced files actually exist before
    fixing this, not assumed."""

    def replace(match: re.Match) -> str:
        code_path = (docs_en_root / match.group(1)).resolve()
        if not code_path.exists():
            return ""  # broken/renamed reference - leave a gap, don't crash the whole import
        return f"```python\n{code_path.read_text(encoding='utf-8')}\n```"

    return _INCLUDE_RE.sub(replace, text)


def clean_doc(text: str, docs_en_root: Path) -> str:
    """Apply all cleaning steps, in an order that keeps injected code
    untouched by the HTML/markup stripping meant for the original prose."""
    text = _HTML_TAG_RE.sub("", text)
    text = _HEADING_ID_RE.sub("", text)
    text = _resolve_includes(text, docs_en_root)
    text = _ADMONITION_RE.sub("", text)
    return text


def import_docs(fastapi_src: Path, output_dir: Path, limit: int | None = None) -> list[Path]:
    docs_en_root = fastapi_src / "docs" / "en"
    docs_root = docs_en_root / "docs"
    md_paths = sorted(
        p for section in ("tutorial", "advanced") for p in (docs_root / section).rglob("*.md")
    )
    if limit is not None:
        md_paths = md_paths[:limit]

    written = []
    for md_path in md_paths:
        cleaned = clean_doc(md_path.read_text(encoding="utf-8"), docs_en_root)
        out_path = output_dir / md_path.relative_to(docs_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(cleaned, encoding="utf-8")
        written.append(out_path)
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fastapi_src", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    paths = import_docs(args.fastapi_src, args.output_dir, args.limit)
    print(f"Imported {len(paths)} files to {args.output_dir}")
    for p in paths:
        print(f"  {p}")
