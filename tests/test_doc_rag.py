"""Documentation RAG: ingest + retrieval against the sample corpus, plus
(from the chunking rework) heading metadata and the relevance cutoff."""
from src.tools.doc_rag import _expand_with_siblings, ingest, retrieve


class _FakeCollection:
    """Stands in for a chromadb Collection's .get(where=...) - lets the
    expansion logic be tested in complete isolation, no embeddings/model
    calls at all."""

    def __init__(self, chunks):
        # chunks: list of (id, text, source, heading, is_code)
        self._chunks = chunks

    def get(self, where):
        source = where["$and"][0]["source"]["$eq"]
        heading = where["$and"][1]["heading"]["$eq"]
        matches = [c for c in self._chunks if c[2] == source and c[3] == heading]
        return {
            "ids": [c[0] for c in matches],
            "documents": [c[1] for c in matches],
            "metadatas": [{"source": c[2], "heading": c[3], "is_code": c[4]} for c in matches],
        }


def _hit(chunk_id, text, source, heading, distance=0.2):
    return {
        "id": chunk_id,
        "text": text,
        "source": source,
        "heading": heading,
        "is_code": False,
        "distance": distance,
        "expanded": False,
    }


def test_ingest_populates_the_collection():
    count = ingest()
    assert count > 0


def test_retrieve_surfaces_the_relevant_doc():
    ingest()

    results = retrieve("How do I configure connection pooling in SQLAlchemy?", top_k=2)

    assert results
    assert results[0]["source"] == "sqlalchemy.md"


def test_retrieve_respects_top_k():
    ingest()

    results = retrieve(
        "How does logging basicConfig work?", top_k=1, max_distance=None, expand_same_section=False
    )

    assert len(results) == 1


def test_retrieve_hits_carry_heading_and_is_code_metadata():
    ingest()

    results = retrieve("How do I configure connection pooling in SQLAlchemy?", top_k=3)

    assert results
    assert all("heading" in hit for hit in results)
    assert all("is_code" in hit for hit in results)
    assert any("Connection Pooling" in hit["heading"] for hit in results)


def test_retrieve_filters_out_irrelevant_hits_by_default():
    ingest()

    results = retrieve("What's the airspeed velocity of an unladen swallow?")

    assert results == []


def test_retrieve_max_distance_none_disables_the_cutoff():
    ingest()

    results = retrieve(
        "What's the airspeed velocity of an unladen swallow?",
        top_k=3,
        max_distance=None,
        expand_same_section=False,
    )

    assert len(results) == 3


def test_retrieve_expands_same_section_by_default():
    # sqlalchemy.md has a single top-level heading and no ### subsections,
    # so every one of its chunks shares the same heading tag - a real,
    # natural case where a query hitting one of its chunks should expand
    # to pull in the rest of the doc's chunks too.
    ingest()

    results = retrieve("How do I configure connection pooling in SQLAlchemy?", top_k=1)

    assert len(results) > 1
    assert all(r["source"] == "sqlalchemy.md" for r in results)
    assert sum(1 for r in results if r["expanded"]) >= 1
    assert sum(1 for r in results if not r["expanded"]) == 1


def test_expand_with_siblings_pulls_in_matching_heading_only():
    collection = _FakeCollection(
        [
            ("a::0", "text0", "a.md", "Dependencies", False),
            ("a::1", "text1", "a.md", "Dependencies", False),
            ("a::2", "text2", "a.md", "Other", False),
        ]
    )
    hits = [_hit("a::0", "text0", "a.md", "Dependencies")]

    expanded = _expand_with_siblings(hits, collection, max_total=10)

    ids = {h["id"] for h in expanded}
    assert ids == {"a::0", "a::1"}


def test_expand_with_siblings_respects_max_total():
    collection = _FakeCollection([(f"a::{i}", f"text{i}", "a.md", "Dependencies", False) for i in range(6)])
    hits = [_hit("a::0", "text0", "a.md", "Dependencies")]

    expanded = _expand_with_siblings(hits, collection, max_total=3)

    assert len(expanded) == 3


def test_expand_with_siblings_marks_new_chunks_as_expanded_and_inherits_distance():
    collection = _FakeCollection(
        [
            ("a::0", "text0", "a.md", "Dependencies", False),
            ("a::1", "text1", "a.md", "Dependencies", False),
        ]
    )
    hits = [_hit("a::0", "text0", "a.md", "Dependencies", distance=0.31)]

    expanded = _expand_with_siblings(hits, collection, max_total=10)

    original = next(h for h in expanded if h["id"] == "a::0")
    sibling = next(h for h in expanded if h["id"] == "a::1")
    assert original["expanded"] is False
    assert sibling["expanded"] is True
    assert sibling["distance"] == 0.31


def test_expand_with_siblings_does_not_duplicate_across_multiple_hits():
    collection = _FakeCollection(
        [
            ("a::0", "text0", "a.md", "Dependencies", False),
            ("a::1", "text1", "a.md", "Dependencies", False),
        ]
    )
    # both original hits already share the same siblings - shouldn't double up
    hits = [_hit("a::0", "text0", "a.md", "Dependencies"), _hit("a::1", "text1", "a.md", "Dependencies")]

    expanded = _expand_with_siblings(hits, collection, max_total=10)

    assert len(expanded) == 2
