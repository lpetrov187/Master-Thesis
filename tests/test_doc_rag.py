"""Documentation RAG: ingest + retrieval against the sample corpus, plus
(from the chunking rework) heading metadata and the relevance cutoff."""
from src.tools.doc_rag import ingest, retrieve


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

    results = retrieve("How does logging basicConfig work?", top_k=1, max_distance=None)

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

    results = retrieve("What's the airspeed velocity of an unladen swallow?", max_distance=None)

    assert len(results) == 3
