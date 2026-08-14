"""Documentation RAG tool: chunk + embed + store the doc corpus in Chroma,
and retrieve the most relevant chunks for a query.

Ingestion and retrieval share one embedding model instance (loaded lazily,
since constructing a SentenceTransformer is slow) and one persistent Chroma
collection on disk, so re-running ingest() picks up doc changes without
restarting anything. Chunking is markdown-aware (src/tools/chunking.py) -
each chunk carries its heading path and whether it's a code block, which
matters once the corpus is real documentation instead of 4 short files.
"""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_DIR, DOCS_DIR, EMBEDDING_MODEL
from src.tools.chunking import chunk_markdown

_COLLECTION_NAME = "docs"

# Cosine distance ranges roughly 0.0 (identical) to 2.0 (opposite); chunks
# beyond this are treated as "nothing relevant" rather than force-returned
# as evidence. Recalibrated after the FastAPI corpus import (2026-08-15):
# the original 0.75 was fit to a 4-doc corpus and turned out too loose at
# real scale - a bigger, topically broader corpus has *some* chunk that's
# moderately close to almost any query just from sheer size, so the
# "irrelevant" floor shifts closer to 0 as the corpus grows. Re-measured
# against the 85-file FastAPI corpus with 5 relevant + 5 irrelevant probe
# queries: relevant top-hits landed at 0.24-0.34, irrelevant ones at
# 0.72-0.80 - 0.5 sits centered in that gap with margin on both sides.
_DEFAULT_MAX_DISTANCE = 0.5

_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _reset_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """Recreate the collection from scratch, forcing cosine distance.

    A collection's distance space is fixed at creation time - clearing its
    contents isn't enough to change it, so this drops and recreates it.
    """
    existing_names = {c.name for c in client.list_collections()}
    if _COLLECTION_NAME in existing_names:
        client.delete_collection(_COLLECTION_NAME)
    return client.create_collection(_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def _load_docs() -> list[tuple[str, str]]:
    """Return (relative_path, raw_text) for every doc under DOCS_DIR,
    searched recursively so a real corpus can be organized into
    subdirectories (e.g. data/docs/fastapi/tutorial/...). The relative
    path (not just the bare filename) is used as the source id, since a
    real multi-section corpus can have same-named files in different
    subdirectories (e.g. multiple index.md)."""
    return [
        (str(path.relative_to(DOCS_DIR)), path.read_text(encoding="utf-8"))
        for path in sorted(Path(DOCS_DIR).rglob("*.md"))
    ]


def ingest(client: chromadb.ClientAPI | None = None) -> int:
    """Chunk, embed, and store every doc under DOCS_DIR. Returns chunk count."""
    client = client or _get_client()
    collection = _reset_collection(client)

    ids, documents, metadatas = [], [], []
    for source, text in _load_docs():
        for i, chunk in enumerate(chunk_markdown(text)):
            ids.append(f"{source}::{i}")
            documents.append(chunk.text)
            metadatas.append({"source": source, "chunk_index": i, "heading": chunk.heading, "is_code": chunk.is_code})

    if not documents:
        return 0

    embeddings = _get_embedder().encode(documents).tolist()
    collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    return len(documents)


def retrieve(
    query: str,
    top_k: int = 3,
    max_distance: float | None = _DEFAULT_MAX_DISTANCE,
    client: chromadb.ClientAPI | None = None,
) -> list[dict]:
    """Return up to `top_k` chunks relevant to `query`, closest first.

    Each result is {"text", "source", "heading", "is_code", "distance"}.
    Chunks farther than `max_distance` are dropped rather than force-
    returned - pass None to disable the cutoff and always get `top_k`.
    """
    client = client or _get_client()
    collection = client.get_or_create_collection(_COLLECTION_NAME)

    query_embedding = _get_embedder().encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    hits = [
        {
            "text": doc,
            "source": meta["source"],
            "heading": meta.get("heading", ""),
            "is_code": meta.get("is_code", False),
            "distance": distance,
        }
        for doc, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ]

    if max_distance is not None:
        hits = [h for h in hits if h["distance"] <= max_distance]
    return hits


if __name__ == "__main__":
    count = ingest()
    print(f"Ingested {count} chunks from {DOCS_DIR}\n")

    sample_queries = [
        "How do I configure connection pooling in SQLAlchemy?",
        "How do I reuse a requests Session?",
        "How do pytest fixture scopes work?",
        "What's the airspeed velocity of an unladen swallow?",  # expect: nothing relevant
    ]
    for query in sample_queries:
        print(f"Query: {query}")
        for hit in retrieve(query, top_k=3, max_distance=None):  # no cutoff, to see raw distances
            print(f"  [{hit['distance']:.3f}] {hit['source']} ({hit['heading']}): {hit['text'][:60]}...")
        print()
