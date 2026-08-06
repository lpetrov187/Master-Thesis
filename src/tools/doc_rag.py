"""Documentation RAG tool: chunk + embed + store the doc corpus in Chroma,
and retrieve the most relevant chunks for a query.

Ingestion and retrieval share one embedding model instance (loaded lazily,
since constructing a SentenceTransformer is slow) and one persistent Chroma
collection on disk, so re-running ingest() picks up doc changes without
restarting anything.
"""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_DIR, DOCS_DIR, EMBEDDING_MODEL

_COLLECTION_NAME = "docs"
_CHUNK_WORDS = 120
_CHUNK_OVERLAP_WORDS = 20

_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _chunk_text(text: str, chunk_words: int = _CHUNK_WORDS, overlap_words: int = _CHUNK_OVERLAP_WORDS) -> list[str]:
    """Split `text` into overlapping word-count windows."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_words
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap_words
    return chunks


def _load_docs() -> list[tuple[str, str]]:
    """Return (filename, raw_text) for every doc under DOCS_DIR."""
    return [(path.name, path.read_text(encoding="utf-8")) for path in sorted(Path(DOCS_DIR).glob("*.md"))]


def ingest(client: chromadb.ClientAPI | None = None) -> int:
    """Chunk, embed, and store every doc under DOCS_DIR. Returns chunk count.

    Clears the collection first so re-running ingest reflects the current
    contents of DOCS_DIR exactly, including removed docs/chunks.
    """
    client = client or _get_client()
    collection = client.get_or_create_collection(_COLLECTION_NAME)

    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    ids, documents, metadatas = [], [], []
    for source, text in _load_docs():
        for i, chunk in enumerate(_chunk_text(text)):
            ids.append(f"{source}::{i}")
            documents.append(chunk)
            metadatas.append({"source": source, "chunk_index": i})

    if not documents:
        return 0

    embeddings = _get_embedder().encode(documents).tolist()
    collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    return len(documents)


def retrieve(query: str, top_k: int = 3, client: chromadb.ClientAPI | None = None) -> list[dict]:
    """Return the `top_k` chunks most relevant to `query`.

    Each result is {"text": str, "source": str, "distance": float}, sorted
    by ascending distance (closest match first).
    """
    client = client or _get_client()
    collection = client.get_or_create_collection(_COLLECTION_NAME)

    query_embedding = _get_embedder().encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    return [
        {"text": doc, "source": meta["source"], "distance": distance}
        for doc, meta, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


if __name__ == "__main__":
    count = ingest()
    print(f"Ingested {count} chunks from {DOCS_DIR}\n")

    sample_queries = [
        "How do I configure connection pooling in SQLAlchemy?",
        "How do I reuse a requests Session?",
        "How do pytest fixture scopes work?",
    ]
    for query in sample_queries:
        print(f"Query: {query}")
        for hit in retrieve(query, top_k=2):
            print(f"  [{hit['distance']:.3f}] {hit['source']}: {hit['text'][:80]}...")
        print()
