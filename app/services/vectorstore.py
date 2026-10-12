"""ChromaDB vector store.

ChromaDB stores the vectors plus the original chunk text and any metadata.
Because it persists to disk, the index survives a server restart.

We generate embeddings ourselves (sentence-transformers) and pass them in
explicitly, so the collection is created with cosine distance: ``distance`` is
0 for identical vectors and 2 for opposite ones, which we convert to a
similarity score with ``1 - distance``.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import chromadb
from chromadb.api.models.Collection import Collection

from app.core.config import get_settings
from app.services.chunking import TextChunk
from app.services.embeddings import embed_texts

logger = logging.getLogger("rag.vectorstore")


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk pulled back from the vector store."""

    id: str
    text: str
    score: float
    source: str | None
    chunk_index: int | None
    metadata: dict


_client: chromadb.ClientAPI | None = None
_collection: Collection | None = None


def get_client() -> chromadb.ClientAPI:
    """Return the cached persistent ChromaDB client."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
    return _client


def get_collection() -> Collection:
    """Return the cached document collection, creating it if needed.

    The collection is created with cosine distance so that ``distance`` is
    0 for identical vectors and 2 for opposite ones, which makes
    ``1 - distance`` a true cosine similarity. Because
    ``get_or_create_collection`` never changes the settings of an existing
    collection, a collection created earlier with a different distance metric
    is detected and rebuilt so the scores stay meaningful.
    """
    global _collection
    if _collection is None:
        settings = get_settings()
        client = get_client()
        metadata = {"hnsw:space": "cosine"}

        existing_names = {collection.name for collection in client.list_collections()}
        if settings.chroma_collection in existing_names:
            existing = client.get_collection(settings.chroma_collection)
            if existing.metadata != metadata:
                logger.warning(
                    "collection '%s' was built with %s distance; recreating it "
                    "with cosine so similarity scores stay meaningful",
                    settings.chroma_collection,
                    existing.metadata or "default L2",
                )
                client.delete_collection(settings.chroma_collection)

        _collection = client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata=metadata,
        )
    return _collection


def reset_client() -> None:
    """Drop cached handles (used by tests)."""
    global _client, _collection
    _client = None
    _collection = None


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def add_chunks(
    chunks: list[TextChunk],
    source: str,
    document_id: str | None = None,
) -> list[str]:
    """Embed and store chunks, returning their generated ids.

    Args:
        chunks: Chunks produced by :func:`app.services.chunking.chunk_text`.
        source: Document name; also used for ``where`` filtering later.
        document_id: Stable id grouping all chunks of one document, so the
            whole document can be replaced or deleted atomically.

    Returns:
        The ids created in the collection.
    """
    if not chunks:
        return []

    collection = get_collection()
    document_id = document_id or str(uuid.uuid4())

    ids = [f"{document_id}:{chunk.index}" for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            **chunk.metadata,
            "source": source,
            "document_id": document_id,
            "chunk_index": chunk.index,
            "characters": len(chunk.text),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        for chunk in chunks
    ]

    embeddings = embed_texts(documents)

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return ids


def delete_document(source: str) -> int:
    """Delete every chunk belonging to ``source``; returns how many were removed."""
    collection = get_collection()
    existing = collection.get(where={"source": source}, include=[])
    ids = existing.get("ids") or []
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def list_documents() -> list[dict]:
    """Summarise the stored chunks grouped by source document."""
    collection = get_collection()
    result = collection.get(include=["metadatas"])

    grouped: dict[str, dict] = {}
    for metadata in result.get("metadatas") or []:
        if not metadata:
            continue
        source = metadata.get("source", "unknown")
        entry = grouped.setdefault(
            source,
            {"source": source, "chunks": 0, "characters": 0, "metadata": {}},
        )
        entry["chunks"] += 1
        entry["characters"] += int(metadata.get("characters", 0))
        entry["ingested_at"] = metadata.get("ingested_at")
        entry["metadata"] = {
            key: value
            for key, value in metadata.items()
            if key not in {"source", "document_id", "chunk_index", "characters", "ingested_at"}
        }

    return sorted(grouped.values(), key=lambda item: item["source"])


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def similarity_search(
    query: str,
    top_k: int | None = None,
    source_filter: str | None = None,
) -> list[RetrievedChunk]:
    """Return the chunks most similar to ``query``.

    Args:
        query: The user question (embedded with the same model as the chunks).
        top_k: Number of chunks to return.
        source_filter: Restrict the search to one document.

    Returns:
        Chunks sorted from most to least similar.
    """
    settings = get_settings()
    limit = top_k or settings.retrieval_top_k

    collection = get_collection()
    if collection.count() == 0:
        return []

    where = {"source": source_filter} if source_filter else None

    query_embedding = embed_texts([query])[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=limit,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    ids = (result.get("ids") or [[]])[0]

    chunks: list[RetrievedChunk] = []
    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        metadata = metadata or {}
        chunks.append(
            RetrievedChunk(
                id=chunk_id,
                text=text or "",
                score=round(1.0 - float(distance), 4),
                source=metadata.get("source"),
                chunk_index=metadata.get("chunk_index"),
                metadata=metadata,
            )
        )

    chunks.sort(key=lambda item: item.score, reverse=True)
    return chunks


def count_chunks() -> int:
    """Total number of stored chunks."""
    return get_collection().count()
