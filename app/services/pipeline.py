"""The RAG pipeline: chunk -> embed -> store -> retrieve -> generate."""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.core.config import get_settings
from app.schemas.rag import QueryResponse, SourceChunk
from app.services import llm
from app.services.chunking import chunk_text
from app.services.vectorstore import (
    RetrievedChunk,
    add_chunks,
    count_chunks,
    delete_document,
    list_documents,
    reset_client,
    similarity_search,
)


@dataclass(slots=True)
class IngestResult:
    """Outcome of ingesting one document."""

    source: str
    chunks_created: int
    characters: int
    chunk_ids: list[str] = field(default_factory=list)
    ingested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class QueryResult:
    """Outcome of one retrieval-augmented generation run."""

    answer: str
    question: str
    model: str
    sources: list[RetrievedChunk]
    context: str
    latency_ms: float


def ingest_document(
    text: str,
    source: str,
    metadata: dict | None = None,
    replace: bool = True,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> IngestResult:
    """Index a document end to end.

    Args:
        text: Raw document content.
        source: Document name.
        metadata: Extra metadata attached to every chunk.
        replace: Delete any earlier chunks for the same source first, so
            re-ingesting a file does not create duplicates.
        chunk_size: Override the default target chunk size in characters.
        chunk_overlap: Override the default overlap between adjacent chunks.

    Returns:
        An :class:`IngestResult`.
    """
    if replace:
        delete_document(source)

    chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source=source,
        base_metadata=metadata,
    )
    chunk_ids = add_chunks(chunks, source=source)

    return IngestResult(
        source=source,
        chunks_created=len(chunk_ids),
        characters=len(text),
        chunk_ids=chunk_ids,
    )


def answer_question(
    question: str,
    top_k: int | None = None,
    source_filter: str | None = None,
    temperature: float | None = None,
    include_sources: bool = True,
) -> QueryResponse:
    """Answer a question using retrieval-augmented generation.

    Steps:
        1. Embed the question and search the vector store for close chunks.
        2. Use those chunks as the grounding context for the LLM.
        3. Return the answer with the sources it was built from.
    """
    started = time.perf_counter()

    chunks = similarity_search(
        query=question,
        top_k=top_k,
        source_filter=source_filter,
    )

    context_blocks = [chunk.text for chunk in chunks]
    context_used = llm.build_prompt(question, context_blocks) if include_sources else None

    answer, model = llm.generate(
        question=question,
        context_blocks=context_blocks,
        temperature=temperature,
    )

    elapsed_ms = (time.perf_counter() - started) * 1000

    return QueryResponse(
        question=question,
        answer=answer,
        model=model,
        sources=[_to_schema(chunk) for chunk in chunks] if include_sources else [],
        context_used=context_used,
        latency_ms=round(elapsed_ms, 2),
    )


def list_indexed_documents() -> list[dict]:
    """Documents currently stored in the vector database."""
    return list_documents()


def reset_store() -> None:
    """Drop cached ChromaDB handles so new settings take effect."""
    reset_client()


def build_prompt(question: str, context_blocks: list[str]) -> str:
    """Format the grounding context and question into a single prompt."""
    return llm.build_prompt(question, context_blocks)


def count_stored_chunks() -> int:
    """Total number of chunks in the vector database."""
    return count_chunks()


def collection_stats() -> dict:
    """Summary numbers for the health endpoint."""
    settings = get_settings()
    documents = list_documents()
    return {
        "documents": len(documents),
        "chunks": sum(item["chunks"] for item in documents),
        "collection": settings.chroma_collection,
    }


def _to_schema(chunk: RetrievedChunk) -> SourceChunk:
    """Convert a store result into its public schema, dropping noisy fields."""
    hidden = {"source", "document_id", "chunk_index", "characters", "ingested_at"}
    return SourceChunk(
        id=chunk.id,
        text=chunk.text,
        score=chunk.score,
        source=chunk.source,
        chunk_index=chunk.chunk_index,
        metadata={k: v for k, v in (chunk.metadata or {}).items() if k not in hidden},
    )
