"""API request/response models.

Schemas are the *contract* of the HTTP API: they validate incoming data and
guarantee the shape of outgoing data. Keep them separate from database models
and from internal service objects.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SourceChunk(BaseModel):
    """A retrieved chunk that was used to build the answer."""

    id: str = Field(description="Chunk id inside the vector store.")
    text: str = Field(description="The chunk content.")
    score: float = Field(description="Similarity score (higher is better, 0-1).")
    source: str | None = Field(default=None, description="Origin document name.")
    chunk_index: int | None = Field(
        default=None, description="Position of the chunk inside its document."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


class IngestTextRequest(BaseModel):
    """Ingest a raw document supplied directly in the request body."""

    text: str = Field(
        min_length=1,
        max_length=2_000_000,
        description="Raw document content.",
    )
    source: str = Field(
        default="inline",
        max_length=255,
        description="Human readable name for the document.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra metadata stored alongside every chunk.",
    )


class IngestResponse(BaseModel):
    """Result of an ingestion run."""

    model_config = ConfigDict(protected_namespaces=())

    source: str
    chunks_created: int
    characters_processed: int
    chunk_ids: list[str] = Field(default_factory=list)
    ingested_at: datetime


class DocumentSummary(BaseModel):
    """A document known to the vector store."""

    source: str
    chunks: int
    characters: int
    ingested_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentListResponse(BaseModel):
    """List of documents in the vector store."""

    total_documents: int
    total_chunks: int
    documents: list[DocumentSummary]


class DeleteDocumentResponse(BaseModel):
    """Result of deleting a document."""

    source: str
    chunks_deleted: int


# ---------------------------------------------------------------------------
# Query / generation
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """A question to answer with retrieval-augmented generation."""

    question: str = Field(
        min_length=1,
        max_length=4000,
        description="The user question.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Override how many chunks to retrieve.",
    )
    source_filter: str | None = Field(
        default=None,
        max_length=255,
        description="Restrict retrieval to a single document.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Override the LLM sampling temperature.",
    )
    include_sources: bool = Field(
        default=True,
        description="Return the retrieved chunks in the response.",
    )


class QueryResponse(BaseModel):
    """The generated answer plus the context it was grounded in."""

    question: str
    answer: str
    model: str
    sources: list[SourceChunk] = Field(default_factory=list)
    context_used: str | None = Field(
        default=None,
        description="The prompt context, useful for debugging retrieval quality.",
    )
    latency_ms: float = Field(description="End-to-end wall time in milliseconds.")


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Service health report."""

    status: str
    app_name: str
    version: str
    embedding_model: str
    llm_model: str
    chroma_collection: str
    documents_indexed: int


class ErrorResponse(BaseModel):
    """Uniform error payload."""

    detail: str
