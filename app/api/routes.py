"""HTTP routes.

Routes stay thin: they validate input with the schemas, call a service, and
shape the response. All business logic lives in ``app/services`` so it can be
reused by scripts, workers or tests without going through HTTP.
"""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.schemas.rag import (
    DeleteDocumentResponse,
    DocumentListResponse,
    HealthResponse,
    IngestResponse,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
)
from app.services import pipeline
from app.services.pipeline import IngestResult
from app.services.vectorstore import get_collection

logger = logging.getLogger("rag.api")

router = APIRouter()

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def _ingest_response(result: IngestResult) -> IngestResponse:
    """Map a service result onto its public schema."""
    return IngestResponse(
        source=result.source,
        chunks_created=result.chunks_created,
        characters_processed=result.characters,
        chunk_ids=result.chunk_ids,
        ingested_at=result.ingested_at,
    )


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Report service status and what is currently indexed."""
    status_value = "ok"
    try:
        get_collection().count()
    except Exception:  # pragma: no cover - defensive
        logger.exception("vector store health check failed")
        status_value = "degraded"

    stats = pipeline.collection_stats()
    config = get_settings()

    return HealthResponse(
        status=status_value,
        app_name=config.app_name,
        version=config.app_version,
        embedding_model=config.embedding_model,
        llm_model=config.llm_model,
        chroma_collection=stats["collection"],
        documents_indexed=stats["documents"],
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "LLM unreachable"}},
    tags=["rag"],
)
def query(request: QueryRequest) -> QueryResponse:
    """Answer a question with retrieval-augmented generation."""
    try:
        return pipeline.answer_question(
            question=request.question,
            top_k=request.top_k,
            source_filter=request.source_filter,
            temperature=request.temperature,
            include_sources=request.include_sources,
        )
    except Exception as exc:
        logger.exception("query failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Generation failed: {exc}",
        ) from exc


@router.post(
    "/documents",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
def ingest_text(request: IngestTextRequest) -> IngestResponse:
    """Ingest a document supplied as raw text in the request body."""
    result = pipeline.ingest_document(
        text=request.text,
        source=request.source,
        metadata=request.metadata,
    )
    return _ingest_response(result)


@router.post(
    "/documents/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
async def upload_document(
    file: UploadFile = File(...),
) -> IngestResponse:
    """Ingest an uploaded ``.txt`` or ``.md`` file."""
    if not (file.filename or "").lower().endswith((".txt", ".md")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .txt and .md files are supported.",
        )

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds the 5 MB limit.",
        )

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be UTF-8 encoded text.",
        ) from exc

    result = pipeline.ingest_document(text=text, source=file.filename or "upload")
    return _ingest_response(result)


@router.get("/documents", response_model=DocumentListResponse, tags=["documents"])
def get_documents() -> DocumentListResponse:
    """List every document currently indexed."""
    documents = pipeline.list_indexed_documents()
    return DocumentListResponse(
        total_documents=len(documents),
        total_chunks=sum(item["chunks"] for item in documents),
        documents=documents,
    )


@router.delete(
    "/documents/{source}",
    response_model=DeleteDocumentResponse,
    tags=["documents"],
)
def remove_document(source: str) -> DeleteDocumentResponse:
    """Delete one document and all of its chunks."""
    deleted = pipeline.delete_document(source)
    if deleted == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No document named '{source}' was found.",
        )
    return DeleteDocumentResponse(source=source, chunks_deleted=deleted)
