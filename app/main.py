"""FastAPI application entrypoint.

``create_app`` is the factory pattern: it builds the app object so tests can
create an isolated instance with different settings, and the ASGI server gets
a fully configured app from ``main:app``.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from app.api.routes import router
from app.core.config import get_settings
from app.services.embeddings import get_embedding_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up expensive resources before the first request arrives.

    Loading the embedding model takes a few seconds, so doing it at startup
    means the first user request is not stuck waiting for it.
    """
    logger.info("loading embedding model...")
    get_embedding_model()
    logger.info("embedding model ready")
    yield
    logger.info("shutting down")


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Retrieval-Augmented Generation API: ingest documents, then ask "
            "questions answered only from those documents."
        ),
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix=settings.api_prefix)

    @app.get("/", tags=["system"], include_in_schema=False)
    def root() -> dict:
        return {
            "message": f"Welcome to the {settings.app_name} API",
            "docs": "/docs",
            "health": f"{settings.api_prefix}/health",
        }

    return app


app = create_app()
