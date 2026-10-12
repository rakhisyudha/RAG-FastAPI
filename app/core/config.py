"""Application settings.

Every value can be overridden with an environment variable or a local ``.env``
file, which keeps secrets (API keys) out of the source code.
"""

from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration object.

    A single :class:`Settings` instance is created lazily and cached, so any
    module can import :func:`get_settings` and receive the same object.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application ---
    app_name: str = "RAG FastAPI"
    app_version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["*"],
        description="Comma-separated list of allowed CORS origins, or '*'.",
    )

    # --- Vector store ---
    chroma_persist_directory: str = "./chroma_store"
    chroma_collection: str = "rag_documents"

    # --- Embeddings ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # --- Chunking ---
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- Retrieval ---
    retrieval_top_k: int = 4

    # --- LLM (OpenAI-compatible endpoint) ---
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "llama3.2"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 120.0

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, value: object) -> object:
        """Allow ``CORS_ORIGINS=http://a.com,http://b.com`` style values."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the cached :class:`Settings` instance."""
    return Settings()


settings = get_settings()


@lru_cache
def get_settings() -> Settings:
    """Return the cached :class:`Settings` instance."""
    return Settings()
