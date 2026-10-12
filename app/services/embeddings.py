"""Embedding model wrapper.

An embedding model turns text into a dense vector of numbers. Texts that mean
similar things end up with vectors that point in similar directions, which is
what makes similarity search possible.

The model is loaded once and cached as a module-level singleton: loading a
transformer takes seconds and a lot of RAM, so doing it per request would be
very slow.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the embedding model."""
    settings = get_settings()
    return SentenceTransformer(
        settings.embedding_model,
        device=settings.embedding_device,
    )


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed a batch of texts into vectors.

    Args:
        texts: Documents or queries to embed.
        batch_size: How many texts to push through the model at once.

    Returns:
        A list of vectors, one per input text, in the same order.
    """
    if not texts:
        return []

    model = get_embedding_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype="float32").tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]


def embedding_dimension() -> int:
    """Return the vector size produced by the configured model."""
    return int(get_embedding_model().get_sentence_embedding_dimension())
