"""Document chunking.

Chunking splits raw text into small, semantically coherent pieces so that
retrieval can return only the paragraphs relevant to a question. A good chunk
is small enough to embed precisely but large enough to carry meaning on its
own, which is why neighbouring chunks overlap slightly.
"""

import re
from dataclasses import dataclass, field

from app.core.config import get_settings

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


@dataclass(slots=True)
class TextChunk:
    """A single piece of a larger document."""

    text: str
    index: int
    metadata: dict = field(default_factory=dict)


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping their trailing whitespace."""
    return _SENTENCE_BOUNDARY.split(text)


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    source: str = "inline",
    base_metadata: dict | None = None,
) -> list[TextChunk]:
    """Recursively split ``text`` into overlapping chunks.

    The splitter tries, in order, natural boundaries (paragraph, newline,
    sentence, whitespace) before falling back to a hard character cut. This
    keeps sentences intact instead of slicing mid-word.

    Args:
        text: Raw document content.
        chunk_size: Target maximum number of characters per chunk.
        chunk_overlap: Characters repeated at the end of a chunk and the start
            of the next one, to avoid losing context at the seam.
        source: Document identifier stored in chunk metadata.
        base_metadata: Extra metadata copied onto every chunk.

    Returns:
        A list of :class:`TextChunk` in document order.
    """
    settings = get_settings()
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    overlap = max(0, min(overlap, size // 2))

    cleaned = _normalise(text)
    if not cleaned:
        return []

    separators = ["\n\n", "\n", ". ", " ", ""]

    pieces = _recursive_split(cleaned, size, separators)
    merged = _merge_with_overlap(pieces, size, overlap)

    base = dict(base_metadata or {})
    base["source"] = source

    return [
        TextChunk(text=piece, index=i, metadata=dict(base))
        for i, piece in enumerate(merged)
        if piece.strip()
    ]


def _normalise(text: str) -> str:
    """Collapse noisy whitespace so it does not distort character counts."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _recursive_split(text: str, size: int, separators: list[str]) -> list[str]:
    """Split ``text`` using the first separator that produces small units."""
    if len(text) <= size:
        return [text] if text.strip() else []

    separator = separators[0]
    remaining_separators = separators[1:]

    if separator == "":
        # Last resort: hard character cut.
        return [text[i : i + size] for i in range(0, len(text), size)]

    parts = [p for p in text.split(separator) if p.strip()]
    pieces: list[str] = []

    for part in parts:
        if len(part) <= size:
            pieces.append(part)
        elif remaining_separators:
            pieces.extend(_recursive_split(part, size, remaining_separators))
        else:
            pieces.extend(part[i : i + size] for i in range(0, len(part), size))

    return pieces


def _merge_with_overlap(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Grow chunks back up to ``size`` characters, carrying ``overlap`` tail."""
    if not pieces:
        return []

    chunks: list[str] = []
    current = ""

    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece

        if len(candidate) <= size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            if overlap > 0:
                current = f"{_snap_to_boundary(current[-overlap:])} {piece}".strip()
            else:
                current = piece
        else:
            chunks.append(piece.strip())
            current = ""

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _snap_to_boundary(tail: str) -> str:
    """Trim an overlap tail so it starts at a sentence or word boundary.

    Without this, the repeated tail would begin mid-word (``"les. Plain
    values"``), which produces a chunk that reads as broken text.
    """
    if not tail:
        return ""

    sentence_cut = _SENTENCE_BOUNDARY.search(tail)
    if sentence_cut:
        return tail[sentence_cut.start() :].strip()

    whitespace_cut = tail.rfind(" ")
    if whitespace_cut > 0:
        return tail[whitespace_cut + 1 :].strip()

    return tail.strip()
