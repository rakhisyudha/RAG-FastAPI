"""Command line data ingestion.

Usage:
    python -m scripts.ingest                       # ingest data/documents/*.txt and *.md
    python -m scripts.ingest --file path/to/doc.md
    python -m scripts.ingest --directory data/documents
"""

import argparse
import sys
from pathlib import Path

from app.services.pipeline import ingest_document

SUPPORTED_SUFFIXES = {".txt", ".md"}


def iter_documents(directory: Path):
    """Yield every supported file in ``directory`` in stable order."""
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument("--file", type=Path, help="A single document to ingest.")
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("data/documents"),
        help="Directory to scan for documents (default: data/documents).",
    )
    args = parser.parse_args()

    if args.file:
        paths = [args.file]
    elif args.directory.is_dir():
        paths = list(iter_documents(args.directory))
        if not paths:
            print(f"No .txt or .md files found in {args.directory}")
            return 1
    else:
        print(f"Directory not found: {args.directory}")
        return 1

    total_chunks = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        result = ingest_document(text=text, source=path.name)
        total_chunks += result.chunks_created
        print(
            f"[ok] {result.source:<28} {result.chunks_created:>3} chunks "
            f"({result.characters} chars)"
        )

    print(f"\nDone. {len(paths)} document(s), {total_chunks} chunks indexed.")
    print("Start the API with:  uvicorn app.main:app --reload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
