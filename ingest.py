"""
ingest.py
---------
One-time (or periodically re-run) ingestion pipeline that:
  1. Reads the raw knowledge base text file (data/hyder_bikes.txt)
  2. Splits it into overlapping chunks
  3. Embeds each chunk with a local sentence-transformers model
  4. Upserts the chunks + embeddings into a persistent ChromaDB collection

Run directly:
    python ingest.py

Re-running this script is safe: chunk IDs are content hashes, so unchanged
chunks are simply re-upserted (no duplicates) and edited/new chunks are
picked up automatically.
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

from config import settings
from logger import get_logger

logger = get_logger(__name__)

_INGEST_SESSION = "ingest"  # pseudo session_id for log correlation


def load_raw_text(path: str) -> str:
    """Read the knowledge base file from disk."""
    if not os.path.exists(path):
        logger.error("Data file not found at %s", path, extra={"session_id": _INGEST_SESSION})
        raise FileNotFoundError(
            f"Knowledge base file not found: {path}. "
            "Create it or update DATA_FILE_PATH in your .env."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks.

    A simple, dependency-free chunker: the knowledge base file uses blank
    lines to separate logical entries (a model's spec sheet, an FAQ, etc).
    We split on those first so a whole entry stays together whenever
    possible, then further split any oversized entry into overlapping
    fixed-size character windows.
    """
    entries = [e.strip() for e in text.split("\n\n") if e.strip()]
    chunks: List[str] = []

    for entry in entries:
        if len(entry) <= chunk_size:
            chunks.append(entry)
            continue

        start = 0
        while start < len(entry):
            end = min(start + chunk_size, len(entry))
            chunks.append(entry[start:end])
            if end == len(entry):
                break
            start = end - overlap  # step forward, keeping overlap

    logger.info("Text split into %d chunks", len(chunks), extra={"session_id": _INGEST_SESSION})
    return chunks


def _chunk_id(chunk: str) -> str:
    """Deterministic ID (content hash) so re-running ingestion updates
    existing entries rather than duplicating them."""
    return hashlib.sha256(chunk.encode("utf-8")).hexdigest()[:16]


def build_vector_store(chunks: List[str]) -> None:
    """Embed chunks and upsert them into a persistent ChromaDB collection."""
    logger.info(
        "Loading embedding model '%s'...",
        settings.EMBEDDING_MODEL_NAME,
        extra={"session_id": _INGEST_SESSION},
    )
    model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

    logger.info("Encoding %d chunks...", len(chunks), extra={"session_id": _INGEST_SESSION})
    embeddings = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)

    client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [_chunk_id(c) for c in chunks]
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[{"source": settings.DATA_FILE_PATH} for _ in chunks],
    )

    logger.info(
        "Ingestion complete. Collection '%s' now has %d items.",
        settings.CHROMA_COLLECTION_NAME,
        collection.count(),
        extra={"session_id": _INGEST_SESSION},
    )


def main() -> None:
    logger.info("Starting knowledge base ingestion...", extra={"session_id": _INGEST_SESSION})
    raw_text = load_raw_text(settings.DATA_FILE_PATH)
    chunks = chunk_text(raw_text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    if not chunks:
        logger.error(
            "No chunks produced from data file; aborting.",
            extra={"session_id": _INGEST_SESSION},
        )
        sys.exit(1)
    build_vector_store(chunks)


if __name__ == "__main__":
    main()
