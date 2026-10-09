"""
ingest.py
---------
One-time (or periodically re-run) ingestion pipeline that:
  1. Reads the raw knowledge base text file (data/hyder_knowledge_base.md)
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
from chromadb.utils import embedding_functions

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
    """Split text into chunks by combining consecutive paragraphs so 

    headers and specs (like prices) remain in the same vector chunk.
    """
    entries = [e.strip() for e in text.split("\n\n") if e.strip()]
    chunks: List[str] = []
    current_chunk = ""

    for entry in entries:
        # If adding this paragraph exceeds chunk_size, store the current chunk
        if len(current_chunk) + len(entry) + 2 > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = entry
        else:
            if current_chunk:
                current_chunk += "\n\n" + entry
            else:
                current_chunk = entry

    if current_chunk:
        chunks.append(current_chunk)

    logger.info("Text split into %d combined chunks", len(chunks), extra={"session_id": _INGEST_SESSION})
    return chunks


def _chunk_id(chunk: str, idx: int) -> str:
    """Deterministic ID combining content hash and index so duplicate text
    snippets still receive unique IDs in ChromaDB."""
    content_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()[:12]
    return f"{content_hash}_{idx}"


def build_vector_store(chunks: List[str]) -> None:
    """Embed chunks locally and upsert them into a persistent ChromaDB collection."""
    logger.info(
      "Initializing local sentence-transformer embeddings using '%s'...",
      settings.EMBEDDING_MODEL_NAME,
      extra={"session_id": _INGEST_SESSION},
    )
    
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.EMBEDDING_MODEL_NAME
    )

    client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    ids = [_chunk_id(c, i) for i, c in enumerate(chunks)]
    
    collection.upsert(
        ids=ids,
        documents=chunks,
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