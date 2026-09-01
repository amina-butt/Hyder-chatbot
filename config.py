"""
config.py
----------
Centralized configuration management for Hyder Assistant.

Loads all environment-driven settings using pydantic-settings, providing
type validation, sane defaults, and a single source of truth for the rest
of the application. Import `settings` anywhere configuration is needed:

    from config import settings
    print(settings.GEMINI_MODEL)
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings, populated from environment variables / .env file."""

    # --- Google Gemini (google-genai SDK) ---
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"
    GEMINI_TEMPERATURE: float = 0.3
    # Urdu script (and to a lesser extent Roman Urdu) tokenizes less
    # efficiently than English under most tokenizers — the same sentence
    # can cost noticeably more tokens. This was previously raised to 4096
    # because 2048 was occasionally tight for longer Urdu answers.
    # Lowered to 510 for faster streaming turnaround — NOTE: this
    # reintroduces the truncation risk the 4096 change existed to avoid,
    # especially for longer Urdu replies and multi-model comparisons.
    # rag_engine._response_was_truncated / _trim_to_last_complete_sentence
    # will kick in more often as a result; watch for cut-off replies and
    # raise this back up if that becomes noticeable.
    GEMINI_MAX_OUTPUT_TOKENS: int = 510
    # Per-HTTP-call timeout to Gemini, in seconds. Bounds a stalled/hanging
    # connection — without this, a call that never returns (rather than
    # erroring) can block a threadpool slot indefinitely.
    GEMINI_TIMEOUT_SECONDS: float = 30.0

    # --- Vector store (ChromaDB) ---
    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "hyder_bikes_kb"

    # --- Embeddings ---
    # NOTE: this model must produce the same embedding dimensionality as
    # whatever is already stored in CHROMA_DB_PATH. If you change this after
    # ingesting data, delete the CHROMA_DB_PATH directory and re-run
    # ingest.py — mixing embeddings from two different models in one
    # collection silently produces meaningless similarity scores, even if
    # the vector dimensions happen to match.
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # --- Knowledge base ingestion ---
    DATA_FILE_PATH: str = "data/hyder_bikes.txt"
    CHUNK_SIZE: int = 1500          # characters per chunk
    CHUNK_OVERLAP: int = 200        # overlap between consecutive chunks
    TOP_K_RESULTS: int = 4         # number of chunks retrieved per query

    # --- Conversation memory ---
    MAX_HISTORY_TURNS: int = 6     # sliding window size (user+assistant pairs)
    SESSION_TTL_HOURS: float = 6.0  # evict a session after this long with no activity
    MAX_SESSIONS: int = 10000       # hard cap on concurrent sessions held in memory

    # --- Logging ---
    LOG_DIR: str = "logs"
    LOG_FILE_NAME: str = "app.log"
    LOG_LEVEL: str = "INFO"
    LOG_MAX_BYTES: int = 5 * 1024 * 1024
    LOG_BACKUP_COUNT: int = 3

    # --- API auth ---
    # Shared secret your frontend must send as X-API-Key. Left as None by
    # default so local dev doesn't need one set; the auth dependency in
    # main.py refuses to start serving the protected endpoint if this is
    # unset in a non-development ENVIRONMENT (see main.py's startup check).
    API_KEY: str | None = None
    ENVIRONMENT: str = "development"  # "development" | "production"

    # --- Company / support info ---
    COMPANY_NAME: str = "Hyder Electric Bikes"
    HUMAN_HANDOFF_CONTACT: str = "0309 9432 432"
    HANDOFF_EMAIL: str = "support@hyderbikes.pk"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance (avoids re-parsing .env
    on every import)."""
    return Settings()


settings = get_settings()