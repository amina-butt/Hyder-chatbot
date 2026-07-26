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
    GEMINI_MODEL: str = "gemini-3.6-flash"
    GEMINI_TEMPERATURE: float = 0.3
    GEMINI_MAX_OUTPUT_TOKENS: int = 1024

    # --- Vector store (ChromaDB) ---
    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "hyder_bikes_kb"

    # --- Embeddings ---
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # --- Knowledge base ingestion ---
    DATA_FILE_PATH: str = "data/hyder_bikes.txt"
    CHUNK_SIZE: int = 500          # characters per chunk
    CHUNK_OVERLAP: int = 80        # overlap between consecutive chunks
    TOP_K_RESULTS: int = 4         # number of chunks retrieved per query

    # --- Conversation memory ---
    MAX_HISTORY_TURNS: int = 6     # sliding window size (user+assistant pairs)

    # --- Logging ---
    LOG_DIR: str = "logs"
    LOG_FILE_NAME: str = "app.log"
    LOG_LEVEL: str = "INFO"
    LOG_MAX_BYTES: int = 5 * 1024 * 1024
    LOG_BACKUP_COUNT: int = 3

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
