"""
logger.py
---------
Structured logging setup for Hyder Assistant.

Provides a single `get_logger(name)` factory that returns a logger writing
to both the console (human-readable) and a rotating log file (logs/app.log),
so logs survive restarts and don't grow unbounded. Every log call can pass
`extra={"session_id": ...}` to correlate log lines with a chat session.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import List

from config import settings

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "session=%(session_id)s | %(message)s"
)


class _SessionContextFilter(logging.Filter):
    """Injects a default session_id field so the formatter never raises a
    KeyError when a log call doesn't pass session-specific `extra` data."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "session_id"):
            record.session_id = "-"
        return True


def _build_handlers() -> List[logging.Handler]:
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    log_path = os.path.join(settings.LOG_DIR, settings.LOG_FILE_NAME)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=settings.LOG_MAX_BYTES,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(_LOG_FORMAT)
    for handler in (file_handler, console_handler):
        handler.setFormatter(formatter)
        handler.addFilter(_SessionContextFilter())

    return [file_handler, console_handler]


_HANDLERS = _build_handlers()


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name.

    Usage:
        logger = get_logger(__name__)
        logger.info("message", extra={"session_id": session_id})
    """
    logger = logging.getLogger(name)
    logger.setLevel(settings.LOG_LEVEL.upper())

    if not logger.handlers:  # avoid duplicate handlers if module re-imported
        for handler in _HANDLERS:
            logger.addHandler(handler)
        logger.propagate = False

    return logger
