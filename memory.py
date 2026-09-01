"""
memory.py
---------
Session-based sliding-window conversation memory.

Keeps the last N conversational turns per session_id in memory with automatic
TTL (Time-To-Live) and max-session LRU eviction to prevent memory leaks.

The eviction sweep is time-gated (see _SWEEP_INTERVAL_SECONDS) rather than
run on every call — a full scan over every session on every single
add_message/get_history would mean each chat turn blocks on an O(n) scan
under the shared lock, which turns into real contention once session count
grows. Sessions may therefore live up to TTL + sweep interval instead of
exactly TTL; that's a fine tradeoff for avoiding per-request lock contention.

IMPORTANT WORKER WARNING:
This is an in-memory store. If running Uvicorn with multiple workers (--workers > 1),
worker processes do NOT share this memory. Requests from the same session_id may hit
different workers and experience conversation amnesia. For multi-worker production,
migrate `SessionMemory` to Redis.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional

from config import settings
from logger import get_logger

logger = get_logger(__name__)

# How often the eviction sweep is allowed to run, regardless of how many
# add_message/get_history calls happen in between. Keeps the O(n) scan off
# the hot path for every chat turn.
_SWEEP_INTERVAL_SECONDS = 300  # 5 minutes


@dataclass
class Message:
    role: str          # "user" or "assistant"
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class _SessionEntry:
    history: Deque[Message]
    last_accessed: float


class SessionMemory:
    """Thread-safe sliding-window memory with TTL and capacity-based eviction."""

    def __init__(self, max_turns: Optional[int] = None) -> None:
        self._max_messages = 2 * (max_turns or settings.MAX_HISTORY_TURNS)
        self._ttl_seconds = settings.SESSION_TTL_HOURS * 3600
        self._max_sessions = settings.MAX_SESSIONS
        self._sessions: Dict[str, _SessionEntry] = {}
        self._lock = threading.Lock()
        self._last_sweep = 0.0

    def _evict_expired_and_overflow_locked(self) -> None:
        """Full eviction scan (must be called while holding self._lock)."""
        now = time.time()

        # 1. Evict sessions inactive beyond TTL
        expired_keys = [
            sid for sid, entry in self._sessions.items()
            if (now - entry.last_accessed) > self._ttl_seconds
        ]
        for sid in expired_keys:
            del self._sessions[sid]

        if expired_keys:
            logger.info(
                "Evicted %d expired sessions (TTL > %.1f hours)",
                len(expired_keys),
                settings.SESSION_TTL_HOURS,
            )

        # 2. If still exceeding MAX_SESSIONS cap, drop oldest-accessed sessions
        while len(self._sessions) > self._max_sessions:
            oldest_sid = min(self._sessions, key=lambda k: self._sessions[k].last_accessed)
            del self._sessions[oldest_sid]
            logger.warning(
                "Session memory cap exceeded (%d). Evicted oldest session: %s",
                self._max_sessions,
                oldest_sid,
            )

    def _maybe_evict_locked(self) -> None:
        """Runs the eviction scan at most once per _SWEEP_INTERVAL_SECONDS.
        Must be called while holding self._lock."""
        now = time.time()
        if now - self._last_sweep < _SWEEP_INTERVAL_SECONDS:
            return
        self._last_sweep = now
        self._evict_expired_and_overflow_locked()

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to a session's history and refresh timestamp."""
        with self._lock:
            self._maybe_evict_locked()
            now = time.time()
            if session_id not in self._sessions:
                entry = _SessionEntry(
                    history=deque(maxlen=self._max_messages),
                    last_accessed=now,
                )
                self._sessions[session_id] = entry
            else:
                entry = self._sessions[session_id]
                entry.last_accessed = now

            entry.history.append(Message(role=role, content=content))
            current_len = len(entry.history)

        logger.debug(
            "Memory updated (role=%s, history_len=%d)",
            role,
            current_len,
            extra={"session_id": session_id},
        )

    def get_history(self, session_id: str) -> List[Message]:
        """Return sliding-window history and update session last-accessed time."""
        with self._lock:
            self._maybe_evict_locked()
            entry = self._sessions.get(session_id)
            if not entry:
                return []
            entry.last_accessed = time.time()
            return list(entry.history)

    def get_history_as_text(self, session_id: str) -> str:
        """Render history as a simple transcript string."""
        history = self.get_history(session_id)
        if not history:
            return "(no prior conversation in this session)"
        lines = [f"{m.role.capitalize()}: {m.content}" for m in history]
        return "\n".join(lines)

    def clear_session(self, session_id: str) -> None:
        """Wipe a session's history."""
        with self._lock:
            self._sessions.pop(session_id, None)
        logger.info("Session memory cleared", extra={"session_id": session_id})

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions


# Singleton instance shared across the app
conversation_memory = SessionMemory()