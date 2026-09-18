import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional

from config import settings
from logger import get_logger

logger = get_logger(__name__)

_SWEEP_INTERVAL_SECONDS = 300


@dataclass
class Message:
    role: str
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class _SessionEntry:
    history: Deque[Message]
    last_accessed: float
    pending_question: Optional[str] = None


class SessionMemory:
    def __init__(self, max_turns: Optional[int] = None) -> None:
        self._max_messages = 2 * (max_turns or settings.MAX_HISTORY_TURNS)
        self._ttl_seconds = settings.SESSION_TTL_HOURS * 3600
        self._max_sessions = settings.MAX_SESSIONS
        self._sessions: Dict[str, _SessionEntry] = {}
        self._lock = threading.Lock()
        self._last_sweep = 0.0

    def _evict_expired_and_overflow_locked(self) -> None:
        now = time.time()

        # --- Evict Expired Sessions ---
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

        # --- Evict Overflow (LRU) ---
        while len(self._sessions) > self._max_sessions:
            oldest_sid = min(self._sessions, key=lambda k: self._sessions[k].last_accessed)
            del self._sessions[oldest_sid]
            logger.warning(
                "Session memory cap exceeded (%d). Evicted oldest session: %s",
                self._max_sessions,
                oldest_sid,
            )

    def _maybe_evict_locked(self) -> None:
        now = time.time()
        if now - self._last_sweep < _SWEEP_INTERVAL_SECONDS:
            return
        self._last_sweep = now
        self._evict_expired_and_overflow_locked()

    def add_message(self, session_id: str, role: str, content: str) -> None:
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
        with self._lock:
            self._maybe_evict_locked()
            entry = self._sessions.get(session_id)
            if not entry:
                return []
            entry.last_accessed = time.time()
            return list(entry.history)

    def get_history_as_text(self, session_id: str) -> str:
        history = self.get_history(session_id)
        if not history:
            return "(no prior conversation in this session)"
        lines = [f"{m.role.capitalize()}: {m.content}" for m in history]
        return "\n".join(lines)

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
        logger.info("Session memory cleared", extra={"session_id": session_id})

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def set_pending_question(self, session_id: str, question: str) -> None:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry:
                entry.pending_question = question
                entry.last_accessed = time.time()

    def get_pending_question(self, session_id: str) -> Optional[str]:
        with self._lock:
            entry = self._sessions.get(session_id)
            return entry.pending_question if entry else None

    def clear_pending_question(self, session_id: str) -> None:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry:
                entry.pending_question = None


# --- Singleton ---
conversation_memory = SessionMemory()