import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple

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
    pending_reason: Optional[str] = None
    pending_set_at: float = 0.0
    is_handed_off: bool = False
    greeted: bool = False


class SessionMemory:
    def __init__(self, max_turns: Optional[int] = None) -> None:
        self._max_messages = 2 * (max_turns or settings.MAX_HISTORY_TURNS)
        self._ttl_seconds = settings.SESSION_TTL_HOURS * 3600
        self._max_sessions = settings.MAX_SESSIONS
        # A pending handoff confirmation goes stale if the customer never answers.
        self._pending_ttl_seconds = (
            getattr(settings, "PENDING_QUESTION_TTL_MINUTES", 30) * 60
        )
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

    def _get_or_create_locked(self, session_id: str, now: float) -> _SessionEntry:
        entry = self._sessions.get(session_id)
        if entry is None:
            entry = _SessionEntry(
                history=deque(maxlen=self._max_messages),
                last_accessed=now,
            )
            self._sessions[session_id] = entry
        else:
            entry.last_accessed = now
        return entry

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            self._maybe_evict_locked()
            entry = self._get_or_create_locked(session_id, time.time())
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

    def set_pending_question(
        self, session_id: str, question: str, reason: Optional[str] = None
    ) -> None:
        """Park the customer's question while we wait for a handoff yes/no.

        `reason` is a handoff-reason key (e.g. "kb_gap", "frustration") used for
        the agent-facing private note. Creates the session if needed: on a
        customer's first turn the entry does not exist yet.
        """
        with self._lock:
            self._maybe_evict_locked()
            now = time.time()
            entry = self._get_or_create_locked(session_id, now)
            entry.pending_question = question
            entry.pending_reason = reason
            entry.pending_set_at = now

    def get_pending_handoff(self, session_id: str) -> Optional[Tuple[str, Optional[str]]]:
        """Return (question, reason) for a live pending confirmation, else None."""
        with self._lock:
            entry = self._sessions.get(session_id)
            if not entry or not entry.pending_question:
                return None
            if (time.time() - entry.pending_set_at) > self._pending_ttl_seconds:
                entry.pending_question = None
                entry.pending_reason = None
                entry.pending_set_at = 0.0
                return None
            return entry.pending_question, entry.pending_reason

    def get_pending_question(self, session_id: str) -> Optional[str]:
        pending = self.get_pending_handoff(session_id)
        return pending[0] if pending else None

    def clear_pending_question(self, session_id: str) -> None:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry:
                entry.pending_question = None
                entry.pending_reason = None
                entry.pending_set_at = 0.0

    def mark_handed_off(self, session_id: str) -> None:
        """Lock the session as handed off to a human and, in the same atomic
        step, drop any pending confirmation so it can never re-trigger."""
        with self._lock:
            self._maybe_evict_locked()
            entry = self._get_or_create_locked(session_id, time.time())
            entry.is_handed_off = True
            entry.pending_question = None
            entry.pending_reason = None
            entry.pending_set_at = 0.0

    def is_handed_off(self, session_id: str) -> bool:
        """True while a human owns the session. Touches last_accessed so an
        active muted chat isn't evicted by the TTL sweep."""
        with self._lock:
            entry = self._sessions.get(session_id)
            if not entry or not entry.is_handed_off:
                return False
            entry.last_accessed = time.time()
            return True

    def claim_greeting(self, session_id: str) -> bool:
        """Atomically decide whether a welcome greeting may be sent.

        True at most once per session, and only for a genuinely fresh one:
        never if the session was already greeted, already has messages, or is
        handed off to a human. The caller sends the greeting iff this is True.
        """
        with self._lock:
            self._maybe_evict_locked()
            entry = self._sessions.get(session_id)
            if entry and (entry.greeted or entry.history or entry.is_handed_off):
                return False
            entry = self._get_or_create_locked(session_id, time.time())
            entry.greeted = True
            return True

    def has_history(self, session_id: str) -> bool:
        with self._lock:
            entry = self._sessions.get(session_id)
            return bool(entry and entry.history)

    def clear_handoff_state(self, session_id: str) -> None:
        """Hand the session back to the bot (agent resolved / chat reopened)."""
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry:
                entry.is_handed_off = False
                entry.pending_question = None
                entry.pending_reason = None
                entry.pending_set_at = 0.0

    def get_recent_history_formatted(
        self, session_id: str, max_turns: int = 4, max_chars: int = 500
    ) -> str:
        """Last `max_turns` turns (2 messages each) as clean 'User: ...' /
        'Assistant: ...' lines, oldest first. Long messages are truncated."""
        history = self.get_history(session_id)
        if not history:
            return "(no prior conversation)"
        recent = history[-(2 * max(max_turns, 1)):]
        lines = []
        for m in recent:
            text = " ".join(m.content.split()) if m.content else ""
            if len(text) > max_chars:
                text = text[: max_chars - 1].rstrip() + "…"
            lines.append(f"{m.role.capitalize()}: {text}")
        return "\n".join(lines)


# --- Singleton ---
conversation_memory = SessionMemory()