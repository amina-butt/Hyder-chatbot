"""
memory.py
---------
Session-based sliding-window conversation memory.

Keeps the last N conversational turns per session_id in memory so the LLM
has short-term context without the prompt growing unbounded. This is an
in-process store suitable for a single-instance deployment; see the
production note at the bottom for scaling to multiple workers.
"""

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional

from config import settings
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class Message:
    role: str          # "user" or "assistant"
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class SessionMemory:
    """Thread-safe sliding-window memory keyed by session_id."""

    def __init__(self, max_turns: Optional[int] = None) -> None:
        # A "turn" = one user message + one assistant reply, so we keep
        # 2 * max_turns messages per session in the deque.
        self._max_messages = 2 * (max_turns or settings.MAX_HISTORY_TURNS)
        self._sessions: Dict[str, Deque[Message]] = {}
        self._lock = threading.Lock()

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to a session's history, automatically evicting
        the oldest message once the sliding-window limit is exceeded."""
        with self._lock:
            history = self._sessions.setdefault(
                session_id, deque(maxlen=self._max_messages)
            )
            history.append(Message(role=role, content=content))
            current_len = len(history)

        logger.debug(
            "Memory updated (role=%s, history_len=%d)",
            role,
            current_len,
            extra={"session_id": session_id},
        )

    def get_history(self, session_id: str) -> List[Message]:
        """Return the current sliding-window history for a session."""
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def get_history_as_text(self, session_id: str) -> str:
        """Render history as a simple 'Role: content' transcript, suitable
        for injecting directly into the LLM prompt."""
        history = self.get_history(session_id)
        if not history:
            return "(no prior conversation in this session)"
        lines = [f"{m.role.capitalize()}: {m.content}" for m in history]
        return "\n".join(lines)

    def clear_session(self, session_id: str) -> None:
        """Wipe a session's history, e.g. on chat exit or a manual reset."""
        with self._lock:
            self._sessions.pop(session_id, None)
        logger.info("Session memory cleared", extra={"session_id": session_id})

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions


# Singleton instance shared across the app
conversation_memory = SessionMemory()

# --- Production note ---
# For multi-worker / multi-instance deployments (e.g. behind a load balancer
# or in a horizontally-scaled container fleet), replace the in-memory
# `_sessions` dict with a Redis-backed store (e.g. RPUSH/LTRIM per session
# key, with a TTL for automatic expiry) so all workers share the same
# conversation state and sessions survive process restarts.
