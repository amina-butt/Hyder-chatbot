"""
app.py
------
Interactive terminal chat loop for Hyder Assistant.

Run:
    python app.py

Features:
  - Generates a unique session_id per run (sliding-window memory keyed on it)
  - Graceful shutdown on Ctrl+C, "exit"/"quit", or EOF (piped input)
  - Logs every turn to logs/app.log (see logger.py)
"""

from __future__ import annotations

import sys
import uuid

from config import settings
from logger import get_logger
from memory import conversation_memory
from rag_engine import generate_reply

logger = get_logger(__name__)

_EXIT_COMMANDS = {"exit", "quit", "bye", "khuda hafiz", "allah hafiz"}
_BANNER = f"""
==================================================
  {settings.COMPANY_NAME} - Hyder Assistant
  Type your question in English, Urdu, or Roman Urdu.
  Type 'exit' to leave the conversation.
==================================================
"""


def run_chat_loop() -> None:
    session_id = str(uuid.uuid4())
    logger.info("New chat session started", extra={"session_id": session_id})
    print(_BANNER)

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                # e.g. input piped from a file or CI environment
                break

            if not user_input:
                continue

            if user_input.lower() in _EXIT_COMMANDS:
                print("Hyder Assistant: Thank you for reaching out. Have a great day!")
                break

            reply, handoff = generate_reply(session_id, user_input)
            print(f"Hyder Assistant: {reply}")
            if handoff:
                print(
                    f"(You can also reach a human agent directly at "
                    f"{settings.HUMAN_HANDOFF_CONTACT})"
                )

    except KeyboardInterrupt:
        print("\nHyder Assistant: Session interrupted. Goodbye!")
    finally:
        conversation_memory.clear_session(session_id)
        logger.info("Chat session ended", extra={"session_id": session_id})


if __name__ == "__main__":
    try:
        run_chat_loop()
    except Exception:
        logger.exception("Fatal error in chat loop", extra={"session_id": "-"})
        sys.exit(1)
