"""
streamlit_app.py
----------------
Streamlit frontend for "Hyder Assistant" — a customer support chatbot for
Hyder Electric Bikes. Wraps the existing RAG backend (rag_engine.py,
memory.py, config.py) in a minimal, single-column, ChatGPT/Claude-style
chat view.

Design notes (why the CSS is built this way):
- No custom colors are forced anywhere. Every previous contrast bug
  (white-on-white / black-on-black text) came from hard-coding hex colors
  that fought Streamlit's native light/dark theme. The only CSS left here
  is layout-only (max width, spacing) — it never sets a `color` or
  `background-color` property, so it can never go invisible under either
  theme.
- No tabs, metric cards, badges, or counters — just the header, the chat
  history, the mic, and the input box.

Run:
    streamlit run streamlit_app.py
"""

import uuid

import streamlit as st
from audio_recorder_streamlit import audio_recorder

from config import settings
from memory import conversation_memory
from rag_engine import generate_reply, transcribe_audio

# --------------------------------------------------------------------------
# Page config (must be the first Streamlit command)
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Hyder Assistant",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------
# Layout-only CSS — spacing and sizing, never color. Safe under both
# light and dark themes.
# --------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 720px;
    }
    /* Small caption above the mic button, spaced tight to the chat input
       right below it. */
    .mic-caption {
        font-size: 0.8rem;
        opacity: 0.65;
        margin-bottom: -0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                f"👋 Welcome to **{settings.COMPANY_NAME}** support! Ask me about "
                "our bike models, pricing, showrooms, or warranty — in English, "
                "Urdu, or Roman Urdu."
            ),
        }
    ]

# Tracks a hash of the last processed recording so a rerun triggered by
# something else (e.g. the "Clear chat" button) doesn't re-transcribe and
# re-send the same audio again — audio_recorder_streamlit keeps returning
# the same bytes across reruns until a new recording is made.
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None


def _process_prompt(prompt: str) -> None:
    """Send a prompt through the RAG pipeline, then render and store the
    exchange. Shared by st.chat_input and voice input."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply, handoff = generate_reply(st.session_state.session_id, prompt)

        display_reply = reply
        if handoff:
            display_reply += (
                f"\n\n*Need more help? Reach our team directly at "
                f"{settings.HUMAN_HANDOFF_CONTACT}.*"
            )

        st.markdown(display_reply)

    st.session_state.messages.append({"role": "assistant", "content": display_reply})


# --------------------------------------------------------------------------
# Header — plain, native, theme-safe
# --------------------------------------------------------------------------

st.title("⚡ Hyder Assistant")
st.caption(f"AI customer support for {settings.COMPANY_NAME}")

# --------------------------------------------------------------------------
# Minimal sidebar — just session control, no branding blocks or forced
# colors.
# --------------------------------------------------------------------------

with st.sidebar:
    st.subheader(settings.COMPANY_NAME)
    st.caption(f"📞 {settings.HUMAN_HANDOFF_CONTACT}")
    if st.button("🗑️ Clear chat"):
        conversation_memory.clear_session(st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.last_audio_hash = None
        st.rerun()

# --------------------------------------------------------------------------
# Chat history
# --------------------------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------------------------------------------------
# Voice input — placed directly above the text chat input, minimal (a
# small caption + the mic icon), no card/border around it.
# --------------------------------------------------------------------------

mic_col, _spacer = st.columns([1, 6])
with mic_col:
    st.markdown('<p class="mic-caption">🎙️ Or speak</p>', unsafe_allow_html=True)
    audio_bytes = audio_recorder(
        text="",
        icon_size="1.5x",
        pause_threshold=2.0,
        key="hyder_voice_recorder",
    )

if audio_bytes:
    # Hash the raw bytes so a brand-new recording is told apart from the
    # component simply re-returning the last recording on an unrelated
    # rerun — without this check, every rerun (e.g. clicking "Clear chat")
    # would re-transcribe and re-send the same audio in a loop.
    audio_hash = hash(audio_bytes)
    if audio_hash != st.session_state.last_audio_hash:
        st.session_state.last_audio_hash = audio_hash

        with st.spinner("Transcribing your audio..."):
            transcribed_text = transcribe_audio(audio_bytes)

        if transcribed_text:
            _process_prompt(transcribed_text)
        else:
            st.warning(
                "Sorry, I couldn't understand that recording. "
                "Please try again or type your question instead."
            )

# --------------------------------------------------------------------------
# Text input
# --------------------------------------------------------------------------

user_prompt = st.chat_input("Ask about Hyder EV bikes...")
if user_prompt:
    _process_prompt(user_prompt)