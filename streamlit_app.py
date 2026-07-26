"""
streamlit_app.py
----------------
Streamlit frontend for "Hyder EV Assistant" — a customer support chatbot
for Hyder Electric Bikes. Wraps the existing RAG backend (rag_engine.py,
memory.py, config.py) in a chat UI with session tracking, quick-action
buttons, and a sidebar with contact/showroom info.

Run:
    streamlit run streamlit_app.py
"""

import uuid

import streamlit as st

from config import settings
from memory import conversation_memory
from rag_engine import generate_reply

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Hyder EV Customer Support",
    page_icon="⚡",
    layout="centered",
)

# --------------------------------------------------------------------------
# Custom CSS — clean, modern chat interface
# --------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Overall app background */
    .stApp {
        background-color: #f5f7fa;
    }

    /* Main content width / padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 780px;
    }

    /* Header title */
    .hyder-header-title {
        font-size: 2rem;
        font-weight: 700;
        color: #10151a;
        margin-bottom: 0.1rem;
    }
    .hyder-header-caption {
        color: #5b6672;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.4rem 0.6rem;
        margin-bottom: 0.4rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #10151a;
    }
    section[data-testid="stSidebar"] * {
        color: #f5f7fa !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #2a323c;
    }

    /* Sidebar brand block */
    .hyder-brand {
        text-align: center;
        padding: 0.5rem 0 1rem 0;
    }
    .hyder-brand-icon {
        font-size: 2.4rem;
    }
    .hyder-brand-name {
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    .hyder-brand-sub {
        font-size: 0.8rem;
        color: #9aa4b0 !important;
    }

    /* Contact card */
    .hyder-contact-card {
        background-color: #1a212b;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin-top: 0.5rem;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    /* Quick action buttons */
    div[data-testid="column"] .stButton button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #dfe3e8;
        background-color: #ffffff;
        padding: 0.5rem 0.4rem;
        font-size: 0.85rem;
        font-weight: 500;
        transition: all 0.15s ease-in-out;
    }
    div[data-testid="column"] .stButton button:hover {
        border-color: #10151a;
        color: #10151a;
    }

    /* Sidebar clear-chat button */
    section[data-testid="stSidebar"] .stButton button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #3a424d;
        background-color: transparent;
        color: #f5f7fa;
        font-weight: 500;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        border-color: #f5f7fa;
        background-color: #1a212b;
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
                f"👋 Welcome to **{settings.COMPANY_NAME}** support! "
                "I'm Hyder EV Assistant — ask me about our bike models, pricing, "
                "showrooms, warranty, or anything else about your Hyder EV. "
                "Aap mujh se Urdu, Roman Urdu, ya English mein bhi pooch sakte hain."
            ),
        }
    ]


def _process_prompt(prompt: str) -> None:
    """Send a prompt through the RAG pipeline, then render and store the
    exchange. Shared by both st.chat_input and the quick-action buttons."""
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
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        f"""
        <div class="hyder-brand">
            <div class="hyder-brand-icon">⚡</div>
            <div class="hyder-brand-name">{settings.COMPANY_NAME}</div>
            <div class="hyder-brand-sub">AI Customer Support</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**📞 Contact & Showroom**")
    st.markdown(
        f"""
        <div class="hyder-contact-card">
            <b>Phone / WhatsApp:</b><br>{settings.HUMAN_HANDOFF_CONTACT}<br><br>
            <b>Showroom:</b><br>Main Blvd, Lahore<br><br>
            <b>Timings:</b><br>10:00 AM - 8:00 PM
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    if st.button("🗑️ Clear Chat History"):
        conversation_memory.clear_session(st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# --------------------------------------------------------------------------
# Main header
# --------------------------------------------------------------------------

st.markdown(
    '<div class="hyder-header-title">⚡ Hyder EV Customer Assistant</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hyder-header-caption">Supporting English, Roman Urdu, and Urdu.</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Quick action buttons
# --------------------------------------------------------------------------

col1, col2, col3 = st.columns(3)

quick_prompt = None
with col1:
    if st.button("🏍️ ELI 100 Specs & Price"):
        quick_prompt = "Can you tell me the specs and price of the ELI 100?"
with col2:
    if st.button("📍 Showroom Location"):
        quick_prompt = "Where is your showroom located?"
with col3:
    if st.button("🛡️ Warranty Info"):
        quick_prompt = "What is the warranty policy on Hyder bikes?"

st.markdown("---")

# --------------------------------------------------------------------------
# Render existing chat history
# --------------------------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------------------------------------------------
# Handle a quick-action click
# --------------------------------------------------------------------------

if quick_prompt:
    _process_prompt(quick_prompt)

# --------------------------------------------------------------------------
# Handle free-text chat input
# --------------------------------------------------------------------------

user_prompt = st.chat_input("Ask about Hyder EV bikes...")
if user_prompt:
    _process_prompt(user_prompt)
