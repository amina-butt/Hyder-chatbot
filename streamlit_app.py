"""
streamlit_app.py
----------------
Streamlit frontend for "Hyder EV Assistant" — a customer support chatbot
for Hyder Electric Bikes. Wraps the existing RAG backend (rag_engine.py,
memory.py, config.py) in a clean, contrast-safe chat UI: header banner,
tabbed layout (Chat / Models & Pricing / Support), and a minimal sidebar.

Design notes (why the CSS is built this way):
- We do NOT override `.stApp`'s background globally — doing so fights
  Streamlit's native light/dark mode toggle and is exactly how apps end up
  with invisible white-on-white or black-on-black text.
- Every custom-colored block (`.custom-card`, header banner, sidebar) is
  self-contained: it sets its OWN background AND explicit text colors for
  every child element (h1-h6, p, span, label) it draws, rather than relying
  on inherited theme colors that may flip under dark mode.
- Native Streamlit widgets (chat bubbles, buttons, metrics) are only ever
  given layout tweaks (radius, padding, spacing) — never forced colors —
  so they keep working correctly in both light and dark mode.

Run:
    streamlit run streamlit_app.py
"""

import uuid

import streamlit as st

from config import settings
from memory import conversation_memory
from rag_engine import generate_reply

# --------------------------------------------------------------------------
# Page config (must be the first Streamlit command)
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Hyder EV Customer Support",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Design tokens & custom CSS
# --------------------------------------------------------------------------
# Palette: Deep Slate Navy (#1E293B), Soft Accent Indigo (#3B82F6),
# Off-White (#F8FAFC), light border (#E2E8F0), dark border (#334155).

st.markdown(
    """
    <style>
    /* ---- Layout spacing (no color overrides here — theme-safe) ---- */
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 780px;
    }

    /* ---- Header banner: self-contained colors, safe in any theme ---- */
    .hyder-header {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .hyder-header h1 {
        color: #F8FAFC !important;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0 0 4px 0;
    }
    .hyder-header p {
        color: #CBD5E1 !important;
        font-size: 0.92rem;
        margin: 0;
    }
    .hyder-badge {
        display: inline-block;
        margin-top: 10px;
        padding: 3px 12px;
        border-radius: 999px;
        background-color: rgba(59, 130, 246, 0.18);
        color: #93C5FD !important;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* ---- Custom card: self-contained colors on an explicit light bg ---- */
    .custom-card {
        background-color: #ffffff;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .custom-card h3 {
        color: #0F172A !important;
        margin-top: 0;
        margin-bottom: 6px;
        font-size: 1.15rem;
        font-weight: 600;
    }
    .custom-card p {
        color: #475569 !important;
        font-size: 0.92rem;
        line-height: 1.55;
        margin: 0;
    }
    .custom-card .price-tag {
        display: inline-block;
        margin-top: 10px;
        color: #1E293B !important;
        background-color: #F1F5F9;
        border-radius: 8px;
        padding: 4px 10px;
        font-weight: 700;
        font-size: 0.95rem;
    }

    /* ---- Metric value contrast fix (theme-agnostic dark text is wrong
       on dark mode, so we only nudge weight/size, not color) ---- */
    div[data-testid="stMetricValue"] {
        font-weight: 700;
    }

    /* ---- Chat bubbles: spacing/radius only — colors stay native ---- */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.3rem 0.5rem;
    }

    /* ---- Quick action buttons: rounded, theme-neutral ---- */
    div[data-testid="column"] .stButton button {
        width: 100%;
        border-radius: 10px;
        padding: 0.5rem 0.4rem;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* ---- Sidebar: self-contained dark theme, explicit light text ---- */
    section[data-testid="stSidebar"] {
        background-color: #1E293B;
    }
    section[data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #334155;
    }
    .hyder-sidebar-brand {
        text-align: center;
        padding: 6px 0 14px 0;
    }
    .hyder-sidebar-brand .icon {
        font-size: 2.2rem;
    }
    .hyder-sidebar-brand .name {
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 2px;
    }
    .hyder-sidebar-brand .sub {
        font-size: 0.78rem;
        color: #94A3B8 !important;
    }
    .hyder-sidebar-note {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 10px 12px;
        font-size: 0.82rem;
        line-height: 1.5;
        color: #CBD5E1 !important;
    }
    section[data-testid="stSidebar"] .stButton button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #334155;
        background-color: transparent;
        font-weight: 500;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        border-color: #94A3B8;
        background-color: #0F172A;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Static reference data (mirrors data/hyder_bikes.txt — display only)
# --------------------------------------------------------------------------

MODEL_SPECS = [
    {
        "name": "ELI 100",
        "tagline": "Entry-level city commuter",
        "motor": "1000W brushless hub motor",
        "battery": "48V 20Ah Lithium-ion",
        "top_speed": "55 km/h",
        "range": "70-80 km (eco)",
        "price": "PKR 285,000",
    },
    {
        "name": "HLI 100",
        "tagline": "Mid-range hybrid pedal-assist",
        "motor": "1200W brushless hub motor",
        "battery": "60V 22Ah Lithium-ion",
        "top_speed": "65 km/h",
        "range": "90-100 km (eco)",
        "price": "PKR 365,000",
    },
    {
        "name": "SLI 100",
        "tagline": "Premium long-range / fleet",
        "motor": "1500W brushless hub motor",
        "battery": "60V 30Ah Lithium-ion",
        "top_speed": "75 km/h",
        "range": "120-130 km (eco)",
        "price": "PKR 465,000",
    },
]

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
    exchange. Shared by st.chat_input and the quick-action buttons."""
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
# Sidebar — branding, quick reference, chat control
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        f"""
        <div class="hyder-sidebar-brand">
            <div class="icon">⚡</div>
            <div class="name">{settings.COMPANY_NAME}</div>
            <div class="sub">AI Customer Support</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("Quick Reference")
    st.markdown(
        f"""
        <div class="hyder-sidebar-note">
            📞 <b>{settings.HUMAN_HANDOFF_CONTACT}</b><br>
            See the <b>Support</b> tab for full contact & showroom details.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("Session")
    if st.button("🗑️ Clear Chat History"):
        conversation_memory.clear_session(st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# --------------------------------------------------------------------------
# Header banner
# --------------------------------------------------------------------------

st.markdown(
    """
    <div class="hyder-header">
        <h1>⚡ Hyder EV Customer Assistant</h1>
        <p>AI-powered support in English, Roman Urdu, and Urdu.</p>
        <span class="hyder-badge">🟢 Assistant Online</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Key facts row (static, verifiable — not fabricated performance metrics)
# --------------------------------------------------------------------------

m1, m2, m3 = st.columns(3)
m1.metric("Models Available", "3")
m2.metric("Languages Supported", "3")
m3.metric("Support Hours", "10 AM–8 PM")

# --------------------------------------------------------------------------
# Tabbed layout
# --------------------------------------------------------------------------

tab_chat, tab_models, tab_support = st.tabs(
    ["💬 Chat", "🏍️ Models & Pricing", "📞 Support"]
)

# ---- Chat tab ----
with tab_chat:
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

    st.markdown("")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if quick_prompt:
        _process_prompt(quick_prompt)

    user_prompt = st.chat_input("Ask about Hyder EV bikes...")
    if user_prompt:
        _process_prompt(user_prompt)

# ---- Models & Pricing tab ----
with tab_models:
    for model in MODEL_SPECS:
        st.markdown(
            f"""
            <div class="custom-card">
                <h3>{model['name']} — {model['tagline']}</h3>
                <p>
                    Motor: {model['motor']}<br>
                    Battery: {model['battery']}<br>
                    Top speed: {model['top_speed']}<br>
                    Range: {model['range']}
                </p>
                <div class="price-tag">{model['price']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Side-by-side comparison")
    st.dataframe(
        [
            {
                "Model": m["name"],
                "Motor": m["motor"],
                "Battery": m["battery"],
                "Top Speed": m["top_speed"],
                "Range": m["range"],
                "Price": m["price"],
            }
            for m in MODEL_SPECS
        ],
        use_container_width=True,
        hide_index=True,
    )

# ---- Support tab ----
with tab_support:
    st.markdown(
        f"""
        <div class="custom-card">
            <h3>📞 Talk to a Human</h3>
            <p>
                Phone / WhatsApp: <b>{settings.HUMAN_HANDOFF_CONTACT}</b><br>
                Email: <b>{settings.HANDOFF_EMAIL}</b>
            </p>
        </div>
        <div class="custom-card">
            <h3>📍 Showroom</h3>
            <p>
                Main Blvd, Lahore<br>
                Timings: 10:00 AM – 8:00 PM (Mon–Sat)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )