"""
Streamlit prototype — Title IV Financial Aid Voice Agent

Simulates the voice call center experience with text input.
Demonstrates the full pipeline: RAG retrieval, safety classification,
confidence scoring, disclaimers, and escalation.

Connects to the FastAPI backend at BACKEND_URL.
"""

from __future__ import annotations

import os
import uuid

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Title IV Financial Aid Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #003087 0%, #0050b3 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .tier-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .tier-1 { background: #d1fae5; color: #065f46; }
    .tier-2 { background: #fef3c7; color: #92400e; }
    .tier-3 { background: #fee2e2; color: #991b1b; }
    .confidence-bar-container {
        background: #e5e7eb;
        border-radius: 6px;
        height: 8px;
        margin: 4px 0;
    }
    .agent-message {
        background: #1e3a5f;
        border-left: 4px solid #4a9eff;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #e8f0fe;
    }
    .user-message {
        background: #2d2d2d;
        border-left: 4px solid #6b7280;
        padding: 0.8rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        color: #d1d5db;
    }
    .escalation-box {
        background: #3d1f00;
        border: 2px solid #f97316;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        color: #fed7aa;
    }
    .source-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #d1d5db;
        font-size: 1rem;
    }
    .stButton > button {
        background: #003087;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: #0050b3;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_meta" not in st.session_state:
    st.session_state.last_meta = None


# ── Helper functions ──────────────────────────────────────────────────────────

def _tier_badge(tier: int) -> str:
    labels = {1: "Simple FAQ", 2: "Complex Regulatory", 3: "Personal — Escalated"}
    css_class = {1: "tier-1", 2: "tier-2", 3: "tier-3"}
    return (
        f'<span class="tier-badge {css_class.get(tier, "tier-1")}">'
        f'Tier {tier}: {labels.get(tier, "Unknown")}</span>'
    )


def _confidence_color(conf: float) -> str:
    if conf >= 0.80:
        return "#10b981"
    if conf >= 0.60:
        return "#f59e0b"
    return "#ef4444"


def _send_message(query: str) -> dict | None:
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{BACKEND_URL}/api/chat",
                json={"query": query, "session_id": st.session_state.session_id},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        st.error("Cannot reach the backend server. Make sure it's running at " + BACKEND_URL)
        return None
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


def _check_backend() -> bool:
    try:
        r = httpx.get(f"{BACKEND_URL}/health", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/US_Federal_Student_Aid_logo.svg/320px-US_Federal_Student_Aid_logo.svg.png", width=120)
    st.markdown("### Title IV Assistant")
    st.markdown("*Financial Aid & University Enrollment*")
    st.divider()

    # Backend status
    backend_ok = _check_backend()
    if backend_ok:
        st.success("Backend connected", icon="✅")
    else:
        st.error("Backend offline", icon="❌")

    st.divider()

    # Session info
    st.markdown("**Session**")
    st.code(st.session_state.session_id[:8] + "...", language=None)

    if st.button("New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.last_meta = None
        st.rerun()

    st.divider()

    # Safety tier legend
    st.markdown("**Safety Tiers**")
    st.markdown(
        '<span class="tier-badge tier-1">Tier 1</span> Simple FAQ<br>'
        '<span class="tier-badge tier-2">Tier 2</span> Regulatory<br>'
        '<span class="tier-badge tier-3">Tier 3</span> Escalated<br>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Sample questions
    st.markdown("**Try these questions:**")
    sample_questions = [
        "What is the FAFSA?",
        "What is a Pell Grant and who qualifies?",
        "How is the Expected Family Contribution calculated?",
        "What are federal student loan interest rates?",
        "What are the SAP requirements?",
        "Am I eligible for financial aid this year?",  # → Tier 3
        "Should I appeal my aid decision?",             # → Tier 3
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True, key=f"sample_{q[:20]}"):
            st.session_state._pending_query = q
            st.rerun()

    st.divider()
    st.caption("Powered by Anthropic Claude + Qdrant + VAPI")


# ── Main content ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h2 style="margin:0; font-size:1.6rem;">🎓 Title IV Financial Aid Assistant</h2>
    <p style="margin:0.3rem 0 0; opacity:0.85; font-size:0.95rem;">
        Ask about FAFSA, federal grants, loans, enrollment verification, and more.
        This assistant is for information purposes — it does not replace your financial aid office.
    </p>
</div>
""", unsafe_allow_html=True)

# Two-column layout: chat | metadata
col_chat, col_meta = st.columns([3, 2])

with col_chat:
    # Render conversation history
    for turn in st.session_state.messages:
        if turn["role"] == "user":
            with st.chat_message("user"):
                st.write(turn["content"])
        else:
            with st.chat_message("assistant", avatar="🎓"):
                if turn.get("escalated"):
                    st.warning("**Transfer to Advisor**\n\n" + turn["content"])
                else:
                    st.write(turn["content"])

    # Handle pending query from sidebar sample buttons
    pending = st.session_state.pop("_pending_query", None)

    # Native chat input — reliable across all Streamlit versions
    user_input = st.chat_input("Ask about FAFSA, grants, loans, enrollment...")

    query = pending or user_input

    if query and query.strip():
        query = query.strip()

        # Show user message immediately
        with st.chat_message("user"):
            st.write(query)

        # Get response
        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner("Alex is thinking..."):
                result = _send_message(query)

            if result:
                if result["escalated"]:
                    st.warning("**Transfer to Advisor**\n\n" + result["response"])
                else:
                    st.write(result["response"])

                st.session_state.messages.append({"role": "user", "content": query})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"],
                    "escalated": result["escalated"],
                })
                st.session_state.last_meta = result


# ── Metadata panel ────────────────────────────────────────────────────────────

with col_meta:
    st.markdown("### Analysis")

    meta = st.session_state.last_meta
    if not meta:
        st.info("Send a message to see RAG analysis here.")
    else:
        # Tier badge
        st.markdown("**Query Classification**")
        st.markdown(_tier_badge(meta["tier"]), unsafe_allow_html=True)
        st.caption(meta["tier_label"])

        st.markdown("---")

        # Confidence
        st.markdown("**Retrieval Confidence**")
        conf = meta["confidence"]
        color = _confidence_color(conf)
        st.markdown(
            f'<div class="confidence-bar-container">'
            f'<div style="background:{color}; height:8px; border-radius:6px; width:{int(conf*100)}%;"></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"{conf:.0%} confidence")

        st.markdown("---")

        # Escalation flag
        if meta["escalated"]:
            st.error("Escalated to human advisor", icon="⚠️")
        else:
            st.success("Answered by AI", icon="✅")

        st.markdown("---")

        # Retrieved sources
        if meta.get("doc_sources"):
            st.markdown("**Retrieved Sources**")
            for src in meta["doc_sources"]:
                score_pct = f"{src['rerank_score']:.0%}"
                doc_type_label = {
                    "fsa_handbook": "FSA Handbook",
                    "university_policy": "University Policy",
                    "faq": "FAQ",
                }.get(src["doc_type"], src["doc_type"])

                with st.expander(f"📄 {src['source'][:45]}... — {score_pct}"):
                    st.caption(f"Type: {doc_type_label}")
                    st.markdown(f"_{src['content_snippet']}..._")
        else:
            st.caption("No sources retrieved (escalated before retrieval).")
