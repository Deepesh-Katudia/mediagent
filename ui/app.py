import sys
import os
import time
import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from agents.orchestrator import OrchestratorAgent, State
import json
from dotenv import load_dotenv
from streamlit_mic_recorder import speech_to_text
from utils.openai_voice import get_openai_client, consult_gpt
load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediAgent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        background-color: #0f0f23;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] span {
        color: #e0e0e0 !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #1e1e3f;
        color: #e0e0e0;
        border: 1px solid #3d3d7a;
        border-radius: 8px;
        text-align: left;
        padding: 8px 12px;
        font-size: 0.85rem;
        transition: background 0.2s;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #2d2d5e;
        border-color: #6666cc;
    }
    .new-chat-btn > button {
        background-color: #4f46e5 !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .new-chat-btn > button:hover {
        background-color: #4338ca !important;
    }
    .delete-btn > button {
        background-color: transparent !important;
        color: #ef4444 !important;
        border: 1px solid #7f1d1d !important;
        border-radius: 6px !important;
        padding: 4px 8px !important;
        font-size: 0.8rem !important;
        min-height: 0 !important;
    }
    .delete-btn > button:hover {
        background-color: #7f1d1d !important;
        color: white !important;
    }
    .main-header {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .thinking-text {
        color: #9ca3af;
        font-style: italic;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ── State helpers ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "orchestrator": OrchestratorAgent(),
        "messages": [],
        "sessions": [],          # list of {id, title, messages, timestamp}
        "session_counter": 0,
        "viewing": None,         # None = current chat, else session id
        "voice_mode": False,
        "voice_history": [],
        "openai_client": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_session_title(messages: list) -> str:
    for msg in messages:
        if msg["role"] == "user":
            text = msg["content"].strip()
            return (text[:42] + "…") if len(text) > 42 else text
    return "Untitled session"


def save_and_new_chat():
    if st.session_state.messages:
        st.session_state.session_counter += 1
        st.session_state.sessions.insert(0, {
            "id": st.session_state.session_counter,
            "title": get_session_title(st.session_state.messages),
            "messages": st.session_state.messages.copy(),
            "timestamp": datetime.datetime.now().strftime("%b %d · %H:%M"),
        })
    st.session_state.orchestrator = OrchestratorAgent()
    st.session_state.messages = []
    st.session_state.viewing = None


init_state()
agent: OrchestratorAgent = st.session_state.orchestrator

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MediAgent")
    st.caption("AI Diagnostic Assistant")
    st.divider()

    # New chat button
    with st.container():
        st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
        if st.button("➕  New Chat", use_container_width=True):
            save_and_new_chat()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 💬 Chat History")

    if not st.session_state.sessions:
        st.caption("_Your past sessions will appear here._")
    else:
        for session in st.session_state.sessions:
            col_title, col_del = st.columns([5, 1])
            with col_title:
                label = f"🗒️ {session['title']}\n{session['timestamp']}"
                if st.button(label, key=f"sess_{session['id']}", use_container_width=True):
                    st.session_state.viewing = session["id"]
                    st.rerun()
            with col_del:
                st.markdown('<div class="delete-btn">', unsafe_allow_html=True)
                if st.button("🗑️", key=f"del_{session['id']}", help="Delete this session"):
                    # If currently viewing this session, go back to active chat
                    if st.session_state.viewing == session["id"]:
                        st.session_state.viewing = None
                    st.session_state.sessions = [
                        s for s in st.session_state.sessions if s["id"] != session["id"]
                    ]
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🎙️ Voice Consult")
    new_voice = st.toggle("Enable Voice Mode", value=st.session_state.voice_mode)
    if new_voice != st.session_state.voice_mode:
        st.session_state.voice_mode = new_voice
        st.session_state.voice_history = []
        st.rerun()
    if st.session_state.voice_mode:
        st.caption("🔴 Voice mode active")
    st.divider()
    st.caption("⚠️ Educational demo — not medical advice.")


# ── History viewer ────────────────────────────────────────────────────────────
if st.session_state.viewing is not None:
    sess = next(
        (s for s in st.session_state.sessions if s["id"] == st.session_state.viewing),
        None,
    )
    if sess:
        st.markdown(f"### 📋 {sess['title']}")
        st.caption(f"Session from {sess['timestamp']}")
        if st.button("← Back to current chat"):
            st.session_state.viewing = None
            st.rerun()
        st.divider()
        for msg in sess["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    st.stop()


# ── Active chat ───────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🏥 MediAgent — Diagnostic Assistant</p>', unsafe_allow_html=True)
st.caption("Describe your symptoms and I'll help identify possible conditions and suggest treatment options.")
st.warning("⚠️ This is an educational AI demo. Always consult a real doctor for medical advice.")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Greeting on first load
if agent.state == State.COLLECTING and not st.session_state.messages:
    greeting = (
        "👋 Hello! I'm **MediAgent**.\n\n"
        "Please describe your symptoms and I'll help assess what might be going on. "
        "For example:\n"
        "> *\"I have fever, headache and feel tired\"*\n\n"
        "What are you experiencing?"
    )
    st.session_state.messages.append({"role": "assistant", "content": greeting})
    with st.chat_message("assistant"):
        st.markdown(greeting)

# ── Voice consultation panel (visible only when voice mode is on) ────────────
if st.session_state.voice_mode:
    st.divider()
    st.markdown("#### 🎙️ Voice Consultation")
    st.caption("Ask the AI doctor anything. Separate from the symptom diagnostic flow.")

    if st.session_state.openai_client is None:
        try:
            st.session_state.openai_client = get_openai_client()
        except ValueError as e:
            st.error(f"⚠️ {e}")
            st.session_state.voice_mode = False
            st.stop()

    for turn in st.session_state.voice_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    transcript = speech_to_text(
        language="en",
        start_prompt="🎤 Click to speak",
        stop_prompt="⏹ Stop",
        just_once=True,
        use_container_width=False,
        key="voice_stt",
    )

    if transcript:
        reply = None
        with st.spinner("Consulting doctor…"):
            try:
                reply = consult_gpt(
                    st.session_state.openai_client,
                    transcript,
                    st.session_state.voice_history,
                )
            except Exception as e:
                err = str(e)
                if "429" in err or "insufficient_quota" in err:
                    st.error(
                        "⚠️ OpenAI quota exceeded. Please add billing at "
                        "https://platform.openai.com/settings/billing to use voice consultation."
                    )
                else:
                    st.error(f"Voice error: {e}")

        if reply:
            with st.chat_message("user"):
                st.markdown(f"🎤 *{transcript}*")
            with st.chat_message("assistant"):
                st.markdown(reply)
                st.components.v1.html(
                    f"<script>window.speechSynthesis.cancel();"
                    f"window.speechSynthesis.speak("
                    f"new SpeechSynthesisUtterance({json.dumps(reply)}));</script>",
                    height=0,
                )
            st.rerun()
    st.divider()

# ── Chat input + thinking animation ──────────────────────────────────────────
THINKING_FRAMES = [
    "🤔 Analyzing your symptoms…",
    "🧠 Running diagnostic inference…",
    "🔍 Checking treatment pathways…",
    "💊 Validating drug safety…",
    "📋 Preparing your report…",
]

if prompt := st.chat_input("Describe your symptoms…"):
    # User bubble
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response with thinking animation
    with st.chat_message("assistant"):
        if agent.state == State.DONE:
            response = (
                "✅ Your consultation is complete.\n\n"
                "Click **➕ New Chat** in the sidebar to start a fresh session, "
                "or browse your history above."
            )
            st.markdown(response)
        else:
            # Thinking animation
            placeholder = st.empty()
            for frame in THINKING_FRAMES:
                placeholder.markdown(f'<p class="thinking-text">{frame}</p>', unsafe_allow_html=True)
                time.sleep(0.35)

            # Process
            response = agent.process(prompt)
            placeholder.empty()
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    # Celebration on completion
    if agent.state == State.DONE:
        st.success("✅ Consultation complete! Start a **New Chat** from the sidebar for another session.")
        st.components.v1.html("""
<canvas id="cc" style="position:fixed;top:0;left:0;width:100%;height:100%;
    pointer-events:none;z-index:9999;"></canvas>
<script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.2/dist/confetti.browser.min.js"></script>
<script>
(function(){
    var c = confetti.create(document.getElementById('cc'), {resize:true,useWorker:true});
    c({particleCount:120,spread:70,angle:60,origin:{x:0.1,y:0.7},
       colors:['#ff6b6b','#ffd93d','#6bcb77','#4d96ff','#ff922b','#cc5de8']});
    c({particleCount:120,spread:70,angle:120,origin:{x:0.9,y:0.7},
       colors:['#ff6b6b','#ffd93d','#6bcb77','#4d96ff','#ff922b','#cc5de8']});
    setTimeout(function(){
        c({particleCount:80,spread:100,origin:{x:0.5,y:0},gravity:1.2,
           colors:['#ff6b6b','#ffd93d','#6bcb77','#4d96ff']});
    }, 400);
})();
</script>
""", height=0)
