import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from agents.orchestrator import OrchestratorAgent, State

st.set_page_config(page_title="MediAgent", page_icon="🏥", layout="centered")
st.title("🏥 MediAgent — Diagnostic Assistant")
st.caption("Describe your symptoms and I'll help identify possible conditions and treatment options.")
st.warning("⚠️ This is an educational AI demo. Always consult a real doctor for medical advice.")

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = OrchestratorAgent()
    st.session_state.messages = []

agent: OrchestratorAgent = st.session_state.orchestrator

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if agent.state == State.COLLECTING and not st.session_state.messages:
    greeting = "Hello! I'm MediAgent. Please describe your symptoms and I'll help assess what might be going on."
    st.session_state.messages.append({"role": "assistant", "content": greeting})
    with st.chat_message("assistant"):
        st.markdown(greeting)

col1, col2 = st.columns([6, 1])
with col2:
    if st.button("Reset", use_container_width=True):
        st.session_state.orchestrator = OrchestratorAgent()
        st.session_state.messages = []
        st.rerun()

if prompt := st.chat_input("Describe your symptoms..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if agent.state == State.DONE:
        response = "Your session is complete. Click **Reset** to start a new consultation."
    else:
        with st.spinner("Analyzing..."):
            response = agent.process(prompt)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

    if agent.state == State.DONE:
        st.success("Consultation complete. You can reset to start a new session.")
