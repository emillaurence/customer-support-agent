"""Streamlit shell for the Bookly support agent.

Thin on purpose: render the transcript, take one message, hand it to the agent.
All logic lives in `agent/` and `tools/`.

Run with: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from agent.orchestrator import BooklyAgent
from agent.state import SessionState

st.set_page_config(page_title="Bookly Support", page_icon="📚")


def get_session() -> SessionState:
    """Fetch the session state for this browser tab, creating it if needed."""
    if "bookly_state" not in st.session_state:
        st.session_state.bookly_state = SessionState()
    return st.session_state.bookly_state


def get_agent() -> BooklyAgent:
    """Fetch the agent, constructing it once per session."""
    if "bookly_agent" not in st.session_state:
        st.session_state.bookly_agent = BooklyAgent()
    return st.session_state.bookly_agent


def main() -> None:
    """Render the chat UI and drive one turn per submission."""
    state = get_session()
    agent = get_agent()

    st.title("📚 Bookly Support")
    st.caption("Scaffold — the agent loop is not wired up yet.")

    with st.sidebar:
        st.subheader("Session")
        st.write("Verified:", state.verified_customer_id or "—")
        st.write("Region:", state.customer_region or "—")
        st.write("Active order:", state.active_order_id or "—")
        st.write("Active item:", state.active_item_id or "—")
        st.write("Eligibility:", state.eligibility.policy_id if state.eligibility else "—")
        st.write("Confirmed:", state.confirmed)
        st.write("Escalated:", state.escalated)
        if st.button("Reset conversation"):
            st.session_state.pop("bookly_state", None)
            st.rerun()

    for message in state.messages:
        with st.chat_message(message.role.value):
            st.markdown(message.content)

    if prompt := st.chat_input("How can we help?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        reply = agent.respond(state, prompt)
        with st.chat_message("assistant"):
            st.markdown(reply)


if __name__ == "__main__":
    main()
