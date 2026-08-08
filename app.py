"""Streamlit shell for the Bookly support agent.

Thin on purpose: render the transcript, take one message, hand it to the agent.
All logic lives in `agent/` and `tools/`.

Still deliberately plain. The agent trace and the model-routing display are
Phase 6 — the data for both is already being captured on `SessionState`
(`tool_traces`, `model_turns`), so that phase is a rendering job and not a
change to the loop.

Run with: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from agent.config import AnthropicConfigError
from agent.orchestrator import BooklyAgent
from agent.state import SessionState

st.set_page_config(page_title="Bookly Support", page_icon="📚")


def get_session() -> SessionState:
    """Fetch the session state for this browser tab, creating it if needed."""
    if "bookly_state" not in st.session_state:
        st.session_state.bookly_state = SessionState()
    return st.session_state.bookly_state


def get_agent() -> BooklyAgent:
    """Fetch the agent, constructing it once per session.

    Construction reads and validates the Anthropic configuration, so a missing
    key or model name stops here with an explanation rather than failing on the
    customer's first message.
    """
    if "bookly_agent" not in st.session_state:
        st.session_state.bookly_agent = BooklyAgent()
    return st.session_state.bookly_agent


def main() -> None:
    """Render the chat UI and drive one turn per submission."""
    state = get_session()

    st.title("📚 Bookly Support")

    try:
        agent = get_agent()
    except AnthropicConfigError as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.subheader("Session")
        st.write("Verified:", state.verified_customer_id or "—")
        st.write("Region:", state.customer_region or "—")
        st.write("Active order:", state.active_order_id or "—")
        st.write("Active item:", state.active_item_id or "—")
        st.write("Eligibility:", state.eligibility.policy_id if state.eligibility else "—")
        st.write("Confirmed:", state.confirmed)
        st.write("Escalated:", state.escalated)

        st.subheader("Routing")
        last_turn = state.model_turns[-1] if state.model_turns else None
        st.write("Last turn:", last_turn.model_tier if last_turn else "—")
        st.write("Because:", last_turn.routing_reason if last_turn else "—")
        st.write("Tool calls:", len(state.tool_traces))

        if st.button("Reset conversation"):
            st.session_state.pop("bookly_state", None)
            st.rerun()

    for message in state.messages:
        with st.chat_message(message.role.value):
            st.markdown(message.content)

    if prompt := st.chat_input("How can we help?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("…"):
            reply = agent.respond(state, prompt)
        with st.chat_message("assistant"):
            st.markdown(reply)
        # The sidebar was drawn before this turn ran; rerun so it reflects it.
        st.rerun()


if __name__ == "__main__":
    main()
