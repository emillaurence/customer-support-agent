"""Streamlit shell for the Bookly support agent.

Thin on purpose: render the transcript, take one message, hand it to the agent.
Logic lives in `agent/` and `policy/`, presentation in `ui.py`.

Two resets, and they are not the same thing. "Reset conversation" forgets the
conversation; the RMA the demo just created is still in `data/returns.json`.
"Reset demo" also puts that data back — it calls the same `agent.tools.reset_demo`
as `scripts/reset_demo.py`, rather than a second copy of the logic.

Run with: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

import ui
from agent.agent import AnthropicConfigError, BooklyAgent
from agent.state import SessionState
from agent.tools import reset_demo

st.set_page_config(page_title=ui.PAGE_TITLE, page_icon=ui.PAGE_ICON, layout="centered")

# Shown if the shell itself fails, which the agent's own error handling should
# already have covered. A comment rather than a docstring on purpose: Streamlit's
# magic renders a bare string at the top level of a script.
UNAVAILABLE = "Something went wrong. Please try again, or reset the conversation."


def get_session() -> SessionState:
    """The session state for this browser tab, created on first use."""
    if "bookly_state" not in st.session_state:
        st.session_state.bookly_state = SessionState()
    return st.session_state.bookly_state


def get_turns() -> list[ui.AssistantTurn]:
    """The per-turn trace records this tab has captured, oldest first.

    Held beside the session rather than on it: they are a view of what the agent
    did, and `SessionState` is what the agent needs to keep working.
    """
    if "bookly_turns" not in st.session_state:
        st.session_state.bookly_turns = []
    return st.session_state.bookly_turns


def get_agent() -> BooklyAgent:
    """The agent, built once per session — which is where a missing API key or
    model name is caught, rather than on the customer's first message."""
    if "bookly_agent" not in st.session_state:
        st.session_state.bookly_agent = BooklyAgent()
    return st.session_state.bookly_agent


def run_turn(agent: BooklyAgent, state: SessionState, prompt: str) -> None:
    """Take one customer message through the agent and record what it did.

    The trace offset is read *before* the turn, so everything the loop appends
    while it runs belongs to this turn.
    """
    offset = len(state.tool_traces)
    try:
        reply = agent.respond(state, prompt)
    except Exception:  # noqa: BLE001 - a customer must never meet a stack trace
        st.session_state.turn_error = UNAVAILABLE
        return

    get_turns().append(ui.capture_turn(state, reply, trace_offset=offset))


def render_sidebar(state: SessionState) -> None:
    """The demo controls and the developer view — beside the chat, not in it."""
    with st.sidebar:
        st.markdown("### Demo controls")

        if st.button("Reset demo", type="primary", width="stretch"):
            summary = reset_demo()
            # Cleared wholesale rather than key by key, so a widget or cache added
            # later cannot survive a reset by being forgotten. The agent carries
            # over: it holds no conversation state, and rebuilding it would only
            # re-read the same configuration.
            agent = st.session_state.get("bookly_agent")
            st.session_state.clear()
            if agent is not None:
                st.session_state.bookly_agent = agent
            st.session_state.bookly_state = SessionState()
            st.session_state.reset_notice = summary
            st.rerun()

        if st.button("Reset conversation", width="stretch"):
            # The conversation only. Any RMA it created stays where it is.
            st.session_state.pop("bookly_state", None)
            st.session_state.pop("bookly_turns", None)
            st.rerun()

        if notice := st.session_state.pop("reset_notice", None):
            st.success(notice)

        st.divider()
        ui.render_developer_state(state)


def main() -> None:
    """Render the chat UI and drive one turn per submission."""
    ui.apply_branding()
    ui.render_header()

    state = get_session()

    try:
        agent = get_agent()
    except AnthropicConfigError as exc:
        st.error(str(exc))
        st.stop()

    render_sidebar(state)

    if not state.messages:
        ui.render_welcome()

    for message, turn in ui.pair_turns(state.messages, get_turns()):
        ui.render_exchange(message.role.value, message.content, turn)

    if error := st.session_state.pop("turn_error", None):
        st.error(error)

    if prompt := st.chat_input(ui.CHAT_PLACEHOLDER):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Working on it…"):
            run_turn(agent, state, prompt)
        # The page was drawn before this turn ran; rerun so the reply, its trace,
        # and the session view all reflect it.
        st.rerun()


if __name__ == "__main__":
    main()
