"""Streamlit shell for the Bookly support agent.

Thin on purpose: render the transcript, take one message, hand it to the agent.
All logic lives in `agent/` and `tools/`; everything about how a turn *looks*
lives in `ui/`. This file is the wiring between them, and it holds no business
rule, no tool, and no routing decision of its own.

The screen is the conversation. Under each assistant reply there is a badge
saying which model handled the turn and a collapsed **Agent trace** describing
that reply — the tools it ran, in order, with real latencies, and the policy and
graph path behind any eligibility decision. The trace is per turn rather than one
list at the bottom of the page, so an action can be connected to the reply that
caused it.

The traces are not new. `SessionState` has been collecting `tool_traces` and
`model_turns` since Phase 4; `ui.turns.capture_turn` slices them into the turn
that produced them, which is a presentation mapping rather than a change to the
loop.

Two resets, and they are not the same thing. "Reset conversation" forgets the
conversation; the RMA the demo just created is still in `data/returns.json`.
"Reset demo" also puts that data back, which is what makes the hero flow
rehearsable — it calls the same `agent.demo.reset_demo` as
`scripts/reset_demo.py`, rather than a second copy of the logic.

Run with: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from agent.config import AnthropicConfigError
from agent.demo import fresh_session, reset_demo
from agent.orchestrator import BooklyAgent
from agent.state import SessionState
from ui import render
from ui.theme import CHAT_PLACEHOLDER, PAGE_ICON, PAGE_TITLE
from ui.turns import AssistantTurn, capture_turn, pair_turns

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

# What the customer sees if the shell itself fails. One sentence, and nothing
# about why: the agent already answers an outage, an unreachable policy graph,
# and a failed tool with a sentence of its own, so this covers only the case
# where the call around it did not come back at all. Written as a comment rather
# than a docstring on purpose — Streamlit's magic renders a bare string at the
# top level of the script, and implementation notes are not customer copy.
UNAVAILABLE = "Something went wrong. Please try again, or reset the conversation."


def get_session() -> SessionState:
    """Fetch the session state for this browser tab, creating it if needed."""
    if "bookly_state" not in st.session_state:
        st.session_state.bookly_state = SessionState()
    return st.session_state.bookly_state


def get_turns() -> list[AssistantTurn]:
    """The per-turn trace records this tab has captured, oldest first.

    Held beside the session rather than on it: they are a view of what the agent
    did, and `SessionState` is what the agent needs to keep working.
    """
    if "bookly_turns" not in st.session_state:
        st.session_state.bookly_turns = []
    return st.session_state.bookly_turns


def get_agent() -> BooklyAgent:
    """Fetch the agent, constructing it once per session.

    Construction reads and validates the Anthropic configuration, so a missing
    key or model name stops here with an explanation rather than failing on the
    customer's first message.
    """
    if "bookly_agent" not in st.session_state:
        st.session_state.bookly_agent = BooklyAgent()
    return st.session_state.bookly_agent


def run_turn(agent: BooklyAgent, state: SessionState, prompt: str) -> None:
    """Take one customer message through the agent and record what it did.

    The trace offset is read *before* the turn, so everything the loop appends
    while it runs belongs to this turn.

    Args:
        agent: The agent for this session.
        state: The live session, mutated in place.
        prompt: What the customer typed.
    """
    offset = len(state.tool_traces)
    try:
        reply = agent.respond(state, prompt)
    except Exception:  # noqa: BLE001 - a customer must never meet a stack trace
        # `respond` is written not to raise: an outage, a refused guard, and a
        # failed tool all come back as an honest sentence. If something got past
        # that, the customer still gets a sentence.
        st.session_state.turn_error = UNAVAILABLE
        return

    get_turns().append(capture_turn(state, reply, trace_offset=offset))


def render_sidebar(state: SessionState) -> None:
    """The demo controls and the developer view — beside the chat, not in it."""
    with st.sidebar:
        st.markdown("### Demo controls")

        if st.button("Reset demo", type="primary", width="stretch"):
            # Everything the conversation reset does, plus the data the
            # conversation wrote. Same function the command line runs — see
            # agent/demo.py — so a rehearsal and a live run start identically.
            result = reset_demo()
            # Cleared wholesale rather than key by key, so a widget or a cache
            # added later cannot survive a reset by being forgotten. The agent is
            # the one thing carried over: it holds no conversation state — see
            # `agent.orchestrator` — and rebuilding it would re-read the
            # configuration to arrive at the same object.
            agent = st.session_state["bookly_agent"] if "bookly_agent" in st.session_state else None
            st.session_state.clear()
            if agent is not None:
                st.session_state.bookly_agent = agent
            st.session_state.bookly_state = fresh_session()
            st.session_state.reset_notice = result.summary
            st.rerun()

        if st.button("Reset conversation", width="stretch"):
            # The conversation only. Any RMA it created stays where it is.
            st.session_state.pop("bookly_state", None)
            st.session_state.pop("bookly_turns", None)
            st.rerun()

        if notice := st.session_state.pop("reset_notice", None):
            st.success(notice)

        st.divider()
        render.render_developer_state(state)


def main() -> None:
    """Render the chat UI and drive one turn per submission."""
    render.apply_branding()
    render.render_header()

    state = get_session()

    try:
        agent = get_agent()
    except AnthropicConfigError as exc:
        st.error(str(exc))
        st.stop()

    render_sidebar(state)

    if not state.messages:
        render.render_welcome()

    for message, turn in pair_turns(state.messages, get_turns()):
        render.render_exchange(message.role.value, message.content, turn)

    if error := st.session_state.pop("turn_error", None):
        st.error(error)

    if prompt := st.chat_input(CHAT_PLACEHOLDER):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Working on it…"):
            run_turn(agent, state, prompt)
        # The sidebar and the transcript were drawn before this turn ran; rerun
        # so the reply, its trace, and the session view all reflect it.
        st.rerun()


if __name__ == "__main__":
    main()
