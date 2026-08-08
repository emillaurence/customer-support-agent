"""The Streamlit calls. Thin, because `ui.format` and `ui.turns` did the thinking.

The shape of the screen is the design decision here: **the conversation is the
page, and the trace is underneath it.** Each assistant reply is followed by one
collapsed "Agent trace" expander describing that reply and nothing else, so a
reviewer can connect an action to the turn that caused it without scrolling to a
console at the bottom.

What the trace shows is observable execution only: which model handled the turn
and why, which tools ran in which order, how long each took, whether it
succeeded, the sanitized arguments, a one-line result, and — for an eligibility
check — the policy and the graph path behind the decision. It does not show
reasoning: no `thinking` content is captured anywhere in this repo, so there is
none here to render.
"""

from __future__ import annotations

import streamlit as st

from agent.state import SessionState
from agent.tracing import ToolStatus, ToolTrace
from ui.format import (
    decision_label,
    format_args,
    format_latency,
    format_rule_path,
    status_label,
    tier_label,
)
from ui.theme import BRAND_NAME, BRAND_TAGLINE, CSS, WELCOME_MESSAGE
from ui.turns import AssistantTurn

TRACE_LABEL = "Agent trace"
DEVELOPER_LABEL = "Developer state"

STATUS_COLOURS: dict[ToolStatus, str] = {
    ToolStatus.OK: "green",
    ToolStatus.BLOCKED: "orange",
    ToolStatus.ERROR: "red",
    ToolStatus.REJECTED: "grey",
}
"""Colour per outcome. Blocked is amber, not red — a guard refusing is not a fault."""

TIER_COLOURS: dict[str, str] = {"haiku": "grey", "sonnet": "violet"}
"""Routing is visible but quiet: two muted badges, neither shouting for attention."""


# --- Branding ------------------------------------------------------------


def apply_branding() -> None:
    """Inject the small stylesheet. Called once per run, before anything renders."""
    st.markdown(CSS, unsafe_allow_html=True)


def render_header() -> None:
    """The Bookly title and one line saying what the agent can do."""
    st.markdown(
        f"""
        <div class="bookly-header">
          <h1><span class="mark">📚</span> {BRAND_NAME}</h1>
          <p>{BRAND_TAGLINE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome() -> None:
    """The opening greeting, above an empty conversation.

    Rendered, never stored: it is not a turn, so it does not reach the model's
    transcript, the confirmation check, or the router's turn count.
    """
    with st.chat_message("assistant"):
        st.markdown(WELCOME_MESSAGE)


# --- The conversation ----------------------------------------------------


def render_exchange(role: str, content: str, turn: AssistantTurn | None) -> None:
    """One message, with its trace underneath when it is an assistant turn.

    Args:
        role: `"user"` or `"assistant"`.
        content: The message text.
        turn: The turn that produced it, for an assistant message with a record.
    """
    with st.chat_message(role):
        st.markdown(content)
        if turn is not None:
            render_trace(turn)


def render_trace(turn: AssistantTurn) -> None:
    """The routing line and the collapsed trace for one assistant turn."""
    render_model_line(turn)

    if not turn.tool_traces:
        # Nothing ran, and an expander promising a trace that says "no tools" is
        # a click for no reason. The model line above already said what handled
        # the turn.
        return

    with st.expander(TRACE_LABEL, expanded=False):
        render_model_summary(turn)
        for trace in turn.tool_traces:
            render_tool_trace(trace, turn)


def render_model_line(turn: AssistantTurn) -> None:
    """The model badge under a reply: visible, deliberately quiet.

    A small badge and a grey line, not a headline — the customer's answer is the
    thing on the page, and the routing decision sits beside it.
    """
    if not turn.model_tier:
        return

    colour = TIER_COLOURS.get(turn.model_tier.lower(), "grey")
    st.markdown(
        f":{colour}-badge[Model: {tier_label(turn.model_tier)}] "
        f":grey[{model_line_detail(turn)}]"
    )


def model_line_detail(turn: AssistantTurn) -> str:
    """The grey text beside the model badge: why this tier, and how much ran."""
    parts = [part for part in (turn.routing_reason,) if part]
    if turn.tool_traces:
        count = len(turn.tool_traces)
        parts.append(f"{count} tool call{'s' if count != 1 else ''}")
    return " · ".join(parts)


def render_model_summary(turn: AssistantTurn) -> None:
    """Inside the trace: which model, which id, and the rule that chose it."""
    st.markdown(f"**Model** · {tier_label(turn.model_tier)}")
    st.caption(f"{turn.model or '—'} — routed because {turn.routing_reason or '—'}")


def render_tool_trace(trace: ToolTrace, turn: AssistantTurn) -> None:
    """One tool call: what was asked, what came back, and how long it took."""
    colour = STATUS_COLOURS.get(trace.status, "grey")
    with st.container(border=True):
        st.markdown(
            f"**{trace.tool_name}** · {format_latency(trace.latency_ms)} "
            f":{colour}-badge[{status_label(trace.status)}]"
        )

        st.caption(f"Arguments · {format_args(trace.tool_args)}")
        st.caption(f"Result · {trace.result_summary or '—'}")

        # The technical detail of a failure belongs here, where a reviewer looked
        # for it, and not in the reply the customer read.
        if trace.error:
            st.caption(f"Detail · {trace.error}")

        if turn.eligibility is not None and trace.tool_name == "check_return_eligibility":
            render_policy_decision(turn)


def render_policy_decision(turn: AssistantTurn) -> None:
    """The graph-backed part of an eligibility check: policy, decision, path.

    The point of showing the path is that the answer was *evaluated*, not
    guessed: these are the hops from the item's category to the policy that won,
    including the region or promotion that made a conditional policy apply. The
    Cypher that walked them is not shown — the traversal is the explanation, the
    query is an implementation detail.
    """
    decision = turn.eligibility
    if decision is None:
        return

    st.markdown(f"**Policy** · {decision.policy_id or '—'}")
    st.markdown(f"**Decision** · {decision_label(decision.eligible)}")

    path = format_rule_path(decision.rule_path)
    if path:
        st.markdown("**Rule path**")
        st.markdown(f'<div class="bookly-path">{path}</div>', unsafe_allow_html=True)


# --- Developer state -----------------------------------------------------


def render_developer_state(state: SessionState) -> None:
    """The debug view: the session's current trusted fields, and nothing else.

    Separate from the trace, and collapsed, because this is not the customer's
    experience. Deliberately absent: the eligibility token's value, the customer
    record, the API key, anything from the environment, and the model's
    reasoning. What is here is what a guard actually reads.
    """
    with st.expander(DEVELOPER_LABEL, expanded=False):
        eligibility = state.eligibility
        rows = {
            "Session": state.session_id,
            "Verified customer": state.verified_customer_id or "—",
            "Region": state.customer_region or "—",
            "Active order": state.active_order_id or "—",
            "Active item": state.active_item_id or "—",
            "Return reason": state.return_reason or "—",
            "Eligibility": (
                f"{eligibility.policy_id or '—'} · {decision_label(eligibility.eligible)}"
                if eligibility
                else "—"
            ),
            # Whether a token is held, never the token. The value is a
            # credential; its presence is the interesting part.
            "Eligibility token": "held" if state.eligibility_token else "none",
            "Awaiting confirmation": _yes_no(state.pending_return is not None),
            "Confirmed": _yes_no(state.confirmed),
            "Escalated": _yes_no(state.escalated),
            "May mutate": _yes_no(state.may_mutate),
            "Tool calls": str(len(state.tool_traces)),
        }
        for label, value in rows.items():
            st.caption(f"**{label}** · {value}")


def _yes_no(value: bool) -> str:
    """A boolean, for a reader rather than for Python."""
    return "yes" if value else "no"
