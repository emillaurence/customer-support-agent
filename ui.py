"""How a turn is shown, not how it is decided.

Reads the records the agent wrote and turns them into something a reviewer can
read. Nothing here decides anything: no tool is called, no policy is evaluated,
no model is chosen, no trusted field is written, and `agent/` and `policy/` do
not import this module.

The shape of the screen is the design decision: **the conversation is the page,
and the trace is underneath it.** Each assistant reply is followed by one
collapsed "Agent trace" describing that reply and nothing else, so an action can
be connected to the turn that caused it without scrolling to a console.

Two registers, deliberately. Outside the trace a turn reads the way it is skimmed
— `Haiku · Policy lookup · 2 tools` — and inside it the router's own reason is
printed verbatim, because that is what a reviewer is checking. The formatters
rename a recorded reason; they never re-decide one.
"""

from __future__ import annotations

import re
from typing import Any

import streamlit as st
from pydantic import BaseModel, Field

from agent.state import (
    Message,
    Role,
    SessionState,
    ToolStatus,
    ToolTrace,
    sanitize_args,
)

# --- Branding ------------------------------------------------------------
#
# The palette lives in `.streamlit/config.toml`, Streamlit's own theming, so it
# applies to widgets without a stylesheet fighting them. What is left here is the
# small amount config cannot express.

PAGE_TITLE = "Bookly Support"
PAGE_ICON = "📚"
BRAND_NAME = "Bookly Support"
BRAND_TAGLINE = "Orders, returns, refunds, and Bookly policies."
BRAND_STATUS = "Online"
CHAT_PLACEHOLDER = "Message Bookly Support…"

WELCOME_MESSAGE = (
    "Hi, I'm Bookly Support. I can help with orders, returns, refunds, and Bookly policies."
)
"""Rendered above an empty conversation, never stored — so it is not in the
transcript the model sees, does not count as an assistant message for the
confirmation check, and does not shift the router's turn count."""

TRACE_LABEL = "Agent trace"
DEVELOPER_LABEL = "Developer state"

ACCENT = "#1F5C4F"
INK = "#1C2321"
MUTED = "#6B7280"
LINE = "#E7E3DB"
SURFACE = "#FFFFFF"
CONTENT_WIDTH = "52rem"

CSS = f"""
<style>
  /* --- The rhythm of the page -------------------------------------------
     Streamlit leaves 6rem of air above the first element and 10rem below the
     last, and its app header — an opaque 3.75rem bar that scrolled content
     hides behind — accounts for most of the first number. The bar is kept,
     because it holds the control that reopens a collapsed sidebar; it is made
     shorter, and the conversation is brought up to just under it. */
  header[data-testid="stHeader"] {{ height: 2.5rem; min-height: 2.5rem; }}
  .stMainBlockContainer, .block-container {{
      padding-top: 2.75rem; padding-bottom: 1.5rem; max-width: {CONTENT_WIDTH};
  }}
  [data-testid="stBottomBlockContainer"] {{
      padding-top: .5rem; padding-bottom: 1.1rem; max-width: {CONTENT_WIDTH};
  }}

  /* --- The brand header ------------------------------------------------- */
  .bookly-header {{ margin: 0 0 1rem; border-bottom: 1px solid {LINE}; padding-bottom: .7rem; }}
  .bookly-header .bookly-title {{ display: flex; align-items: baseline; gap: .55rem; flex-wrap: wrap; }}
  .bookly-header h1 {{ font-size: 1.45rem; font-weight: 650; margin: 0; padding: 0; color: {INK}; letter-spacing: -.01em; }}
  .bookly-header h1 span.mark {{ color: {ACCENT}; }}
  .bookly-header p {{ margin: .2rem 0 0; color: {MUTED}; font-size: .9rem; }}
  .bookly-status {{ font-size: .72rem; font-weight: 500; letter-spacing: .02em; color: {ACCENT}; white-space: nowrap; }}
  .bookly-status::before {{ content: "●"; font-size: .55rem; vertical-align: .1em; margin-right: .28rem; }}

  /* --- The messages -----------------------------------------------------
     The customer's turn keeps the filled bubble Streamlit gives it; the
     assistant's gets a white surface and a hairline, so the two read apart
     without either becoming a box. */
  [data-testid="stChatMessage"] {{ padding: .7rem .85rem; border-radius: .7rem; }}
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{
      background: {SURFACE}; border: 1px solid {LINE};
  }}

  /* Vertical space between two stacked elements is the bottom margin of the
     paragraph above, and nothing else: Streamlit offsets every markdown block
     by -1rem and relies on the block's gap to cancel it, so touching the gap
     makes elements overlap. So the reply, its metadata line, and its trace are
     drawn closer by shortening that margin — and only on the last paragraph, so
     a structured answer keeps the space between its own paragraphs. */
  [data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] > p:last-child {{
      margin-bottom: .35rem;
  }}

  /* --- The trace ---------------------------------------------------------
     Secondary to the reply above it, and dense enough that three tool calls
     do not push the next message off the screen. */
  [data-testid="stExpander"] details {{ border-color: {LINE}; }}
  [data-testid="stExpander"] summary {{ padding: .3rem .7rem; }}
  [data-testid="stExpander"] summary p {{ font-size: .82rem; color: {MUTED}; }}
  [data-testid="stExpander"] details > div[class] {{ padding: .3rem .7rem .45rem; }}
  [data-testid="stExpander"] [data-testid="stMarkdownContainer"] > p {{ margin-bottom: .15rem; }}
  /* The one place a smaller gap is safe: a trace is captions and one-line
     headings, so there is no paragraph margin for it to eat into. */
  [data-testid="stExpander"] [class*="stVerticalBlock"] {{ gap: .7rem; }}

  /* The rule path, and only the rule path, is monospace. Wraps rather than
     scrolls, so a long traversal survives a narrow window. Its bottom margin is
     a paragraph's, because the -1rem above it is what the block expects to
     cancel — without it the last line of a trace hangs out of the card. */
  .bookly-path {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: .78rem; line-height: 1.5; color: {INK};
      overflow-wrap: anywhere; margin: .05rem 0 1rem;
  }}

  /* --- The sidebar ------------------------------------------------------
     Demo controls, not a control panel. */
  section[data-testid="stSidebar"] {{ width: 17rem !important; min-width: 17rem !important; }}
  [data-testid="stSidebarHeader"] {{ height: 2.5rem; }}
  [data-testid="stSidebarUserContent"] {{ padding-top: .5rem; }}
  [data-testid="stSidebarContent"] h3 {{
      font-size: .78rem; font-weight: 600; letter-spacing: .05em;
      text-transform: uppercase; color: {MUTED};
  }}
  [data-testid="stSidebarContent"] [class*="stVerticalBlock"] {{ gap: .6rem; }}
  [data-testid="stSidebarContent"] button {{ min-height: 2.1rem; }}
  [data-testid="stSidebarContent"] hr {{ margin: .55rem 0; border-color: {LINE}; }}
</style>
"""


# --- Attaching traces to the reply that caused them ----------------------
#
# `SessionState` keeps two flat lists — every tool call, and every model turn.
# That is the right shape for the loop, which appends without knowing a UI
# exists, and the wrong shape for a reviewer, who wants one reply and what
# produced *it*. So this is a mapping rather than a change to the loop.


class AssistantTurn(BaseModel):
    """One assistant reply, and the observable activity behind it.

    Presentation only. Nothing reads this to make a decision.
    """

    reply: str = Field(description="The text the customer was shown.")
    model: str = ""
    model_tier: str = ""
    routing_reason: str = ""
    tool_traces: list[ToolTrace] = Field(
        default_factory=list, description="The tools this turn ran, in execution order."
    )

    # Latency, carried over from `ModelTurn` rather than re-measured here — this
    # module renders, it does not time anything. See `agent.state.ModelTurn` for
    # what each one means; `iterations` is the round trips this turn made.
    iterations: int = 0
    total_latency_ms: float = 0.0
    model_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    time_to_first_token_ms: float | None = None
    timed_out: bool = False

    @property
    def tool_names(self) -> list[str]:
        return [trace.tool_name for trace in self.tool_traces]


def capture_turn(state: SessionState, reply: str, *, trace_offset: int) -> AssistantTurn:
    """Build the record for the turn that just finished.

    `trace_offset` is `len(state.tool_traces)` read *before* the turn ran:
    everything appended since belongs to this turn, in execution order, because
    the loop only ever appends.
    """
    turn = state.model_turns[-1] if state.model_turns else None
    return AssistantTurn(
        reply=reply,
        model=turn.model if turn else "",
        model_tier=turn.model_tier if turn else "",
        routing_reason=turn.routing_reason if turn else "",
        tool_traces=list(state.tool_traces[trace_offset:]),
        iterations=turn.iterations if turn else 0,
        total_latency_ms=turn.total_latency_ms if turn else 0.0,
        model_latency_ms=turn.model_latency_ms if turn else 0.0,
        tool_latency_ms=turn.tool_latency_ms if turn else 0.0,
        time_to_first_token_ms=turn.time_to_first_token_ms if turn else None,
        timed_out=turn.timed_out if turn else False,
    )


def pair_turns(
    messages: list[Message], turns: list[AssistantTurn]
) -> list[tuple[Message, AssistantTurn | None]]:
    """Pair each assistant message with its turn, by position rather than content
    — so two identical replies are still two different turns. A message with no
    turn renders without a trace rather than borrowing the next one's."""
    remaining = iter(turns)
    return [
        (message, next(remaining, None) if message.role is Role.ASSISTANT else None)
        for message in messages
    ]


# --- Formatting ----------------------------------------------------------
#
# Pure functions. Every number the trace shows was measured in the loop and
# passed in; nothing here invents a latency, a status, or a policy.

STATUS_LABELS: dict[ToolStatus, str] = {
    ToolStatus.OK: "Success",
    ToolStatus.BLOCKED: "Blocked",
    ToolStatus.ERROR: "Failed",
    ToolStatus.REJECTED: "Rejected",
}
"""Plain English per outcome. "Blocked" and "Failed" are kept apart on purpose: a
guard refusing a return is the system working, and an unreachable database is not."""

STATUS_MARKS: dict[ToolStatus, str] = {
    ToolStatus.OK: "✓", ToolStatus.BLOCKED: "⊘", ToolStatus.ERROR: "✕", ToolStatus.REJECTED: "–",
}

STATUS_COLOURS: dict[ToolStatus, str] = {
    ToolStatus.OK: "green", ToolStatus.BLOCKED: "orange",
    ToolStatus.ERROR: "red", ToolStatus.REJECTED: "grey",
}

TIER_LABELS: dict[str, str] = {"haiku": "Haiku", "sonnet": "Sonnet"}
TIER_COLOURS: dict[str, str] = {"haiku": "grey", "sonnet": "violet"}
"""Routing is visible but quiet: two muted badges, neither shouting for attention."""

ACTIVITY_LABELS: dict[str, str] = {
    "informational policy lookup": "Policy lookup",
    "return or refund workflow": "Return workflow",
    "a return workflow is open": "Return workflow",
    "a return is awaiting confirmation": "Return confirmation",
    "a confirmed return is pending": "Return confirmation",
    "conversation is escalated": "Escalated",
    "escalation intent": "Escalation",
    "request to depart from policy": "Policy exception",
    "ambiguous reference to resolve": "Clarification",
}
"""A short name for what a turn was, per routing reason.

The line under a reply is read at a glance; the router's own sentence is written
to explain a decision, not to be skimmed. The full reason is still in the trace.
"""

MULTI_TURN_PREFIX = "multi-turn context"
"""The one reason carrying a number, so it is matched by prefix rather than whole."""

MULTI_TURN_LABEL = "Extended conversation"

ACTIVITY_BY_TOOL: dict[str, str] = {
    "search_policy": "Policy lookup",
    "check_return_eligibility": "Eligibility check",
    "initiate_return": "Return workflow",
    "escalate_to_human": "Escalation",
    "lookup_order": "Order lookup",
    "verify_identity": "Identity check",
}
"""What a turn was, when the routing reason does not say — first match wins,
most decisive first: a turn that evaluated eligibility is an eligibility check
even though it read an order to get there."""

CONVERSATION_LABEL = "Support question"

HOP_PATTERN = re.compile(r"^\((?P<start>[^)]+)\)-\[:(?P<rel>[^\]]+)\]->\((?P<end>[^)]+)\)$")
"""One hop of a rule path, e.g. `(PhysicalBook)-[:GOVERNED_BY]->(STANDARD_30_DAY)`."""


def format_latency(latency_ms: float) -> str:
    """Render a measured duration: `"0.4 ms"`, `"31 ms"`, `"1.2 s"`.

    Three bands, so it never rounds to zero — the tools answer in a fraction of a
    millisecond, and "0 ms" reads as a broken clock rather than a fast lookup.
    """
    if latency_ms >= 1000:
        return f"{latency_ms / 1000:.1f} s"
    if latency_ms >= 10:
        return f"{round(latency_ms)} ms"
    return f"{latency_ms:.1f} ms"


def latency_summary(turn: AssistantTurn) -> str:
    """One line proving the turn was fast, or saying why it was not.

    Three numbers, each answering a distinct question a reviewer would ask:
    how long before anything appeared (`time_to_first_token_ms` — the first
    user-visible assistant text, not the first Anthropic response), how long
    the whole turn took (`total_latency_ms`), and how many Anthropic model
    calls it took (`iterations`). Cumulative model and tool time are not
    shown here — they read as single-call durations even when a turn made
    several calls, which is misleading at a glance. Tool time stays visible
    beside each tool's own trace; model time, when more than one call was
    made, is available via `round_trip_breakdown` as secondary text. A
    timed-out turn says so, since a fallback that reads like any other reply
    would hide the one thing this line exists to surface.
    """
    parts = []
    if turn.time_to_first_token_ms is not None:
        parts.append(f"{format_latency(turn.time_to_first_token_ms)} to first text")
    parts.append(f"{format_latency(turn.total_latency_ms)} total")
    parts.append(f"{turn.iterations} model call{'s' if turn.iterations != 1 else ''}")
    line = "Timing · " + " · ".join(parts)
    return f"{line} · timed out" if turn.timed_out else line


def round_trip_breakdown(turn: AssistantTurn) -> str | None:
    """Secondary diagnostic text for a multi-call turn: the elapsed time of each
    Anthropic model call, in call order, e.g. `"Model calls · 5.8s + 1.2s"`.

    Only shown when per-call latencies were actually recorded and there is more
    than one — a single call has nothing to break down, and this never estimates
    a split that was not measured. The rounded sum reconciles with
    `model_latency_ms`, subject to the same rounding `format_latency` always has.

    `round_trip_latencies_ms` is read defensively (`getattr`, not a field on
    `AssistantTurn`) because nothing in the loop records individual call
    latencies today — only their sum, `model_latency_ms`. This returns `None`
    until that per-call recording exists; it does not synthesize a split from
    the aggregate.
    """
    latencies = getattr(turn, "round_trip_latencies_ms", None)
    if not latencies or len(latencies) < 2:
        return None
    return "Model calls · " + " + ".join(format_latency(ms) for ms in latencies)


def status_label(status: ToolStatus | str) -> str:
    return STATUS_LABELS.get(ToolStatus(status), str(status).title())


def status_mark(status: ToolStatus | str) -> str:
    """The glyph shown beside the word. A glyph alone would ask the reader to
    remember a legend."""
    return STATUS_MARKS.get(ToolStatus(status), "·")


def tier_label(model_tier: str) -> str:
    """`'haiku'` → `'Haiku'`. An unrecognised tier is shown as it was recorded."""
    return TIER_LABELS.get(model_tier.lower(), model_tier or "—")


def decision_label(eligible: bool) -> str:
    return "Eligible" if eligible else "Not eligible"


def activity_label(routing_reason: str, tool_names: list[str] | None = None) -> str:
    """A short name for what a turn did, for the line under the reply.

    The reason is looked up first, because that is the decision the system made;
    the tools are consulted only for reasons that say nothing about subject
    matter — the Haiku default, and anything unrecognised.
    """
    reason = routing_reason.strip().lower()
    if label := ACTIVITY_LABELS.get(reason):
        return label
    if reason.startswith(MULTI_TURN_PREFIX):
        return MULTI_TURN_LABEL

    ran = tool_names or []
    for tool, label in ACTIVITY_BY_TOOL.items():
        if tool in ran:
            return label
    return CONVERSATION_LABEL


def tool_count_label(count: int) -> str:
    """`"2 tools"`, `"1 tool"`, or `""` when a turn ran none — an empty string
    rather than "0 tools", so the metadata line drops the segment entirely."""
    if count <= 0:
        return ""
    return f"{count} tool{'s' if count != 1 else ''}"


def display_args(tool_args: dict[str, Any]) -> dict[str, Any]:
    """The arguments a trace may show, sanitized and stripped of empties.

    Sanitized a second time — the loop did it on the way in, and this is the last
    gate before a screen, where another pass costs nothing.
    """
    return {
        key: value
        for key, value in sanitize_args(tool_args).items()
        if value is not None and value != ""
    }


def format_args(tool_args: dict[str, Any]) -> str:
    """The arguments as one readable line, or a dash when there were none."""
    shown = display_args(tool_args)
    if not shown:
        return "—"
    return ", ".join(f"{key}={value}" for key, value in shown.items())


def rule_path_nodes(rule_path: list[str]) -> list[str]:
    """The graph nodes a rule path visits, in the order it reached them.

    A hop that does not parse is kept verbatim rather than dropped — a path with a
    hop missing would misrepresent the decision.
    """
    nodes: list[str] = []
    for hop in rule_path:
        match = HOP_PATTERN.match(hop.strip())
        for part in ([match["start"], match["end"]] if match else [hop]):
            if part not in nodes:
                nodes.append(part)
    return nodes


def format_rule_path(rule_path: list[str]) -> str:
    """The path as one chain — `PhysicalBook → STANDARD_30_DAY`.

    Empty when there is no path: a refusal that never reached a policy has none,
    and inventing one would suggest a traversal that never happened.

    Kept for the raw graph traversal (debugging, tests); the trace itself shows
    `format_policy_path` instead, which does not flatten override/region edges
    into a fake linear chain.
    """
    return " → ".join(rule_path_nodes(rule_path))


def policy_path_steps(decision: dict[str, Any]) -> list[str]:
    """The business-friendly decision path for one eligibility check: input
    context, the policy that applies, what blocked it (if anything), and the
    outcome — never the raw graph traversal behind the match.

    Built only from this call's own structured fields (`product_type`,
    `region`, `policy_id`, `return_window_days`, `existing_return_id`), so it
    cannot drift into implying the policy itself caused a rejection that was
    really an existing return, or borrow a sibling item's decision.
    """
    steps: list[str] = []

    product_type = decision.get("product_type")
    region = decision.get("region")
    if product_type:
        steps.append(f"{product_type} + {region}" if region else product_type)

    if policy_id := decision.get("policy_id"):
        window_days = decision.get("return_window_days")
        steps.append(f"{policy_id} ({window_days} days)" if window_days else policy_id)

    if existing_return_id := decision.get("existing_return_id"):
        steps.append(f"Existing return: {existing_return_id}")

    if steps:
        steps.append(decision_label(bool(decision.get("eligible"))))
    return steps


def format_policy_path(decision: dict[str, Any]) -> str:
    """`policy_path_steps` joined for display, or empty when nothing applied."""
    return " → ".join(policy_path_steps(decision))


# --- Rendering -----------------------------------------------------------
#
# Thin, because the formatting above did the thinking.


def apply_branding() -> None:
    """Inject the small stylesheet. Called once per run, before anything renders."""
    st.markdown(CSS, unsafe_allow_html=True)


def render_header() -> None:
    """The Bookly title, a quiet status, and one line saying what the agent does."""
    st.markdown(
        f"""
        <div class="bookly-header">
          <div class="bookly-title">
            <h1><span class="mark">📚</span> {BRAND_NAME}</h1>
            <span class="bookly-status">{BRAND_STATUS}</span>
          </div>
          <p>{BRAND_TAGLINE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome() -> None:
    """The opening greeting, above an empty conversation."""
    with st.chat_message("assistant"):
        st.markdown(WELCOME_MESSAGE)


def render_exchange(role: str, content: str, turn: AssistantTurn | None) -> None:
    """One message, with its trace underneath when it is an assistant turn."""
    with st.chat_message(role):
        st.markdown(content)
        if turn is not None:
            render_trace(turn)


def render_trace(turn: AssistantTurn) -> None:
    """The metadata line and the collapsed trace for one assistant turn.

    Observable execution only: the model and why, the tools in order, latencies,
    statuses, sanitized arguments, and — for an eligibility check — the policy and
    graph path. No reasoning; none is captured anywhere in this repo.
    """
    if turn.model_tier:
        # `Haiku · Policy lookup · 2 tools`, with the tier as a small badge.
        # Secondary by construction: the customer's answer is the page.
        colour = TIER_COLOURS.get(turn.model_tier.lower(), "grey")
        parts = [activity_label(turn.routing_reason, turn.tool_names)]
        if count := tool_count_label(len(turn.tool_traces)):
            parts.append(count)
        st.caption(f":{colour}-badge[{tier_label(turn.model_tier)}] {' · '.join(parts)}")

    if not turn.tool_traces:
        # Nothing ran, and an expander promising a trace that says "no tools" is a
        # click for no reason.
        return

    with st.expander(TRACE_LABEL, expanded=False):
        # The deterministic routing reason is kept verbatim here. The line outside
        # renames it for reading; this is the one the router actually recorded.
        st.markdown(f"**Model** · {tier_label(turn.model_tier)} · `{turn.model or '—'}`")
        st.caption(f"Routing · {turn.routing_reason or '—'}")
        st.caption(latency_summary(turn))
        if breakdown := round_trip_breakdown(turn):
            st.caption(breakdown)
        for position, trace in enumerate(turn.tool_traces, start=1):
            render_tool_trace(trace, position=position)


def render_tool_trace(trace: ToolTrace, *, position: int) -> None:
    """One tool call: what was asked, what came back, and how long it took.

    Two lines rather than a card — a scannable headline and a caption — so a turn
    with three tool calls still fits on a screen.
    """
    colour = STATUS_COLOURS.get(trace.status, "grey")
    st.markdown(
        f"**{position} · {trace.tool_name}** "
        f":{colour}-badge[{status_mark(trace.status)} {status_label(trace.status)}] "
        f":grey[{format_latency(trace.latency_ms)}]"
    )
    st.caption(f"{trace.result_summary or '—'} · {format_args(trace.tool_args)}")

    # The technical detail of a failure belongs here, where a reviewer looked for
    # it, and not in the reply the customer read.
    if trace.error:
        st.caption(f"Detail · {trace.error}")

    # Read off this call, not off the session: a turn that checks two items holds
    # only the last decision in state, and the first check must still show its own.
    if trace.policy_decision:
        render_policy_decision(trace.policy_decision)


def render_policy_decision(decision: dict[str, Any]) -> None:
    """The graph-backed part of an eligibility check: policy, decision, path.

    Showing the path is what distinguishes an *evaluated* answer from a guessed
    one. The Cypher is not shown: the traversal is the explanation, the query is
    an implementation detail. The path itself is the business-friendly decision
    path (input context → applicable policy → blocker → outcome), not the raw
    Neo4j hops — an applicable policy and the reason it was blocked are kept
    distinct, so `AU_BOOKLY_EXTENDED_RETURN` never reads as having caused a
    rejection that was really an existing return.
    """
    st.markdown(
        f"**Policy** · {decision.get('policy_id') or '—'} "
        f"· **Decision** · {decision_label(bool(decision.get('eligible')))}"
    )
    if path := format_policy_path(decision):
        st.caption("Policy path")
        st.markdown(f'<div class="bookly-path">{path}</div>', unsafe_allow_html=True)


def render_developer_state(state: SessionState) -> None:
    """The debug view: the session's trusted fields, and nothing else.

    Deliberately absent: the eligibility token's value, the customer record, the
    API key, the environment. What is here is what a guard reads.
    """
    with st.expander(DEVELOPER_LABEL, expanded=False):
        eligibility = state.eligibility
        confirmed_count = sum(1 for pending in state.pending_returns if pending.confirmed)
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
            # Counts, never the tokens — a session can hold more than one pending
            # return, each with its own, and the value itself is a credential.
            "Pending returns": str(len(state.pending_returns)),
            "Confirmed returns": str(confirmed_count),
            "Escalated": _yes_no(state.escalated),
            "May mutate": _yes_no(state.may_mutate),
            "Tool calls": str(len(state.tool_traces)),
        }
        for label, value in rows.items():
            st.caption(f"**{label}** · {value}")


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
