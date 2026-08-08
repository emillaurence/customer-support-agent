"""Pure formatting for the agent trace. Strings in, strings out.

No Streamlit import, no session, no side effects — which is what makes the trace
testable without a browser. Every number the trace shows is measured somewhere
else and passed in here; nothing in this module invents a latency, a status, or a
policy.

Two things it is careful about:

* **It never widens what a trace exposes.** `display_args` re-runs
  `agent.tracing.sanitize_args` even though the orchestrator already sanitized
  what it stored, because this is the last place before a value reaches a screen
  and the cost of the second pass is nothing.
* **It reads the graph path, it does not re-derive it.** `rule_path_nodes` parses
  the hops `check_return_eligibility` recorded. If a hop is in a shape it does
  not recognise it is shown verbatim rather than dropped — a path with a hop
  missing would misrepresent the decision.
"""

from __future__ import annotations

import re
from typing import Any

from agent.tracing import ToolStatus, sanitize_args

HOP_PATTERN = re.compile(r"^\((?P<start>[^)]+)\)-\[:(?P<rel>[^\]]+)\]->\((?P<end>[^)]+)\)$")
"""One hop of a rule path, in the arrow notation `tools.search_policy` writes.

For example `(PhysicalBook)-[:GOVERNED_BY]->(STANDARD_30_DAY)`.
"""

STATUS_LABELS: dict[ToolStatus, str] = {
    ToolStatus.OK: "Success",
    ToolStatus.BLOCKED: "Blocked",
    ToolStatus.ERROR: "Failed",
    ToolStatus.REJECTED: "Rejected",
}
"""Plain English for each outcome.

"Blocked" and "Failed" are kept apart on purpose: a guard refusing a return is
the system working, and an unreachable database is not.
"""

STATUS_MARKS: dict[ToolStatus, str] = {
    ToolStatus.OK: "✓",
    ToolStatus.BLOCKED: "⊘",
    ToolStatus.ERROR: "✕",
    ToolStatus.REJECTED: "–",
}
"""One glyph per outcome, so a trace of three tools can be scanned in a glance.

The word is kept beside it — a glyph alone asks the reader to remember a legend.
"""

TIER_LABELS: dict[str, str] = {"haiku": "Haiku", "sonnet": "Sonnet"}
"""Tier names as they are written for a reader. Anything else is shown as given."""

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

The line under a reply is read at a glance, and the router's own sentence — "a
return is awaiting confirmation" — is written to explain a decision rather than
to be skimmed. The full reason is still shown inside the trace, where a reviewer
went looking for it.
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
"""What a turn was, when the routing reason does not say — first match wins.

Ordered most decisive first: a turn that evaluated eligibility is an eligibility
check even though it read an order to get there.
"""

CONVERSATION_LABEL = "Support question"
"""A turn that ran nothing and was routed on nothing in particular."""


def format_latency(latency_ms: float) -> str:
    """Render a measured duration for reading.

    Three bands so the number stays short without ever rounding to zero: a
    sub-10ms call keeps one decimal, a millisecond call is a whole number, and
    anything over a second is expressed in seconds.

    Args:
        latency_ms: Milliseconds, as recorded on the trace.

    Returns:
        For example `"0.4 ms"`, `"31 ms"`, `"1.2 s"`.
    """
    if latency_ms >= 1000:
        return f"{latency_ms / 1000:.1f} s"
    if latency_ms >= 10:
        return f"{round(latency_ms)} ms"
    return f"{latency_ms:.1f} ms"


def status_label(status: ToolStatus | str) -> str:
    """The reader-facing word for a tool outcome."""
    return STATUS_LABELS.get(ToolStatus(status), str(status).title())


def status_mark(status: ToolStatus | str) -> str:
    """The glyph for a tool outcome, shown beside the word."""
    return STATUS_MARKS.get(ToolStatus(status), "·")


def tier_label(model_tier: str) -> str:
    """`'haiku'` → `'Haiku'`. An unrecognised tier is shown as it was recorded."""
    return TIER_LABELS.get(model_tier.lower(), model_tier or "—")


def activity_label(routing_reason: str, tool_names: list[str] | None = None) -> str:
    """A short name for what a turn did, for the line under the reply.

    Presentation only: it renames the router's reason, it does not second-guess
    it. The reason is looked up first, because that is the decision the system
    actually made; the tools are consulted only for the reasons that say nothing
    about subject matter — the Haiku default, and anything unrecognised.

    Args:
        routing_reason: `ModelDecision.reason`, as recorded on the turn.
        tool_names: The tools the turn ran, in execution order.

    Returns:
        For example `"Policy lookup"`, `"Return workflow"`, `"Order lookup"`.
    """
    reason = routing_reason.strip()
    if label := ACTIVITY_LABELS.get(reason.lower()):
        return label
    if reason.lower().startswith(MULTI_TURN_PREFIX):
        return MULTI_TURN_LABEL

    ran = tool_names or []
    for tool, label in ACTIVITY_BY_TOOL.items():
        if tool in ran:
            return label
    return CONVERSATION_LABEL


def tool_count_label(count: int) -> str:
    """`"2 tools"`, `"1 tool"`, or `""` when a turn ran none.

    An empty string rather than "0 tools": a turn that ran nothing has nothing to
    report, and the metadata line drops the segment entirely.
    """
    if count <= 0:
        return ""
    return f"{count} tool{'s' if count != 1 else ''}"


def display_args(tool_args: dict[str, Any]) -> dict[str, Any]:
    """The arguments a trace may show, sanitized and stripped of empties.

    Sanitized a second time — the orchestrator already did it on the way in, and
    this is the last gate on the way out. An argument that is None or blank is
    dropped rather than shown as an empty row; it says nothing about the call.

    Args:
        tool_args: The trace's recorded arguments.

    Returns:
        A new dict, safe to display.
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

    The recorded path is a list of hops; a reader wants the chain. Each hop
    contributes its start and its end, and a node already in the chain is not
    repeated — so the regional hop that made a conditional policy apply shows up
    once, next to the policy it granted.

    A hop that does not parse is kept verbatim as its own entry. Dropping it
    would show a shorter path than the one the decision actually walked.

    Args:
        rule_path: `EligibilityDecision.rule_path`.

    Returns:
        Node names, first to last. Empty when the path is empty.
    """
    nodes: list[str] = []
    for hop in rule_path:
        match = HOP_PATTERN.match(hop.strip())
        parts = [match["start"], match["end"]] if match else [hop]
        for part in parts:
            if part not in nodes:
                nodes.append(part)
    return nodes


def format_rule_path(rule_path: list[str]) -> str:
    """The rule path as the trace prints it: one chain, arrows between the hops.

    For example::

        PhysicalBook → STANDARD_30_DAY

    Written on one line so the traversal reads as a chain rather than a list, and
    so a narrow window wraps it instead of scrolling it.

    Args:
        rule_path: `EligibilityDecision.rule_path`.

    Returns:
        The nodes joined by arrows, or `""` when there is no path — a refusal
        that never reached a policy has none, and inventing one would suggest a
        traversal that never happened.
    """
    return " → ".join(rule_path_nodes(rule_path))


def decision_label(eligible: bool) -> str:
    """`Eligible` or `Not eligible` — the decision, in one word each way."""
    return "Eligible" if eligible else "Not eligible"
