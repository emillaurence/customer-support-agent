"""Attaching the traces to the assistant turn that caused them.

`SessionState` keeps two flat lists — every tool call this session, and every
model turn. That is the right shape for the orchestrator, which appends to them
without knowing a UI exists, and the wrong shape for a reviewer, who wants to
look at one reply and see what produced *it*.

This module is that mapping, and it is deliberately a mapping rather than a
change to the loop: the orchestrator keeps its flat lists, and the UI records
where each turn started.

`capture_turn` is called once per reply, with the number of traces the session
held *before* the turn ran. Everything appended since belongs to this turn, in
execution order, because the loop only ever appends.

The eligibility decision is the one thing that is snapshotted rather than sliced.
A `ToolTrace` records the call, not the policy behind it, so the graph path is
read off `state.eligibility` and matched against the call's own order and item
before it is attached — a decision that has since been cleared or replaced is not
shown against a turn it did not belong to.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent.models import EligibilityDecision, Role
from agent.state import Message, SessionState
from agent.tracing import ToolStatus, ToolTrace

ELIGIBILITY_TOOL = "check_return_eligibility"
"""The one tool whose trace has a graph decision worth showing beside it."""


class AssistantTurn(BaseModel):
    """One assistant reply, and the observable activity behind it.

    Presentation only. Nothing reads this to make a decision; it exists so the
    trace under a reply describes that reply.
    """

    reply: str = Field(description="The text the customer was shown.")
    model: str = Field(default="", description="The model id that handled the turn.")
    model_tier: str = Field(default="", description="'haiku' or 'sonnet'.")
    routing_reason: str = Field(default="", description="The rule that chose the tier.")
    tool_traces: list[ToolTrace] = Field(
        default_factory=list, description="The tools this turn ran, in execution order."
    )
    eligibility: EligibilityDecision | None = Field(
        default=None,
        description="The decision behind this turn's eligibility check, when it made one.",
    )

    @property
    def tool_names(self) -> list[str]:
        """The tools this turn ran, oldest first."""
        return [trace.tool_name for trace in self.tool_traces]

    @property
    def had_failure(self) -> bool:
        """Whether anything this turn asked for did not succeed.

        Includes a blocked guard: the customer's request did not happen, which is
        worth flagging in the trace even though the system behaved correctly.
        """
        return any(trace.status is not ToolStatus.OK for trace in self.tool_traces)


def capture_turn(state: SessionState, reply: str, *, trace_offset: int) -> AssistantTurn:
    """Build the record for the turn that just finished.

    Args:
        state: The session, immediately after `BooklyAgent.respond` returned.
        reply: The reply that was shown to the customer.
        trace_offset: `len(state.tool_traces)` read *before* the turn ran.

    Returns:
        The turn, with its own traces and — when it ran an eligibility check —
        the decision that check produced.
    """
    turn = state.model_turns[-1] if state.model_turns else None
    traces = list(state.tool_traces[trace_offset:])
    return AssistantTurn(
        reply=reply,
        model=turn.model if turn else "",
        model_tier=turn.model_tier if turn else "",
        routing_reason=turn.routing_reason if turn else "",
        tool_traces=traces,
        eligibility=eligibility_for(state, traces),
    )


def eligibility_for(
    state: SessionState, traces: list[ToolTrace]
) -> EligibilityDecision | None:
    """The decision belonging to this turn's eligibility check, if there is one.

    Three conditions, all required, because a decision shown against the wrong
    turn is worse than no decision at all:

    * the turn actually ran a successful `check_return_eligibility`;
    * the session still holds a decision — a return opened later in the same turn
      clears it, and a cleared decision is not this turn's to display;
    * the held decision is for the order and item that were checked.

    Args:
        state: The session after the turn.
        traces: That turn's traces.

    Returns:
        The decision, or None.
    """
    checks = [
        trace
        for trace in traces
        if trace.tool_name == ELIGIBILITY_TOOL and trace.status is ToolStatus.OK
    ]
    if not checks or state.eligibility is None:
        return None

    args = checks[-1].tool_args
    if args.get("order_id") != state.active_order_id:
        return None
    if args.get("item_id") != state.active_item_id:
        return None
    return state.eligibility


def pair_turns(
    messages: list[Message], turns: list[AssistantTurn]
) -> list[tuple[Message, AssistantTurn | None]]:
    """Walk the visible transcript, pairing each assistant message with its turn.

    Assistant messages and captured turns are produced one for one, in order, so
    they are matched by position rather than by content — two identical replies in
    one conversation are still two different turns.

    A message with no turn is paired with None. That happens for a user message,
    and for an assistant message the UI has no record for — a session restored
    without its traces, for instance. It renders without a trace rather than
    borrowing the next one.

    Args:
        messages: `SessionState.messages`.
        turns: The turns captured by this UI, oldest first.

    Returns:
        One entry per message, in transcript order.
    """
    remaining = iter(turns)
    return [
        (message, next(remaining, None) if message.role is Role.ASSISTANT else None)
        for message in messages
    ]
