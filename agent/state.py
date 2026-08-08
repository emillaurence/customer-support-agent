"""Typed session state for a single Bookly conversation.

One `SessionState` per Streamlit session. It is the only mutable thing the
agent carries between turns — tools stay stateless.

Two transcripts, deliberately. `messages` is what the customer sees: plain text,
one entry per turn. `transcript` is what Anthropic sees: the same conversation
plus the tool_use and tool_result blocks, in the Messages API's own shape. They
are not the same thing and merging them would mean either showing the customer
tool plumbing or hiding tool results from the model.

Everything else here is *trusted* — written only by the orchestrator, and only
from a tool result that actually succeeded. The model can ask for a tool; it
cannot set `verified_customer_id`, mint an `eligibility_token`, or flip
`confirmed`. That separation is the point of the file.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from agent.models import EligibilityDecision, Role
from agent.tracing import ModelTurn, ToolTrace


class Message(BaseModel):
    """One turn of the visible conversation."""

    role: Role
    content: str


class PendingReturn(BaseModel):
    """A return that passed eligibility and is waiting on the customer's yes.

    Created only from a successful, eligible `check_return_eligibility`. Its
    existence is what makes a later "yes" mean something — see
    `agent.confirmation`.

    It carries the order and item so the confirmation is *for a specific return*.
    If the customer switches item, the pending action is dropped along with the
    token, and the old yes cannot be spent on the new item.
    """

    order_id: str
    item_id: str
    eligibility_token: str
    asked: bool = Field(
        default=False,
        description="True once the agent has actually asked the customer to confirm this return.",
    )


class SessionState(BaseModel):
    """Everything the agent knows about the conversation so far."""

    session_id: str = Field(default_factory=lambda: f"SESS-{uuid.uuid4().hex[:8].upper()}")

    messages: list[Message] = Field(
        default_factory=list, description="The visible transcript, for the UI."
    )
    transcript: list[dict[str, Any]] = Field(
        default_factory=list,
        description="The Anthropic Messages API conversation, including tool blocks.",
    )

    verified_customer_id: str | None = Field(
        default=None,
        description="Set once identity is confirmed. Tools that expose order data require it.",
    )
    customer_region: str | None = Field(
        default=None,
        description="ISO country code of the verified customer. Selects regional policy overrides.",
    )
    active_order_ids: list[str] = Field(
        default_factory=list,
        description="The verified customer's live orders. More than one means the agent must ask which.",
    )
    active_order_id: str | None = Field(
        default=None,
        description="The order under discussion. None when the customer has several and has not chosen.",
    )
    active_item_id: str | None = Field(
        default=None,
        description="The line item under discussion. Returns are per item, not per order.",
    )
    return_reason: str | None = Field(
        default=None,
        description="The customer's stated reason, in their own words. Not paraphrased.",
    )
    return_intent_expressed: bool = Field(
        default=False,
        description=(
            "True once the customer has asked about a return, even before an item is "
            "chosen. Keeps the turns spent working out which book they mean on Sonnet."
        ),
    )
    eligibility: EligibilityDecision | None = Field(
        default=None,
        description="The last decision from check_return_eligibility, kept so it can be quoted.",
    )
    eligibility_token: str | None = Field(
        default=None,
        description="Token from an eligible decision. initiate_return will not act without it.",
    )
    pending_return: PendingReturn | None = Field(
        default=None,
        description="The specific return a 'yes' would authorise. None means a 'yes' authorises nothing.",
    )
    confirmed: bool = Field(
        default=False,
        description=(
            "True once the customer has explicitly said yes to the described action. "
            "Bookkeeping only — it must be passed explicitly to initiate_return, which "
            "does its own check and never reads session state."
        ),
    )
    escalated: bool = Field(
        default=False,
        description="True once handed to a human — the agent should stop acting.",
    )

    tool_traces: list[ToolTrace] = Field(
        default_factory=list, description="Every tool call this session, oldest first."
    )
    model_turns: list[ModelTurn] = Field(
        default_factory=list, description="Which model handled each turn, and why."
    )

    @property
    def is_verified(self) -> bool:
        """Whether identity has been confirmed this session."""
        return self.verified_customer_id is not None

    @property
    def user_turn_count(self) -> int:
        """How many times the customer has spoken. Used by the router."""
        return sum(1 for message in self.messages if message.role is Role.USER)

    @property
    def return_workflow_active(self) -> bool:
        """Whether a return is being worked on right now.

        True from the moment the customer says they want to return something, not
        just once a token exists. Picking which book they mean is part of the
        workflow, and "Designing Data-Intensive Applications" is not a sentence
        with a return keyword in it — so without this the turn that chooses the
        item, and therefore the turn that runs the eligibility check, would drop
        back to the cheaper model in the middle of a return.
        """
        return any(
            (
                self.return_intent_expressed,
                self.active_item_id is not None,
                self.return_reason is not None,
                self.eligibility is not None,
                self.eligibility_token is not None,
                self.pending_return is not None,
            )
        )

    @property
    def may_mutate(self) -> bool:
        """Whether a write is permitted right now.

        Three gates, all required: identity, a passing eligibility check with a
        token, and an explicit confirmation from the customer.

        This is the orchestrator's own check, not the safety boundary. The write
        tool re-checks: `initiate_return` takes `eligibility_token` and
        `confirmed` as arguments, so a bug here cannot let a mutation through.
        """
        return self.is_verified and self.eligibility_token is not None and self.confirmed

    def add_message(self, role: Role, content: str) -> None:
        """Append a turn to the visible transcript."""
        self.messages.append(Message(role=role, content=content))

    def last_assistant_message(self) -> str | None:
        """The agent's most recent reply, if it has spoken.

        Read by the confirmation check: a "yes" only counts if what it answers
        was a question.
        """
        return next(
            (m.content for m in reversed(self.messages) if m.role is Role.ASSISTANT), None
        )

    def clear_return_context(self) -> None:
        """Drop everything tied to one return attempt.

        Called when the customer switches order or item, so a token issued for
        one item can never be spent on another, and a yes given for one item
        cannot authorise a return of the next.
        """
        self.active_order_id = None
        self.active_item_id = None
        self.return_reason = None
        self.return_intent_expressed = False
        self.eligibility = None
        self.eligibility_token = None
        self.pending_return = None
        self.confirmed = False
