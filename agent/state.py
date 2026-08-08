"""Typed session state for a single Bookly conversation.

One `SessionState` per Streamlit session. It is the only mutable thing the
agent carries between turns — tools stay stateless.

Deliberately minimal: just enough to run the planned flow

    verify → find the order → pick the item → check eligibility → confirm → act

and nothing more. No analytics, no timers, no tool-call history yet.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent.models import EligibilityDecision, Role


class Message(BaseModel):
    """One turn of the visible conversation."""

    role: Role
    content: str


class SessionState(BaseModel):
    """Everything the agent knows about the conversation so far."""

    messages: list[Message] = Field(default_factory=list)

    verified_customer_id: str | None = Field(
        default=None,
        description="Set once identity is confirmed. Tools that expose order data require it.",
    )
    customer_region: str | None = Field(
        default=None,
        description="ISO country code of the verified customer. Selects regional policy overrides.",
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
    eligibility: EligibilityDecision | None = Field(
        default=None,
        description="The last decision from check_return_eligibility, kept so it can be quoted.",
    )
    eligibility_token: str | None = Field(
        default=None,
        description="Token from an eligible decision. initiate_return will not act without it.",
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

    @property
    def is_verified(self) -> bool:
        """Whether identity has been confirmed this session."""
        return self.verified_customer_id is not None

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
        """Append a turn to the transcript."""
        self.messages.append(Message(role=role, content=content))

    def clear_return_context(self) -> None:
        """Drop everything tied to one return attempt.

        Called when the customer switches order or item, so a token issued for
        one item can never be spent on another.
        """
        self.active_order_id = None
        self.active_item_id = None
        self.return_reason = None
        self.eligibility = None
        self.eligibility_token = None
        self.confirmed = False
