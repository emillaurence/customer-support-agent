"""What the agent remembers, and what it writes down.

Three things, in this order: **domain records** mirroring the mock JSON in
`data/` (policy is not among them — it comes from Neo4j); **execution records**,
`ToolTrace` and `ModelTurn` plus the sanitizing that keeps a trace safe to store
and to show; and **`SessionState`**, the only mutable thing the agent carries
between turns.

Two transcripts, deliberately. `messages` is what the customer sees; `transcript`
is what Anthropic sees, the same conversation plus tool blocks. Merging them
would mean either showing the customer tool plumbing or hiding tool results from
the model.

Everything else on `SessionState` is *trusted* — written only by the agent loop,
and only from a tool result that actually succeeded. The model can ask for a
tool; it cannot set `verified_customer_id`, mint an `eligibility_token`, or flip
`confirmed`. That separation is the point of the file.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from policy.policy import ProductType


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class OrderStatus(StrEnum):
    PLACED = "placed"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class ReturnStatus(StrEnum):
    REQUESTED = "requested"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"


# --- Domain records ------------------------------------------------------


class Customer(BaseModel):
    customer_id: str
    name: str
    email: str
    country: str = Field(description="ISO 3166-1 alpha-2. Drives regional overrides.")
    note: str | None = Field(default=None, description="Fixture annotation. Never shown.")


class Item(BaseModel):
    item_id: str
    title: str
    product_type: ProductType
    price: float
    currency: str = "USD"


class OrderItem(BaseModel):
    item_id: str
    quantity: int
    unit_price: float


class Order(BaseModel):
    """`delivered_at` is the clock the return window runs from — None while the
    order is in transit, which means no window has started yet."""

    order_id: str
    customer_id: str
    status: OrderStatus
    placed_at: date
    delivered_at: date | None = None
    items: list[OrderItem] = Field(default_factory=list)
    promotion_code: str | None = None
    scenario: str | None = Field(default=None, description="Fixture annotation. Never shown.")


class ReturnRecord(BaseModel):
    return_id: str
    order_id: str
    item_id: str
    customer_id: str
    status: ReturnStatus
    reason: str
    created_at: datetime
    scenario: str | None = Field(default=None, description="Fixture annotation. Never shown.")


class EligibilityDecision(BaseModel):
    """The outcome of an eligibility check, with the reasoning kept explicit.

    The agent quotes `explanation` rather than inventing a justification, and the
    loop passes `eligibility_token` to `initiate_return` so a mutation cannot
    happen without a check that said yes — necessary but not sufficient, since
    that tool also requires `confirmed=True`.
    """

    eligible: bool
    policy_id: str | None = None
    explanation: str = Field(default="", description="Customer-facing. Safe to read aloud.")
    rule_path: list[str] = Field(
        default_factory=list,
        description=(
            "Raw graph hops behind the match. Kept for debugging and backward "
            "compatibility; the trace displays the friendlier `policy_path` "
            "derived from the fields below instead of this traversal."
        ),
    )
    product_type: str | None = Field(
        default=None, description="The item's category, for the displayed policy path."
    )
    region: str | None = Field(
        default=None, description="Set only when the winning policy is region-specific."
    )
    return_window_days: int | None = Field(
        default=None, description="The winning policy's window, if it offers one."
    )
    existing_return_id: str | None = Field(
        default=None, description="The open return that blocked this check, if any."
    )
    eligibility_token: str | None = Field(
        default=None, description="Issued only when eligible. Required by initiate_return."
    )
    days_remaining: int | None = None


# --- Execution records ---------------------------------------------------
#
# These trace execution, not reasoning: that a tool was called with an order and
# an item and returned `eligible=False` in 4ms. No `thinking` content is captured
# anywhere in this repo, so there is none to leak here.


class ToolStatus(StrEnum):
    OK = "ok"
    """The tool ran and returned a result. Says nothing about whether the answer
    was yes — a refused eligibility check is a successful tool call."""

    BLOCKED = "blocked"
    """A guard refused. `initiate_return` without confirmation lands here."""

    ERROR = "error"
    """The tool raised, or a dependency was unavailable. Nothing can be concluded."""

    REJECTED = "rejected"
    """The model asked for a tool that does not exist, or with unusable arguments."""


class ToolTrace(BaseModel):
    """One tool call, as an observer would see it."""

    trace_id: str = Field(default_factory=lambda: f"TRC-{uuid.uuid4().hex[:8].upper()}")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    session_id: str
    model: str = Field(description="The model id that requested the call.")
    model_tier: str = Field(description="'haiku' or 'sonnet'.")
    tool_name: str
    tool_args: dict[str, Any] = Field(
        default_factory=dict, description="Sanitized. Never the raw arguments."
    )
    status: ToolStatus
    latency_ms: float
    result_summary: str = ""
    error: str | None = Field(default=None, description="The message, never a stack.")
    policy_decision: dict[str, Any] | None = Field(
        default=None,
        description=(
            "For an eligibility check: what *this* call decided. Held per invocation "
            "because one turn can check several items, and the session only keeps the "
            "last decision. Rendering fields only — never the eligibility token."
        ),
    )


class ModelTurn(BaseModel):
    """One user turn and which model handled it, recorded whether or not any tool
    ran — so routing is visible on a plain policy question as well as a return."""

    turn_id: str = Field(default_factory=lambda: f"TURN-{uuid.uuid4().hex[:8].upper()}")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    session_id: str
    model_tier: str
    model: str
    routing_reason: str = Field(description="The one rule that decided the tier.")
    tool_calls: int = 0
    iterations: int = Field(default=0, description="Round trips to Anthropic this turn.")

    # Prompt-cache accounting, summed over this turn's round trips. Read off the
    # response's own `usage` — nothing here estimates. Both zero means the model
    # was never called, or the prefix was too short for the cache to engage.
    cache_creation_input_tokens: int = Field(
        default=0, description="Prefix written to the cache, at a premium."
    )
    cache_read_input_tokens: int = Field(
        default=0, description="Prefix served from the cache. The saving."
    )

    # Latency, measured where it happens rather than estimated afterwards — see
    # `agent.agent._run_tool_loop_stream`, the only writer of these three.
    model_latency_ms: float = Field(
        default=0.0, description="Time spent inside Anthropic calls, summed over this turn."
    )
    tool_latency_ms: float = Field(
        default=0.0, description="Time spent inside tool calls, summed over this turn."
    )
    total_latency_ms: float = Field(
        default=0.0, description="Wall-clock time for the whole turn, model and tools together."
    )
    time_to_first_token_ms: float | None = Field(
        default=None,
        description=(
            "How long the customer waited before the first visible character. None "
            "means nothing was ever streamed to them — an outage before any text arrived."
        ),
    )
    timed_out: bool = Field(
        default=False, description="True when the turn stopped because of TURN_TIMEOUT_SECONDS."
    )


MAX_SUMMARY_CHARS = 200

SENSITIVE_KEYS = frozenset(
    {"api_key", "anthropic_api_key", "password", "neo4j_password", "token",
     "eligibility_token", "secret", "authorization"}
)
"""Argument names redacted whole, whatever their value.

`eligibility_token` is listed even though the model never sees one — it is
injected by the loop — so a trace stays safe if that ever changes.
"""

EMAIL_PATTERN = re.compile(r"([^@\s]+)@([^@\s]+)")


def sanitize_args(args: dict[str, Any]) -> dict[str, Any]:
    """Make tool arguments safe to record: redact credentials, mask emails,
    truncate. Returns a new dict; the input is not modified."""
    clean: dict[str, Any] = {}
    for key, value in args.items():
        if key.lower() in SENSITIVE_KEYS:
            clean[key] = "***"
        elif isinstance(value, str):
            clean[key] = _truncate(mask_email(value))
        else:
            clean[key] = value
    return clean


def mask_email(value: str) -> str:
    """`ada@bookly.test` becomes `a***@bookly.test`. The domain is kept because it
    helps when reading a trace and is not personal; the local part is not."""
    return EMAIL_PATTERN.sub(lambda m: f"{m.group(1)[:1]}***@{m.group(2)}", value)


def summarize(value: Any) -> str:
    """Turn a tool result into one readable line for the trace list."""
    if value is None:
        return "no result"

    if isinstance(value, BaseModel):
        fields = value.model_dump()
        interesting = {
            key: fields[key]
            for key in ("verified", "eligible", "matched", "created", "policy_id", "case_id")
            if key in fields and fields[key] is not None
        }
        summary = ", ".join(f"{k}={v}" for k, v in interesting.items()) or type(value).__name__
        return _truncate(mask_email(summary))

    return _truncate(mask_email(str(value)))


def _truncate(text: str) -> str:
    if len(text) <= MAX_SUMMARY_CHARS:
        return text
    return f"{text[: MAX_SUMMARY_CHARS - 1]}…"


# --- Session state -------------------------------------------------------


class Message(BaseModel):
    """One turn of the visible conversation."""

    role: Role
    content: str


class PendingReturn(BaseModel):
    """One return that passed eligibility and is waiting on the customer's yes.

    Created only from a successful, eligible check, so its existence is what makes
    a later "yes" mean something. Everything a mutation needs to trust *this*
    return is carried on the return itself — customer, order, item, token,
    whether the agent has asked, whether the customer has agreed — so a session
    can hold several of these at once (one per eligible item) with no shared
    field one of them could borrow from another. A token issued for ORD-1008
    authorises nothing on ORD-1006, and being confirmed on one does not confirm
    the other.
    """

    customer_id: str
    order_id: str
    item_id: str
    eligibility_token: str
    asked: bool = Field(
        default=False, description="True once the agent has actually asked about this return."
    )
    confirmed: bool = Field(
        default=False,
        description=(
            "True once the customer has explicitly said yes to this exact return. "
            "Bookkeeping only — it must be passed explicitly to initiate_return, which "
            "does its own check and never reads session state."
        ),
    )


class SessionState(BaseModel):
    """Everything the agent knows about the conversation so far."""

    session_id: str = Field(default_factory=lambda: f"SESS-{uuid.uuid4().hex[:8].upper()}")

    messages: list[Message] = Field(default_factory=list, description="Visible transcript, for the UI.")
    transcript: list[dict[str, Any]] = Field(
        default_factory=list, description="The Anthropic conversation, including tool blocks."
    )

    verified_customer_id: str | None = Field(
        default=None, description="Set once identity is confirmed. Account tools require it."
    )
    customer_region: str | None = Field(
        default=None, description="ISO country code. Selects regional policy overrides."
    )
    active_order_ids: list[str] = Field(
        default_factory=list, description="Live orders. More than one means the agent must ask."
    )
    active_order_id: str | None = Field(
        default=None, description="The order under discussion. None until the customer chooses."
    )
    active_item_id: str | None = Field(
        default=None, description="Returns are per item, not per order."
    )
    return_reason: str | None = Field(
        default=None, description="The customer's stated reason, in their own words."
    )
    return_intent_expressed: bool = Field(
        default=False,
        description=(
            "True once the customer has asked about a return, even before an item is "
            "chosen. Keeps the turns spent working out which book they mean on Sonnet."
        ),
    )
    eligibility: EligibilityDecision | None = Field(
        default=None, description="The last decision, kept so it can be quoted."
    )
    pending_returns: list[PendingReturn] = Field(
        default_factory=list,
        description=(
            "What a 'yes' would authorise, one entry per eligible item still open. "
            "Empty means it authorises nothing. A turn that checks several items can "
            "leave several of these; each carries its own token and its own "
            "confirmation, so a 'yes' to one does not authorise another."
        ),
    )
    escalated: bool = Field(default=False, description="True once handed to a human.")

    tool_traces: list[ToolTrace] = Field(default_factory=list)
    model_turns: list[ModelTurn] = Field(default_factory=list)

    @property
    def is_verified(self) -> bool:
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
        workflow, and "Designing Data-Intensive Applications" carries no return
        keyword — so without this, the turn that chooses the item, and therefore
        the turn that runs the eligibility check, would drop to the cheaper model
        mid-return.
        """
        return any(
            (
                self.return_intent_expressed,
                self.active_item_id is not None,
                self.return_reason is not None,
                self.eligibility is not None,
                bool(self.pending_returns),
            )
        )

    @property
    def may_mutate(self) -> bool:
        """Whether at least one pending return is currently safe to write.

        Not itself the authority to write — `initiate_return` re-checks the exact
        matching pending return from its own arguments, so a bug here cannot let a
        mutation through. This just says whether *any* pending return is confirmed
        and holds a token; it does not say which one.
        """
        return self.is_verified and any(
            pending.confirmed and pending.eligibility_token for pending in self.pending_returns
        )

    def add_message(self, role: Role, content: str) -> None:
        self.messages.append(Message(role=role, content=content))

    def last_assistant_message(self) -> str | None:
        """The agent's most recent reply. Read by the confirmation check: a "yes"
        only counts if what it answers was a question."""
        return next((m.content for m in reversed(self.messages) if m.role is Role.ASSISTANT), None)

    def clear_return_context(self) -> None:
        """Drop every pending return and everything said about it.

        Called when the customer switches order or item — so a token issued for
        one item can never be spent on another, and a yes given for one cannot
        authorise a return of the next — and once the last pending return has
        been written, so the workflow does not linger with nothing left to do.
        """
        self.active_order_id = None
        self.active_item_id = None
        self.return_reason = None
        self.return_intent_expressed = False
        self.eligibility = None
        self.pending_returns = []

    def clear_account_context(self) -> None:
        """Drop every trusted field scoped to the previously verified customer.

        Called before adopting a new `verified_customer_id`, so a pending return,
        a token, or a confirmation minted for one account can never be spent
        under another.
        """
        self.clear_return_context()
        self.active_order_ids = []
