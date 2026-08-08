"""Shared domain models for the Bookly support agent.

`Customer`, `Item`, `Order`, and `ReturnRecord` mirror the mock JSON fixtures in
`data/`. `Policy` mirrors a `:Policy` node in Neo4j — policy data is read from
the graph at runtime, never from JSON. Types and shapes only — no business
logic. `SessionState` lives in `agent/state.py` because it is conversation state
rather than domain data.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ProductType(StrEnum):
    """Item category. Decides which policy governs a return.

    Matches the `:Category` nodes in the policy graph.
    """

    PHYSICAL_BOOK = "PhysicalBook"
    EBOOK = "EBook"


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


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class Customer(BaseModel):
    """A Bookly account holder."""

    customer_id: str
    name: str
    email: str
    country: str = Field(description="ISO 3166-1 alpha-2, e.g. 'AU'. Drives regional overrides.")
    note: str | None = Field(default=None, description="Fixture annotation. Never shown to a customer.")


class Item(BaseModel):
    """A catalogue entry — one book, in one format."""

    item_id: str
    title: str
    product_type: ProductType
    price: float
    currency: str = "USD"


class OrderItem(BaseModel):
    """One line on an order: a catalogue item, as purchased."""

    item_id: str
    quantity: int
    unit_price: float


class Order(BaseModel):
    """A purchase made by a customer.

    `delivered_at` is the clock the return window runs from. It is None while
    the order is still in transit, which means no window has started yet.
    """

    order_id: str
    customer_id: str
    status: OrderStatus
    placed_at: date
    delivered_at: date | None = None
    items: list[OrderItem] = Field(default_factory=list)
    promotion_code: str | None = Field(
        default=None,
        description="Promotion the order was placed under, e.g. 'MIDYEAR_HOLIDAY_SALE_2026'.",
    )
    scenario: str | None = Field(default=None, description="Fixture annotation. Never shown to a customer.")


class Policy(BaseModel):
    """A return policy rule, as it exists in Neo4j.

    One `:Policy` node's properties. Which categories a policy governs, which
    regions override into it, and which policies it outranks are *edges*, not
    fields — they are answered by traversal, never by a list on this model.

    `window_days` of None means returns are not offered at all: the absence of
    a window, not a window of zero length.
    """

    policy_id: str = Field(description="e.g. 'STANDARD_30_DAY', 'AU_BOOKLY_EXTENDED_RETURN'.")
    name: str
    summary: str
    window_days: int | None = None
    window_starts_from: str | None = Field(
        default=None, description="Which date the window runs from, e.g. 'delivered_at'."
    )
    precedence: int = Field(default=0, description="Higher wins when several policies match.")
    exceptions: list[str] = Field(
        default_factory=list, description="Reason codes the window does not apply to, e.g. 'DAMAGED_ON_ARRIVAL'."
    )
    promotion_code: str | None = Field(
        default=None, description="Set when the policy is granted by a promotion."
    )
    promotion_active_from: date | None = None
    promotion_active_to: date | None = None


class EligibilityDecision(BaseModel):
    """The outcome of an eligibility check, with the reasoning kept explicit.

    The agent quotes `explanation` to the customer rather than inventing its
    own justification, and passes `eligibility_token` to `initiate_return` so a
    mutation cannot happen without a check that actually said yes. The token is
    necessary but not sufficient: `initiate_return` also requires an explicit
    `confirmed=True`.

    `rule_path` is the graph traversal behind the answer — the ordered hops
    from item category to winning policy — so the decision is explainable
    rather than asserted.
    """

    eligible: bool
    policy_id: str | None = None
    explanation: str = Field(default="", description="Customer-facing. Safe to read aloud verbatim.")
    rule_path: list[str] = Field(
        default_factory=list,
        description=(
            "One string per graph hop, e.g. '(PhysicalBook)-[:GOVERNED_BY]->(AU_BOOKLY_EXTENDED_RETURN)'. "
            "Includes the region or promotion hop that made a conditional policy apply, and what it outranked."
        ),
    )
    eligibility_token: str | None = Field(
        default=None,
        description="Issued only when eligible. Required by initiate_return, alongside confirmed=True.",
    )
    days_remaining: int | None = None
    requires_human: bool = Field(
        default=False, description="True when a rule matched but only a human may approve it.",
    )


class ReturnRecord(BaseModel):
    """A return the customer has asked for."""

    return_id: str
    order_id: str
    item_id: str
    customer_id: str
    status: ReturnStatus
    reason: str
    created_at: datetime
    scenario: str | None = Field(default=None, description="Fixture annotation. Never shown to a customer.")
