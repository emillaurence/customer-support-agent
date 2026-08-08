"""Shared domain models for the Bookly support agent.

These mirror the JSON fixtures in `data/` and the policy graph in `neo4j/`.
Types and shapes only — no business logic. `SessionState` lives in
`agent/state.py` because it is conversation state rather than domain data.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ProductType(StrEnum):
    """Item category. Decides which policy governs a return.

    Matches the `:ProductType` nodes in the policy graph.
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
    """A return policy rule.

    Sourced from the policy graph, not from the order data. `window_days` of
    None means returns are not offered at all — the absence of a window, not a
    window of zero length.
    """

    policy_id: str = Field(description="e.g. 'STANDARD_30_DAY', 'AU_BOOKLY_EXTENDED_RETURN'.")
    name: str
    window_days: int | None = None
    summary: str
    precedence: int = Field(default=0, description="Higher wins when several policies match.")
    applies_to_product_types: list[ProductType] | None = None
    applies_to_regions: list[str] | None = Field(
        default=None, description="ISO country codes, or None for 'everywhere'."
    )
    granted_by_promotion: str | None = None


class EligibilityDecision(BaseModel):
    """The outcome of an eligibility check, with the reasoning kept explicit.

    The agent quotes `explanation` to the customer rather than inventing its
    own justification, and passes `eligibility_token` to `initiate_return` so a
    mutation cannot happen without a check that actually said yes.

    `rule_path` is the graph traversal behind the answer — the ordered hops
    from item category to winning policy — so the decision is explainable
    rather than asserted.
    """

    eligible: bool
    policy_id: str | None = None
    explanation: str = Field(default="", description="Customer-facing. Safe to read aloud verbatim.")
    rule_path: list[str] = Field(
        default_factory=list,
        description="e.g. ['PhysicalBook', 'GOVERNED_BY', 'STANDARD_30_DAY', 'HAS_WINDOW', 'WINDOW_30_DAY'].",
    )
    eligibility_token: str | None = Field(
        default=None,
        description="Issued only when eligible. Required by initiate_return.",
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
