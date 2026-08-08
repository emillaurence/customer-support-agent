"""Tool: confirm the caller is who they say they are.

**Mocked identity verification.** A matching email address is treated as proof of
identity. That is deliberate for this take-home and is not production
authentication: an email address is a public identifier, not a secret, so anyone
who knows a customer's address could pass this check.

Real authentication belongs outside the agent — a session token or an
authenticated user id handed to the tools by the application, or at minimum a
one-time code sent to the address on file. The shape here is the part worth
keeping: identity is established once, by a tool, and every tool that reads
customer data demands the resulting `customer_id`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from tools import fixtures


class VerifyIdentityResult(BaseModel):
    """Outcome of an identity check.

    Carries only what the agent needs to proceed: who the customer is, which
    region's policy applies to them, and which orders are theirs to talk about.
    No order contents — that is `lookup_order`'s job, and it asks for the
    `customer_id` from here.
    """

    verified: bool
    customer_id: str | None = None
    region: str | None = Field(
        default=None,
        description="ISO 3166-1 alpha-2 code of the verified customer. Becomes SessionState.customer_region.",
    )
    active_order_ids: list[str] = Field(
        default_factory=list,
        description="Order ids only — no dates, items, or status. More than one means the agent must ask which.",
    )
    message: str = ""
    """Customer-facing line."""


def verify_identity(email: str) -> VerifyIdentityResult:
    """Verify a customer by the email address on their account.

    Mocked, as described in the module docstring: a match verifies. Compared
    case-insensitively and with surrounding whitespace stripped, because
    customers type their address rather than paste it.

    Args:
        email: The email address the customer gave.

    Returns:
        On a match: verified, the customer id, their region, and the ids of their
        active orders. On no match, or on empty input: `verified=False` with
        every other field left empty.
    """
    normalised = email.strip().lower()
    if not normalised:
        return VerifyIdentityResult(
            verified=False,
            message="I need the email address on the account to look you up.",
        )

    customer = next(
        (c for c in fixtures.load_customers() if c.email.strip().lower() == normalised),
        None,
    )
    if customer is None:
        # Says only that it did not match. Confirming that an address is *not* a
        # Bookly customer is itself a disclosure.
        return VerifyIdentityResult(
            verified=False,
            message="I can't find an account for that email address. Could you double-check it?",
        )

    return VerifyIdentityResult(
        verified=True,
        customer_id=customer.customer_id,
        region=customer.country,
        active_order_ids=active_order_ids(customer.customer_id),
        message=f"Thanks {customer.name.split()[0]} — I've found your account.",
    )


def active_order_ids(customer_id: str) -> list[str]:
    """The customer's live orders, most recently placed first.

    Active means not cancelled: an order in transit and an order delivered last
    week are both things a customer might be calling about. Sorted so the agent
    has a sensible order to read them in — but the agent must still *ask* when
    there is more than one, rather than taking the first.

    Args:
        customer_id: The verified customer.

    Returns:
        Their order ids, newest first.
    """
    orders = [
        order
        for order in fixtures.load_orders()
        if order.customer_id == customer_id and order.status != "cancelled"
    ]
    orders.sort(key=lambda order: order.placed_at, reverse=True)
    return [order.order_id for order in orders]
