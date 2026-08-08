"""Tool: confirm the caller is who they say they are. Stub only."""

from __future__ import annotations

from pydantic import BaseModel


class VerifyIdentityResult(BaseModel):
    """Outcome of an identity check."""

    verified: bool
    customer_id: str | None = None
    country: str | None = None
    """ISO code of the verified customer — becomes SessionState.customer_region."""
    message: str = ""


def verify_identity(email: str, order_id: str) -> VerifyIdentityResult:
    """Match an email against an order to confirm identity.

    Two weak factors together: the customer must know both the email on the
    account and an order number belonging to it. Enough for a demo, not for
    production.

    Args:
        email: Email the customer gave.
        order_id: Order number the customer gave.

    Returns:
        Whether verification passed, and the customer id and country if it did.
    """
    # TODO: load data/customers.json and data/orders.json, check the email
    #       matches the order's customer, compare case-insensitively.
    # TODO: on failure say only "that didn't match" — never which half was wrong.
    raise NotImplementedError("verify_identity is a scaffold stub")
