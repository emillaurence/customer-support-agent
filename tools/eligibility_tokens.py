"""Server-side store for eligibility tokens.

`check_return_eligibility` mints a token when — and only when — it decides an
item is returnable, and records what that token was issued *for*.
`initiate_return` looks the token up and refuses if the request does not match
the grant exactly.

The point is that the token carries no meaning of its own. It is an opaque
`uuid4`, and everything that matters about it — which customer, which order,
which item, under which policy — is held here, server-side. A model that invents
a plausible-looking token finds nothing on the other end, and a model that
replays a real token for a different item finds a grant that does not match. No
JWT, no signing, no claims to validate: for a prototype an unguessable key into
a server-side record is the simpler equivalent.

In-memory, so tokens do not survive a restart. That is acceptable — an
eligibility decision is only meaningful inside the conversation that produced it,
and a fresh process can mint a new one.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field


class EligibilityGrant(BaseModel):
    """What one token permits: exactly one item, on one order, for one customer."""

    token: str
    customer_id: str
    order_id: str
    item_id: str
    policy_id: str = Field(description="The policy the eligible decision was made under.")
    issued_at: datetime


_GRANTS: dict[str, EligibilityGrant] = {}


def issue(customer_id: str, order_id: str, item_id: str, policy_id: str) -> EligibilityGrant:
    """Mint a token bound to one customer, order, item, and policy.

    Args:
        customer_id: The verified customer the decision was made for.
        order_id: The order the decision was made against.
        item_id: The item found eligible.
        policy_id: The policy that allowed it.

    Returns:
        The stored grant, including the new token.
    """
    grant = EligibilityGrant(
        token=str(uuid.uuid4()),
        customer_id=customer_id,
        order_id=order_id,
        item_id=item_id,
        policy_id=policy_id,
        issued_at=datetime.now(UTC),
    )
    _GRANTS[grant.token] = grant
    return grant


def lookup(token: str) -> EligibilityGrant | None:
    """Find the grant a token was issued as, if it was issued by this process.

    Args:
        token: The token the caller presented.

    Returns:
        The grant, or None for an unknown, malformed, or invented token.
    """
    return _GRANTS.get(token)


def matches(grant: EligibilityGrant, customer_id: str, order_id: str, item_id: str) -> bool:
    """Whether a grant authorises this exact request.

    All three must match. A token issued for one item on one order cannot be
    spent on another, and cannot be spent by another customer.
    """
    return (
        grant.customer_id == customer_id
        and grant.order_id == order_id
        and grant.item_id == item_id
    )


def clear() -> None:
    """Drop every grant. For tests, so one case cannot leak a token into another."""
    _GRANTS.clear()
