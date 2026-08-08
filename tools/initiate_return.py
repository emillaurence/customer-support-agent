"""Tool: create a return record.

The only write in the tool set, so its preconditions are part of its signature
rather than something the caller is trusted to have arranged. `confirmed` is
passed in explicitly: `SessionState.confirmed` records that the customer said
yes, but session state is orchestrator bookkeeping, and a tool that mutates data
must not depend on it being correct.

Every guard is enforced here, in this function. Not in the prompt, not in the
session, not in the future agent loop — a model that skips a step, hallucinates a
token, or asks to return someone else's book must fail against the tool itself.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel

from agent.models import ReturnRecord, ReturnStatus
from tools import eligibility_tokens, fixtures
from tools.check_return_eligibility import BLOCKING_RETURN_STATUSES
from tools.lookup_order import find_line_item, lookup_order


class ReturnResult(BaseModel):
    """The outcome of a return request that passed every guard.

    `created` distinguishes the two success cases. False means a return already
    existed and is being reported back rather than duplicated — the customer's
    request is satisfied either way, and Bookly's records are unchanged.
    """

    created: bool
    return_record: ReturnRecord
    message: str = ""
    """Customer-facing line."""


class ReturnBlockedError(ValueError):
    """A guard failed, so nothing was written.

    A `ValueError` subclass so callers can catch it specifically or generically.
    The message is written for the agent, not the customer: it names the guard
    that failed so the failure is debuggable, and the agent rephrases.
    """


def initiate_return(
    order_id: str,
    item_id: str,
    customer_id: str,
    reason: str,
    eligibility_token: str,
    confirmed: bool,
) -> ReturnResult:
    """Open a return for an item already found eligible and explicitly confirmed.

    The only tool in the set that writes. It does not trust that the agent
    checked first: the token from `check_return_eligibility` is required, and is
    valid only for the exact customer, order, and item it was issued against. Nor
    does it trust that the agent asked first — `confirmed` must be True, and the
    caller has to say so as an argument.

    Every one of these must hold before anything is written. Any one of them
    false raises, and no record is created:

    1. The customer exists and is identified.
    2. The order belongs to that customer.
    3. The item is on that order.
    4. `eligibility_token` was issued by this server.
    5. The token was issued for this exact customer, order, and item.
    6. The token represents a decision that said eligible.
    7. The customer explicitly confirmed (`confirmed` is True).
    8. No return already exists for this order and item.

    Guard 8 is the one that does not raise: an existing return means the customer
    is asking for something they already have, so the existing RMA is returned
    with `created=False`. Nothing is written either way, and repeated requests
    cannot produce a second RMA.

    Args:
        order_id: The order being returned against.
        item_id: The item being returned.
        customer_id: The verified customer.
        reason: The customer's stated reason, in their own words.
        eligibility_token: Token from an eligible decision for this order+item.
        confirmed: True only when the customer was shown the action and said yes.
            Never defaulted, never inferred from session state.

    Returns:
        The return record, and whether this call created it.

    Raises:
        ReturnBlockedError: If any of guards 1-7 fails. Nothing is written.
    """
    # Confirmation first: the cheapest check, and the one whose absence is least
    # recoverable, since a return the customer never asked for is visible to them.
    if not confirmed:
        raise ReturnBlockedError(
            "explicit customer confirmation required: initiate_return was called with "
            "confirmed=False. Ask the customer to confirm the return, then call again."
        )

    if not eligibility_token:
        raise ReturnBlockedError(
            "no eligibility_token: run check_return_eligibility first and pass the token it issues."
        )

    customer = next(
        (c for c in fixtures.load_customers() if c.customer_id == customer_id), None
    )
    if customer is None:
        raise ReturnBlockedError(f"unknown customer {customer_id}: identity is not verified.")

    # Ownership, re-checked here. lookup_order returns None both for an order
    # that does not exist and one belonging to someone else.
    details = lookup_order(order_id, customer_id)
    if details is None:
        raise ReturnBlockedError(
            f"order {order_id} does not belong to {customer_id}, or does not exist."
        )

    if not find_line_item(details.order, item_id):
        raise ReturnBlockedError(f"item {item_id} is not on order {order_id}.")

    grant = eligibility_tokens.lookup(eligibility_token)
    if grant is None:
        # An invented, expired, or copied-from-somewhere token looks exactly like
        # this: nothing on the server was ever issued under it.
        raise ReturnBlockedError(
            "eligibility_token is not a token this server issued. "
            "Run check_return_eligibility and use the token it returns."
        )

    # Guard 6 needs no separate check: a token exists only on the eligible path of
    # check_return_eligibility, so a grant being found *is* an eligible decision.
    # `grant.policy_id` records which policy allowed it.
    if not eligibility_tokens.matches(grant, customer_id, order_id, item_id):
        raise ReturnBlockedError(
            f"eligibility_token was issued for {grant.customer_id} / {grant.order_id} / "
            f"{grant.item_id}, not {customer_id} / {order_id} / {item_id}. "
            "A token cannot be spent on a different customer, order, or item."
        )

    # Guard 8. Checked last, immediately before the write, so the window between
    # the check and the append is as small as it can be.
    existing = next(
        (
            record
            for record in fixtures.load_returns()
            if record.order_id == order_id
            and record.item_id == item_id
            and record.status in BLOCKING_RETURN_STATUSES
        ),
        None,
    )
    if existing is not None:
        return ReturnResult(
            created=False,
            return_record=existing,
            message=(
                f"There's already a return open for that item — reference {existing.return_id}. "
                f"I haven't started a second one."
            ),
        )

    record = ReturnRecord(
        return_id=fixtures.next_return_id(),
        order_id=order_id,
        item_id=item_id,
        customer_id=customer_id,
        status=ReturnStatus.REQUESTED,
        reason=reason,
        created_at=datetime.now(UTC),
    )
    fixtures.append_return(record)

    return ReturnResult(
        created=True,
        return_record=record,
        message=(
            f"Your return is open — reference {record.return_id}. "
            f"You'll get an email with the next steps."
        ),
    )
