"""Tool: create a return record. Stub only."""

from __future__ import annotations

from agent.models import ReturnRecord


def initiate_return(
    order_id: str,
    item_id: str,
    customer_id: str,
    reason: str,
    eligibility_token: str,
) -> ReturnRecord:
    """Open a return for an item already found eligible.

    The only tool in the set that writes. It does not trust that the agent
    checked first: the token from `check_return_eligibility` is required, and is
    valid only for the exact order and item it was issued against.

    Args:
        order_id: The order being returned against.
        item_id: The item being returned.
        customer_id: The verified customer.
        reason: The customer's stated reason, in their own words.
        eligibility_token: Token from an eligible decision for this order+item.

    Returns:
        The created return record.

    Raises:
        ValueError: If the token is missing, does not match this order and item,
            or a return is already open.
    """
    # TODO: validate eligibility_token against order_id + item_id + customer_id.
    # TODO: re-check for an existing open return (RET-5001 blocks ORD-1007).
    # TODO: append to data/returns.json with a generated return_id.
    raise NotImplementedError("initiate_return is a scaffold stub")
