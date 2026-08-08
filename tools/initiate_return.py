"""Tool: create a return record. Stub only.

The only write in the tool set, so its preconditions are part of its signature
rather than something the caller is trusted to have arranged. `confirmed` is
passed in explicitly: `SessionState.confirmed` records that the customer said
yes, but session state is orchestrator bookkeeping, and a tool that mutates data
must not depend on it being correct.
"""

from __future__ import annotations

from agent.models import ReturnRecord


def initiate_return(
    order_id: str,
    item_id: str,
    customer_id: str,
    reason: str,
    eligibility_token: str,
    confirmed: bool,
) -> ReturnRecord:
    """Open a return for an item already found eligible and explicitly confirmed.

    The only tool in the set that writes. It does not trust that the agent
    checked first: the token from `check_return_eligibility` is required, and is
    valid only for the exact order and item it was issued against. Nor does it
    trust that the agent asked first — `confirmed` must be True, and the caller
    has to say so as an argument.

    Every one of these must hold before anything is written. Any one of them
    false is a refusal, not a warning:

    1. Customer identity is verified.
    2. The order belongs to that customer.
    3. The item is eligible for return.
    4. `eligibility_token` is valid, and bound to this order + item + customer.
    5. The customer explicitly confirmed this action (`confirmed` is True).
    6. No duplicate RMA already exists for this order + item.

    Args:
        order_id: The order being returned against.
        item_id: The item being returned.
        customer_id: The verified customer.
        reason: The customer's stated reason, in their own words.
        eligibility_token: Token from an eligible decision for this order+item.
        confirmed: True only when the customer was shown the action and said yes.
            Never defaulted, never inferred from session state.

    Returns:
        The created return record.

    Raises:
        ValueError: If confirmation is absent, the token is missing or does not
            match this order, item, and customer, the order is not the
            customer's, or a return is already open.
    """
    # Confirmation is checked first: it is the cheapest gate and the one whose
    # absence is least recoverable, since a write here is customer-visible.
    if not confirmed:
        # TODO: raise ValueError("explicit customer confirmation required") once
        #       the tool's error contract is wired into the agent loop.
        raise NotImplementedError(
            "initiate_return is a scaffold stub — explicit customer confirmation required"
        )

    # TODO: verify customer_id is a verified identity for this session.
    # TODO: verify order_id belongs to customer_id (CUST-004 must not touch ORD-1008).
    # TODO: validate eligibility_token against order_id + item_id + customer_id.
    # TODO: re-check for an existing open return (RET-5001 blocks ORD-1007).
    # TODO: append to data/returns.json with a generated return_id.
    raise NotImplementedError("initiate_return is a scaffold stub")
