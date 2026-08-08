"""Tool: fetch an order and the items on it. Stub only."""

from __future__ import annotations

from pydantic import BaseModel

from agent.models import Item, Order


class OrderDetails(BaseModel):
    """An order, with its line items resolved to catalogue entries."""

    order: Order
    items: list[Item]


def lookup_order(order_id: str, customer_id: str) -> OrderDetails | None:
    """Look up one order belonging to a verified customer.

    `customer_id` is required, not optional: it stops the agent reading out
    someone else's order if the model hallucinates or is fed an order number.

    Args:
        order_id: The order to fetch.
        customer_id: The verified customer the order must belong to.

    Returns:
        The order and its items, or None if there is no such order for that
        customer. A real order owned by someone else returns None too — the
        same answer as a number that does not exist.
    """
    # TODO: read data/orders.json + data/items.json, filter by customer_id.
    # TODO: never leak `scenario` / `note` fixture annotations to the model.
    # TODO: the two-active-orders flow (CUST-003) needs a way to offer a choice.
    #       Decide in Phase 3 whether that is a second tool or an `order_id=None`
    #       call on this one. Do not guess an order on the agent's behalf.
    raise NotImplementedError("lookup_order is a scaffold stub")
