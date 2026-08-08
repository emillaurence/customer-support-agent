"""Tool: fetch an order and the items on it.

Reads the mock order data in `data/`. Everything the agent says about an order —
status, dates, titles, prices — comes from here, so that nothing has to be
inferred from the conversation.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from pydantic import BaseModel, Field

from agent.models import Item, Order, OrderStatus
from tools import fixtures


class Shipment(BaseModel):
    """Where the order is, as plainly as the data supports.

    Separate from `Order` because this is the derived view — `has_arrived` and
    `days_since_delivery` are computed, and the return clock depends on them.
    Nothing is estimated: an order with no `delivered_at` reports None, not a
    guessed arrival date.
    """

    status: OrderStatus
    placed_at: date
    delivered_at: date | None = None
    has_arrived: bool = Field(description="True only when a delivery date is on record.")
    days_since_delivery: int | None = Field(
        default=None, description="None until it has arrived. No return window has started before then.",
    )


class OrderDetails(BaseModel):
    """An order, its line items resolved to catalogue entries, and its shipment."""

    order: Order
    items: list[Item]
    shipment: Shipment


def lookup_order(order_id: str, customer_id: str, now: datetime | None = None) -> OrderDetails | None:
    """Look up one order belonging to a verified customer.

    `customer_id` is required, not optional: it stops the agent reading out
    someone else's order if the model hallucinates an order number or a customer
    quotes one they saw somewhere. Ownership is enforced here rather than trusted
    to the caller.

    Args:
        order_id: The order to fetch.
        customer_id: The verified customer the order must belong to.
        now: Clock for `days_since_delivery`. Defaults to the current UTC time;
            tests pass a fixed value.

    Returns:
        The order, its items, and its shipment — or None if there is no such
        order for that customer. A real order owned by someone else returns None
        too: the same answer as a number that does not exist, so the response
        cannot be used to discover whether an order id is real.
    """
    order = next(
        (
            candidate
            for candidate in fixtures.load_orders()
            if candidate.order_id == order_id and candidate.customer_id == customer_id
        ),
        None,
    )
    if order is None:
        return None

    catalogue = fixtures.load_items()
    items = [catalogue[line.item_id] for line in order.items if line.item_id in catalogue]

    return OrderDetails(
        # `scenario` is a note to whoever reads the JSON. Dropped here so it can
        # never reach the model, let alone the customer.
        order=order.model_copy(update={"scenario": None}),
        items=items,
        shipment=build_shipment(order, now),
    )


def build_shipment(order: Order, now: datetime | None = None) -> Shipment:
    """Derive the shipment view from an order.

    Args:
        order: The order to describe.
        now: Clock for the day count. Defaults to the current UTC time.

    Returns:
        The shipment, with `days_since_delivery` set only if it has arrived.
    """
    today = (now or datetime.now(UTC)).date()
    arrived = order.delivered_at is not None
    return Shipment(
        status=order.status,
        placed_at=order.placed_at,
        delivered_at=order.delivered_at,
        has_arrived=arrived,
        days_since_delivery=(today - order.delivered_at).days if arrived else None,
    )


def find_line_item(order: Order, item_id: str) -> bool:
    """Whether an item is actually on an order.

    Its own function because two tools ask the question: eligibility, to refuse a
    check on an item that was never bought, and `initiate_return`, to refuse a
    write for the same reason.
    """
    return any(line.item_id == item_id for line in order.items)
