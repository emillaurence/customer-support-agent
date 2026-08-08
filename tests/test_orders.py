"""Order lookup and identity verification.

All skipped — the tools are stubs. Each test names a behaviour worth locking
down before the logic lands, against the fixtures in ../data/.
"""

import pytest

pytestmark = pytest.mark.skip(reason="scaffold — tools not implemented yet")


def test_lookup_order_returns_items() -> None:
    """A known order resolves its line items to catalogue entries."""
    # TODO: lookup_order("ORD-1003", "CUST-002") -> 2 items (ITEM-102, ITEM-201)
    ...


def test_lookup_order_unknown_id_returns_none() -> None:
    """An order number that does not exist yields None, not an error."""
    # TODO: lookup_order("ORD-9999", "CUST-001") is None
    ...


def test_lookup_order_rejects_other_customers_order() -> None:
    """A real order belonging to someone else is not readable."""
    # TODO: lookup_order("ORD-1008", "CUST-001") is None — ORD-1008 is CUST-004's
    ...


def test_lookup_order_does_not_leak_fixture_annotations() -> None:
    """The `scenario` / `note` fields are for humans reading the JSON."""
    # TODO: nothing returned to the model contains a `scenario` string
    ...


def test_customer_with_two_active_orders_is_not_guessed() -> None:
    """CUST-003 has ORD-1004 and ORD-1005; the agent must ask which."""
    # TODO: unresolved order -> a clarifying question, not a lookup
    ...


def test_in_transit_order_has_no_delivery_date() -> None:
    """ORD-1005 is still in transit, so no return clock has started."""
    # TODO: order.delivered_at is None, status is OrderStatus.IN_TRANSIT
    ...


def test_verify_identity_matching_email_and_order() -> None:
    """Correct email plus a matching order verifies, and yields the region."""
    # TODO: verify_identity("ada@example.com", "ORD-1001") -> verified,
    #       customer_id == "CUST-001", country == "GB"
    ...


def test_verify_identity_mismatched_pair_fails() -> None:
    """A real email with someone else's order does not verify."""
    # TODO: verify_identity("ada@example.com", "ORD-1003").verified is False
    ...
