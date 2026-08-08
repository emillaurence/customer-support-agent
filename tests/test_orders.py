"""Identity verification and order lookup.

Neither tool touches Neo4j — both read the mock data in `data/`. The behaviours
worth locking down are the refusals: an unknown email, an order number that does
not exist, and an order that exists but belongs to someone else.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from agent.models import OrderStatus
from tools import lookup_order, verify_identity
from tools.verify_identity import active_order_ids

# --- verify_identity ----------------------------------------------------


def test_known_email_verifies() -> None:
    result = verify_identity("ada@example.com")
    assert result.verified is True
    assert result.customer_id == "CUST-001"
    assert result.region == "GB"


def test_result_carries_customer_region_and_active_orders() -> None:
    """The three things the agent needs to carry forward, and nothing else."""
    result = verify_identity("bruce@example.com")
    assert (result.customer_id, result.region) == ("CUST-002", "AU")
    assert set(result.active_order_ids) == {"ORD-1003", "ORD-1007"}


def test_unknown_email_is_rejected() -> None:
    result = verify_identity("nobody@example.com")
    assert result.verified is False
    assert result.customer_id is None
    assert result.region is None
    assert result.active_order_ids == []


@pytest.mark.parametrize("email", ["", "   ", "ada@", "ada"])
def test_invalid_email_fails_cleanly(email: str) -> None:
    """No exception, no partial result — just an unverified answer."""
    result = verify_identity(email)
    assert result.verified is False
    assert result.message


def test_email_match_ignores_case_and_whitespace() -> None:
    """Customers type their address; they do not paste it."""
    result = verify_identity("  Ada@Example.COM  ")
    assert result.verified is True
    assert result.customer_id == "CUST-001"


def test_verification_exposes_no_order_detail() -> None:
    """Order ids only. Dates, items, and status are lookup_order's to give out."""
    dumped = verify_identity("ada@example.com").model_dump()
    assert set(dumped) == {"verified", "customer_id", "region", "active_order_ids", "message"}
    assert dumped["active_order_ids"] == ["ORD-1001", "ORD-1002"]


def test_rejection_does_not_say_whether_the_address_is_known() -> None:
    """Confirming an address is not a Bookly customer is itself a disclosure."""
    message = verify_identity("nobody@example.com").message.lower()
    assert "cust-" not in message
    assert "ord-" not in message


def test_customer_with_two_active_orders_gets_both(now: datetime) -> None:
    """CUST-003 has ORD-1004 and ORD-1005.

    The tool hands back both so the agent can ask which. Guessing one is the
    failure this fixture exists to make visible — and it can only ask if it was
    given the choice.
    """
    result = verify_identity("sofia@example.com")
    assert set(result.active_order_ids) == {"ORD-1004", "ORD-1005"}
    assert len(result.active_order_ids) == 2


def test_active_orders_are_newest_first() -> None:
    """ORD-1005 was placed 2026-08-05, ORD-1004 on 2026-08-01."""
    assert active_order_ids("CUST-003") == ["ORD-1005", "ORD-1004"]


# --- lookup_order -------------------------------------------------------


def test_lookup_order_returns_items(now: datetime) -> None:
    details = lookup_order("ORD-1003", "CUST-002", now=now)
    assert details is not None
    assert [item.item_id for item in details.items] == ["ITEM-102", "ITEM-201"]
    assert details.items[0].title == "A Short History of Nearly Everything"


def test_lookup_order_unknown_id_returns_none(now: datetime) -> None:
    """An order number that does not exist yields None, not an error."""
    assert lookup_order("ORD-9999", "CUST-001", now=now) is None


def test_lookup_order_rejects_other_customers_order(now: datetime) -> None:
    """ORD-1008 is CUST-004's. CUST-001 quoting the real number gets nothing."""
    assert lookup_order("ORD-1008", "CUST-001", now=now) is None


def test_missing_and_forbidden_orders_are_indistinguishable(now: datetime) -> None:
    """Same answer either way, so a response cannot confirm an order id is real."""
    assert lookup_order("ORD-1008", "CUST-001", now=now) == lookup_order(
        "ORD-9999", "CUST-001", now=now
    )


def test_lookup_order_does_not_leak_fixture_annotations(now: datetime) -> None:
    """`scenario` is a note to whoever reads the JSON, not data for the model."""
    details = lookup_order("ORD-1007", "CUST-002", now=now)
    assert details is not None
    assert details.order.scenario is None
    assert "RET-5001" not in details.model_dump_json()


def test_shipment_reports_days_since_delivery(now: datetime) -> None:
    """ORD-1001 was delivered 2026-07-28 — 11 days before the fixed clock."""
    details = lookup_order("ORD-1001", "CUST-001", now=now)
    assert details is not None
    assert details.shipment.has_arrived is True
    assert details.shipment.days_since_delivery == 11
    assert details.shipment.status == OrderStatus.DELIVERED


def test_in_transit_order_has_no_delivery_date(now: datetime) -> None:
    """ORD-1005 is still in transit, so no return clock has started."""
    details = lookup_order("ORD-1005", "CUST-003", now=now)
    assert details is not None
    assert details.order.delivered_at is None
    assert details.order.status == OrderStatus.IN_TRANSIT
    assert details.shipment.has_arrived is False
    # Not estimated, not zero. Absent.
    assert details.shipment.days_since_delivery is None


def test_promotional_order_keeps_its_promotion_code(now: datetime) -> None:
    """Eligibility needs it to decide whether the holiday policy applies at all."""
    details = lookup_order("ORD-1006", "CUST-004", now=now)
    assert details is not None
    assert details.order.promotion_code == "MIDYEAR_HOLIDAY_SALE_2026"
