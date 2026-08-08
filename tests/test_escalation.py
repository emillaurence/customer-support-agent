"""Escalation: the escape hatch has to work without anything else working.

Mocked, so there is little to assert about integration. What matters is that it
never refuses: an agent that can only escalate for verified customers cannot
escalate the person who failed verification.
"""

from __future__ import annotations

from tools import escalate_to_human


def test_escalation_returns_a_case_id() -> None:
    result = escalate_to_human("customer asked for a person")
    assert result.case_id.startswith("CASE-")
    assert len(result.case_id) > len("CASE-")


def test_case_ids_are_distinct() -> None:
    ids = {escalate_to_human("asked for a person").case_id for _ in range(10)}
    assert len(ids) == 10


def test_escalation_records_the_context_a_human_needs() -> None:
    result = escalate_to_human(
        "asked whether Australian consumer law overrides the return window",
        customer_id="CUST-002",
        order_id="ORD-1003",
    )
    assert result.customer_id == "CUST-002"
    assert result.order_id == "ORD-1003"
    assert "consumer law" in result.reason
    assert result.created_at.tzinfo is not None


def test_escalation_works_without_identity() -> None:
    """Whoever could not be verified is exactly who needs a human."""
    result = escalate_to_human("could not verify identity, customer is upset")
    assert result.case_id
    assert result.customer_id is None
    assert result.order_id is None


def test_customer_facing_message_promises_nothing() -> None:
    """It says a person will pick it up, not what they will decide."""
    message = escalate_to_human("wants a refund outside the window").message.lower()
    assert "refund" not in message
    for promise in ("will approve", "guarantee", "you'll get your money"):
        assert promise not in message


def test_message_quotes_the_case_id() -> None:
    result = escalate_to_human("asked for a person")
    assert result.case_id in result.message
