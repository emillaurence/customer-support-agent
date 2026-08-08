"""Return eligibility, including how overlapping policies resolve.

These cases are the reason the policy lives in a graph. Each one is a fixture
scenario: a physical book inside the window, one outside it, an ebook, an AU
customer, a promotional order, an order still in transit, and an item that already
has a return open.

Dates are measured against `FIXED_NOW` (2026-08-08), which is what the order
fixtures were written against — "day 11", "day 34", "day 67" stay true.

The graph is stubbed from `neo4j/policy_graph.json` so these run offline; the same
scenarios run against the live database in `test_neo4j_integration.py`.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from agent.graph import PolicyGraphUnavailableError
from agent.models import EligibilityDecision
from tests.conftest import break_policy_graph
from tools import check_return_eligibility, eligibility_tokens
from tools.check_return_eligibility import applicable_policies, policy_applies

pytestmark = pytest.mark.usefixtures("seeded_graph")


def decide(order_id: str, item_id: str, customer_id: str, now: datetime) -> EligibilityDecision:
    return check_return_eligibility(order_id, item_id, customer_id, now=now)


# --- The default path ---------------------------------------------------


def test_physical_book_within_30_days_is_eligible(now: datetime) -> None:
    """ORD-1001 / ITEM-100, day 11: STANDARD_30_DAY allows it."""
    decision = decide("ORD-1001", "ITEM-100", "CUST-001", now)
    assert decision.eligible is True
    assert decision.policy_id == "STANDARD_30_DAY"
    assert decision.days_remaining == 19


def test_physical_book_after_window_is_not_eligible(now: datetime) -> None:
    """ORD-1002 / ITEM-101, day 67: past 30 days with no override."""
    decision = decide("ORD-1002", "ITEM-101", "CUST-001", now)
    assert decision.eligible is False
    assert decision.policy_id == "STANDARD_30_DAY"
    # The explanation names the window rather than just refusing.
    assert "30 days" in decision.explanation
    assert "67 days ago" in decision.explanation
    assert decision.eligibility_token is None


# --- Ebooks -------------------------------------------------------------


def test_ebook_is_never_eligible(now: datetime) -> None:
    """ORD-1004 / ITEM-200, day 7 — well inside any window, still refused."""
    decision = decide("ORD-1004", "ITEM-200", "CUST-003", now)
    assert decision.eligible is False
    assert decision.policy_id == "DIGITAL_NO_RETURN"
    assert decision.eligibility_token is None


def test_ebook_not_rescued_by_regional_override(now: datetime) -> None:
    """ITEM-201 is an ebook on an AU customer's order.

    AU_BOOKLY_EXTENDED_RETURN governs PhysicalBook only, so there is no path from
    an ebook to it however high its precedence.
    """
    decision = decide("ORD-1003", "ITEM-201", "CUST-002", now)
    assert decision.eligible is False
    assert decision.policy_id == "DIGITAL_NO_RETURN"


def test_ebook_and_physical_on_one_order_decide_differently(now: datetime) -> None:
    """ORD-1003 holds both. The decision is per item, never per order."""
    physical = decide("ORD-1003", "ITEM-102", "CUST-002", now)
    ebook = decide("ORD-1003", "ITEM-201", "CUST-002", now)
    assert physical.eligible is True
    assert ebook.eligible is False


# --- Regional override --------------------------------------------------


def test_au_customer_gets_extended_window(now: datetime) -> None:
    """ORD-1003 / ITEM-102, CUST-002 (AU), day 34.

    Past STANDARD_30_DAY, inside AU_BOOKLY_EXTENDED_RETURN's 45 days.
    """
    decision = decide("ORD-1003", "ITEM-102", "CUST-002", now)
    assert decision.eligible is True
    assert decision.policy_id == "AU_BOOKLY_EXTENDED_RETURN"
    assert decision.days_remaining == 11


def test_non_au_customer_never_gets_the_au_policy(now: datetime) -> None:
    """The guard the filter-then-rank order exists for.

    CUST-001 is in GB. AU_BOOKLY_EXTENDED_RETURN has precedence 10 against
    STANDARD_30_DAY's 0, and its 45-day window would make this day-67 return...
    still expired — so the assertion that matters is the policy named, not the
    verdict. A GB customer must be answered under the GB rule.
    """
    decision = decide("ORD-1002", "ITEM-101", "CUST-001", now)
    assert decision.policy_id == "STANDARD_30_DAY"
    assert "AU_BOOKLY_EXTENDED_RETURN" not in " ".join(decision.rule_path)


def test_au_policy_is_not_offered_to_a_us_customer(now: datetime) -> None:
    """CUST-003 is in US; ORD-1004 is decided without the AU extension."""
    decision = decide("ORD-1004", "ITEM-200", "CUST-003", now)
    assert decision.policy_id != "AU_BOOKLY_EXTENDED_RETURN"


# --- Promotional override ----------------------------------------------


def test_promotional_order_gets_extended_window(now: datetime) -> None:
    """ORD-1006 / ITEM-103, day 41, placed under MIDYEAR_HOLIDAY_SALE_2026.

    Past STANDARD_30_DAY, inside the promotion's 60 days.
    """
    decision = decide("ORD-1006", "ITEM-103", "CUST-004", now)
    assert decision.eligible is True
    assert decision.policy_id == "HOLIDAY_EXTENDED_RETURN"
    assert decision.days_remaining == 19


def test_non_promotional_order_does_not_get_the_extension(now: datetime) -> None:
    """The promotion is not a blanket 60 days for everyone.

    ORD-1002 carries no promotion_code, so at day 67 it is refused under the
    30-day rule — not allowed under the 60-day one.
    """
    decision = decide("ORD-1002", "ITEM-101", "CUST-001", now)
    assert decision.eligible is False
    assert decision.policy_id == "STANDARD_30_DAY"
    assert "HOLIDAY_EXTENDED_RETURN" not in " ".join(decision.rule_path)


def test_promotion_must_cover_the_order_date(now: datetime) -> None:
    """A promotion code alone is not enough — the order must fall in its dates.

    ORD-1001 is moved outside MIDYEAR_HOLIDAY_SALE_2026 (15 Jun – 15 Jul) while
    still carrying the code, and must not receive the extension.
    """
    from tools import fixtures

    order = next(o for o in fixtures.load_orders() if o.order_id == "ORD-1001")
    # Placed 2026-07-24 — after the promotion closed on 2026-07-15.
    order.promotion_code = "MIDYEAR_HOLIDAY_SALE_2026"

    applicable = applicable_policies("PhysicalBook", order, "GB")
    assert [policy.policy_id for policy, _, _ in applicable] == ["STANDARD_30_DAY"]


def test_region_gate_holds_at_any_precedence() -> None:
    """The applicability rule on its own, with precedence turned up to absurd.

    A region-granted policy with precedence 999 still does not reach a customer
    outside that region. Precedence ranks what applies; it cannot make something
    apply.
    """
    from agent.models import Policy
    from tools import fixtures

    order = next(o for o in fixtures.load_orders() if o.order_id == "ORD-1002")
    regional = Policy(
        policy_id="SOME_REGIONAL_POLICY",
        name="Regional",
        summary="A regional extension.",
        window_days=365,
        precedence=999,
    )

    assert policy_applies(regional, ["AU"], order, "AU") is True
    assert policy_applies(regional, ["AU"], order, "GB") is False
    # No region gate at all: applies to everyone in the category.
    assert policy_applies(regional, [], order, "GB") is True


# --- Nothing to decide about -------------------------------------------


def test_in_transit_order_has_no_window_yet(now: datetime) -> None:
    """ORD-1005 has no delivered_at, so the clock has not started."""
    decision = decide("ORD-1005", "ITEM-101", "CUST-003", now)
    assert decision.eligible is False
    assert "hasn't arrived" in decision.explanation
    assert decision.eligibility_token is None


def test_existing_return_blocks_a_second_one(now: datetime) -> None:
    """RET-5001 is already open against ORD-1007 / ITEM-100, at day 19."""
    decision = decide("ORD-1007", "ITEM-100", "CUST-002", now)
    assert decision.eligible is False
    assert "RET-5001" in decision.explanation
    assert decision.eligibility_token is None


def test_another_customers_order_is_not_decided(now: datetime) -> None:
    """ORD-1008 is CUST-004's. CUST-001 gets a refusal, not a decision."""
    decision = decide("ORD-1008", "ITEM-101", "CUST-001", now)
    assert decision.eligible is False
    assert decision.policy_id is None


def test_item_not_on_the_order_is_refused(now: datetime) -> None:
    """ITEM-103 was never bought on ORD-1001."""
    decision = decide("ORD-1001", "ITEM-103", "CUST-001", now)
    assert decision.eligible is False
    assert decision.policy_id is None


def test_unverified_customer_is_refused(now: datetime) -> None:
    decision = decide("ORD-1001", "ITEM-100", "CUST-999", now)
    assert decision.eligible is False
    assert decision.policy_id is None


# --- Tokens ------------------------------------------------------------


def test_eligible_decision_issues_a_token(now: datetime) -> None:
    decision = decide("ORD-1001", "ITEM-100", "CUST-001", now)
    assert decision.eligibility_token is not None


@pytest.mark.parametrize(
    ("order_id", "item_id", "customer_id"),
    [
        ("ORD-1002", "ITEM-101", "CUST-001"),  # outside the window
        ("ORD-1004", "ITEM-200", "CUST-003"),  # ebook
        ("ORD-1005", "ITEM-101", "CUST-003"),  # in transit
        ("ORD-1007", "ITEM-100", "CUST-002"),  # already returned
        ("ORD-1008", "ITEM-101", "CUST-001"),  # someone else's order
    ],
)
def test_ineligible_decision_never_issues_a_token(
    order_id: str, item_id: str, customer_id: str, now: datetime
) -> None:
    """Every refusal path, checked for the same thing: no token escapes."""
    decision = decide(order_id, item_id, customer_id, now)
    assert decision.eligible is False
    assert decision.eligibility_token is None


def test_token_is_bound_to_the_decision_it_came_from(now: datetime) -> None:
    """The token means nothing by itself — the server holds what it permits."""
    decision = decide("ORD-1003", "ITEM-102", "CUST-002", now)
    assert decision.eligibility_token is not None
    grant = eligibility_tokens.lookup(decision.eligibility_token)
    assert grant is not None
    assert (grant.customer_id, grant.order_id, grant.item_id) == (
        "CUST-002",
        "ORD-1003",
        "ITEM-102",
    )
    assert grant.policy_id == "AU_BOOKLY_EXTENDED_RETURN"


def test_each_decision_mints_a_distinct_token(now: datetime) -> None:
    first = decide("ORD-1001", "ITEM-100", "CUST-001", now)
    second = decide("ORD-1003", "ITEM-102", "CUST-002", now)
    assert first.eligibility_token != second.eligibility_token


def test_token_is_not_derived_from_the_ids(now: datetime) -> None:
    """A uuid4, not a recipe a model could reproduce from the arguments."""
    token = decide("ORD-1001", "ITEM-100", "CUST-001", now).eligibility_token
    assert token is not None
    for identifier in ("ORD-1001", "ITEM-100", "CUST-001", "STANDARD_30_DAY"):
        assert identifier.lower() not in token.lower()


# --- Explainability ----------------------------------------------------


def test_decision_includes_an_explainable_rule_path(now: datetime) -> None:
    """The traversal is reported, not a hardcoded sentence."""
    decision = decide("ORD-1003", "ITEM-102", "CUST-002", now)
    assert decision.rule_path[0] == "(PhysicalBook)-[:GOVERNED_BY]->(AU_BOOKLY_EXTENDED_RETURN)"
    # The region hop is what justifies the AU policy being used at all.
    assert "(AU)-[:HAS_OVERRIDE]->(AU_BOOKLY_EXTENDED_RETURN)" in decision.rule_path
    # And what it displaced.
    assert "(AU_BOOKLY_EXTENDED_RETURN)-[:OVERRIDES]->(STANDARD_30_DAY)" in decision.rule_path


def test_rule_path_starts_at_the_product_category(now: datetime) -> None:
    for order_id, item_id, customer_id, category in [
        ("ORD-1001", "ITEM-100", "CUST-001", "PhysicalBook"),
        ("ORD-1004", "ITEM-200", "CUST-003", "EBook"),
    ]:
        decision = decide(order_id, item_id, customer_id, now)
        assert decision.rule_path[0].startswith(f"({category})-[:GOVERNED_BY]->")


def test_explanation_is_safe_to_read_to_a_customer(now: datetime) -> None:
    """No policy ids, no tool names, no talk of precedence or the graph."""
    for order_id, item_id, customer_id in [
        ("ORD-1001", "ITEM-100", "CUST-001"),
        ("ORD-1002", "ITEM-101", "CUST-001"),
        ("ORD-1003", "ITEM-102", "CUST-002"),
        ("ORD-1004", "ITEM-200", "CUST-003"),
        ("ORD-1006", "ITEM-103", "CUST-004"),
    ]:
        explanation = decide(order_id, item_id, customer_id, now).explanation
        assert explanation
        for leak in (
            "STANDARD_30_DAY",
            "AU_BOOKLY_EXTENDED_RETURN",
            "HOLIDAY_EXTENDED_RETURN",
            "DIGITAL_NO_RETURN",
            "precedence",
            "check_return_eligibility",
            "GOVERNED_BY",
            "Neo4j",
        ):
            assert leak not in explanation


# --- No graph, no decision ---------------------------------------------


def test_unavailable_graph_fails_and_does_not_guess(
    now: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An eligibility answer from a stale fixture would be worse than an error."""
    break_policy_graph(monkeypatch)

    with pytest.raises(PolicyGraphUnavailableError, match="cannot reach Neo4j"):
        decide("ORD-1001", "ITEM-100", "CUST-001", now)


def test_no_token_is_issued_when_the_graph_is_unavailable(
    now: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    break_policy_graph(monkeypatch)

    with pytest.raises(PolicyGraphUnavailableError):
        decide("ORD-1001", "ITEM-100", "CUST-001", now)
    assert eligibility_tokens.lookup("") is None
