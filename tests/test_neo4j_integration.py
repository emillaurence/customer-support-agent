"""Integration tests against the real, seeded Neo4j database.

    pytest -m integration

Everything else in the suite stubs the graph from `neo4j/policy_graph.json`. These
tests take no stub: they open a driver, run the actual Cypher, and re-check the
decisions that matter against what is really in the database. That is what makes
the stub trustworthy — if the ingested graph and the seed file ever diverge, the
unit tests keep passing and these stop.

Skipped, not failed, when Neo4j is unreachable: a contributor without a database
running should still get a green unit suite. Run `python neo4j/ingest.py` first.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from agent.graph import PolicyGraphUnavailableError, fetch_policies_for_category, get_driver
from tools import check_return_eligibility, search_policy

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def live_graph() -> None:
    """Skip the module unless a seeded database answers.

    Checks that policies are actually present, not just that the server is up —
    an empty database would otherwise produce confusing no-match failures rather
    than an honest "run the ingest script".
    """
    try:
        driver = get_driver()
        records, _, _ = driver.execute_query("MATCH (p:Policy) RETURN count(p) AS policies")
    except PolicyGraphUnavailableError as exc:
        pytest.skip(f"Neo4j not available: {exc}")

    if not records or records[0]["policies"] == 0:
        pytest.skip("Neo4j is reachable but holds no policies — run `python neo4j/ingest.py`")


# --- The graph itself ---------------------------------------------------


def test_physical_book_policies_come_back_from_the_database() -> None:
    rows = fetch_policies_for_category("PhysicalBook")
    ids = [row["policy"]["policy_id"] for row in rows]
    assert set(ids) == {
        "STANDARD_30_DAY",
        "HOLIDAY_EXTENDED_RETURN",
        "AU_BOOKLY_EXTENDED_RETURN",
    }
    # Ordered by precedence, which is what eligibility ranks on.
    assert ids[0] == "AU_BOOKLY_EXTENDED_RETURN"


def test_ebook_category_has_only_the_digital_policy() -> None:
    from agent.models import Policy

    rows = fetch_policies_for_category("EBook")
    assert [row["policy"]["policy_id"] for row in rows] == ["DIGITAL_NO_RETURN"]
    # Neo4j does not store null properties, so `window_days` is absent from the
    # row rather than present-and-None. Either way the model reads it as None,
    # which is what "returns are not offered" means.
    assert Policy.model_validate(rows[0]["policy"]).window_days is None


def test_region_edge_is_present_in_the_database() -> None:
    """The HAS_OVERRIDE edge is what gates the AU policy. Without it, everyone gets it."""
    rows = fetch_policies_for_category("PhysicalBook")
    au = next(r for r in rows if r["policy"]["policy_id"] == "AU_BOOKLY_EXTENDED_RETURN")
    assert au["granted_to_regions"] == ["AU"]
    standard = next(r for r in rows if r["policy"]["policy_id"] == "STANDARD_30_DAY")
    assert standard["granted_to_regions"] == []


def test_override_edges_are_present_in_the_database() -> None:
    rows = fetch_policies_for_category("PhysicalBook")
    au = next(r for r in rows if r["policy"]["policy_id"] == "AU_BOOKLY_EXTENDED_RETURN")
    assert set(au["outranks"]) == {"STANDARD_30_DAY", "HOLIDAY_EXTENDED_RETURN"}


def test_unknown_category_returns_nothing_rather_than_erroring() -> None:
    assert fetch_policies_for_category("VinylRecord") == []


def test_promotional_dates_parse_off_the_database() -> None:
    """Read back as dates, since eligibility compares them to an order date."""
    from agent.models import Policy

    rows = fetch_policies_for_category("PhysicalBook")
    holiday = next(r for r in rows if r["policy"]["policy_id"] == "HOLIDAY_EXTENDED_RETURN")
    policy = Policy.model_validate(holiday["policy"])
    assert policy.promotion_code == "MIDYEAR_HOLIDAY_SALE_2026"
    assert policy.promotion_active_from is not None
    assert policy.promotion_active_from < policy.promotion_active_to


# --- search_policy against the live graph ------------------------------


def test_search_finds_the_standard_window_live() -> None:
    result = search_policy("what is the standard return window", product_type="PhysicalBook")
    assert result.matched is True
    standard = next(m for m in result.matches if m.policy.policy_id == "STANDARD_30_DAY")
    assert standard.policy.window_days == 30


def test_search_finds_the_ebook_policy_live() -> None:
    result = search_policy("can ebooks be returned")
    assert [m.policy.policy_id for m in result.matches] == ["DIGITAL_NO_RETURN"]


def test_search_finds_the_australian_policy_live() -> None:
    result = search_policy("what is Bookly's Australian return policy")
    assert result.matches[0].policy.policy_id == "AU_BOOKLY_EXTENDED_RETURN"
    assert result.matches[0].granted_by_region == "AU"


def test_search_no_match_live() -> None:
    result = search_policy("can I return a vinyl record", product_type="VinylRecord")
    assert result.matched is False
    assert result.matches == []


# --- The full decision, live ------------------------------------------


@pytest.mark.parametrize(
    ("order_id", "item_id", "customer_id", "eligible", "policy_id"),
    [
        ("ORD-1001", "ITEM-100", "CUST-001", True, "STANDARD_30_DAY"),
        ("ORD-1002", "ITEM-101", "CUST-001", False, "STANDARD_30_DAY"),
        ("ORD-1004", "ITEM-200", "CUST-003", False, "DIGITAL_NO_RETURN"),
        ("ORD-1003", "ITEM-102", "CUST-002", True, "AU_BOOKLY_EXTENDED_RETURN"),
        ("ORD-1003", "ITEM-201", "CUST-002", False, "DIGITAL_NO_RETURN"),
        ("ORD-1006", "ITEM-103", "CUST-004", True, "HOLIDAY_EXTENDED_RETURN"),
    ],
)
def test_eligibility_against_the_live_graph(
    order_id: str,
    item_id: str,
    customer_id: str,
    eligible: bool,
    policy_id: str,
    now: datetime,
) -> None:
    """The same six scenarios the unit tests cover, decided by the real database."""
    decision = check_return_eligibility(order_id, item_id, customer_id, now=now)
    assert decision.eligible is eligible
    assert decision.policy_id == policy_id


def test_non_au_customer_never_gets_the_au_policy_live() -> None:
    """The precedence trap, against real data: AU is precedence 10, and unreachable from GB."""
    from tools.check_return_eligibility import applicable_policies
    from tools.fixtures import load_orders

    order = next(o for o in load_orders() if o.order_id == "ORD-1002")
    gb = [policy.policy_id for policy, _, _ in applicable_policies("PhysicalBook", order, "GB")]
    au = [policy.policy_id for policy, _, _ in applicable_policies("PhysicalBook", order, "AU")]

    assert "AU_BOOKLY_EXTENDED_RETURN" not in gb
    assert "AU_BOOKLY_EXTENDED_RETURN" in au


def test_informational_and_eligibility_agree_on_australia_live(now: datetime) -> None:
    """The Australia regression, against the real database rather than the stub.

    The informational answer and the decision are selected by the same mechanism,
    so if they ever disagree about which policy governs an AU physical book, they
    disagree here first.
    """
    informational = search_policy(
        "what is the return policy for Australian customers?", product_type="PhysicalBook"
    )
    decision = check_return_eligibility("ORD-1003", "ITEM-102", "CUST-002", now=now)

    assert informational.region == "AU"
    assert informational.region_policy_found is True
    assert informational.resolved.policy_id == decision.policy_id == "AU_BOOKLY_EXTENDED_RETURN"
    assert informational.resolved.return_window_days == 45
    assert informational.resolved.rule_path == decision.rule_path
    assert decision.eligible is True
    assert decision.days_remaining == 11


def test_rule_path_reflects_the_live_traversal(now: datetime) -> None:
    decision = check_return_eligibility("ORD-1003", "ITEM-102", "CUST-002", now=now)
    assert "(AU)-[:HAS_OVERRIDE]->(AU_BOOKLY_EXTENDED_RETURN)" in decision.rule_path
