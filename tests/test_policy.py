"""Policy: which rule governs, for whom, and why — answered the same way twice.

These cases are the reason the policy lives in a graph. A physical book inside
the window, one outside it, an ebook, an Australian customer, a promotional
order, an order still in transit.

The regression the Australia section exists for: informational lookup and
customer-specific eligibility used to disagree. `search_policy` had a region
filter of its own and was called with the *verified customer's* region regardless
of what the customer had just asked about, so a question naming Australia was
answered for GB — and the AU policy, filtered out, looked like an absence.
Meanwhile `check_return_eligibility` applied the AU 45-day window to the same
account. Two properties are pinned here, and they are the whole fix:

* **Region precedence.** A region named in the current question outranks the
  session's region, which outranks nothing at all.
* **One policy truth.** Both tools select through `policy/policy.py`, so the
  window an informational answer quotes is the window a decision was made on.

Dates are measured against the fixed clock (2026-08-08), which is what the order
fixtures were written against. The graph is stubbed from
`neo4j/policy_graph.json` so these run offline; the same scenarios run against
the live database in `test_neo4j_integration.py`.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from agent.state import SessionState
from agent.tools import check_return_eligibility, invoke_tool
from policy import graph
from policy.graph import PolicyGraphUnavailableError
from policy.policy import (
    Policy,
    PolicyContext,
    PolicySearchResult,
    ProductType,
    applicable_policies,
    normalize_region,
    policy_applies,
    region_from_text,
    resolve_region,
    search_policy,
)
from tests.conftest import FIXED_NOW, break_policy_graph

pytestmark = pytest.mark.usefixtures("seeded_graph")

ROOT = Path(__file__).resolve().parent.parent
GRAPH_SEED = ROOT / "neo4j" / "policy_graph.json"

AU_POLICY = "AU_BOOKLY_EXTENDED_RETURN"
AU_WINDOW_DAYS = 45
STANDARD_POLICY = "STANDARD_30_DAY"

BRUCE, BRUCE_REGION = "CUST-002", "AU"
BRUCE_ORDER, BRUCE_BOOK, BRUCE_EBOOK = "ORD-1003", "ITEM-102", "ITEM-201"

AU_QUESTION = "What is the return policy for Australian customers?"
UK_QUESTION = "What is the return policy in the UK?"


def ids(result: PolicySearchResult) -> list[str]:
    return [match.policy.policy_id for match in result.matches]


def decide(order_id: str, item_id: str, customer_id: str, now: datetime = FIXED_NOW):
    return check_return_eligibility(order_id, item_id, customer_id, now=now)


@pytest.fixture
def gb_session() -> SessionState:
    """A verified GB customer — the session region the AU question must outrank."""
    return SessionState(
        verified_customer_id="CUST-001", customer_region="GB", active_order_ids=["ORD-1001"]
    )


@pytest.fixture
def au_session() -> SessionState:
    """Bruce, verified. Two active orders, so neither is adopted."""
    return SessionState(
        verified_customer_id=BRUCE,
        customer_region=BRUCE_REGION,
        active_order_ids=[BRUCE_ORDER, "ORD-1007"],
    )


# =========================================================================
# Regions
# =========================================================================


@pytest.mark.parametrize(
    ("written", "code"),
    [
        ("Australia", "AU"), ("Australian", "AU"), ("AU", "AU"), ("au", "AU"),
        ("United Kingdom", "GB"), ("UK", "GB"), ("GB", "GB"),
    ],
)
def test_supported_regions_normalize_deterministically(written: str, code: str) -> None:
    """A table, not a judgement the model makes."""
    assert normalize_region(written) == code


def test_unknown_region_names_resolve_to_nothing() -> None:
    """Better no region than a guessed one."""
    assert normalize_region("somewhere") is None
    assert normalize_region("") is None
    assert region_from_text("what is your return policy") is None


def test_region_is_read_from_the_question_on_word_boundaries() -> None:
    assert region_from_text(AU_QUESTION) == "AU"
    assert region_from_text(UK_QUESTION) == "GB"
    # Not a region, and a substring match would have said otherwise.
    assert region_from_text("my aunt ordered it for me") is None


def test_precedence_order_is_explicit() -> None:
    assert resolve_region(AU_QUESTION, "GB") == "AU"  # the question wins
    assert resolve_region("what is your return window", "GB") == "GB"  # session falls back
    assert resolve_region("what is your return window", None) is None  # global context


# =========================================================================
# search_policy: what the rules are
# =========================================================================


def test_standard_window_is_retrieved() -> None:
    """"What is the standard return window?" → STANDARD_30_DAY, with its window."""
    result = search_policy("what is the standard return window", product_type="PhysicalBook")
    assert result.matched is True
    standard = next(m for m in result.matches if m.policy.policy_id == STANDARD_POLICY)
    assert standard.policy.window_days == 30
    assert standard.policy.window_starts_from == "delivered_at"


def test_ebook_question_retrieves_the_digital_policy() -> None:
    """The category is inferred from the question."""
    result = search_policy("can I return an ebook")
    assert result.searched_categories == ["EBook"]
    assert ids(result) == ["DIGITAL_NO_RETURN"]
    assert result.matches[0].policy.window_days is None
    assert "no return window" in " ".join(result.matches[0].conditions)


def test_australian_question_retrieves_the_regional_policy() -> None:
    result = search_policy("what is Bookly's Australian return policy")
    assert result.matches[0].policy.policy_id == AU_POLICY
    assert result.matches[0].granted_by_region == "AU"
    assert result.matches[0].policy.window_days == AU_WINDOW_DAYS


def test_matches_include_policy_id_and_rule_path() -> None:
    result = search_policy("return window", product_type="PhysicalBook", country="AU")
    for match in result.matches:
        assert match.policy.policy_id
        assert match.rule_path[0].startswith("(PhysicalBook)-[:GOVERNED_BY]->")


def test_regional_policy_states_its_condition() -> None:
    """Returned as conditional, not as the rule. The agent must not present it flatly."""
    result = search_policy("return window", product_type="PhysicalBook")
    au = next(m for m in result.matches if m.policy.policy_id == AU_POLICY)
    assert au.granted_by_region == "AU"
    assert any("only to customers in AU" in condition for condition in au.conditions)


def test_promotional_policy_states_its_condition() -> None:
    result = search_policy("return window", product_type="PhysicalBook")
    holiday = next(m for m in result.matches if m.policy.policy_id == "HOLIDAY_EXTENDED_RETURN")
    assert any("MIDYEAR_HOLIDAY_SALE_2026" in condition for condition in holiday.conditions)


def test_default_policy_is_unconditional() -> None:
    result = search_policy("return window", product_type="PhysicalBook")
    standard = next(m for m in result.matches if m.policy.policy_id == STANDARD_POLICY)
    assert standard.conditions == []
    assert standard.granted_by_region is None


def test_overrides_are_reported() -> None:
    """Why one policy displaces another comes from the edges, not from prose."""
    result = search_policy("return window", product_type="PhysicalBook", country="AU")
    au = next(m for m in result.matches if m.policy.policy_id == AU_POLICY)
    assert set(au.outranks) == {STANDARD_POLICY, "HOLIDAY_EXTENDED_RETURN"}


def test_source_is_always_the_graph() -> None:
    assert search_policy("return window").source == "neo4j"


def test_naming_a_region_excludes_other_regions_policies() -> None:
    """A GB customer's question must not surface the AU extension as available."""
    result = search_policy("return window", product_type="PhysicalBook", country="GB")
    assert AU_POLICY not in ids(result)
    assert STANDARD_POLICY in ids(result)


def test_category_ordering_beats_raw_precedence() -> None:
    """DIGITAL_NO_RETURN has precedence 100, but it is not the answer about paperbacks."""
    assert "DIGITAL_NO_RETURN" not in ids(search_policy("return window", product_type="PhysicalBook"))


def test_question_naming_neither_searches_both_categories() -> None:
    """Better to return both rules than to guess which the customer meant."""
    result = search_policy("what is your return policy")
    assert result.searched_categories == ["PhysicalBook", "EBook"]
    assert {STANDARD_POLICY, "DIGITAL_NO_RETURN"} <= set(ids(result))


def test_unknown_category_is_a_structured_no_match() -> None:
    """A no-match is a result, not an exception, and not an invented policy."""
    result = search_policy("can I return a vinyl record", product_type="VinylRecord")
    assert result.matched is False
    assert result.matches == []
    assert result.message


def test_no_match_invents_nothing() -> None:
    dumped = search_policy("do you price match", product_type="VinylRecord").model_dump_json()
    for invented in ("30", "window_days", "STANDARD"):
        assert invented not in dumped


def test_search_does_not_decide_eligibility() -> None:
    """The result has no verdict and no token — that is the other tool's job."""
    dumped = search_policy("can I return my book", product_type="PhysicalBook").model_dump()
    assert "eligible" not in dumped
    assert "eligibility_token" not in dumped


# =========================================================================
# check_return_eligibility: the decision
# =========================================================================


def test_physical_book_within_30_days_is_eligible() -> None:
    """ORD-1001 / ITEM-100, day 11: STANDARD_30_DAY allows it."""
    decision = decide("ORD-1001", "ITEM-100", "CUST-001")
    assert decision.eligible is True
    assert decision.policy_id == STANDARD_POLICY
    assert decision.days_remaining == 19


def test_physical_book_after_window_is_not_eligible() -> None:
    """ORD-1002 / ITEM-101, day 67: past 30 days with no override."""
    decision = decide("ORD-1002", "ITEM-101", "CUST-001")
    assert decision.eligible is False
    assert decision.policy_id == STANDARD_POLICY
    # The explanation names the window rather than just refusing.
    assert "30 days" in decision.explanation
    assert "67 days ago" in decision.explanation
    assert decision.eligibility_token is None


def test_ebook_is_never_eligible() -> None:
    """ORD-1004 / ITEM-200, day 7 — well inside any window, still refused."""
    decision = decide("ORD-1004", "ITEM-200", "CUST-003")
    assert decision.eligible is False
    assert decision.policy_id == "DIGITAL_NO_RETURN"
    assert decision.eligibility_token is None


def test_ebook_not_rescued_by_regional_override() -> None:
    """ITEM-201 is an ebook on an AU customer's order. The AU policy governs
    PhysicalBook only, so there is no path to it however high its precedence."""
    decision = decide(BRUCE_ORDER, BRUCE_EBOOK, BRUCE)
    assert decision.eligible is False
    assert decision.policy_id == "DIGITAL_NO_RETURN"
    assert decision.eligibility_token is None


def test_ebook_and_physical_on_one_order_decide_differently() -> None:
    """ORD-1003 holds both. The decision is per item, never per order."""
    assert decide(BRUCE_ORDER, BRUCE_BOOK, BRUCE).eligible is True
    assert decide(BRUCE_ORDER, BRUCE_EBOOK, BRUCE).eligible is False


def test_au_customer_gets_extended_window() -> None:
    """Day 34: past STANDARD_30_DAY, inside AU_BOOKLY_EXTENDED_RETURN's 45 days."""
    decision = decide(BRUCE_ORDER, BRUCE_BOOK, BRUCE)
    assert decision.eligible is True
    assert decision.policy_id == AU_POLICY
    assert decision.days_remaining == 11
    assert f"(AU)-[:HAS_OVERRIDE]->({AU_POLICY})" in decision.rule_path


def test_non_au_customer_never_gets_the_au_policy() -> None:
    """The guard the filter-then-rank order exists for.

    CUST-001 is in GB. The AU policy has precedence 10 against STANDARD_30_DAY's
    0, so the assertion that matters is the policy named, not the verdict.
    """
    decision = decide("ORD-1002", "ITEM-101", "CUST-001")
    assert decision.policy_id == STANDARD_POLICY
    assert AU_POLICY not in " ".join(decision.rule_path)


def test_au_policy_is_not_offered_to_a_us_customer() -> None:
    assert decide("ORD-1004", "ITEM-200", "CUST-003").policy_id != AU_POLICY


def test_promotional_order_gets_extended_window() -> None:
    """ORD-1006 / ITEM-103, day 41, placed under MIDYEAR_HOLIDAY_SALE_2026."""
    decision = decide("ORD-1006", "ITEM-103", "CUST-004")
    assert decision.eligible is True
    assert decision.policy_id == "HOLIDAY_EXTENDED_RETURN"
    assert decision.days_remaining == 19


def test_non_promotional_order_does_not_get_the_extension() -> None:
    """The promotion is not a blanket 60 days for everyone."""
    decision = decide("ORD-1002", "ITEM-101", "CUST-001")
    assert decision.eligible is False
    assert decision.policy_id == STANDARD_POLICY
    assert "HOLIDAY_EXTENDED_RETURN" not in " ".join(decision.rule_path)


def test_promotion_must_cover_the_order_date() -> None:
    """A promotion code alone is not enough — the order must fall in its dates.

    ORD-1001 was placed 2026-07-24, after the promotion closed on 2026-07-15, so
    carrying the code must not earn it the extension.
    """
    from agent.tools import _load_orders

    order = next(o for o in _load_orders() if o.order_id == "ORD-1001")
    order.promotion_code = "MIDYEAR_HOLIDAY_SALE_2026"

    applicable = applicable_policies(
        "PhysicalBook",
        PolicyContext(region="GB", promotion_code=order.promotion_code, placed_at=order.placed_at),
    )
    assert [c.policy.policy_id for c in applicable] == [STANDARD_POLICY]


def test_region_gate_holds_at_any_precedence() -> None:
    """The applicability rule on its own, with precedence turned up to absurd.

    A region-granted policy with precedence 999 still does not reach a customer
    outside that region. Precedence ranks what applies; it cannot make something
    apply.
    """
    regional = Policy(
        policy_id="SOME_REGIONAL_POLICY",
        name="Regional",
        summary="A regional extension.",
        window_days=365,
        precedence=999,
    )

    assert policy_applies(regional, ["AU"], PolicyContext(region="AU")) is True
    assert policy_applies(regional, ["AU"], PolicyContext(region="GB")) is False
    # No region gate at all: applies to everyone in the category.
    assert policy_applies(regional, [], PolicyContext(region="GB")) is True


def test_in_transit_order_has_no_window_yet() -> None:
    """ORD-1005 has no delivered_at, so the clock has not started."""
    decision = decide("ORD-1005", "ITEM-101", "CUST-003")
    assert decision.eligible is False
    assert "hasn't arrived" in decision.explanation
    assert decision.eligibility_token is None


def test_existing_return_blocks_a_second_one() -> None:
    """RET-5001 is already open against ORD-1007 / ITEM-100, at day 19."""
    decision = decide("ORD-1007", "ITEM-100", BRUCE)
    assert decision.eligible is False
    assert "RET-5001" in decision.explanation
    assert decision.eligibility_token is None


@pytest.mark.parametrize(
    ("order_id", "item_id", "customer_id"),
    [
        ("ORD-1008", "ITEM-101", "CUST-001"),  # someone else's order
        ("ORD-1001", "ITEM-103", "CUST-001"),  # item never bought on that order
        ("ORD-1001", "ITEM-100", "CUST-999"),  # unverified customer
    ],
)
def test_a_refusal_that_never_reached_a_policy_names_none(
    order_id: str, item_id: str, customer_id: str
) -> None:
    decision = decide(order_id, item_id, customer_id)
    assert decision.eligible is False
    assert decision.policy_id is None


def test_rule_path_starts_at_the_product_category() -> None:
    for order_id, item_id, customer_id, category in [
        ("ORD-1001", "ITEM-100", "CUST-001", "PhysicalBook"),
        ("ORD-1004", "ITEM-200", "CUST-003", "EBook"),
    ]:
        decision = decide(order_id, item_id, customer_id)
        assert decision.rule_path[0].startswith(f"({category})-[:GOVERNED_BY]->")


def test_decision_includes_an_explainable_rule_path() -> None:
    """The traversal is reported, not a hardcoded sentence: what governs, what
    granted it, and what it displaced."""
    decision = decide(BRUCE_ORDER, BRUCE_BOOK, BRUCE)
    assert decision.rule_path[0] == f"(PhysicalBook)-[:GOVERNED_BY]->({AU_POLICY})"
    assert f"(AU)-[:HAS_OVERRIDE]->({AU_POLICY})" in decision.rule_path
    assert f"({AU_POLICY})-[:OVERRIDES]->({STANDARD_POLICY})" in decision.rule_path


def test_explanation_is_safe_to_read_to_a_customer() -> None:
    """No policy ids, no tool names, no talk of precedence or the graph."""
    for order_id, item_id, customer_id in [
        ("ORD-1001", "ITEM-100", "CUST-001"),
        ("ORD-1002", "ITEM-101", "CUST-001"),
        (BRUCE_ORDER, BRUCE_BOOK, BRUCE),
        ("ORD-1004", "ITEM-200", "CUST-003"),
        ("ORD-1006", "ITEM-103", "CUST-004"),
    ]:
        explanation = decide(order_id, item_id, customer_id).explanation
        assert explanation
        for leak in (
            STANDARD_POLICY, AU_POLICY, "HOLIDAY_EXTENDED_RETURN", "DIGITAL_NO_RETURN",
            "precedence", "check_return_eligibility", "GOVERNED_BY", "Neo4j",
        ):
            assert leak not in explanation


# =========================================================================
# Australia, answered the same way twice
# =========================================================================


def test_explicit_region_outranks_the_session_region(gb_session: SessionState) -> None:
    """A GB customer asking about Australia is asking about Australia."""
    outcome = invoke_tool("search_policy", {"query": AU_QUESTION}, gb_session, now=FIXED_NOW)

    # The call the tool actually received — this is what the trace shows.
    assert outcome.args_used["country"] == "AU"

    result = outcome.payload
    assert result.region == "AU"
    assert result.resolved.policy_id == AU_POLICY
    assert result.resolved.return_window_days == AU_WINDOW_DAYS
    assert "GB" not in outcome.content


def test_session_region_is_used_only_as_a_fallback(gb_session: SessionState) -> None:
    """With no region in the question, the verified customer's region decides."""
    outcome = invoke_tool(
        "search_policy",
        {"query": "what is your return window", "product_type": "PhysicalBook"},
        gb_session,
        now=FIXED_NOW,
    )
    assert outcome.args_used["country"] == "GB"
    assert outcome.payload.resolved.policy_id == STANDARD_POLICY


def test_an_unverified_session_can_still_ask_about_a_region() -> None:
    """Policy is public, so the region has to come out of the question."""
    outcome = invoke_tool("search_policy", {"query": AU_QUESTION}, SessionState())
    assert outcome.args_used["country"] == "AU"
    assert outcome.payload.resolved.policy_id == AU_POLICY


def test_australian_customer_gets_the_au_policy(au_session: SessionState) -> None:
    result = invoke_tool(
        "search_policy", {"query": AU_QUESTION}, au_session, now=FIXED_NOW
    ).payload

    assert result.matched is True
    assert result.region == "AU"
    assert result.region_policy_found is True
    assert result.resolved.policy_id == AU_POLICY
    assert result.resolved.region == "AU"
    assert result.resolved.return_window_days == AU_WINDOW_DAYS
    assert result.resolved.category == "PhysicalBook"
    assert result.resolved.precedence == 10
    assert set(result.resolved.overrides) == {STANDARD_POLICY, "HOLIDAY_EXTENDED_RETURN"}
    assert result.resolved.rule_path[:2] == [
        f"(PhysicalBook)-[:GOVERNED_BY]->({AU_POLICY})",
        f"(AU)-[:HAS_OVERRIDE]->({AU_POLICY})",
    ]


def test_the_result_says_whether_a_regional_policy_was_found(au_session: SessionState) -> None:
    """The structural fix for "I don't see any region-specific policy for Australia".

    The absence of an AU entry in a list is not the same statement as "there is no
    AU policy", and an agent cannot be relied on to tell them apart. So the result
    says which it is, in a field.
    """
    found = invoke_tool("search_policy", {"query": AU_QUESTION}, au_session, now=FIXED_NOW).payload
    assert found.region_policy_found is True
    assert "region-specific" in found.region_note
    assert "No AU" not in found.region_note

    absent = invoke_tool("search_policy", {"query": UK_QUESTION}, au_session, now=FIXED_NOW).payload
    assert absent.region_policy_found is False
    assert absent.region_note.startswith("No GB-specific policy")


def test_the_australian_answer_is_not_the_same_everywhere(au_session: SessionState) -> None:
    """45 days in AU and 30 in GB, from the same tool on the same session."""
    au = invoke_tool("search_policy", {"query": AU_QUESTION}, au_session, now=FIXED_NOW).payload
    gb = invoke_tool("search_policy", {"query": UK_QUESTION}, au_session, now=FIXED_NOW).payload

    assert (au.resolved.return_window_days, gb.resolved.return_window_days) == (45, 30)
    assert au.resolved.policy_id != gb.resolved.policy_id


def test_ebook_restrictions_still_apply_in_australia(au_session: SessionState) -> None:
    """No region rescues an ebook."""
    result = invoke_tool(
        "search_policy",
        {"query": "can I return an ebook in Australia", "product_type": "EBook"},
        au_session,
        now=FIXED_NOW,
    ).payload

    assert result.resolved.policy_id == "DIGITAL_NO_RETURN"
    assert result.resolved.return_window_days is None
    assert AU_POLICY not in ids(result)


def test_search_and_eligibility_resolve_the_same_policy(au_session: SessionState) -> None:
    """The property the fix exists for: one policy truth, two presentations."""
    informational = invoke_tool(
        "search_policy",
        {"query": AU_QUESTION, "product_type": "PhysicalBook"},
        au_session,
        now=FIXED_NOW,
    ).payload
    decision = decide(BRUCE_ORDER, BRUCE_BOOK, BRUCE)

    assert informational.resolved.policy_id == decision.policy_id == AU_POLICY
    assert informational.resolved.return_window_days == AU_WINDOW_DAYS
    assert informational.resolved.rule_path == decision.rule_path

    # Day 34 of a 45-day window.
    assert decision.eligible is True
    assert decision.days_remaining == 11
    assert "45" in decision.explanation


def test_an_au_customer_asking_about_the_uk_gets_the_uk_answer(au_session: SessionState) -> None:
    outcome = invoke_tool("search_policy", {"query": UK_QUESTION}, au_session, now=FIXED_NOW)

    assert outcome.args_used["country"] == "GB"
    result = outcome.payload
    assert result.region == "GB"
    assert result.resolved.policy_id == STANDARD_POLICY
    assert result.resolved.return_window_days == 30
    assert AU_POLICY not in ids(result)
    # The customer's own region did not leak into the answer.
    assert result.resolved.region is None


def test_asking_about_the_uk_does_not_change_the_customers_own_eligibility(
    au_session: SessionState,
) -> None:
    """An informational question is not a change of region."""
    invoke_tool("search_policy", {"query": UK_QUESTION}, au_session, now=FIXED_NOW)

    assert au_session.customer_region == BRUCE_REGION
    decision = decide(BRUCE_ORDER, BRUCE_BOOK, BRUCE)
    assert decision.policy_id == AU_POLICY
    assert decision.days_remaining == 11


def test_a_uk_customer_is_never_handed_the_australian_window() -> None:
    """The other half of one policy truth: shared selection did not widen access.

    CUST-001's ORD-1002 is day 67: outside 30, and inside 45 — so a GB customer
    reaching the AU policy would show up here as an eligible return.
    """
    informational = search_policy("return window", product_type="PhysicalBook", country="GB")
    assert informational.resolved.policy_id == STANDARD_POLICY
    assert AU_POLICY not in ids(informational)

    decision = decide("ORD-1002", "ITEM-101", "CUST-001")
    assert decision.eligible is False
    assert decision.policy_id == STANDARD_POLICY


def test_a_promotional_override_still_outranks_the_standard_policy() -> None:
    """Higher-precedence applicable policies are unaffected by the region fix."""
    decision = decide("ORD-1006", "ITEM-103", "CUST-004")
    assert decision.policy_id == "HOLIDAY_EXTENDED_RETURN"
    assert decision.eligible is True


# =========================================================================
# No graph, no answer
#
# The property the whole design rests on. Not an empty result, not a cached one,
# not a policy read from a file — an error the agent has to tell the customer
# about.
# =========================================================================


def test_search_fails_and_does_not_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    break_policy_graph(monkeypatch)

    with pytest.raises(PolicyGraphUnavailableError, match="cannot reach Neo4j"):
        search_policy("what is the standard return window", product_type="PhysicalBook")


def test_eligibility_fails_and_does_not_guess(monkeypatch: pytest.MonkeyPatch) -> None:
    """An eligibility answer from a stale fixture would be worse than an error,
    and no token is minted on the way out."""
    break_policy_graph(monkeypatch)

    with pytest.raises(PolicyGraphUnavailableError, match="cannot reach Neo4j"):
        decide("ORD-1001", "ITEM-100", "CUST-001")

    from agent.tools import _GRANTS

    assert _GRANTS == {}


@pytest.fixture
def unconfigured(monkeypatch: pytest.MonkeyPatch):
    """No NEO4J_* in the environment, and no .env to pick them up from."""
    monkeypatch.setattr(graph, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(graph, "_driver", None)
    for name in graph.REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)


def test_missing_configuration_raises(unconfigured) -> None:
    with pytest.raises(PolicyGraphUnavailableError) as exc:
        graph.get_driver()
    assert "NEO4J_URI" in str(exc.value)


def test_partial_configuration_still_raises(unconfigured, monkeypatch) -> None:
    """Half a config is not a config — no silent default for the rest."""
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    with pytest.raises(PolicyGraphUnavailableError) as exc:
        graph.get_driver()
    assert "NEO4J_PASSWORD" in str(exc.value)


def test_unreachable_server_raises(unconfigured, monkeypatch) -> None:
    """A driver that cannot verify connectivity fails loudly and is closed."""
    closed = []

    class DeadDriver:
        def verify_connectivity(self) -> None:
            raise OSError("connection refused")

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(graph.GraphDatabase, "driver", lambda *a, **k: DeadDriver())
    for name in graph.REQUIRED_ENV:
        monkeypatch.setenv(name, "x")

    with pytest.raises(PolicyGraphUnavailableError, match="cannot reach Neo4j"):
        graph.get_driver()
    assert closed, "the driver must be closed when connectivity fails"
    assert graph._driver is None, "a failed connection must not be cached"


# --- One driver, many queries -------------------------------------------
#
# A Neo4j `Driver` owns a connection pool and is meant to be built once. Opening
# one per query would pay a TCP, TLS, and Bolt handshake on every policy lookup —
# far more than the query itself costs. These use a stand-in driver, so nothing
# here connects to anything.


class RecordingDriver:
    """A driver that counts what was asked of it, and hands back no rows."""

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.sessions = 0
        self.closed = False

    def verify_connectivity(self) -> None:
        pass

    def execute_query(self, query: str, **_: Any):
        self.queries.append(query)
        return [], None, None

    def session(self, **_: Any):
        self.sessions += 1
        return object()

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def recorded_driver(unconfigured, monkeypatch: pytest.MonkeyPatch) -> RecordingDriver:
    """Make `GraphDatabase.driver` hand back one recording driver, and count how
    many times it is asked to build one."""
    driver = RecordingDriver()
    built = []

    def build(*args: Any, **kwargs: Any) -> RecordingDriver:
        built.append(kwargs)
        return driver

    monkeypatch.setattr(graph.GraphDatabase, "driver", build)
    for name in graph.REQUIRED_ENV:
        monkeypatch.setenv(name, "x")
    driver.built = built  # type: ignore[attr-defined]
    return driver


def test_the_driver_is_built_once_and_reused(recorded_driver) -> None:
    """Ten policy reads, one driver. The second `get_driver` is a module read."""
    for _ in range(10):
        graph.fetch_policies_for_category("PhysicalBook")

    assert len(recorded_driver.built) == 1
    assert len(recorded_driver.queries) == 10
    assert graph.get_driver() is graph.get_driver() is recorded_driver


def test_connectivity_is_verified_once_not_per_query(recorded_driver, monkeypatch) -> None:
    """The handshake is first-use only — it is the cost being avoided."""
    verified = []
    monkeypatch.setattr(
        recorded_driver, "verify_connectivity", lambda: verified.append(True)
    )

    graph.get_driver()
    for _ in range(5):
        graph.fetch_policies_for_category("EBook")

    assert len(verified) == 1


def test_the_pool_is_configured_and_bounded(recorded_driver) -> None:
    """Pool size, and a ceiling on how long a query waits for a connection — so a
    wedged database surfaces as an outage, not a hang."""
    graph.get_driver()

    (kwargs,) = recorded_driver.built
    assert kwargs["max_connection_pool_size"] == graph.POOL_SIZE
    assert kwargs["connection_acquisition_timeout"] == graph.ACQUISITION_TIMEOUT_SECONDS
    assert kwargs["max_transaction_retry_time"] == graph.TRANSACTION_RETRY_SECONDS
    assert 0 < graph.ACQUISITION_TIMEOUT_SECONDS <= 30


def test_sessions_are_still_opened_independently(recorded_driver) -> None:
    """Reusing the driver is not sharing a session. Sessions stay short-lived and
    per-caller; the pooled connections underneath them are what is shared."""
    driver = graph.get_driver()
    first, second = driver.session(), driver.session()

    assert first is not second
    assert recorded_driver.sessions == 2


def test_closing_the_driver_lets_the_next_call_open_a_fresh_one(recorded_driver) -> None:
    """For shutdown, and for a test that wants a clean slate. Idempotent."""
    graph.get_driver()
    graph.close_driver()

    assert recorded_driver.closed
    assert graph._driver is None
    graph.close_driver()  # no second close, no error

    graph.get_driver()
    assert len(recorded_driver.built) == 2


SOURCE_FILES = [
    path
    for pattern in ("*.py", "*.toml", "*.md", "*.json", "*.cypher", ".env.example")
    for path in ROOT.glob(f"**/{pattern}")
    if not {".venv", "__pycache__", ".git"} & set(path.parts)
]


@pytest.mark.parametrize("token", ["USE_NEO4J", "no_graph", "policies.json"])
def test_no_bypass_token_survives(token: str) -> None:
    """Ban the names the old JSON fallback went by, this test file aside."""
    offenders = [
        path.relative_to(ROOT)
        for path in SOURCE_FILES
        if path != Path(__file__) and token in path.read_text()
    ]
    assert not offenders, f"{token} still referenced in {offenders}"


# =========================================================================
# The policy graph seed
#
# `neo4j/policy_graph.json` is seed data for ingestion, not a runtime store.
# Checked here as seed data — nothing below connects to Neo4j.
# =========================================================================


@pytest.fixture(scope="module")
def seed() -> dict:
    return json.loads(GRAPH_SEED.read_text())


@pytest.fixture(scope="module")
def policies(seed: dict) -> list[Policy]:
    return [Policy.model_validate(p) for p in seed["policies"]]


def test_policy_graph_has_every_section(seed: dict) -> None:
    for section in ("categories", "policies", "regions", "relationships"):
        assert seed[section], section


def test_every_policy_node_validates(policies: list[Policy]) -> None:
    """The seed's Policy nodes match the model the tools read them back as."""
    assert {p.policy_id for p in policies} == {
        STANDARD_POLICY, "DIGITAL_NO_RETURN", "HOLIDAY_EXTENDED_RETURN", AU_POLICY
    }
    assert len({p.policy_id for p in policies}) == len(policies), "duplicate policy id"
    for policy in policies:
        assert policy.name and policy.summary
        # A window without a start date, or a start date without a window, would
        # be undecidable at eligibility time.
        assert (policy.window_days is None) == (policy.window_starts_from is None)


def test_promotional_policy_has_an_active_window(policies: list[Policy]) -> None:
    for policy in policies:
        if policy.promotion_code:
            assert policy.promotion_active_from and policy.promotion_active_to
            assert policy.promotion_active_from <= policy.promotion_active_to


def test_digital_policy_has_no_window_and_no_overrides(seed: dict) -> None:
    """The absence of these is the rule that ebooks cannot be rescued."""
    digital = next(p for p in seed["policies"] if p["policy_id"] == "DIGITAL_NO_RETURN")
    assert digital["window_days"] is None
    assert digital["exceptions"] == []
    assert not [
        r for r in seed["relationships"]
        if r["to"] == "DIGITAL_NO_RETURN" and r["type"] != "GOVERNED_BY"
    ]


def test_expected_policy_relationships_are_present(seed: dict) -> None:
    """The four edges the demo turns on."""
    rels = {(r["type"], r["from"], r["to"]) for r in seed["relationships"]}
    assert ("GOVERNED_BY", "PhysicalBook", STANDARD_POLICY) in rels
    assert ("GOVERNED_BY", "EBook", "DIGITAL_NO_RETURN") in rels
    assert ("OVERRIDES", "HOLIDAY_EXTENDED_RETURN", STANDARD_POLICY) in rels
    assert ("HAS_OVERRIDE", "AU", AU_POLICY) in rels


def test_categories_match_the_product_type_enum(seed: dict) -> None:
    from agent.tools import _load_items

    names = {c["name"] for c in seed["categories"]}
    assert names == {t.value for t in ProductType}
    assert {i.product_type.value for i in _load_items().values()} <= names


def test_customer_countries_exist_as_regions(seed: dict) -> None:
    from agent.tools import _load_customers

    codes = {r["code"] for r in seed["regions"]}
    for customer in _load_customers():
        assert customer.country in codes, customer.customer_id


def test_order_promotions_line_up_with_the_graph(seed: dict) -> None:
    """A promotional extension is only coherent if the code and dates line up."""
    from agent.tools import _load_orders

    promos = {p["promotion_code"]: p for p in seed["policies"] if p.get("promotion_code")}
    for order in _load_orders():
        if not order.promotion_code:
            continue
        assert order.promotion_code in promos, order.order_id
        promo = promos[order.promotion_code]
        placed = order.placed_at.isoformat()
        assert promo["promotion_active_from"] <= placed <= promo["promotion_active_to"], order.order_id


def test_seed_cypher_mentions_every_node(seed: dict, policies: list[Policy]) -> None:
    """Cheap consistency check between the fixture and the reference Cypher."""
    cypher = (ROOT / "neo4j" / "seed.cypher").read_text()
    for policy in policies:
        assert policy.policy_id in cypher, policy.policy_id
    for category in seed["categories"]:
        assert category["name"] in cypher
    for region in seed["regions"]:
        assert f"'{region['code']}'" in cypher


# --- Ingestion validates before it connects -----------------------------
#
# `ingest.py` is a script in neo4j/, not part of an installed package, so it is
# loaded by path. That also avoids putting the repo root on sys.path, where the
# neo4j/ directory would shadow the neo4j driver package.


def _load_ingest():
    spec = importlib.util.spec_from_file_location("bookly_ingest", ROOT / "neo4j" / "ingest.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ingest = _load_ingest()


@pytest.fixture()
def fixture() -> dict[str, Any]:
    return json.loads(GRAPH_SEED.read_text())


def test_the_real_fixture_loads_and_validates() -> None:
    loaded = ingest.load_fixture()
    assert loaded["policies"] and loaded["relationships"]
    index = ingest.build_index(loaded)
    ingest.validate_relationships(loaded, index)
    assert index["PhysicalBook"] == "Category"
    assert index[STANDARD_POLICY] == "Policy"
    assert index["AU"] == "Region"


def test_missing_section_is_rejected(tmp_path: Path, fixture: dict[str, Any]) -> None:
    del fixture["regions"]
    path = tmp_path / "policy_graph.json"
    path.write_text(json.dumps(fixture))
    with pytest.raises(ingest.IngestError, match="missing the 'regions' section"):
        ingest.load_fixture(path)


def test_unparseable_fixture_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "policy_graph.json"
    path.write_text("{not json")
    with pytest.raises(ingest.IngestError, match="not valid JSON"):
        ingest.load_fixture(path)


def test_missing_fixture_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ingest.IngestError, match="not found"):
        ingest.load_fixture(tmp_path / "nope.json")


def test_unknown_relationship_type_is_rejected(fixture: dict[str, Any]) -> None:
    """The requirement that matters: no silent skipping."""
    fixture["relationships"].append({"type": "HAS_WINDOW", "from": STANDARD_POLICY, "to": "AU"})
    with pytest.raises(ingest.IngestError, match="unknown relationship type 'HAS_WINDOW'"):
        ingest.validate_relationships(fixture, ingest.build_index(fixture))


def test_unknown_endpoint_is_rejected(fixture: dict[str, Any]) -> None:
    fixture["relationships"].append(
        {"type": "OVERRIDES", "from": STANDARD_POLICY, "to": "TYPO_POLICY"}
    )
    with pytest.raises(ingest.IngestError, match="matches no category, policy, or region"):
        ingest.validate_relationships(fixture, ingest.build_index(fixture))


def test_wrong_endpoint_labels_are_rejected(fixture: dict[str, Any]) -> None:
    """GOVERNED_BY must go Category -> Policy, not Region -> Policy."""
    fixture["relationships"].append({"type": "GOVERNED_BY", "from": "AU", "to": STANDARD_POLICY})
    with pytest.raises(ingest.IngestError, match=r"must go \(Category\)->\(Policy\)"):
        ingest.validate_relationships(fixture, ingest.build_index(fixture))


def test_duplicate_node_key_is_rejected(fixture: dict[str, Any]) -> None:
    """Endpoints are bare keys, so a key shared across sections is ambiguous."""
    fixture["regions"].append({"code": "PhysicalBook", "name": "Nonsense"})
    with pytest.raises(ingest.IngestError, match="duplicate node key"):
        ingest.build_index(fixture)


def test_node_without_its_key_property_is_rejected(fixture: dict[str, Any]) -> None:
    fixture["policies"].append({"name": "No id"})
    with pytest.raises(ingest.IngestError, match="no 'policy_id'"):
        ingest.build_index(fixture)
