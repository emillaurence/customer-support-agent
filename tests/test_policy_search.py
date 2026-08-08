"""Informational policy retrieval.

`search_policy` reads Neo4j and reports. It does not decide anything about an
order — that boundary is asserted here, because collapsing the two is the easiest
way for an agent to start answering "can I return this" without an order.

The graph is stubbed from `neo4j/policy_graph.json` via the `seeded_graph`
fixture, so these run offline. The same retrieval against the live database is in
`test_neo4j_integration.py`. The last test in this file is the important one: with
no graph, there is nothing to return.
"""

from __future__ import annotations

import pytest

from agent.graph import PolicyGraphUnavailableError
from tests.conftest import break_policy_graph
from tools import search_policy
from tools.search_policy import PolicySearchResult

pytestmark = pytest.mark.usefixtures("seeded_graph")


def _ids(result: PolicySearchResult) -> list[str]:
    return [match.policy.policy_id for match in result.matches]


# --- The three questions the demo asks ----------------------------------


def test_standard_window_is_retrieved() -> None:
    """"What is the standard return window?" -> STANDARD_30_DAY, with its window."""
    result = search_policy("what is the standard return window", product_type="PhysicalBook")
    assert result.matched is True
    assert "STANDARD_30_DAY" in _ids(result)
    standard = next(m for m in result.matches if m.policy.policy_id == "STANDARD_30_DAY")
    assert standard.policy.window_days == 30
    assert standard.policy.window_starts_from == "delivered_at"


def test_ebook_question_retrieves_the_digital_policy() -> None:
    """"Can ebooks be returned?" — the category is inferred from the question."""
    result = search_policy("can I return an ebook")
    assert result.searched_categories == ["EBook"]
    assert _ids(result) == ["DIGITAL_NO_RETURN"]
    match = result.matches[0]
    assert match.policy.window_days is None
    assert "no return window" in " ".join(match.conditions)


def test_australian_question_retrieves_the_regional_policy() -> None:
    """"What is Bookly's Australian return policy?" leads with the AU policy."""
    result = search_policy("what is Bookly's Australian return policy")
    assert result.matches[0].policy.policy_id == "AU_BOOKLY_EXTENDED_RETURN"
    assert result.matches[0].granted_by_region == "AU"
    assert result.matches[0].policy.window_days == 45


# --- What a match carries -----------------------------------------------


def test_matches_include_policy_id_and_rule_path() -> None:
    result = search_policy("return window", product_type="PhysicalBook", country="AU")
    for match in result.matches:
        assert match.policy.policy_id
        assert match.rule_path
        assert match.rule_path[0].startswith("(PhysicalBook)-[:GOVERNED_BY]->")


def test_regional_policy_states_its_condition() -> None:
    """Returned as conditional, not as the rule. The agent must not present it flatly."""
    result = search_policy("return window", product_type="PhysicalBook")
    au = next(m for m in result.matches if m.policy.policy_id == "AU_BOOKLY_EXTENDED_RETURN")
    assert au.granted_by_region == "AU"
    assert any("only to customers in AU" in condition for condition in au.conditions)


def test_promotional_policy_states_its_condition() -> None:
    result = search_policy("return window", product_type="PhysicalBook")
    holiday = next(m for m in result.matches if m.policy.policy_id == "HOLIDAY_EXTENDED_RETURN")
    assert any("MIDYEAR_HOLIDAY_SALE_2026" in condition for condition in holiday.conditions)


def test_default_policy_is_unconditional() -> None:
    result = search_policy("return window", product_type="PhysicalBook")
    standard = next(m for m in result.matches if m.policy.policy_id == "STANDARD_30_DAY")
    assert standard.conditions == []
    assert standard.granted_by_region is None


def test_overrides_are_reported() -> None:
    """Why one policy displaces another comes from the edges, not from prose."""
    result = search_policy("return window", product_type="PhysicalBook", country="AU")
    au = next(m for m in result.matches if m.policy.policy_id == "AU_BOOKLY_EXTENDED_RETURN")
    assert set(au.outranks) == {"STANDARD_30_DAY", "HOLIDAY_EXTENDED_RETURN"}


def test_source_is_always_the_graph() -> None:
    assert search_policy("return window").source == "neo4j"


# --- Filtering and ordering ---------------------------------------------


def test_naming_a_region_excludes_other_regions_policies() -> None:
    """A GB customer's question must not surface the AU extension as available."""
    result = search_policy("return window", product_type="PhysicalBook", country="GB")
    assert "AU_BOOKLY_EXTENDED_RETURN" not in _ids(result)
    assert "STANDARD_30_DAY" in _ids(result)


def test_category_ordering_beats_raw_precedence() -> None:
    """DIGITAL_NO_RETURN has precedence 100, but it is not the answer about paperbacks."""
    result = search_policy("return window", product_type="PhysicalBook")
    assert "DIGITAL_NO_RETURN" not in _ids(result)


def test_question_naming_neither_searches_both_categories() -> None:
    """Better to return both rules than to guess which the customer meant."""
    result = search_policy("what is your return policy")
    assert result.searched_categories == ["PhysicalBook", "EBook"]
    assert {"STANDARD_30_DAY", "DIGITAL_NO_RETURN"} <= set(_ids(result))


# --- No match, and no graph ---------------------------------------------


def test_unknown_category_is_a_structured_no_match() -> None:
    """A no-match is a result, not an exception, and not an invented policy."""
    result = search_policy("can I return a vinyl record", product_type="VinylRecord")
    assert result.matched is False
    assert result.matches == []
    assert result.message
    assert isinstance(result, PolicySearchResult)


def test_no_match_invents_nothing() -> None:
    result = search_policy("do you price match", product_type="VinylRecord")
    dumped = result.model_dump_json()
    for invented in ("30", "window_days", "STANDARD"):
        assert invented not in dumped


def test_search_does_not_decide_eligibility() -> None:
    """The result has no verdict and no token — that is the other tool's job."""
    dumped = search_policy("can I return my book", product_type="PhysicalBook").model_dump()
    assert "eligible" not in dumped
    assert "eligibility_token" not in dumped


def test_unavailable_graph_fails_and_does_not_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """The property the whole design rests on: no graph, no answer.

    Not an empty result, not a cached one, not a policy read from a file — an
    error the agent has to tell the customer about.
    """
    break_policy_graph(monkeypatch)

    with pytest.raises(PolicyGraphUnavailableError, match="cannot reach Neo4j"):
        search_policy("what is the standard return window", product_type="PhysicalBook")
