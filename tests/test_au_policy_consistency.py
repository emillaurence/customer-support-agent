"""Australia, answered the same way twice.

The regression this file exists for: informational policy lookup and
customer-specific eligibility disagreed about Australia. `search_policy` had a
region filter of its own and was called with the *verified customer's* region
regardless of what the customer had just asked about, so a question naming
Australia was answered for GB — and the AU policy, filtered out, looked like an
absence. Meanwhile `check_return_eligibility`, reading the graph through its own
applicability logic, applied the AU 45-day window to the same account.

Two properties are pinned here, and they are the whole fix:

* **Region precedence.** A region named in the current question outranks the
  session's region, which outranks nothing at all — see
  `policy_rules.resolve_region`.
* **One policy truth.** Both tools select through `policy_rules`, so the window
  an informational answer quotes is the window a decision was made on.

Bruce (CUST-002, AU) is the customer these run against, with ORD-1003:
`A Short History of Nearly Everything` delivered 34 days before the fixed clock,
which is outside `STANDARD_30_DAY` and inside `AU_BOOKLY_EXTENDED_RETURN`.
"""

from __future__ import annotations

import pytest

from agent import tool_registry
from agent.demo import fresh_session
from agent.graph import PolicyGraphUnavailableError
from agent.state import SessionState
from agent.tracing import ToolStatus
from tests.conftest import FIXED_NOW, break_policy_graph, text, tool_call
from tools import check_return_eligibility, search_policy
from tools.policy_rules import normalize_region, region_from_text, resolve_region

pytestmark = pytest.mark.usefixtures("seeded_graph")

AU_POLICY = "AU_BOOKLY_EXTENDED_RETURN"
AU_WINDOW_DAYS = 45
STANDARD_POLICY = "STANDARD_30_DAY"

BRUCE, BRUCE_REGION = "CUST-002", "AU"
BRUCE_ORDER, BRUCE_BOOK = "ORD-1003", "ITEM-102"
BRUCE_EBOOK = "ITEM-201"

AU_QUESTION = "What is the return policy for Australian customers?"
UK_QUESTION = "What is the return policy in the UK?"


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


def policy_trace(state: SessionState):
    """The last `search_policy` call this session made, as it was traced."""
    return next(
        trace for trace in reversed(state.tool_traces) if trace.tool_name == "search_policy"
    )


# --- Region normalization ------------------------------------------------


@pytest.mark.parametrize(
    ("written", "code"),
    [
        ("Australia", "AU"),
        ("Australian", "AU"),
        ("AU", "AU"),
        ("au", "AU"),
        ("United Kingdom", "GB"),
        ("UK", "GB"),
        ("GB", "GB"),
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


# --- Test A: explicit region overrides session region -------------------


def test_explicit_region_outranks_the_session_region(gb_session: SessionState) -> None:
    """A GB customer asking about Australia is asking about Australia."""
    outcome = tool_registry.invoke(
        "search_policy", {"query": AU_QUESTION}, gb_session, now=FIXED_NOW
    )

    assert outcome.status is ToolStatus.OK
    # The call the tool actually received — this is what the trace shows.
    assert outcome.args_used["country"] == "AU"

    result = outcome.payload
    assert result.region == "AU"
    assert result.resolved.policy_id == AU_POLICY
    assert result.resolved.return_window_days == AU_WINDOW_DAYS
    # And the answer is not the customer's own region's policy.
    assert result.resolved.policy_id != STANDARD_POLICY
    assert "GB" not in outcome.content


def test_session_region_is_used_only_as_a_fallback(gb_session: SessionState) -> None:
    """With no region in the question, the verified customer's region still decides."""
    outcome = tool_registry.invoke(
        "search_policy",
        {"query": "what is your return window", "product_type": "PhysicalBook"},
        gb_session,
        now=FIXED_NOW,
    )
    assert outcome.args_used["country"] == "GB"
    assert outcome.payload.resolved.policy_id == STANDARD_POLICY


def test_precedence_order_is_explicit() -> None:
    assert resolve_region(AU_QUESTION, "GB") == "AU"  # question wins
    assert resolve_region("what is your return window", "GB") == "GB"  # session falls back
    assert resolve_region("what is your return window", None) is None  # global context


def test_an_unverified_session_can_still_ask_about_a_region() -> None:
    """Policy is public, so the region has to come out of the question."""
    outcome = tool_registry.invoke("search_policy", {"query": AU_QUESTION}, SessionState())
    assert outcome.args_used["country"] == "AU"
    assert outcome.payload.resolved.policy_id == AU_POLICY


# --- Test B: an Australian customer's informational lookup --------------


def test_australian_customer_gets_the_au_policy(au_session: SessionState) -> None:
    outcome = tool_registry.invoke(
        "search_policy", {"query": AU_QUESTION}, au_session, now=FIXED_NOW
    )
    result = outcome.payload

    assert outcome.args_used["country"] == "AU"
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


def test_the_result_says_a_regional_policy_was_found(au_session: SessionState) -> None:
    """The structural fix for "I don't see any region-specific policy for Australia".

    The absence of an AU entry in a list is not the same statement as "there is no
    AU policy", and an agent cannot be relied on to tell them apart. So the result
    says which it is, in a field.
    """
    found = tool_registry.invoke(
        "search_policy", {"query": AU_QUESTION}, au_session, now=FIXED_NOW
    ).payload
    assert found.region_policy_found is True
    assert "region-specific" in found.region_note
    assert "No AU" not in found.region_note

    absent = tool_registry.invoke(
        "search_policy", {"query": UK_QUESTION}, au_session, now=FIXED_NOW
    ).payload
    assert absent.region_policy_found is False
    assert absent.region_note.startswith("No GB-specific policy")


def test_the_australian_answer_is_not_the_same_everywhere(au_session: SessionState) -> None:
    """45 days in AU and 30 in GB, from the same tool on the same session."""
    au = tool_registry.invoke(
        "search_policy", {"query": AU_QUESTION}, au_session, now=FIXED_NOW
    ).payload
    gb = tool_registry.invoke(
        "search_policy", {"query": UK_QUESTION}, au_session, now=FIXED_NOW
    ).payload

    assert (au.resolved.return_window_days, gb.resolved.return_window_days) == (45, 30)
    assert au.resolved.policy_id != gb.resolved.policy_id


def test_ebook_restrictions_still_apply_in_australia(au_session: SessionState) -> None:
    """No region rescues an ebook."""
    result = tool_registry.invoke(
        "search_policy",
        {"query": "can I return an ebook in Australia", "product_type": "EBook"},
        au_session,
        now=FIXED_NOW,
    ).payload

    assert result.resolved.policy_id == "DIGITAL_NO_RETURN"
    assert result.resolved.return_window_days is None
    assert AU_POLICY not in [match.policy.policy_id for match in result.matches]


def test_an_informational_lookup_does_not_escalate(
    make_agent, au_session: SessionState
) -> None:
    """A policy question is a read. It does not take the agent out of service."""
    agent, client = make_agent(
        tool_call("search_policy", {"query": AU_QUESTION}),
        text("Physical books can be returned within 45 days of delivery in Australia."),
    )

    agent.respond(au_session, AU_QUESTION)

    assert au_session.escalated is False
    assert policy_trace(au_session).tool_args["country"] == "AU"
    assert au_session.model_turns[-1].routing_reason != "conversation is escalated"


# --- Test C: informational and eligibility agree ------------------------


def test_search_and_eligibility_resolve_the_same_policy(au_session: SessionState) -> None:
    """The property the fix exists for: one policy truth, two presentations."""
    informational = tool_registry.invoke(
        "search_policy",
        {"query": AU_QUESTION, "product_type": "PhysicalBook"},
        au_session,
        now=FIXED_NOW,
    ).payload

    decision = check_return_eligibility(
        BRUCE_ORDER, BRUCE_BOOK, BRUCE, now=FIXED_NOW
    )

    assert informational.resolved.policy_id == decision.policy_id == AU_POLICY
    assert informational.resolved.return_window_days == AU_WINDOW_DAYS
    assert informational.resolved.rule_path == decision.rule_path

    # Day 34 of a 45-day window.
    assert decision.eligible is True
    assert decision.days_remaining == 11
    assert "45" in decision.explanation


def test_bruce_is_eligible_because_of_the_australian_window() -> None:
    """Not because of the standard one — 34 days is outside that."""
    decision = check_return_eligibility(BRUCE_ORDER, BRUCE_BOOK, BRUCE, now=FIXED_NOW)
    assert decision.policy_id == AU_POLICY
    assert f"(AU)-[:HAS_OVERRIDE]->({AU_POLICY})" in decision.rule_path


def test_the_ebook_on_the_same_order_stays_ineligible() -> None:
    decision = check_return_eligibility(BRUCE_ORDER, BRUCE_EBOOK, BRUCE, now=FIXED_NOW)
    assert decision.eligible is False
    assert decision.policy_id == "DIGITAL_NO_RETURN"
    assert decision.eligibility_token is None


def test_the_existing_open_return_is_still_respected() -> None:
    """RET-5001 on ORD-1007 blocks a second return, AU window or not."""
    decision = check_return_eligibility("ORD-1007", "ITEM-100", BRUCE, now=FIXED_NOW)
    assert decision.eligible is False
    assert "RET-5001" in decision.explanation


def test_a_uk_customer_is_never_handed_the_australian_window() -> None:
    """The other half of one policy truth: shared selection did not widen access."""
    informational = search_policy("return window", product_type="PhysicalBook", country="GB")
    assert informational.resolved.policy_id == STANDARD_POLICY
    assert AU_POLICY not in [match.policy.policy_id for match in informational.matches]

    # CUST-001's ORD-1002 is day 67: outside 30, and inside 45 — so a GB customer
    # reaching the AU policy would show up here as an eligible return.
    decision = check_return_eligibility("ORD-1002", "ITEM-101", "CUST-001", now=FIXED_NOW)
    assert decision.eligible is False
    assert decision.policy_id == STANDARD_POLICY


def test_a_promotional_override_still_outranks_the_standard_policy() -> None:
    """Higher-precedence applicable policies are unaffected by the region fix."""
    decision = check_return_eligibility("ORD-1006", "ITEM-103", "CUST-004", now=FIXED_NOW)
    assert decision.policy_id == "HOLIDAY_EXTENDED_RETURN"
    assert decision.eligible is True


# --- Test D: an explicit region query for another region ----------------


def test_an_au_customer_asking_about_the_uk_gets_the_uk_answer(
    au_session: SessionState,
) -> None:
    outcome = tool_registry.invoke(
        "search_policy", {"query": UK_QUESTION}, au_session, now=FIXED_NOW
    )

    assert outcome.args_used["country"] == "GB"
    result = outcome.payload
    assert result.region == "GB"
    assert result.resolved.policy_id == STANDARD_POLICY
    assert result.resolved.return_window_days == 30
    assert AU_POLICY not in [match.policy.policy_id for match in result.matches]
    # The customer's own region did not leak into the answer.
    assert result.resolved.region is None


def test_asking_about_the_uk_does_not_change_the_customers_own_eligibility(
    au_session: SessionState,
) -> None:
    """An informational question is not a change of region."""
    tool_registry.invoke("search_policy", {"query": UK_QUESTION}, au_session, now=FIXED_NOW)

    assert au_session.customer_region == BRUCE_REGION
    decision = check_return_eligibility(BRUCE_ORDER, BRUCE_BOOK, BRUCE, now=FIXED_NOW)
    assert decision.policy_id == AU_POLICY
    assert decision.days_remaining == 11


# --- Test E: escalation state -------------------------------------------


def test_a_normal_policy_lookup_leaves_escalation_alone(au_session: SessionState) -> None:
    outcome = tool_registry.invoke(
        "search_policy", {"query": AU_QUESTION}, au_session, now=FIXED_NOW
    )
    tool_registry.apply_to_state("search_policy", {"query": AU_QUESTION}, outcome, au_session)
    assert au_session.escalated is False


def test_a_failed_policy_lookup_does_not_escalate(
    au_session: SessionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An outage is something to say, not a handoff the agent performs itself."""
    break_policy_graph(monkeypatch)

    outcome = tool_registry.invoke(
        "search_policy", {"query": AU_QUESTION}, au_session, now=FIXED_NOW
    )
    assert outcome.status is ToolStatus.ERROR
    assert au_session.escalated is False


def test_offering_escalation_is_not_escalation(make_agent, au_session: SessionState) -> None:
    """Saying "I can pass you to a colleague" does not take the agent out of service."""
    agent, _ = make_agent(
        tool_call("search_policy", {"query": AU_QUESTION}),
        text("Australian orders have 45 days. I can pass you to a colleague if you'd like."),
    )
    agent.respond(au_session, AU_QUESTION)
    assert au_session.escalated is False


def test_a_complex_question_on_sonnet_does_not_escalate(
    make_agent, au_session: SessionState
) -> None:
    """Routing decides which model answers. It never decides escalation state."""
    agent, _ = make_agent(
        tool_call("search_policy", {"query": "can I return both of these"}),
        text("Here is what the policy says."),
    )
    agent.respond(au_session, "I'm not sure — is the policy the same for both of my orders?")

    assert au_session.model_turns[-1].model_tier == "sonnet"
    assert au_session.escalated is False


def test_escalation_is_set_only_by_a_created_case(au_session: SessionState) -> None:
    outcome = tool_registry.invoke(
        "escalate_to_human", {"reason": "customer asked for a person"}, au_session
    )
    assert outcome.payload.case_id
    tool_registry.apply_to_state("escalate_to_human", {}, outcome, au_session)
    assert au_session.escalated is True


def test_a_new_conversation_starts_unescalated() -> None:
    assert fresh_session().escalated is False


def test_reset_clears_escalation(au_session: SessionState) -> None:
    au_session.escalated = True
    assert fresh_session().escalated is False


# --- The graph is still the only source ---------------------------------


def test_the_shared_selection_still_requires_neo4j(monkeypatch: pytest.MonkeyPatch) -> None:
    """One mechanism, and it is still a mechanism that reads the graph or fails."""
    break_policy_graph(monkeypatch)

    with pytest.raises(PolicyGraphUnavailableError):
        search_policy(AU_QUESTION, product_type="PhysicalBook", country="AU")
    with pytest.raises(PolicyGraphUnavailableError):
        check_return_eligibility(BRUCE_ORDER, BRUCE_BOOK, BRUCE, now=FIXED_NOW)
