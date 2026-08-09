"""The things the agent must not do.

Every guard in `initiate_return` is checked here against a real token from a real
eligible decision, because a guard that only holds for obviously-wrong input is
not a guard. The `data_dir` fixture points the tools at a temporary copy of
`data/`, so these exercise the actual write without leaving an RMA behind.

Five properties, and each has more than one thing holding it up:

* **Ownership** — enforced by `lookup_order`, re-checked by the write.
* **The eligibility token** — issued only on the eligible path, bound to one
  customer, order, and item, and never shown to the model.
* **Confirmation** — decided in Python from the conversation, and required again
  as an argument by the tool.
* **Idempotency** — a second request reports the existing RMA, never a second one.
* **Escalation** — set only by a case that was actually created.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from agent.agent import asks_for_confirmation, is_affirmative
from agent.state import PendingReturn, SessionState, ToolStatus
from agent.tools import (
    TOOL_SCHEMAS,
    ReturnBlockedError,
    _create_eligibility_token,
    _GRANTS,
    _load_returns,
    apply_tool_result,
    check_return_eligibility,
    initiate_return,
    invoke_tool,
)
from tests.conftest import (
    CONFIRM_QUESTION,
    EXPIRED_ITEM,
    EXPIRED_ORDER,
    HERO_CUSTOMER,
    IN_WINDOW_ITEM,
    IN_WINDOW_ORDER,
    returns_in,
    text,
    tool_call,
)

pytestmark = pytest.mark.usefixtures("seeded_graph")

# ORD-1001 / ITEM-100 / CUST-001 — a physical book at day 11, the clean eligible
# case. Every guard test starts from a genuinely valid token, so the guard is the
# only thing under test.
ELIGIBLE = (IN_WINDOW_ORDER, IN_WINDOW_ITEM, HERO_CUSTOMER)

REASON = "Changed my mind."


@pytest.fixture
def token(now: datetime) -> str:
    """A real token from a real eligible decision."""
    decision = check_return_eligibility(*ELIGIBLE, now=now)
    assert decision.eligible and decision.eligibility_token
    return decision.eligibility_token


def returns_on_disk(data_dir: Path) -> str:
    return (data_dir / "returns.json").read_text()


def write(**overrides):
    """Call `initiate_return` for the eligible case, with fields overridden."""
    order_id, item_id, customer_id = ELIGIBLE
    return initiate_return(
        **{
            "order_id": order_id,
            "item_id": item_id,
            "customer_id": customer_id,
            "reason": REASON,
            "confirmed": True,
            **overrides,
        }
    )


# =========================================================================
# Confirmation
#
# The loop's half of this — reading a "yes" against what was actually pending —
# is in test_agent.py. What is here is which phrases count, and that the tool
# refuses regardless of what the session believes.
# =========================================================================


@pytest.mark.parametrize(
    "message",
    ["yes", "Yes.", "yes please", "go ahead", "please do", "confirm it", "proceed", "yep", "do it"],
)
def test_affirmative_phrases_are_recognised(message: str) -> None:
    assert is_affirmative(message)


@pytest.mark.parametrize(
    "message",
    [
        "no",
        "not yet",
        "yes, but how long do I have?",  # carries a question
        "yes but not that one",  # negated
        "hold on",
        "ok wait",
        "what's the return window?",
        "",
    ],
)
def test_non_affirmatives_are_rejected(message: str) -> None:
    """A qualified, negated, or questioning reply needs another turn, not a write."""
    assert not is_affirmative(message)


@pytest.mark.parametrize(
    "message",
    [
        CONFIRM_QUESTION,
        "Would you like me to open a return for that?",
        "Do you want me to go ahead?",
        "Can I start the return for you?",
    ],
)
def test_confirmation_requests_are_recognised(message: str) -> None:
    assert asks_for_confirmation(message)


@pytest.mark.parametrize(
    "message",
    [
        "That book is eligible for a return — you have 19 days left.",
        "Your order arrived on the 28th of July.",
        "Which of your two orders did you mean?",  # a question, but not this one
    ],
)
def test_statements_are_not_confirmation_requests(message: str) -> None:
    """Agreeing with a fact is not authorising an action."""
    assert not asks_for_confirmation(message)


def test_return_without_explicit_confirmation_is_blocked(token: str, data_dir: Path) -> None:
    """A valid token is not enough — the customer must have said yes."""
    before = returns_on_disk(data_dir)

    with pytest.raises(ReturnBlockedError, match="confirmation required"):
        write(eligibility_token=token, confirmed=False)

    assert returns_on_disk(data_dir) == before, "a blocked return must not write"


def test_confirmation_is_not_taken_from_session_state(token: str) -> None:
    """A pending return marked confirmed on the session does not by itself
    authorise a write.

    Guards against the confirmation gate drifting back into `SessionState`: the
    value has to arrive as an argument to the tool.
    """
    state = SessionState(
        verified_customer_id=HERO_CUSTOMER,
        pending_returns=[
            PendingReturn(
                customer_id=HERO_CUSTOMER,
                order_id=IN_WINDOW_ORDER,
                item_id=IN_WINDOW_ITEM,
                eligibility_token=token,
                confirmed=True,
            )
        ],
    )
    assert state.may_mutate is True  # the loop would allow it...

    with pytest.raises(ReturnBlockedError, match="confirmation required"):
        # ...and the tool still refuses, because it was told confirmed=False.
        write(eligibility_token=token, confirmed=False)


def test_session_may_mutate_needs_all_three_gates(token: str) -> None:
    """The loop's own check, for completeness. Not the safety boundary."""
    assert SessionState().may_mutate is False
    assert SessionState(verified_customer_id=HERO_CUSTOMER).may_mutate is False

    unconfirmed = SessionState(
        verified_customer_id=HERO_CUSTOMER,
        pending_returns=[
            PendingReturn(
                customer_id=HERO_CUSTOMER,
                order_id=IN_WINDOW_ORDER,
                item_id=IN_WINDOW_ITEM,
                eligibility_token=token,
            )
        ],
    )
    assert unconfirmed.may_mutate is False


def test_the_model_cannot_express_confirmation_or_a_token() -> None:
    """None of the three trusted arguments is a schema field, so a model trying
    to set one is passing something that goes nowhere."""
    initiate = next(s for s in TOOL_SCHEMAS if s["name"] == "initiate_return")
    properties = initiate["input_schema"]["properties"]

    assert "confirmed" not in properties
    assert "eligibility_token" not in properties
    assert "customer_id" not in properties


def test_model_supplied_confirmation_is_ignored(make_agent, verified_state) -> None:
    """Asked for explicitly, it still does not reach the tool: the value
    `initiate_return` receives comes from session state, which says False."""
    agent, _ = make_agent(
        tool_call(
            "initiate_return",
            {
                "order_id": IN_WINDOW_ORDER,
                "item_id": IN_WINDOW_ITEM,
                "reason": "damaged",
                "confirmed": True,  # ignored: not part of the contract
                "eligibility_token": "made-up-token",
            },
        ),
        text("I'll need to check that first."),
    )
    agent.respond(verified_state, "return it now")

    assert verified_state.tool_traces[0].status is ToolStatus.BLOCKED


def test_initiate_return_is_refused_without_confirmation(make_agent, verified_state) -> None:
    """A model that skips the question is stopped by the tool itself.

    Eligibility passed, so a token exists — but the customer was never asked. The
    loop passes `confirmed=False`, and `initiate_return` raises.
    """
    agent, client = make_agent(
        tool_call("check_return_eligibility", {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM}),
        tool_call(
            "initiate_return",
            {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM, "reason": "damaged"},
            block_id="toolu_2",
        ),
        text("I can't open that without checking with you first — shall I go ahead?"),
    )

    agent.respond(verified_state, "just return it, don't ask me")

    trace = verified_state.tool_traces[-1]
    assert trace.tool_name == "initiate_return"
    assert trace.status is ToolStatus.BLOCKED
    assert "confirmation required" in (trace.error or "")
    assert client.calls[-1]["messages"][-1]["content"][0]["is_error"] is True


def test_a_refused_write_is_reported_as_refused(make_agent, verified_state) -> None:
    """No eligibility check has run, so there is no token. The tool refuses, the
    session is unchanged, and the model is told what happened."""
    agent, _ = make_agent(
        tool_call(
            "initiate_return",
            {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM, "reason": "changed my mind"},
        ),
        text("I need to check whether that's returnable first."),
    )

    agent.respond(verified_state, "open a return for ITEM-100")

    assert verified_state.tool_traces[0].status is ToolStatus.BLOCKED
    assert verified_state.pending_returns == []


# =========================================================================
# Eligibility tokens
# =========================================================================


def test_missing_eligibility_token_is_blocked(data_dir: Path) -> None:
    before = returns_on_disk(data_dir)

    with pytest.raises(ReturnBlockedError, match="no eligibility_token"):
        write(eligibility_token="")

    assert returns_on_disk(data_dir) == before


@pytest.mark.parametrize(
    "invented",
    [
        "eligibility-token",
        "d0f4f1e0-0000-4000-8000-000000000000",  # a well-formed uuid4 nobody issued
        "ORD-1001-ITEM-100-OK",
        "null",
    ],
)
def test_invalid_eligibility_token_is_blocked(invented: str, data_dir: Path) -> None:
    """An invented token looks exactly like an unknown one: nothing was issued."""
    before = returns_on_disk(data_dir)

    with pytest.raises(ReturnBlockedError, match="not a token this server issued"):
        write(eligibility_token=invented)

    assert returns_on_disk(data_dir) == before


def test_token_for_another_order_is_blocked(token: str, data_dir: Path) -> None:
    """Both orders are CUST-001's, so ownership passes and the token binding is
    the only thing left to stop it."""
    before = returns_on_disk(data_dir)

    with pytest.raises(ReturnBlockedError, match="issued for"):
        write(order_id=EXPIRED_ORDER, item_id=EXPIRED_ITEM, eligibility_token=token)

    assert returns_on_disk(data_dir) == before


def test_token_for_another_item_is_blocked(now: datetime, data_dir: Path) -> None:
    """ORD-1003 holds two items. A token for the paperback is not a token for the ebook."""
    decision = check_return_eligibility("ORD-1003", "ITEM-102", "CUST-002", now=now)
    assert decision.eligibility_token
    before = returns_on_disk(data_dir)

    with pytest.raises(ReturnBlockedError, match="issued for"):
        write(
            order_id="ORD-1003",
            item_id="ITEM-201",
            customer_id="CUST-002",
            eligibility_token=decision.eligibility_token,
        )

    assert returns_on_disk(data_dir) == before


def test_token_for_another_customer_is_blocked(now: datetime, data_dir: Path) -> None:
    """CUST-002's token, spent by CUST-001, on CUST-002's order."""
    decision = check_return_eligibility("ORD-1003", "ITEM-102", "CUST-002", now=now)
    assert decision.eligibility_token
    before = returns_on_disk(data_dir)

    with pytest.raises(ReturnBlockedError):
        write(
            order_id="ORD-1003",
            item_id="ITEM-102",
            customer_id=HERO_CUSTOMER,
            eligibility_token=decision.eligibility_token,
        )

    assert returns_on_disk(data_dir) == before


@pytest.mark.parametrize(
    ("order_id", "item_id", "customer_id"),
    [
        (EXPIRED_ORDER, EXPIRED_ITEM, HERO_CUSTOMER),  # outside the window
        ("ORD-1004", "ITEM-200", "CUST-003"),  # ebook
        ("ORD-1005", "ITEM-101", "CUST-003"),  # in transit
        ("ORD-1007", "ITEM-100", "CUST-002"),  # already returned
        ("ORD-1008", "ITEM-101", HERO_CUSTOMER),  # someone else's order
    ],
)
def test_a_token_is_never_issued_by_a_refusal(
    order_id: str, item_id: str, customer_id: str, now: datetime
) -> None:
    """Every refusal path, checked for the same thing: no token escapes.

    The store is the proof — an ineligible check leaves nothing behind, so there
    is nothing for a later `initiate_return` to spend.
    """
    decision = check_return_eligibility(order_id, item_id, customer_id, now=now)
    assert decision.eligible is False
    assert decision.eligibility_token is None
    assert _GRANTS == {}


def test_a_token_records_the_decision_it_came_from(now: datetime) -> None:
    """The token means nothing by itself — the server holds what it permits."""
    decision = check_return_eligibility("ORD-1003", "ITEM-102", "CUST-002", now=now)
    assert decision.eligibility_token

    grant = _GRANTS[decision.eligibility_token]
    assert (grant.customer_id, grant.order_id, grant.item_id) == (
        "CUST-002", "ORD-1003", "ITEM-102"
    )
    assert grant.policy_id == "AU_BOOKLY_EXTENDED_RETURN"


def test_each_decision_mints_a_distinct_undeducible_token(now: datetime) -> None:
    """A uuid4, not a recipe a model could reproduce from the arguments."""
    first = check_return_eligibility(*ELIGIBLE, now=now).eligibility_token
    second = check_return_eligibility("ORD-1003", "ITEM-102", "CUST-002", now=now).eligibility_token

    assert first and second and first != second
    for identifier in (IN_WINDOW_ORDER, IN_WINDOW_ITEM, HERO_CUSTOMER, "STANDARD_30_DAY"):
        assert identifier.lower() not in first.lower()


def test_the_traced_call_never_records_the_token(make_agent, verified_state) -> None:
    """The token is injected into the real call but never written to a trace, and
    never sent to the model: a credential on a screen, or in a transcript the
    model could quote back, is a credential leaked."""
    agent, _ = make_agent(
        tool_call("check_return_eligibility", {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM}),
        text(CONFIRM_QUESTION),
    )
    agent.respond(verified_state, "I'd like to return the paperback")
    assert len(verified_state.pending_returns) == 1
    token = verified_state.pending_returns[0].eligibility_token
    assert token

    agent2, _ = make_agent(
        tool_call(
            "initiate_return",
            {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM, "reason": "damaged"},
        ),
        text("Your return is open."),
    )
    agent2.respond(verified_state, "yes please")

    traced = "".join(trace.model_dump_json() for trace in verified_state.tool_traces)
    assert token not in traced
    assert verified_state.tool_traces[-1].tool_args["eligibility_token"] == "***"
    assert token not in str(verified_state.transcript)


# =========================================================================
# Ownership
# =========================================================================


def test_wrong_order_ownership_is_blocked(token: str, data_dir: Path) -> None:
    """ORD-1008 is CUST-004's, and CUST-001 has a valid token of their own."""
    before = returns_on_disk(data_dir)

    with pytest.raises(ReturnBlockedError, match="does not belong to"):
        write(order_id="ORD-1008", item_id="ITEM-101", eligibility_token=token)

    assert returns_on_disk(data_dir) == before


def test_unknown_customer_is_blocked(token: str, data_dir: Path) -> None:
    before = returns_on_disk(data_dir)

    with pytest.raises(ReturnBlockedError, match="unknown customer"):
        write(customer_id="CUST-999", eligibility_token=token)

    assert returns_on_disk(data_dir) == before


def test_item_not_on_the_order_is_blocked(token: str, data_dir: Path) -> None:
    before = returns_on_disk(data_dir)

    with pytest.raises(ReturnBlockedError, match="is not on order"):
        write(item_id="ITEM-103", eligibility_token=token)

    assert returns_on_disk(data_dir) == before


# =========================================================================
# Idempotency
# =========================================================================


def test_repeating_the_request_returns_the_existing_rma(token: str) -> None:
    """A second call is not a second return."""
    first = write(eligibility_token=token)
    count_after_first = len(_load_returns())

    second = write(eligibility_token=token)

    assert first.created is True
    assert second.created is False
    assert second.return_record.return_id == first.return_record.return_id
    assert len(_load_returns()) == count_after_first, "no second RMA"


def test_preexisting_rma_is_returned_not_duplicated() -> None:
    """RET-5001 is already open against ORD-1007 / ITEM-100 in the fixtures.

    Eligibility refuses this, so a token has to be minted for it directly — which
    is the point: the write tool holds the line even when the caller got a token
    some other way.
    """
    token = _create_eligibility_token("CUST-002", "ORD-1007", "ITEM-100", "STANDARD_30_DAY")
    before = len(_load_returns())

    result = write(
        order_id="ORD-1007",
        item_id="ITEM-100",
        customer_id="CUST-002",
        eligibility_token=token,
    )

    assert result.created is False
    assert result.return_record.return_id == "RET-5001"
    assert len(_load_returns()) == before


def test_ineligible_item_cannot_be_returned_end_to_end(now: datetime, data_dir: Path) -> None:
    """The whole path, as the agent would walk it: an ebook is refused by
    eligibility, which issues nothing, so there is no token to pass on."""
    before = returns_on_disk(data_dir)
    decision = check_return_eligibility("ORD-1004", "ITEM-200", "CUST-003", now=now)
    assert decision.eligible is False
    assert decision.eligibility_token is None

    with pytest.raises(ReturnBlockedError):
        write(
            order_id="ORD-1004",
            item_id="ITEM-200",
            customer_id="CUST-003",
            eligibility_token=decision.eligibility_token or "",
        )

    assert returns_on_disk(data_dir) == before


# =========================================================================
# Escalation
#
# An offer is a sentence, and a sentence does not take the agent out of service.
# Only a case that was actually created does.
# =========================================================================


def test_escalation_marks_the_session(make_agent) -> None:
    agent, _ = make_agent(
        tool_call("escalate_to_human", {"reason": "customer asked for a person"}),
        text("I'm passing you to a colleague."),
    )
    state = SessionState()

    agent.respond(state, "I want to speak to a human")

    assert state.escalated is True


def test_escalation_is_set_only_by_a_created_case(verified_state) -> None:
    outcome = invoke_tool(
        "escalate_to_human", {"reason": "customer asked for a person"}, verified_state
    )
    assert outcome.payload.case_id
    apply_tool_result("escalate_to_human", {}, outcome, verified_state)
    assert verified_state.escalated is True


def test_a_normal_policy_lookup_leaves_escalation_alone(verified_state) -> None:
    args = {"query": "what is the return policy for Australian customers?"}
    outcome = invoke_tool("search_policy", args, verified_state)
    apply_tool_result("search_policy", args, outcome, verified_state)
    assert verified_state.escalated is False


def test_a_failed_policy_lookup_does_not_escalate(verified_state, monkeypatch) -> None:
    """An outage is something to say, not a handoff the agent performs itself."""
    from tests.conftest import break_policy_graph

    break_policy_graph(monkeypatch)

    outcome = invoke_tool("search_policy", {"query": "what is the return policy"}, verified_state)
    assert outcome.status is ToolStatus.ERROR
    assert verified_state.escalated is False


def test_an_informational_lookup_does_not_escalate(make_agent, verified_state) -> None:
    """A policy question is a read. It does not take the agent out of service —
    and saying "I can pass you to a colleague" is an offer, not a handoff."""
    agent, _ = make_agent(
        tool_call("search_policy", {"query": "What is the return policy for Australian customers?"}),
        text("Australian orders have 45 days. I can pass you to a colleague if you'd like."),
    )
    agent.respond(verified_state, "What is the return policy for Australian customers?")

    assert verified_state.escalated is False
    assert verified_state.model_turns[-1].routing_reason != "conversation is escalated"
    # The region the question named reached the tool, and the trace shows it.
    policy_trace = next(
        t for t in reversed(verified_state.tool_traces) if t.tool_name == "search_policy"
    )
    assert policy_trace.tool_args["country"] == "AU"


def test_a_complex_question_on_sonnet_does_not_escalate(make_agent, verified_state) -> None:
    """Routing decides which model answers. It never decides escalation state."""
    agent, _ = make_agent(
        tool_call("search_policy", {"query": "can I return both of these"}),
        text("Here is what the policy says."),
    )
    agent.respond(verified_state, "I'm not sure — is the policy the same for both of my orders?")

    assert verified_state.model_turns[-1].model_tier == "sonnet"
    assert verified_state.escalated is False


def test_a_new_conversation_starts_unescalated() -> None:
    assert SessionState().escalated is False


def test_initiate_return_is_blocked_deterministically_once_escalated(
    token: str, data_dir: Path
) -> None:
    """Once handed to a human, no return may open — even a fully confirmed,
    eligible one already sitting on the session. The model is told not to keep
    acting after an escalation, but this holds regardless of what it does: the
    guard runs before dispatch, so the mutation implementation is never called."""
    order_id, item_id, customer_id = ELIGIBLE
    state = SessionState(
        verified_customer_id=customer_id,
        escalated=True,
        pending_returns=[
            PendingReturn(
                customer_id=customer_id,
                order_id=order_id,
                item_id=item_id,
                eligibility_token=token,
                asked=True,
                confirmed=True,
            )
        ],
    )
    before = returns_on_disk(data_dir)

    outcome = invoke_tool("initiate_return", {"order_id": order_id, "item_id": item_id}, state)

    assert outcome.status is ToolStatus.BLOCKED
    assert "escalated" in (outcome.error or "")
    assert returns_on_disk(data_dir) == before, "a blocked return must not write"
    assert token not in outcome.content, "the token must never leak into a tool result"
    # The pending return itself is untouched — still confirmed, still spendable
    # if escalation were ever lifted, and not silently converted into an RMA.
    assert state.pending_returns[0].confirmed is True
    assert not any(
        r.order_id == order_id and r.item_id == item_id for r in _load_returns()
    ), "no RMA was created for the blocked return"


# =========================================================================
# The outside-the-window ending
#
# Same customer, same tools, different fixture dates — and no special case
# anywhere in the code.
# =========================================================================


def test_outside_window_return_is_refused_with_a_reason(make_agent, hero_verified) -> None:
    """The deterministic check says no, and no token is minted."""
    agent, _ = make_agent(
        tool_call(
            "check_return_eligibility",
            {"order_id": EXPIRED_ORDER, "item_id": EXPIRED_ITEM, "reason": "Didn't get on with it."},
        ),
        text("That one was delivered too long ago to return, I'm afraid."),
    )

    agent.respond(hero_verified, "I want to return this book")

    decision = hero_verified.eligibility
    assert decision is not None
    assert decision.eligible is False
    assert "outside the window" in decision.explanation
    assert hero_verified.pending_returns == []


def test_outside_window_yes_authorises_nothing(make_agent, hero_verified, data_dir) -> None:
    """Two gates, independently: the confirmation check has nothing pending to
    attach a yes to, and `initiate_return` refuses on its own when the model tries
    the write regardless."""
    before = returns_in(data_dir)

    agent, _ = make_agent(
        tool_call(
            "check_return_eligibility",
            {"order_id": EXPIRED_ORDER, "item_id": EXPIRED_ITEM, "reason": "Didn't want it."},
        ),
        text("That's outside the return window. Shall I pass you to a colleague?"),
    )
    agent.respond(hero_verified, "Can I send this back?")

    agent2, _ = make_agent(
        tool_call(
            "initiate_return",
            {"order_id": EXPIRED_ORDER, "item_id": EXPIRED_ITEM, "reason": "Didn't want it."},
        ),
        text("I can't open that return — it's outside the window."),
    )
    agent2.respond(hero_verified, "Yes, go ahead")

    assert hero_verified.may_mutate is False
    write_trace = hero_verified.tool_traces[-1]
    assert write_trace.tool_name == "initiate_return"
    assert write_trace.status is ToolStatus.BLOCKED
    assert returns_in(data_dir) == before


def test_outside_window_can_be_escalated(make_agent, hero_verified) -> None:
    """The dead end has somewhere to go: a person."""
    agent, _ = make_agent(
        tool_call(
            "escalate_to_human", {"reason": "return refused as outside window; customer unhappy"}
        ),
        text("I'm passing you to a colleague who can take another look."),
    )

    agent.respond(hero_verified, "Then I'd like to speak to someone about it")

    assert hero_verified.escalated is True
    assert hero_verified.tool_traces[-1].status is ToolStatus.OK


@pytest.mark.parametrize("phrasing", ["Yes", "Go ahead", "Please do"])
def test_the_same_words_authorise_nothing_out_of_context(
    make_agent, hero_verified, phrasing
) -> None:
    """The affirmatives are not a global approval switch."""
    agent, _ = make_agent(text("Sorry — what would you like me to do?"))

    agent.respond(hero_verified, phrasing)

    assert hero_verified.pending_returns == []
    assert hero_verified.may_mutate is False


def test_agreeing_with_a_statement_is_not_a_confirmation(make_agent, hero_verified) -> None:
    """The agent reported eligibility without asking permission, so the customer's
    "yes please" agrees with a fact and `pending_return.asked` stays False."""
    agent, _ = make_agent(
        tool_call(
            "check_return_eligibility",
            {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM, "reason": "Not for me."},
        ),
        text("Good news — that book is still inside its return window."),
        text("Would you like me to start the return?"),
    )
    agent.respond(hero_verified, "I'd like to return it")
    agent.respond(hero_verified, "yes please")

    assert len(hero_verified.pending_returns) == 1
    assert hero_verified.pending_returns[0].asked is False
    assert hero_verified.may_mutate is False
