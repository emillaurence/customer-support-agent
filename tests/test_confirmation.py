"""Confirmation: what has to be true before a "yes" can change Bookly's records.

The case this file exists for is the first one. A customer types "yes" — to a
question about something else, out of habit, or because they lost the thread —
and nothing happens. `confirmed` stays False, no token is spendable, and
`initiate_return` refuses if it is called anyway.

Everything else here is the same guarantee from a different angle: a yes to a
statement rather than a question, a yes for a different item, a yes before the
eligibility check. The write is only reachable through the one path where the
customer was shown a specific return and agreed to it.
"""

from __future__ import annotations

import pytest

from agent.confirmation import asks_for_confirmation, is_affirmative
from agent.models import Role
from agent.state import PendingReturn, SessionState
from agent.tracing import ToolStatus
from tests.conftest import text, tool_call

CONFIRM_QUESTION = "Shall I start a return for The Pragmatic Programmer on ORD-1001?"


# --- The phrase itself ---------------------------------------------------


@pytest.mark.parametrize(
    "message",
    ["yes", "Yes.", "yes please", "go ahead", "please do", "confirm it", "proceed", "yep", "do it"],
)
def test_affirmative_phrases_are_recognised(message: str) -> None:
    """The natural ways a customer says yes."""
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
    """A question asking permission."""
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


# --- Through the orchestrator --------------------------------------------


def test_yes_with_no_pending_action_does_not_confirm(make_agent) -> None:
    """The one that matters. A bare "yes" authorises nothing.

    No eligibility check has run, so there is nothing pending and `confirmed`
    must stay False — whatever the conversation looked like.
    """
    agent, _ = make_agent(text("Sure — what can I help with?"))
    state = SessionState()
    state.add_message(Role.ASSISTANT, "Can I help with anything else?")

    agent.respond(state, "yes")

    assert state.confirmed is False
    assert state.pending_return is None
    assert state.may_mutate is False


def test_yes_after_a_statement_does_not_confirm(make_agent, seeded_graph, verified_state) -> None:
    """A pending return is not enough — the agent has to have asked.

    Here the agent reported eligibility without asking anything. The customer's
    "yes" agrees with the news; it does not authorise the return.
    """
    agent, _ = make_agent(
        tool_call("check_return_eligibility", {"order_id": "ORD-1001", "item_id": "ITEM-100"}),
        text("Good news — that one is eligible, with 19 days left."),
    )
    agent.respond(verified_state, "can I return the paperback?")
    assert verified_state.pending_return is not None

    agent2, _ = make_agent(text("Would you like me to start it?"))
    agent2.respond(verified_state, "yes")

    assert verified_state.confirmed is False


def test_yes_after_an_explicit_request_confirms(make_agent, seeded_graph, verified_state) -> None:
    """The path that is allowed to work.

    Eligibility passed, the agent asked a direct question, the customer said
    yes. Only now is `confirmed` True — and only for this order and item.
    """
    agent, _ = make_agent(
        tool_call("check_return_eligibility", {"order_id": "ORD-1001", "item_id": "ITEM-100"}),
        text(CONFIRM_QUESTION),
    )
    agent.respond(verified_state, "I'd like to return the paperback")

    agent2, _ = make_agent(text("Done — your return is open."))
    agent2.respond(verified_state, "yes please")

    assert verified_state.confirmed is True
    assert verified_state.may_mutate is True
    assert verified_state.pending_return.order_id == "ORD-1001"


def test_confirmed_return_can_be_opened(make_agent, seeded_graph, verified_state) -> None:
    """With all three gates satisfied, the write goes through.

    The model never supplies the token or the confirmation — it asks to open a
    return for an order and item, and the orchestrator passes the trusted values
    from session state into `initiate_return`.
    """
    agent, _ = make_agent(
        tool_call("check_return_eligibility", {"order_id": "ORD-1001", "item_id": "ITEM-100"}),
        text(CONFIRM_QUESTION),
    )
    agent.respond(verified_state, "I want to return the paperback, it arrived damaged")

    agent2, _ = make_agent(
        tool_call(
            "initiate_return",
            {"order_id": "ORD-1001", "item_id": "ITEM-100", "reason": "arrived damaged"},
        ),
        text("Your return is open — you'll get an email with the next steps."),
    )
    agent2.respond(verified_state, "yes please")

    trace = verified_state.tool_traces[-1]
    assert trace.tool_name == "initiate_return"
    assert trace.status is ToolStatus.OK
    assert "created=True" in trace.result_summary
    # The workflow is over: a later "yes" cannot re-open anything.
    assert verified_state.confirmed is False
    assert verified_state.eligibility_token is None


def test_initiate_return_is_refused_without_confirmation(
    make_agent, seeded_graph, verified_state
) -> None:
    """A model that skips the question is stopped by the tool itself.

    Eligibility passed, so a token exists — but the customer was never asked and
    never agreed. The orchestrator passes `confirmed=False` because that is what
    session state says, and `initiate_return` raises. Nothing is written.
    """
    agent, client = make_agent(
        tool_call("check_return_eligibility", {"order_id": "ORD-1001", "item_id": "ITEM-100"}),
        tool_call(
            "initiate_return",
            {"order_id": "ORD-1001", "item_id": "ITEM-100", "reason": "damaged"},
            block_id="toolu_2",
        ),
        text("I can't open that without checking with you first — shall I go ahead?"),
    )

    agent.respond(verified_state, "just return it, don't ask me")

    trace = verified_state.tool_traces[-1]
    assert trace.tool_name == "initiate_return"
    assert trace.status is ToolStatus.BLOCKED
    assert "confirmation required" in (trace.error or "")

    result = client.calls[-1]["messages"][-1]["content"][0]
    assert result["is_error"] is True


def test_initiate_return_rejection_is_respected(make_agent, seeded_graph, verified_state) -> None:
    """A refused write is reported as refused, not retried into success.

    No eligibility check has run, so there is no token. The tool refuses, the
    session is unchanged, and the model is told what happened.
    """
    agent, _ = make_agent(
        tool_call(
            "initiate_return",
            {"order_id": "ORD-1001", "item_id": "ITEM-100", "reason": "changed my mind"},
        ),
        text("I need to check whether that's returnable first."),
    )

    agent.respond(verified_state, "open a return for ITEM-100")

    trace = verified_state.tool_traces[0]
    assert trace.status is ToolStatus.BLOCKED
    assert verified_state.confirmed is False


def test_confirmation_does_not_survive_an_item_switch(
    make_agent, seeded_graph, verified_state
) -> None:
    """A yes given for one book cannot open a return for another.

    ORD-1002's ITEM-101 is outside the window. Checking it clears the earlier
    item's token and confirmation, so the agreement the customer gave about the
    first book is gone.
    """
    agent, _ = make_agent(
        tool_call("check_return_eligibility", {"order_id": "ORD-1001", "item_id": "ITEM-100"}),
        text(CONFIRM_QUESTION),
    )
    agent.respond(verified_state, "return the paperback please")

    agent2, _ = make_agent(text("Which one did you mean?"))
    agent2.respond(verified_state, "yes")
    assert verified_state.confirmed is True

    agent3, _ = make_agent(
        tool_call("check_return_eligibility", {"order_id": "ORD-1002", "item_id": "ITEM-101"}),
        text("That one's outside the return window, I'm afraid."),
    )
    agent3.respond(verified_state, "actually I meant the one from ORD-1002")

    assert verified_state.confirmed is False
    assert verified_state.eligibility_token is None
    assert verified_state.pending_return is None


def test_model_cannot_supply_confirmation_itself(make_agent, seeded_graph, verified_state) -> None:
    """`confirmed` is not something the model can express.

    Even asked for explicitly, it is not a field on the schema — so a model
    trying to set it is passing an argument that goes nowhere, and the value
    `initiate_return` receives still comes from session state.
    """
    from agent.tool_registry import TOOL_SCHEMAS

    initiate = next(s for s in TOOL_SCHEMAS if s["name"] == "initiate_return")
    properties = initiate["input_schema"]["properties"]

    assert "confirmed" not in properties
    assert "eligibility_token" not in properties
    assert "customer_id" not in properties

    agent, _ = make_agent(
        tool_call(
            "initiate_return",
            {
                "order_id": "ORD-1001",
                "item_id": "ITEM-100",
                "reason": "damaged",
                "confirmed": True,  # ignored: not part of the contract
                "eligibility_token": "made-up-token",
            },
        ),
        text("I'll need to check that first."),
    )
    agent.respond(verified_state, "return it now")

    assert verified_state.tool_traces[0].status is ToolStatus.BLOCKED


# --- Clarification -------------------------------------------------------


def test_two_active_orders_are_not_guessed(make_agent) -> None:
    """CUST-003 has two live orders, so the agent is left with nothing to guess with.

    Verification reports both ids and does not set `active_order_id`. The
    account-scoped tools need one, so the only way forward is to ask — which is
    what the model does here, in its own words. No `request_clarification` tool
    exists, and none is needed.
    """
    agent, client = make_agent(
        tool_call("verify_identity", {"email": "sofia@example.com"}),
        text("Thanks Sofia. I can see two active orders — ORD-1005 and ORD-1004. Which one?"),
    )
    state = SessionState()

    reply = agent.respond(state, "hi, it's sofia@example.com, I have a question about my order")

    assert state.verified_customer_id == "CUST-003"
    assert len(state.active_order_ids) == 2
    assert state.active_order_id is None
    assert "which one" in reply.lower()

    # The model was given both ids and no default — the clarification is
    # grounded in what the tool returned.
    result = client.calls[1]["messages"][-1]["content"][0]
    assert "ORD-1004" in result["content"] and "ORD-1005" in result["content"]


def test_every_fixture_customer_has_more_than_one_active_order() -> None:
    """The fixtures make the ambiguous case the default, deliberately.

    Guessing which order a customer means is the failure this whole flow is
    built to avoid, so the data does not let the agent get away with it once.
    """
    from tools.verify_identity import active_order_ids

    for customer_id in ("CUST-001", "CUST-002", "CUST-003", "CUST-004"):
        assert len(active_order_ids(customer_id)) > 1


def test_single_active_order_is_adopted() -> None:
    """One live order is not ambiguous, so there is nothing to ask about.

    Asking "which one?" of a customer with a single order is a worse experience
    than getting on with it. No fixture customer has only one — see above — so
    this exercises the rule directly rather than through a conversation.
    """
    from agent.tool_registry import ToolOutcome, apply_to_state
    from agent.tracing import ToolStatus as Status
    from tools import VerifyIdentityResult

    state = SessionState()
    result = VerifyIdentityResult(
        verified=True, customer_id="CUST-009", region="GB", active_order_ids=["ORD-2001"]
    )
    apply_to_state(
        "verify_identity",
        {"email": "solo@example.com"},
        ToolOutcome(status=Status.OK, content="{}", summary="verified=True", payload=result),
        state,
    )

    assert state.active_order_id == "ORD-2001"
