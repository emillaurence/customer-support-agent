"""The hero conversation, end to end, and the edge case beside it.

One customer, one session, six turns:

    order status → verify → clarify between two orders → look one up
    → switch to a return → eligibility → confirm → a real RMA

Anthropic is scripted, as everywhere else in the suite, because what is under
test is the *orchestration*: that the session carries what it should between
turns, that the router moves to Sonnet when the customer switches to a return,
that a "yes" is only a confirmation when something is pending, and that the
write at the end is real. The model's wording is not under test and cannot be —
which is also why the language tests below check that several phrasings reach
the same place rather than that one exact script does.

The hero customer is **CUST-001 (Ada)**, who has two active orders: ORD-1001
holds a physical book delivered 11 days before the fixed clock, and ORD-1002
holds one delivered 67 days before it. The first is the hero return; the second
is the outside-the-window case. Both come out of the fixtures — no test here
tells a tool what to decide, and no production code knows which order is which.
"""

from __future__ import annotations

import json

import pytest

from agent.routing import ModelTier
from agent.state import SessionState
from agent.tracing import ToolStatus
from tests.conftest import text, tool_call, tool_calls
from tools.verify_identity import active_order_ids

HERO_EMAIL = "ada@example.com"
HERO_CUSTOMER = "CUST-001"

IN_WINDOW_ORDER, IN_WINDOW_ITEM = "ORD-1001", "ITEM-100"
"""The hero return: a physical book inside the standard window."""

EXPIRED_ORDER, EXPIRED_ITEM = "ORD-1002", "ITEM-101"
"""The alternate demonstration: the same customer, a book long out of window."""

CONFIRM_QUESTION = (
    "That one can be returned. Shall I start a return for The Pragmatic Programmer "
    "on order ORD-1001?"
)
"""The agent's confirmation question — a cue plus a question mark, which is what
`agent.confirmation.asks_for_confirmation` looks for."""


def returns_in(data_dir) -> list[dict]:
    """Every RMA currently on disk, read from the test's temporary data copy."""
    return json.loads((data_dir / "returns.json").read_text())


def tool_names(state: SessionState) -> list[str]:
    """The tools this session has run, oldest first."""
    return [trace.tool_name for trace in state.tool_traces]


def tiers(state: SessionState) -> list[str]:
    """The model tier chosen for each turn, oldest first."""
    return [turn.model_tier for turn in state.model_turns]


# --- The precondition the whole demo rests on ---------------------------


def test_hero_customer_really_has_two_active_orders() -> None:
    """The clarification step is a property of the fixtures, not of the script.

    If someone edits `data/orders.json` and leaves Ada with one live order, the
    hero flow stops demonstrating anything and this says so here rather than
    three tests later.
    """
    assert len(active_order_ids(HERO_CUSTOMER)) >= 2


# --- The hero flow -------------------------------------------------------


@pytest.fixture
def hero_script():
    """Anthropic's side of the whole hero conversation, in order.

    Six turns' worth of responses. The tool calls are the ones a model that read
    the schemas would make; the text is what it would say around them.
    """
    return (
        # Turn 1 — unverified, so the only thing to do is ask.
        text("Happy to check. What's the email address on your Bookly account?"),
        # Turn 2 — verify, then read both live orders so the question can name them.
        tool_call("verify_identity", {"email": HERO_EMAIL}),
        tool_calls(
            ("lookup_order", {"order_id": IN_WINDOW_ORDER}),
            ("lookup_order", {"order_id": EXPIRED_ORDER}),
        ),
        text(
            "Thanks Ada. You've got two orders with us — The Pragmatic Programmer, "
            "delivered on 28 July, and Designing Data-Intensive Applications, delivered "
            "on 2 June. Which one did you mean?"
        ),
        # Turn 3 — the customer picks one by title; the agent opens that order.
        tool_call("lookup_order", {"order_id": IN_WINDOW_ORDER}, block_id="toolu_pick"),
        text("The Pragmatic Programmer was delivered on 28 July and is with you."),
        # Turn 4 — intent changes; eligibility decides, and the agent asks.
        tool_call(
            "check_return_eligibility",
            {
                "order_id": IN_WINDOW_ORDER,
                "item_id": IN_WINDOW_ITEM,
                "reason": "Not what I expected.",
            },
            block_id="toolu_elig",
        ),
        text(CONFIRM_QUESTION),
        # Turn 5 — confirmed, so the write happens.
        tool_call(
            "initiate_return",
            {
                "order_id": IN_WINDOW_ORDER,
                "item_id": IN_WINDOW_ITEM,
                "reason": "Not what I expected.",
            },
            block_id="toolu_write",
        ),
        text("Your return is open. You'll get an email with the next steps."),
    )


def run_hero_flow(agent, state: SessionState) -> None:
    """Drive the five customer turns of the hero conversation."""
    agent.respond(state, "Where's my book?")
    agent.respond(state, HERO_EMAIL)
    agent.respond(state, "The Pragmatic Programmer one")
    agent.respond(state, "Actually, I want to return it.")
    agent.respond(state, "Yes please")


def test_hero_flow_end_to_end(make_agent, seeded_graph, hero_script, data_dir) -> None:
    """The whole journey, in one session, ending in a real RMA."""
    agent, _ = make_agent(*hero_script)
    state = SessionState()

    run_hero_flow(agent, state)

    assert tool_names(state) == [
        "verify_identity",
        "lookup_order",
        "lookup_order",
        "lookup_order",
        "check_return_eligibility",
        "initiate_return",
    ]
    assert all(trace.status is ToolStatus.OK for trace in state.tool_traces)

    created = [row for row in returns_in(data_dir) if row["order_id"] == IN_WINDOW_ORDER]
    assert len(created) == 1
    assert created[0]["item_id"] == IN_WINDOW_ITEM
    assert created[0]["customer_id"] == HERO_CUSTOMER
    assert created[0]["reason"] == "Not what I expected."


def test_two_active_orders_are_not_resolved_by_guessing(
    make_agent, seeded_graph, hero_script
) -> None:
    """After verification and a look at both orders, none of them is 'the' order.

    Reading two orders out is the agent building its clarifying question. If the
    second one silently became the active order, the agent would be answering
    the question it is in the middle of asking.
    """
    agent, _ = make_agent(*hero_script)
    state = SessionState()

    agent.respond(state, "Where's my book?")
    agent.respond(state, HERO_EMAIL)

    assert state.verified_customer_id == HERO_CUSTOMER
    assert len(state.active_order_ids) == 2
    assert state.active_order_id is None


def test_customer_selection_sets_the_active_order(
    make_agent, seeded_graph, hero_script
) -> None:
    """A single lookup after the customer chose is what records the choice."""
    agent, _ = make_agent(*hero_script)
    state = SessionState()

    agent.respond(state, "Where's my book?")
    agent.respond(state, HERO_EMAIL)
    agent.respond(state, "The Pragmatic Programmer one")

    assert state.active_order_id == IN_WINDOW_ORDER


def test_status_to_return_transition_reuses_trusted_state(
    make_agent, seeded_graph, hero_script
) -> None:
    """Switching to a return re-asks for nothing the session already holds.

    Identity, region, and the selected order survive the change of intent, and
    `verify_identity` runs exactly once in the whole conversation.
    """
    agent, _ = make_agent(*hero_script)
    state = SessionState()

    run_hero_flow(agent, state)

    assert tool_names(state).count("verify_identity") == 1
    assert state.verified_customer_id == HERO_CUSTOMER
    assert state.customer_region == "GB"

    eligibility_call = next(
        trace for trace in state.tool_traces if trace.tool_name == "check_return_eligibility"
    )
    assert eligibility_call.tool_args["order_id"] == IN_WINDOW_ORDER


def test_eligible_return_leaves_something_to_confirm(
    make_agent, seeded_graph, hero_script
) -> None:
    """A passing check mints a token and puts a specific return in the balance."""
    agent, _ = make_agent(*hero_script)
    state = SessionState()

    agent.respond(state, "Where's my book?")
    agent.respond(state, HERO_EMAIL)
    agent.respond(state, "The Pragmatic Programmer one")
    agent.respond(state, "Actually, I want to return it.")

    assert state.eligibility is not None and state.eligibility.eligible
    assert state.eligibility_token is not None
    assert state.pending_return is not None
    assert state.pending_return.order_id == IN_WINDOW_ORDER
    assert state.pending_return.item_id == IN_WINDOW_ITEM
    assert state.confirmed is False  # asked, but not yet answered
    assert state.active_item_id == IN_WINDOW_ITEM
    assert state.return_reason == "Not what I expected."


def test_return_workflow_state_is_cleared_after_the_write(
    make_agent, seeded_graph, hero_script
) -> None:
    """Once the RMA exists there is nothing left for a later 'yes' to spend.

    Identity survives — the customer is still the customer — but the token, the
    pending action, and the confirmation do not.
    """
    agent, _ = make_agent(*hero_script)
    state = SessionState()

    run_hero_flow(agent, state)

    assert state.verified_customer_id == HERO_CUSTOMER
    assert state.eligibility_token is None
    assert state.pending_return is None
    assert state.confirmed is False


def test_hero_flow_routing_is_the_generic_router(
    make_agent, seeded_graph, hero_script
) -> None:
    """Status turns stay on Haiku; the return and the confirmation are Sonnet.

    Nothing here forces a tier. These are the decisions `agent.routing` makes
    from the message and the session, and the demo inherits them.
    """
    agent, _ = make_agent(*hero_script)
    state = SessionState()

    run_hero_flow(agent, state)

    assert tiers(state) == [
        ModelTier.HAIKU,  # "Where's my book?"
        ModelTier.HAIKU,  # the email
        ModelTier.HAIKU,  # picking the order
        ModelTier.SONNET,  # "Actually, I want to return it."
        ModelTier.SONNET,  # confirming, with a return pending
    ]
    assert state.model_turns[3].routing_reason == "return or refund intent"
    assert state.model_turns[4].routing_reason == "a return workflow is open"


def test_return_intent_keeps_the_following_turns_on_sonnet(
    make_agent, seeded_graph, hero_verified
) -> None:
    """Naming a book is not a return keyword, but it is still part of a return.

    The customer asks to return something, then answers "which one?" with a
    title. That second turn is where the eligibility check runs, and on its own
    text it looks like a simple lookup — so without the session remembering the
    intent it would drop to the cheaper model mid-workflow.
    """
    agent, _ = make_agent(
        text("Which one — The Pragmatic Programmer, or Designing Data-Intensive Applications?"),
        tool_call(
            "check_return_eligibility",
            {"order_id": EXPIRED_ORDER, "item_id": EXPIRED_ITEM},
        ),
        text("That one's outside the return window, I'm afraid."),
    )

    agent.respond(hero_verified, "I want to return a book")
    agent.respond(hero_verified, "Designing Data-Intensive Applications")

    assert tiers(hero_verified) == [ModelTier.SONNET, ModelTier.SONNET]
    assert hero_verified.tool_traces[-1].model_tier == ModelTier.SONNET


def test_hero_flow_trace_is_readable_and_leaks_nothing(
    make_agent, seeded_graph, hero_script
) -> None:
    """The trace tells the story, without the customer's address or a token."""
    agent, _ = make_agent(*hero_script)
    state = SessionState()

    run_hero_flow(agent, state)

    for trace in state.tool_traces:
        assert trace.model  # which model asked
        assert trace.model_tier in {"haiku", "sonnet"}
        assert trace.latency_ms >= 0
        assert trace.result_summary
        assert HERO_EMAIL not in json.dumps(trace.tool_args)
        assert "eligibility_token" not in trace.result_summary

    verification = state.tool_traces[0]
    assert verification.tool_args["email"] == "a***@example.com"
    assert "verified=True" in verification.result_summary
    assert "created=True" in state.tool_traces[-1].result_summary


def test_repeating_the_hero_return_does_not_create_a_second_rma(
    make_agent, seeded_graph, hero_script, data_dir
) -> None:
    """Idempotency, from the customer's side: asking twice yields one RMA.

    The second pass runs the same script against the same data. Eligibility now
    refuses — a return is already open — so the write is never reached, and the
    file still holds exactly one return for that order.
    """
    agent, _ = make_agent(*hero_script)
    state = SessionState()
    run_hero_flow(agent, state)

    agent2, _ = make_agent(
        tool_call(
            "check_return_eligibility",
            {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM, "reason": "Changed my mind."},
        ),
        text("There's already a return open for that one."),
    )
    second = SessionState(
        verified_customer_id=HERO_CUSTOMER,
        customer_region="GB",
        active_order_ids=[IN_WINDOW_ORDER, EXPIRED_ORDER],
        active_order_id=IN_WINDOW_ORDER,
    )
    agent2.respond(second, "I want to return the Pragmatic Programmer")

    assert second.eligibility is not None and second.eligibility.eligible is False
    assert second.eligibility_token is None
    assert len([r for r in returns_in(data_dir) if r["order_id"] == IN_WINDOW_ORDER]) == 1


def test_a_confirmed_return_is_not_held_up_for_a_reason(
    make_agent, seeded_graph, data_dir
) -> None:
    """A customer who never says why still gets their return.

    `reason` is recorded on the RMA, not checked by any guard, so it must not be
    able to block a confirmed, eligible return. When the model omits it the
    session's own recorded reason is used, and when there is none the record
    simply carries none.
    """
    agent, _ = make_agent(
        tool_call(
            "check_return_eligibility",
            {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM},
        ),
        text(CONFIRM_QUESTION),
        tool_call(
            "initiate_return",
            {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM},
            block_id="toolu_write",
        ),
        text("Your return is open."),
    )
    state = SessionState(
        verified_customer_id=HERO_CUSTOMER,
        customer_region="GB",
        active_order_ids=[IN_WINDOW_ORDER],
        active_order_id=IN_WINDOW_ORDER,
    )

    agent.respond(state, "I'd like to send this one back")
    agent.respond(state, "Go ahead")

    assert state.tool_traces[-1].status is ToolStatus.OK
    created = [r for r in returns_in(data_dir) if r["order_id"] == IN_WINDOW_ORDER]
    assert len(created) == 1
    assert created[0]["reason"] == ""


def test_a_reason_given_earlier_is_not_asked_for_again(
    make_agent, seeded_graph, data_dir
) -> None:
    """The reason the customer gave at the eligibility step reaches the RMA.

    The model omits it on the write. The session already holds it, so it is
    filled in from there rather than lost or re-requested.
    """
    agent, _ = make_agent(
        tool_call(
            "check_return_eligibility",
            {
                "order_id": IN_WINDOW_ORDER,
                "item_id": IN_WINDOW_ITEM,
                "reason": "The spine was cracked.",
            },
        ),
        text(CONFIRM_QUESTION),
        tool_call(
            "initiate_return",
            {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM},
            block_id="toolu_write",
        ),
        text("Your return is open."),
    )
    state = SessionState(
        verified_customer_id=HERO_CUSTOMER,
        customer_region="GB",
        active_order_ids=[IN_WINDOW_ORDER],
        active_order_id=IN_WINDOW_ORDER,
    )

    agent.respond(state, "The spine was cracked, I want to return it")
    agent.respond(state, "Yes")

    created = [r for r in returns_in(data_dir) if r["order_id"] == IN_WINDOW_ORDER]
    assert created[0]["reason"] == "The spine was cracked."


# --- The outside-the-window case ----------------------------------------


@pytest.fixture
def hero_verified() -> SessionState:
    """The hero customer, already verified, with neither order chosen yet."""
    return SessionState(
        verified_customer_id=HERO_CUSTOMER,
        customer_region="GB",
        active_order_ids=[IN_WINDOW_ORDER, EXPIRED_ORDER],
    )


def test_outside_window_return_is_refused_with_a_reason(
    make_agent, seeded_graph, hero_verified
) -> None:
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
    assert hero_verified.eligibility_token is None
    assert hero_verified.pending_return is None


def test_outside_window_yes_authorises_nothing(
    make_agent, seeded_graph, hero_verified, data_dir
) -> None:
    """A customer who says yes anyway gets no return, and the file is untouched.

    Two gates, independently: the confirmation check has nothing pending to
    attach a yes to, and `initiate_return` refuses on its own when the model
    tries the write regardless.
    """
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

    assert hero_verified.confirmed is False
    write = hero_verified.tool_traces[-1]
    assert write.tool_name == "initiate_return"
    assert write.status is ToolStatus.BLOCKED
    assert returns_in(data_dir) == before


def test_outside_window_can_be_escalated(make_agent, seeded_graph, hero_verified) -> None:
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


# --- Language flexibility ------------------------------------------------


@pytest.mark.parametrize(
    "phrasing",
    [
        "Where's my book?",
        "Can you check my delivery?",
        "What's happening with my order?",
        "I haven't received my book yet.",
    ],
)
def test_status_phrasings_all_reach_an_order_lookup(
    make_agent, seeded_graph, phrasing
) -> None:
    """Four ways of asking the same thing, one route through the agent.

    None of them is special-cased anywhere: the router sees a simple read-only
    request and the model reaches for `lookup_order`.
    """
    agent, _ = make_agent(
        tool_call("lookup_order", {"order_id": IN_WINDOW_ORDER}),
        text("It was delivered on 28 July."),
    )
    state = SessionState(
        verified_customer_id=HERO_CUSTOMER,
        customer_region="GB",
        active_order_ids=[IN_WINDOW_ORDER],
        active_order_id=IN_WINDOW_ORDER,
    )

    agent.respond(state, phrasing)

    assert tool_names(state) == ["lookup_order"]
    assert state.model_turns[0].model_tier == ModelTier.HAIKU


@pytest.mark.parametrize(
    "phrasing",
    [
        "Actually, I want to return it.",
        "Can I send this back?",
        "I don't want it anymore.",
        "Can I get a refund?",
    ],
)
def test_return_phrasings_all_reach_the_eligibility_check(
    make_agent, seeded_graph, hero_verified, phrasing
) -> None:
    """Four ways of changing intent, all promoted and all deciding the same way.

    "I don't want it anymore" carries no return keyword, so it reaches Sonnet
    the other way — as a message the agent has to resolve. Which route it takes
    is the router's business; that it does not land on the cheap model is the
    property worth holding.
    """
    agent, _ = make_agent(
        tool_call(
            "check_return_eligibility",
            {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM, "reason": phrasing},
        ),
        text(CONFIRM_QUESTION),
    )

    agent.respond(hero_verified, phrasing)

    assert tool_names(hero_verified) == ["check_return_eligibility"]
    assert hero_verified.pending_return is not None
    assert hero_verified.model_turns[0].model_tier == ModelTier.SONNET


@pytest.mark.parametrize(
    "phrasing", ["Yes", "yes please", "Go ahead", "Please do", "Confirm it", "proceed"]
)
def test_confirmation_phrasings_work_in_a_pending_context(
    make_agent, seeded_graph, hero_verified, phrasing, data_dir
) -> None:
    """Six ways of agreeing, all of which open the return that was asked about."""
    agent, _ = make_agent(
        tool_call(
            "check_return_eligibility",
            {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM, "reason": "Not for me."},
        ),
        text(CONFIRM_QUESTION),
        tool_call(
            "initiate_return",
            {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM, "reason": "Not for me."},
            block_id="toolu_write",
        ),
        text("Your return is open."),
    )
    agent.respond(hero_verified, "I'd like to return it")
    agent.respond(hero_verified, phrasing)

    write = hero_verified.tool_traces[-1]
    assert write.tool_name == "initiate_return"
    assert write.status is ToolStatus.OK
    assert len([r for r in returns_in(data_dir) if r["order_id"] == IN_WINDOW_ORDER]) == 1


@pytest.mark.parametrize("phrasing", ["Yes", "Go ahead", "Please do"])
def test_the_same_words_authorise_nothing_out_of_context(
    make_agent, seeded_graph, hero_verified, phrasing
) -> None:
    """The affirmatives are not a global approval switch.

    Same words, no pending return, and nothing is confirmed — so the write tool
    would refuse even if the model asked for it.
    """
    agent, _ = make_agent(text("Sorry — what would you like me to do?"))

    agent.respond(hero_verified, phrasing)

    assert hero_verified.confirmed is False
    assert hero_verified.may_mutate is False


def test_agreeing_with_a_statement_is_not_a_confirmation(
    make_agent, seeded_graph, hero_verified
) -> None:
    """A "yes" answering something that was not a question authorises nothing.

    The agent reported eligibility without asking permission. The customer's
    "yes please" agrees with a fact, and `pending_return.asked` stays False.
    """
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

    assert hero_verified.pending_return is not None
    assert hero_verified.pending_return.asked is False
    assert hero_verified.confirmed is False
