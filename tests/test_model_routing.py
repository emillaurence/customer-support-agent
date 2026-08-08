"""Model routing: the right tier, for the right reason, every time.

Two things are being checked. The obvious one is that simple turns go to Haiku
and consequential ones go to Sonnet. The one that matters more is that the
decision is *deterministic* — same state, same message, same answer — because a
router that sometimes picks Sonnet is a router nobody can demo or debug.
"""

from __future__ import annotations

import pytest

from agent.models import EligibilityDecision, Role
from agent.routing import COMPLEX_TURN_COUNT, ModelTier, select_model
from agent.state import PendingReturn, SessionState


def route(message: str, state: SessionState | None = None) -> ModelTier:
    """The tier chosen for one message against a state."""
    return select_model(state or SessionState(), message).tier


# --- Haiku: retrieval ----------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "What's Bookly's policy on ebooks?",
        "How long do I have to send something back",  # 'send something back' is not a keyword
        "Do you ship to Ireland?",
        "Where is my order ORD-1001?",
        "Has my order been delivered yet?",
        "hi",
    ],
)
def test_simple_requests_use_haiku(message: str) -> None:
    """A policy question or an order-status check is retrieval, not reasoning."""
    assert route(message) is ModelTier.HAIKU


def test_order_status_on_a_verified_session_stays_on_haiku(verified_state: SessionState) -> None:
    """Being verified is not complexity. Reading out a status is still a lookup."""
    assert route("Where's my order?", verified_state) is ModelTier.HAIKU


# --- Haiku: informational policy questions -------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "What is the return policy?",
        "What is the return policy for Australian customers?",
        "Can ebooks be returned?",
        "How long is the holiday return window?",
        "How long do I have to return a physical book?",
        "What are your refund rules?",
        "Do you allow returns on audiobooks?",
    ],
)
def test_informational_policy_questions_use_haiku(message: str) -> None:
    """"Return" in a question about the rules is a topic, not an intent.

    Every one of these is a read of the policy graph with no customer in it.
    Matching the keyword alone would send them all to Sonnet.
    """
    assert route(message) is ModelTier.HAIKU


def test_the_australian_policy_question_is_a_lookup_not_a_workflow() -> None:
    """The case the change exists for, asserted on the reason as well as the tier."""
    decision = select_model(SessionState(), "What is the return policy for Australian customers?")

    assert decision.tier is ModelTier.HAIKU
    assert decision.reason == "informational policy lookup"
    assert decision.return_intent is False  # nothing to remember for the next turn


def test_a_policy_question_on_a_verified_session_is_still_a_lookup(
    verified_state: SessionState,
) -> None:
    """Knowing who is asking does not turn the question into their transaction."""
    assert route("What's the return policy for ebooks?", verified_state) is ModelTier.HAIKU


# --- Sonnet: intent ------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "I want to return this book",
        "Can I get a refund?",
        "The book arrived damaged",
        "I'd like my money back please",
        "This is faulty, can I exchange it",
    ],
)
def test_return_intent_uses_sonnet(message: str) -> None:
    """Anything that could become a return is reasoned about, not looked up."""
    assert route(message) is ModelTier.SONNET


@pytest.mark.parametrize(
    "message",
    [
        "Can I return my order?",
        "Am I eligible for a return?",
        "I want to return ORD-1003",
        "Start the return",
        "Can I get a refund for ORD-1003?",
        "Why was my return rejected?",
        "Can you make an exception?",
        "Am I eligible to return my book?",
    ],
)
def test_customer_specific_return_questions_use_sonnet(message: str) -> None:
    """The same topic, asked about the customer's own record, is the workflow.

    First person, a named order, or an instruction to act — each one is enough
    on its own, and none of them may be demoted by informational phrasing.
    """
    assert route(message) is ModelTier.SONNET


def test_the_eligibility_question_reads_as_a_workflow_in_the_trace() -> None:
    """The reason has to name the workflow, so the trace distinguishes the two."""
    decision = select_model(SessionState(), "Am I eligible to return my book?")

    assert decision.tier is ModelTier.SONNET
    assert decision.reason == "return or refund workflow"
    assert decision.return_intent is True


def test_informational_phrasing_does_not_rescue_a_specific_order() -> None:
    """A message with both signals is promoted, not demoted.

    "What is the return policy" would demote on its own. The order id is a veto,
    because uncertainty resolves towards the stronger model.
    """
    assert route("What is the return policy for my order ORD-1003?") is ModelTier.SONNET


@pytest.mark.parametrize(
    "message",
    [
        "I want to speak to a human",
        "Let me talk to your manager",
        "I'm going to raise a chargeback",
        "What are my rights under consumer law?",
    ],
)
def test_escalation_intent_uses_sonnet(message: str) -> None:
    """A handoff is a judgement call, and often a distressed customer."""
    assert route(message) is ModelTier.SONNET


@pytest.mark.parametrize(
    "message",
    ["Which one was the paperback?", "I'm not sure which order it was", "actually, the other one"],
)
def test_ambiguity_uses_sonnet(message: str) -> None:
    """A reference the agent has to resolve is exactly where guessing costs."""
    assert route(message) is ModelTier.SONNET


# --- Sonnet: state -------------------------------------------------------


def test_active_return_workflow_keeps_sonnet(verified_state: SessionState) -> None:
    """Once an item is under discussion, the workflow owns the routing.

    The message is a bare "ok" with no keyword in it. Without the state check
    this would drop to Haiku halfway through a return.
    """
    verified_state.active_item_id = "ITEM-100"
    assert route("ok", verified_state) is ModelTier.SONNET


def test_eligibility_decision_keeps_sonnet(verified_state: SessionState) -> None:
    """A decision on file means the workflow is live, eligible or not."""
    verified_state.eligibility = EligibilityDecision(eligible=False, explanation="outside window")
    assert route("I see", verified_state) is ModelTier.SONNET


def test_pending_confirmation_uses_sonnet(verified_state: SessionState) -> None:
    """The turn that might authorise a write is the last one to run cheaply."""
    verified_state.pending_return = PendingReturn(
        order_id="ORD-1001", item_id="ITEM-100", eligibility_token="tok", asked=True
    )
    assert route("yes please", verified_state) is ModelTier.SONNET


def test_go_ahead_with_a_confirmation_pending_uses_sonnet(verified_state: SessionState) -> None:
    """The turn that spends the customer's yes is a write, whatever its wording."""
    verified_state.pending_return = PendingReturn(
        order_id="ORD-1003", item_id="ITEM-100", eligibility_token="tok", asked=True
    )
    assert route("Yes, go ahead", verified_state) is ModelTier.SONNET


def test_a_generic_follow_up_mid_workflow_stays_on_sonnet(verified_state: SessionState) -> None:
    """An open workflow outranks the text of the message.

    "What happens next?" has informational phrasing and nothing customer-specific
    in it, so on its own text it would demote. The state check runs first, so it
    cannot.
    """
    verified_state.return_intent_expressed = True
    decision = select_model(verified_state, "What happens next?")

    assert decision.tier is ModelTier.SONNET
    assert decision.reason == "a return workflow is open"


def test_a_policy_question_mid_workflow_stays_on_sonnet(verified_state: SessionState) -> None:
    """Mid-return, even the policy question belongs to the return."""
    verified_state.active_item_id = "ITEM-100"
    assert route("What is the return policy for ebooks?", verified_state) is ModelTier.SONNET


def test_confirmed_return_uses_sonnet(verified_state: SessionState) -> None:
    """A confirmed return is about to be written."""
    verified_state.confirmed = True
    verified_state.eligibility_token = "tok"
    assert route("thanks", verified_state) is ModelTier.SONNET


def test_escalated_conversation_stays_on_sonnet(verified_state: SessionState) -> None:
    """After a handoff the agent must stop acting — not the moment to economise."""
    verified_state.escalated = True
    assert route("ok thanks", verified_state) is ModelTier.SONNET


def test_long_conversation_promotes_to_sonnet() -> None:
    """A conversation still going after several turns is not a lookup."""
    state = SessionState()
    for _ in range(COMPLEX_TURN_COUNT):
        state.add_message(Role.USER, "and?")
        state.add_message(Role.ASSISTANT, "...")
    assert route("what about that", state) is ModelTier.SONNET


# --- Properties ----------------------------------------------------------


def test_routing_is_deterministic(verified_state: SessionState) -> None:
    """The same input routes the same way, every time.

    The property the demo depends on: nothing here samples, and nothing depends
    on a clock or a model.
    """
    for message in ("what's your return policy", "I want a refund", "ok"):
        decisions = {select_model(verified_state, message).tier for _ in range(20)}
        assert len(decisions) == 1


def test_every_decision_carries_a_reason() -> None:
    """A tier with no stated reason cannot be shown in a trace."""
    for message in ("hello", "refund please", "I want a human"):
        assert select_model(SessionState(), message).reason


def test_promotion_is_one_way_within_a_workflow(verified_state: SessionState) -> None:
    """A return workflow never drops back to the cheaper model.

    Walks the whole workflow with messages that would each route to Haiku on
    their own, and checks every step stays on Sonnet.
    """
    steps = [
        ("ITEM-100", None, "ok"),
        ("ITEM-100", "tok", "right"),
    ]
    for item_id, token, message in steps:
        verified_state.active_item_id = item_id
        verified_state.eligibility_token = token
        assert route(message, verified_state) is ModelTier.SONNET
