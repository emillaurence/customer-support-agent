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
