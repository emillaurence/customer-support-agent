"""Deterministic model selection: Haiku or Sonnet, decided in plain Python.

One function, a keyword list, and a handful of session-state checks. No model
call, no classifier, no routing agent — asking an LLM which LLM to use costs a
round trip to answer a question a boolean can answer, and makes the decision
unreproducible.

The rule underneath the list: **Haiku answers, Sonnet acts.** Reading out an
order status or quoting the standard return window is retrieval. Deciding
whether a return is allowed, holding a multi-turn workflow together, or writing
to Bookly's records is not, and it is where a cheaper model's mistakes cost the
customer something.

Which turns go where:

* **Haiku** — simple, read-only interactions. A policy question, an order
  status, a greeting. Nothing is decided and nothing is written.
* **Sonnet** — returns and refunds, ambiguity the agent has to resolve,
  state-changing actions, escalation, and any conversation that has grown long
  enough to be carrying real context.

**The asymmetry is the design.** The two ways to be wrong do not cost the same.

* A *false promotion* — sending a simple question to Sonnet — costs a fraction
  of a cent and a little latency. The customer gets a correct answer from a
  stronger model than they needed.
* A *false demotion* — sending a return, an ambiguous reference, or a
  half-finished workflow to Haiku — risks a wrong eligibility explanation, a
  dropped thread, or a confident answer about a record the agent should have
  reasoned harder about. That is the failure a customer actually notices.

So every rule below is written to promote when uncertain. The keyword lists are
deliberately broad and the state checks come first: a turn only reaches the
Haiku default by failing every reason to be on Sonnet. Over-promotion is the
expected, accepted cost of that.

Two properties follow from it:

* **Deterministic.** The same state and the same message always route the same
  way, so a demo behaves the same twice and a test can assert on it.
* **One-way.** Once a return workflow is open, the turn stays on Sonnet until
  the workflow is cleared. Promotion is cheap; a workflow that drops back to the
  cheaper model halfway through is not.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from agent.state import SessionState


class ModelTier(StrEnum):
    """Which model handles a turn. Names a tier, never a model id.

    The id comes from the environment — see `agent.config`.
    """

    HAIKU = "haiku"
    SONNET = "sonnet"


RETURN_KEYWORDS: tuple[str, ...] = (
    "return", "refund", "money back", "send it back", "send this back",
    "exchange", "replace", "replacement", "damaged", "faulty", "broken",
    "defective", "wrong item", "cancel my order", "rma",
    # Return intent with none of the vocabulary. "I don't want it anymore" is a
    # customer opening a return; it just doesn't say so in the word the list
    # was built around.
    "don't want it", "dont want it", "do not want it", "no longer want",
    "take it back",
)
"""Anything that could open or continue a return. Matched as substrings."""

ESCALATION_KEYWORDS: tuple[str, ...] = (
    "speak to", "talk to", "human", "person", "agent", "manager", "supervisor",
    "complain", "complaint", "legal", "lawyer", "consumer law", "ombudsman",
    "chargeback", "dispute", "fraud", "unacceptable",
)
"""Requests for a person, and the topics that must become one."""

AMBIGUITY_KEYWORDS: tuple[str, ...] = (
    "both", "either", "which one", "the other one", "not sure", "don't know",
    "dont know", "can't remember", "cant remember", "confused", "actually",
    "instead",
)
"""Signals the customer is referring to something the agent has to disambiguate."""

COMPLEX_TURN_COUNT = 6
"""User turns after which a conversation counts as complex on length alone.

A conversation still going after six turns is not a lookup. Something is being
worked through, and the cheaper model is the wrong place to work it through.
"""


class ModelDecision(BaseModel):
    """Which tier was chosen, and the one reason that decided it.

    `reason` is recorded on the turn so the demo can show *why* a turn was routed
    the way it was, not just where it went.
    """

    tier: ModelTier
    reason: str
    return_intent: bool = False
    """Whether this turn was promoted because the customer asked about a return.

    Recorded on the session by the orchestrator, so the turns that follow stay on
    Sonnet while the customer is still picking which book they mean — see
    `SessionState.return_intent_expressed`.
    """

    @property
    def is_sonnet(self) -> bool:
        return self.tier is ModelTier.SONNET


def select_model(state: SessionState, user_message: str) -> ModelDecision:
    """Choose the model for one turn.

    Checked in order, first match wins. Session state is checked before the
    message text: an open return workflow outranks whatever the customer just
    typed, so a mid-workflow "ok" does not fall back to Haiku.

    Args:
        state: The live session, before this turn's message is processed.
        user_message: What the customer just said.

    Returns:
        The tier and the reason it was chosen. Never raises: a message that
        matches nothing is a simple message, and gets Haiku. Every genuinely
        uncertain case is caught by one of the checks above that default,
        because a false promotion is cheaper than a false demotion.
    """
    text = f" {user_message.lower().strip()} "

    # --- State: an open workflow keeps the turn on Sonnet ------------------

    if state.escalated:
        return ModelDecision(tier=ModelTier.SONNET, reason="conversation is escalated")

    if state.return_workflow_active:
        return ModelDecision(tier=ModelTier.SONNET, reason="a return workflow is open")

    if state.pending_return is not None:
        return ModelDecision(tier=ModelTier.SONNET, reason="a return is awaiting confirmation")

    if state.confirmed:
        return ModelDecision(tier=ModelTier.SONNET, reason="a confirmed return is pending")

    # --- Message: intent that needs reasoning or will change records -------

    if _mentions(text, RETURN_KEYWORDS):
        return ModelDecision(
            tier=ModelTier.SONNET, reason="return or refund intent", return_intent=True
        )

    if _mentions(text, ESCALATION_KEYWORDS):
        return ModelDecision(tier=ModelTier.SONNET, reason="escalation intent")

    if _mentions(text, AMBIGUITY_KEYWORDS):
        return ModelDecision(tier=ModelTier.SONNET, reason="ambiguous reference to resolve")

    # --- Shape: a long conversation is a complex one -----------------------

    if state.user_turn_count >= COMPLEX_TURN_COUNT:
        return ModelDecision(
            tier=ModelTier.SONNET,
            reason=f"multi-turn context ({state.user_turn_count} turns)",
        )

    return ModelDecision(tier=ModelTier.HAIKU, reason="simple read-only request")


def _mentions(padded_text: str, keywords: tuple[str, ...]) -> bool:
    """Whether any keyword appears in the message.

    Args:
        padded_text: The lowercased message, with a space at each end so a
            keyword can be written with word boundaries if it needs them.
        keywords: Substrings to look for.
    """
    return any(keyword in padded_text for keyword in keywords)
