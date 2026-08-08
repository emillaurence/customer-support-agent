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

The word "return" is not the signal — the *subject* is. "What is the return
policy for Australian customers?" and "Can I return my order?" share a keyword
and nothing else: the first is a policy lookup with no customer in it, the
second is an eligibility question about a specific record. So the router asks
which of the two a turn is, and only the second one is a workflow. The trace
says which: *informational policy lookup* versus *return or refund workflow*.

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


INFORMATIONAL_KEYWORDS: tuple[str, ...] = (
    "what is", "what's", "whats", "what are", "how long", "how many",
    "do you allow", "do you accept", "do you offer", "is there a policy",
    "policy", "policies", "rules", "return window", "refund window",
    # Questions about a *class* of product rather than the customer's copy of
    # one. "Can ebooks be returned?" is the policy, asked in the passive voice.
    "can ebooks", "can e-books", "can digital", "can physical", "can audiobooks",
    "are ebooks", "are e-books", "are digital", "in general", "generally",
)
"""Phrasing that asks what the rules *are*, rather than what happens to an order."""

CUSTOMER_SPECIFIC_KEYWORDS: tuple[str, ...] = (
    "my order", "my book", "my item", "my purchase", "my return", "my refund",
    "my case", "my copy", "this order", "this book", "this item", "that order",
    "return this", "return it", "return mine", "refund me", "refund this",
    "am i", "can i", "could i", "may i", "do i qualify", "eligible",
    "eligibility", "i want", "i'd like", "i would like", "i need to",
    "start the return", "start a return", "start my return", "go ahead",
    "for me", "was rejected",
)
"""First-person, this-order, take-action language. Blocks the Haiku demotion."""

OVERRIDE_KEYWORDS: tuple[str, ...] = (
    "make an exception", "an exception", "exception", "special case", "waive",
    "override", "just this once", "bend the rules", "anything you can do",
)
"""Asking for the policy *not* to apply.

Its own group because it is the one case where the customer already knows the
rule. There is nothing to look up — the turn is a judgement about whether to
depart from the answer, which is the opposite of retrieval.
"""

ORDER_REFERENCE_PREFIXES: tuple[str, ...] = ("ord-", "item-", "rma-")
"""Naming a record makes a question about that record, not about the policy."""

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

    # Asking what the return policy *is* is a read, even though it says
    # "return". The keyword alone does not decide it — see
    # `_is_informational_policy_question`.
    if _is_informational_policy_question(text, state):
        return ModelDecision(tier=ModelTier.HAIKU, reason="informational policy lookup")

    if _mentions(text, RETURN_KEYWORDS):
        return ModelDecision(
            tier=ModelTier.SONNET, reason="return or refund workflow", return_intent=True
        )

    if _mentions(text, OVERRIDE_KEYWORDS):
        return ModelDecision(tier=ModelTier.SONNET, reason="request to depart from policy")

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


def _is_informational_policy_question(padded_text: str, state: SessionState) -> bool:
    """Whether this turn is a question about the rules and nothing more.

    Four conditions, all required. The message has to *ask what the policy is*,
    and it has to carry none of the signals that would make it about a specific
    customer's order — so the demotion is only taken on a turn where there is
    genuinely nothing to decide and nothing to write.

    Written as a veto rather than a match: any customer-specific, escalation, or
    ambiguity signal keeps the turn on Sonnet, and so does a conversation that is
    already long enough to be carrying context. That keeps the asymmetry in
    `select_model` intact — this function can only demote a turn it is sure about.

    Args:
        padded_text: The lowercased message, space-padded.
        state: The live session, for the turn-count check.
    """
    if _mentions(padded_text, CUSTOMER_SPECIFIC_KEYWORDS):
        return False
    if _mentions(padded_text, ORDER_REFERENCE_PREFIXES):
        return False
    if _mentions(padded_text, OVERRIDE_KEYWORDS):
        return False
    if _mentions(padded_text, ESCALATION_KEYWORDS):
        return False
    if _mentions(padded_text, AMBIGUITY_KEYWORDS):
        return False
    if state.user_turn_count >= COMPLEX_TURN_COUNT:
        return False
    return _mentions(padded_text, INFORMATIONAL_KEYWORDS)


def _mentions(padded_text: str, keywords: tuple[str, ...]) -> bool:
    """Whether any keyword appears in the message.

    Args:
        padded_text: The lowercased message, with a space at each end so a
            keyword can be written with word boundaries if it needs them.
        keywords: Substrings to look for.
    """
    return any(keyword in padded_text for keyword in keywords)
