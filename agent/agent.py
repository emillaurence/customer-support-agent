"""The Bookly agent: one user turn in, one reply out.

A hand-written tool loop against the Anthropic Messages API. No framework, no
second agent — one class, one loop, one prompt, in the order it happens:

    configuration → the system prompt → model routing → confirmation → the loop

**Claude owns language**; **Python owns truth.** Which model handles the turn,
whether identity is verified, what the policy says, whether a return is eligible,
whether the customer actually agreed, and whether anything is written are all
decided here. Claude can ask for a tool; it cannot decide the answer, and it
cannot set a trusted field.

One turn: read the message and decide whether it confirms a pending return; route
to Haiku or Sonnet; call Anthropic; run whatever tools it asks for, trace them,
update trusted state from the results that succeeded, and loop; return the first
plain-text reply. Nothing here invents a result — every failure reaches the
customer as what it was.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from enum import StrEnum
from typing import Any

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agent.state import (
    EligibilityDecision,
    ModelTurn,
    Role,
    SessionState,
    ToolStatus,
    ToolTrace,
    sanitize_args,
)
from agent.tools import (
    TOOL_SCHEMAS,
    EscalationResult,
    ReturnResult,
    apply_tool_result,
    invoke_tool,
)

LOG = logging.getLogger("bookly")
"""Operational logging: what ran, whether it worked, how long it took.

Given a handler by `app.py` — the library only writes records, so a test or a
script that never configures logging pays nothing and cannot be broken by it.
The lines are written from the same sanitized values as the Agent Trace, so a
credential or a whole email address cannot reach the file by this route either.
No prompt, no reply, no reasoning: those are the conversation, not operations.
"""

# --- Configuration -------------------------------------------------------
#
# No model id is hardcoded anywhere. Which Claude models Bookly runs on is a
# deployment decision: the router picks a *tier*, and the environment decides
# what that resolves to. Both names are required, because the agent routes
# between them. Missing configuration fails at construction, not on the
# customer's first message.

REQUIRED_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL_HAIKU", "ANTHROPIC_MODEL_SONNET")

TEMPERATURE_ENV = "ANTHROPIC_TEMPERATURE"
"""Optional sampling temperature, for both tiers — one value, never one per model.

Unset by default, deliberately: current Claude models manage their own sampling
and reject an explicit temperature, so unset means the parameter is not sent at
all. The setting stays because a model that does accept one should be
configurable without touching the loop.
"""


class AnthropicConfigError(RuntimeError):
    """Anthropic is not configured. Raised at startup, never mid-conversation."""


class AnthropicConfig(BaseModel):
    """The values the loop needs to talk to Anthropic."""

    api_key: str
    haiku_model: str = Field(description="Model id for simple, read-only turns.")
    sonnet_model: str = Field(description="Model id for reasoning and state-changing turns.")
    temperature: float | None = Field(
        default=None, description="None means the parameter is not sent at all."
    )

    def __repr__(self) -> str:
        """Redact the key. A config object can end up in a traceback."""
        return (
            f"AnthropicConfig(api_key='***', haiku_model={self.haiku_model!r}, "
            f"sonnet_model={self.sonnet_model!r}, temperature={self.temperature!r})"
        )

    __str__ = __repr__


def load_anthropic_config() -> AnthropicConfig:
    """Read and validate the Anthropic configuration from the environment.

    Raises:
        AnthropicConfigError: If a required variable is unset or blank — the
            message names every missing one at once — or if the temperature is
            not a number. Silently ignoring a typo would hide it behind
            behaviour that looks deliberate.
    """
    load_dotenv()
    values = {name: (os.getenv(name) or "").strip() for name in REQUIRED_ENV}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise AnthropicConfigError(
            f"Anthropic is required to run the agent, but "
            f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not set. "
            f"Copy .env.example to .env and fill it in. Both model names are "
            f"required: the agent routes between them."
        )

    raw_temperature = (os.getenv(TEMPERATURE_ENV) or "").strip()
    try:
        temperature = float(raw_temperature) if raw_temperature else None
    except ValueError:
        raise AnthropicConfigError(
            f"{TEMPERATURE_ENV} must be a number, but it is set to {raw_temperature!r}. "
            f"Leave it unset to let the model manage its own sampling."
        ) from None

    return AnthropicConfig(
        api_key=values["ANTHROPIC_API_KEY"],
        haiku_model=values["ANTHROPIC_MODEL_HAIKU"],
        sonnet_model=values["ANTHROPIC_MODEL_SONNET"],
        temperature=temperature,
    )


# --- The system prompt ---------------------------------------------------
#
# **No business rule is written here.** There is no return window in this prompt,
# no regional override, no precedence order — those are in Neo4j, and the tools
# read them. A rule stated in the prompt as well as the graph is a rule with two
# versions, and one of them will be wrong.
#
# The gates are the same story. The prompt asks for explicit confirmation before
# a return because that is the behaviour a customer should experience; it is not
# what makes the gate hold. `confirmed` is decided below and enforced in
# `initiate_return`. If the model ignored every line, no return could open.

SYSTEM_PROMPT = """\
You are the customer support agent for Bookly, an online bookstore selling
physical books and ebooks. You handle order status, returns and refunds, and
questions about Bookly's policy. Anything materially outside that — payment
disputes, address changes, account changes, legal or consumer-law questions,
another person's account — gets one honest sentence and an escalation.

## Facts come from tools
Never guess, never fill a gap with something plausible, and never answer from
your own knowledge of how returns usually work. Order status, dates, and titles
come from `lookup_order`. Return rules come from `search_policy`. Eligibility is
decided by `check_return_eligibility` — report its explanation; do not overrule
it, soften it, or hint at an exception.

## Identity first
Anything account-specific needs a verified customer: ask for the email on their
Bookly account and call `verify_identity`. Share no account details until it
succeeds.

## Find the order, then ask only if it is still unclear
Once identity is settled, call `lookup_order` with no order id: it lists their
orders with the books on each and the item ids to act on. If what the customer
named matches exactly one item, that is the one — use its order id and item id
straight away. Do not ask them to confirm a choice they already made, and do not
look the order up again for detail you were not asked for.

Ask when it is genuinely ambiguous: the title matches items on two orders, they
said "my book" and several could be it, or they have not named one at all. Then
name the books and ask which — never pick the recent or likely one. One question
at a time. Read a single order with `lookup_order` when the customer wants its
dates, delivery, or status.

## Confirm before acting
A return changes Bookly's records. Before opening one, name the item and the
order and ask a direct question — "Shall I start a return for <item> on
<order>?" — then wait. An earlier "ok" is not agreement to this. Ask that and
nothing else: a reason is optional, so never make them supply one first.

## Never claim something happened unless it did
Say an action succeeded only if the tool succeeded. If a tool refuses, explain
what it said rather than working around it; if it fails, say so plainly and
offer a colleague.

## Escalate when unsupported
Use `escalate_to_human` when the customer asks for a person, the request is
outside your tools, a tool has failed twice on the same thing, or the customer
is distressed. Say you are handing over and why; do not promise what they will
decide. When you must refuse something the customer wants, say no once, plainly,
offer a colleague, and do not suggest they will say yes.

## Never expose internal reasoning
Do not narrate your reasoning or your tool plan, and do not mention tool names,
policy ids, precedence, or the graph. Give the conclusion and the reason it
holds, in ordinary language.

Tone: plain, warm, brief. No filler apologies. No emoji.
"""


# --- Model routing -------------------------------------------------------
#
# Keyword lists and session-state checks. No model call, no classifier — asking
# an LLM which LLM to use costs a round trip to answer a question a boolean can
# answer, and makes the decision unreproducible.
#
# The rule is **Haiku answers, Sonnet acts**, and the word "return" is not the
# signal — the *subject* is. "What is the return policy for Australian
# customers?" and "Can I return my order?" share a keyword and nothing else.
#
# **The asymmetry is the design.** A false promotion costs a fraction of a cent;
# a false demotion risks a wrong eligibility explanation or a dropped thread
# mid-workflow. So every rule below promotes when uncertain: the lists are broad,
# the state checks come first, and a turn reaches the Haiku default only by
# failing every reason to be on Sonnet.


class ModelTier(StrEnum):
    """Which model handles a turn. Names a tier, never a model id."""

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
"""Asking for the policy *not* to apply. The customer already knows the rule, so
there is nothing to look up."""

ORDER_REFERENCE_PREFIXES: tuple[str, ...] = ("ord-", "item-", "rma-")
"""Naming a record makes a question about that record, not about the policy."""

RETURN_KEYWORDS: tuple[str, ...] = (
    "return", "refund", "money back", "send it back", "send this back",
    "exchange", "replace", "replacement", "damaged", "faulty", "broken",
    "defective", "wrong item", "cancel my order", "rma",
    # Return intent with none of the vocabulary. "I don't want it anymore" is a
    # customer opening a return; it just doesn't say so in the expected word.
    "don't want it", "dont want it", "do not want it", "no longer want",
    "take it back",
)

ESCALATION_KEYWORDS: tuple[str, ...] = (
    "speak to", "talk to", "human", "person", "agent", "manager", "supervisor",
    "complain", "complaint", "legal", "lawyer", "consumer law", "ombudsman",
    "chargeback", "dispute", "fraud", "unacceptable",
)

AMBIGUITY_KEYWORDS: tuple[str, ...] = (
    "both", "either", "which one", "the other one", "not sure", "don't know",
    "dont know", "can't remember", "cant remember", "confused", "actually",
    "instead",
)
"""Signals the customer is referring to something the agent has to disambiguate."""

COMPLEX_TURN_COUNT = 6
"""A conversation still going after six user turns is not a lookup. Something is
being worked through, and the cheaper model is the wrong place to do it."""


class ModelDecision(BaseModel):
    """Which tier was chosen, and the one reason that decided it.

    `reason` is recorded on the turn, so the demo shows *why* a turn went where it
    did rather than just where.
    """

    tier: ModelTier
    reason: str
    return_intent: bool = Field(
        default=False,
        description=(
            "True when this turn was promoted because the customer asked about a "
            "return. Recorded on the session, so the turns spent working out which "
            "book they mean stay on Sonnet."
        ),
    )


def select_model(state: SessionState, user_message: str) -> ModelDecision:
    """Choose the model for one turn. First match wins.

    Session state is checked before the message text: an open return workflow
    outranks whatever the customer just typed, so a mid-workflow "ok" does not
    fall back to Haiku. Never raises — a message that matches nothing is a simple
    message and gets Haiku.
    """
    text = f" {user_message.lower().strip()} "

    # An open workflow keeps the turn on Sonnet.
    if state.escalated:
        return ModelDecision(tier=ModelTier.SONNET, reason="conversation is escalated")
    if state.return_workflow_active:
        return ModelDecision(tier=ModelTier.SONNET, reason="a return workflow is open")
    if state.pending_return is not None:
        return ModelDecision(tier=ModelTier.SONNET, reason="a return is awaiting confirmation")
    if state.confirmed:
        return ModelDecision(tier=ModelTier.SONNET, reason="a confirmed return is pending")

    # Intent that needs reasoning, or will change records.
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

    # A long conversation is a complex one.
    if state.user_turn_count >= COMPLEX_TURN_COUNT:
        return ModelDecision(
            tier=ModelTier.SONNET, reason=f"multi-turn context ({state.user_turn_count} turns)"
        )

    return ModelDecision(tier=ModelTier.HAIKU, reason="simple read-only request")


def _is_informational_policy_question(padded_text: str, state: SessionState) -> bool:
    """Whether this turn is a question about the rules and nothing more.

    A veto rather than a match: any customer-specific, escalation, or ambiguity
    signal keeps the turn on Sonnet, and so does a conversation already carrying
    context. So this can only demote a turn it is sure about.
    """
    for keywords in (
        CUSTOMER_SPECIFIC_KEYWORDS,
        ORDER_REFERENCE_PREFIXES,
        OVERRIDE_KEYWORDS,
        ESCALATION_KEYWORDS,
        AMBIGUITY_KEYWORDS,
    ):
        if _mentions(padded_text, keywords):
            return False
    if state.user_turn_count >= COMPLEX_TURN_COUNT:
        return False
    return _mentions(padded_text, INFORMATIONAL_KEYWORDS)


def _mentions(padded_text: str, keywords: tuple[str, ...]) -> bool:
    """Whether any keyword appears in the message, which is space-padded so a
    keyword can be written with word boundaries if it needs them."""
    return any(keyword in padded_text for keyword in keywords)


# --- Confirmation --------------------------------------------------------
#
# `confirmed=True` is the last gate before Bookly's records change, so it is set
# in Python, from the conversation, never by the model asking for it. Three
# conditions, all required: a pending action, which only a successful eligible
# check creates; the agent having *asked*, since a customer who says "yes" after
# being told their book is eligible has agreed with a fact rather than authorised
# a write; and a genuine affirmative.
#
# Miss one and `confirmed` stays False — the case this is designed against is a
# bare "yes" with nothing pending. And this is still only the loop's gate;
# `initiate_return` refuses on its own.

AFFIRMATIVES: frozenset[str] = frozenset(
    {
        "yes", "yes please", "yep", "yeah", "yup", "sure", "ok", "okay",
        "go ahead", "please do", "please go ahead", "do it", "confirm",
        "confirm it", "confirmed", "proceed", "proceed please", "affirmative",
        "yes do it", "yes go ahead", "yes confirm", "sounds good", "that's right",
        "thats right", "correct", "start it", "start the return", "do that",
    }
)

CONFIRMATION_CUES: tuple[str, ...] = (
    "shall i", "should i", "would you like me to", "do you want me to",
    "confirm", "go ahead", "is that right", "can i start", "may i start",
    "start the return", "start a return", "open the return", "open a return",
)
"""Phrases that make an assistant message a request for permission. Paired with a
question mark: the difference between the agent stating a fact and asking."""

_NEGATIONS = (" not ", " don't ", " dont ", " no ", " wait", " hold on", " actually", " but ")


def is_affirmative(message: str) -> bool:
    """Whether a reply is agreement, on its own terms. False for anything carrying
    a question, a negation, or a condition — those need another turn, not a write."""
    text = message.strip().lower().strip(" .!,\t\n")
    if not text:
        return False

    # A reply with a question in it is not a confirmation, whatever it starts
    # with. "Yes, but how long do I have?" is a question.
    if "?" in message:
        return False

    if text in AFFIRMATIVES:
        return True

    # "yes please, the paperback" — agreement plus a detail. Allowed, as long as
    # the opening is unambiguous and nothing negates it.
    if any(marker in text for marker in _NEGATIONS):
        return False
    return any(
        text.startswith(f"{phrase} ") or text.startswith(f"{phrase},") for phrase in AFFIRMATIVES
    )


def asks_for_confirmation(assistant_message: str) -> bool:
    """Whether the agent's last message asked the customer to confirm an action.
    A message that reports eligibility without asking returns False, so agreeing
    with it confirms nothing."""
    text = assistant_message.lower()
    return "?" in text and any(cue in text for cue in CONFIRMATION_CUES)


# --- The loop ------------------------------------------------------------

MAX_TOOL_ITERATIONS = 6
"""Safety stop, comfortably above the longest real path — verify, look up, check
eligibility, open the return. Hitting it means something is wrong."""

MAX_TOKENS = 1024
"""Support replies are short. A cap this size also bounds a runaway turn."""

REQUEST_TIMEOUT_SECONDS = 30.0
"""How long one Anthropic call may take before it is abandoned.

A support customer is waiting on this, so the useful failure is a fast, honest
"I can't reach our systems" — not a request still open a minute later against a
chat window nobody is watching. Comfortably above a normal turn, which is a
second or two at this `max_tokens`.
"""

MAX_RETRIES = 2
"""Retries per call, by the SDK's own bounded exponential backoff.

Two, not more: the timeout above is a wall-clock budget per attempt, and three
attempts of thirty seconds is already longer than a customer will wait. This
retries *the model call*, which is safe to repeat — a tool call is not, and is
never retried. `initiate_return` is idempotent on top of that, so even a
duplicate request cannot open a second RMA.
"""

CACHE_CONTROL: dict[str, str] = {"type": "ephemeral"}
"""Turn on Anthropic's automatic prompt caching, at the request level.

Every turn re-sends the same prefix — six tool schemas, then the system prompt,
then the whole conversation so far — and a support conversation only grows. With
this set, the API caches the longest stable prefix and later turns read it back
instead of reprocessing it.

Which is why nothing in that prefix may wobble. Caching is a *prefix match*: one
changed byte anywhere invalidates everything after it. `TOOL_SCHEMAS` is a
literal in a fixed order, the system prompt is a constant with no interpolated
date or session id, and the transcript is only ever appended to. The default
short-lived cache is right for a live conversation — the next turn is seconds
away, not hours.
"""

FALLBACK_UNAVAILABLE = (
    "I'm having trouble reaching our systems right now, so I can't look that up. "
    "Please try again in a moment, or reply and I'll pass you to a colleague."
)
FALLBACK_STUCK = (
    "I'm going in circles on this one rather than getting you an answer. "
    "Let me hand you to a colleague who can pick it up."
)
FALLBACK_EMPTY = "Sorry — I didn't manage to put that into words. Could you say that again?"


class BooklyAgent:
    """A single support agent with a flat set of tools.

    Stateless between turns: everything that persists is on the `SessionState`
    passed to `respond`, so one agent can serve many conversations.
    """

    def __init__(
        self,
        config: AnthropicConfig | None = None,
        client: Any | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        clock: datetime | None = None,
    ) -> None:
        """Build an agent, failing now if Anthropic is not configured.

        `config` is read from the environment when omitted. Tests pass a stand-in
        `client` so the loop runs offline, and a fixed `clock` so the fixture
        scenarios keep meaning what they say.
        """
        self.config = config or load_anthropic_config()
        self.client = client or anthropic.Anthropic(
            api_key=self.config.api_key,
            # Bounded on both axes, by the SDK rather than by a retry loop here:
            # one call cannot hang, and a flaky connection cannot be retried
            # forever. See REQUEST_TIMEOUT_SECONDS and MAX_RETRIES.
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=MAX_RETRIES,
        )
        self.system_prompt = system_prompt
        self.clock = clock

    def respond(self, state: SessionState, user_message: str) -> str:
        """Take one user turn and produce one assistant reply.

        Mutates `state` in place. Always returns something sayable: an outage or a
        stuck loop produces an honest message, never an exception and never a
        fabricated success.
        """
        # Confirmation is read before the model sees the message, and from the
        # state as it was when the agent asked its question. Doing it here means a
        # "yes" is judged against what was actually pending.
        self._update_confirmation(state, user_message)

        state.add_message(Role.USER, user_message)
        state.transcript.append({"role": "user", "content": user_message})

        decision = select_model(state, user_message)
        if decision.return_intent:
            # The router found return intent. Remember it, so the turns spent
            # working out which book they mean stay on Sonnet — the eligibility
            # check usually happens on one of them.
            state.return_intent_expressed = True

        model_id = self._model_id(decision)
        turn = ModelTurn(
            session_id=state.session_id,
            model_tier=decision.tier.value,
            model=model_id,
            routing_reason=decision.reason,
        )
        state.model_turns.append(turn)
        LOG.info(
            'agent session=%s model=%s route="%s"',
            state.session_id,
            decision.tier.value,
            decision.reason,
        )

        # The loop owns `state.transcript` — every assistant turn it produces,
        # including a fallback, is written there by the loop itself. Here we only
        # mirror the reply into the visible transcript.
        reply = self._run_tool_loop(state, decision, model_id, turn)
        state.add_message(Role.ASSISTANT, reply)
        return reply

    def _run_tool_loop(
        self, state: SessionState, decision: ModelDecision, model_id: str, turn: ModelTurn
    ) -> str:
        """Call Anthropic, run whatever tools it asks for, repeat until it replies."""
        for _ in range(MAX_TOOL_ITERATIONS):
            turn.iterations += 1

            try:
                response = self.client.messages.create(
                    model=model_id,
                    max_tokens=MAX_TOKENS,
                    # Tools first, then the system prompt, then the conversation:
                    # the render order the API caches against, and the reason the
                    # first two are constants.
                    system=self.system_prompt,
                    tools=TOOL_SCHEMAS,
                    messages=state.transcript,
                    cache_control=CACHE_CONTROL,
                    # One temperature, both tiers — there is a single exit to
                    # Anthropic, so there is nowhere for a turn to be sampled
                    # differently. Omitted entirely unless a deployment set one.
                    **({} if self.config.temperature is None else {"temperature": self.config.temperature}),
                )
            except Exception as exc:  # noqa: BLE001 - any API failure ends the turn safely
                # Rate limit, outage, bad key, dropped connection. The customer
                # gets one honest sentence; the transcript is left as it was.
                self._trace(
                    state,
                    decision,
                    model_id,
                    tool_name="anthropic.messages.create",
                    tool_args={},
                    status=ToolStatus.ERROR,
                    latency_ms=0.0,
                    summary="model call failed",
                    # The type and message, never the request — a request body
                    # carries the transcript, and the key lives on the client.
                    error=f"{type(exc).__name__}: {exc}",
                )
                LOG.error("model call=failed error=%s", type(exc).__name__)
                state.transcript.append(_assistant_text(FALLBACK_UNAVAILABLE))
                return FALLBACK_UNAVAILABLE

            _record_usage(turn, response)
            state.transcript.append({"role": "assistant", "content": _assistant_blocks(response)})

            tool_uses = [b for b in response.content if getattr(b, "type", "") == "tool_use"]
            if not tool_uses:
                if text := _text_of(response):
                    return text
                # A turn with neither text nor a tool call. Rather than leave an
                # empty assistant turn for the next turn to build on, replace it
                # with what the customer will actually see.
                state.transcript[-1] = _assistant_text(FALLBACK_EMPTY)
                return FALLBACK_EMPTY

            # Every tool Claude asked for runs, and all the results go back in one
            # user message — what the Messages API expects, and splitting them
            # teaches the model not to ask for more than one at a time.
            adopt_active_order = not _browsing_orders(tool_uses)
            results = []
            for block in tool_uses:
                turn.tool_calls += 1
                results.append(self._run_one_tool(state, decision, model_id, block, adopt_active_order))
            state.transcript.append({"role": "user", "content": results})

        # Out of iterations. Something is looping; say so rather than trying
        # again, and point at the escape hatch.
        state.transcript.append(_assistant_text(FALLBACK_STUCK))
        return FALLBACK_STUCK

    def _run_one_tool(
        self,
        state: SessionState,
        decision: ModelDecision,
        model_id: str,
        block: Any,
        adopt_active_order: bool,
    ) -> dict[str, Any]:
        """Run one requested tool, trace it, and update state from what it returned."""
        name = getattr(block, "name", "")
        args = dict(getattr(block, "input", None) or {})

        started = time.perf_counter()
        outcome = invoke_tool(name, args, state, now=self.clock)
        latency_ms = (time.perf_counter() - started) * 1000

        # State is updated only from a tool that actually succeeded. A blocked
        # guard or a failed call leaves the session exactly as it was.
        if outcome.status is ToolStatus.OK:
            apply_tool_result(name, args, outcome, state, adopt_active_order=adopt_active_order)

        # The arguments as they were actually used — trusted values included, so
        # the trace shows the real call — with anything sensitive masked. The log
        # line below reads the same values, rather than sanitizing them twice.
        traced_args = sanitize_args(outcome.args_used or args)

        self._trace(
            state,
            decision,
            model_id,
            tool_name=name or "<unnamed>",
            tool_args=traced_args,
            status=outcome.status,
            latency_ms=round(latency_ms, 2),
            summary=outcome.summary,
            error=outcome.error,
            policy_decision=_policy_decision(outcome.payload),
        )
        LOG.info(
            "tool name=%s status=%s latency_ms=%.1f args=%s%s",
            name or "<unnamed>",
            "success" if outcome.status is ToolStatus.OK else outcome.status.value,
            latency_ms,
            traced_args,
            _log_facts(outcome.payload),
        )

        return {
            "type": "tool_result",
            "tool_use_id": getattr(block, "id", ""),
            "content": outcome.content,
            "is_error": outcome.is_error,
        }

    def _update_confirmation(self, state: SessionState, user_message: str) -> None:
        """Decide whether this message confirms the pending return. A bare "yes"
        with nothing pending writes no field at all."""
        pending = state.pending_return
        if pending is None:
            return

        last_assistant = state.last_assistant_message()
        if last_assistant is None:
            return

        # Did the agent actually ask? Recorded on the pending action so it
        # survives the turn and cannot be re-derived from a later message.
        if asks_for_confirmation(last_assistant):
            pending.asked = True

        if pending.asked and is_affirmative(user_message):
            state.confirmed = True

    def _model_id(self, decision: ModelDecision) -> str:
        """Resolve a tier to the model id the environment configured for it."""
        return (
            self.config.sonnet_model
            if decision.tier is ModelTier.SONNET
            else self.config.haiku_model
        )

    def _trace(
        self,
        state: SessionState,
        decision: ModelDecision,
        model_id: str,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        status: ToolStatus,
        latency_ms: float,
        summary: str,
        error: str | None,
        policy_decision: dict[str, Any] | None = None,
    ) -> None:
        """Record one observable event on the session.

        An Anthropic failure is filed here too, under a pseudo tool name: a turn
        that produced nothing needs to show *why* where everything else is shown.
        """
        state.tool_traces.append(
            ToolTrace(
                session_id=state.session_id,
                model=model_id,
                model_tier=decision.tier.value,
                tool_name=tool_name,
                tool_args=tool_args,
                status=status,
                latency_ms=latency_ms,
                result_summary=summary,
                error=error,
                policy_decision=policy_decision,
            )
        )


def _log_facts(payload: Any) -> str:
    """The outcome worth having in the log for this tool, as ` key=value` pairs.

    The three results an operator looks for after the fact: what a policy decided,
    which RMA was opened, which case was raised. Read off the same payload the
    trace uses, and never the token on it.
    """
    if isinstance(payload, EligibilityDecision):
        return f" policy={payload.policy_id} eligible={str(payload.eligible).lower()}"
    if isinstance(payload, ReturnResult):
        return (
            f" return_id={payload.return_record.return_id} "
            f"created={str(payload.created).lower()}"
        )
    if isinstance(payload, EscalationResult):
        return f" case_id={payload.case_id}"
    return ""


def _policy_decision(payload: Any) -> dict[str, Any] | None:
    """The renderable part of one eligibility decision, or None for any other tool.

    Taken from the payload of *this* call, so a turn that checks two items shows
    each item's own policy. The token is deliberately left behind.
    """
    if not isinstance(payload, EligibilityDecision):
        return None
    return {
        "eligible": payload.eligible,
        "policy_id": payload.policy_id,
        "rule_path": payload.rule_path,
        "days_remaining": payload.days_remaining,
    }


# --- Reading an Anthropic response ---------------------------------------
#
# Written against the block fields rather than the SDK's classes, so the loop
# behaves the same on a real response and on a stand-in in a test — and so the
# transcript holds plain, serializable data a Streamlit session can keep across
# a rerun.


def _record_usage(turn: ModelTurn, response: Any) -> None:
    """Add one response's prompt-cache counters to the turn.

    Read defensively, by attribute: `usage` is absent on a test stand-in and its
    cache fields are None when the account or model reports none. Missing is
    zero, never an exception — a turn must not fail over its own bookkeeping.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    turn.cache_creation_input_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
    turn.cache_read_input_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0


def _browsing_orders(tool_uses: list[Any]) -> bool:
    """Whether this response is reading out several orders rather than opening one.

    To ask "which one did you mean?" well, the agent has to say something about
    each — so it looks both up in the same turn. Adopting the last one read as
    "the order under discussion" would answer the question it is still asking.
    """
    order_ids = {
        (getattr(block, "input", None) or {}).get("order_id")
        for block in tool_uses
        if getattr(block, "name", "") == "lookup_order"
    }
    return len(order_ids - {None}) > 1


def _assistant_text(text: str) -> dict[str, Any]:
    """A plain assistant turn, for the fallbacks: the customer was told something,
    so the transcript has to say so — otherwise the next turn is built on a
    conversation where the agent appears to have said nothing."""
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def _assistant_blocks(response: Any) -> list[dict[str, Any]]:
    """Convert an assistant response into transcript blocks.

    Keeps text and tool_use, which is what the conversation needs to continue.
    Anything else is dropped — including thinking, which is not the agent's to
    store, quote, or show.
    """
    blocks: list[dict[str, Any]] = []
    for block in response.content:
        kind = getattr(block, "type", "")
        if kind == "text":
            blocks.append({"type": "text", "text": getattr(block, "text", "")})
        elif kind == "tool_use":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "input": getattr(block, "input", None) or {},
                }
            )
    return blocks


def _text_of(response: Any) -> str:
    """Join the text blocks of a response into the customer-facing reply."""
    return "\n\n".join(
        getattr(block, "text", "")
        for block in response.content
        if getattr(block, "type", "") == "text"
    ).strip()
