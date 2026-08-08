"""The six deterministic tools, described to Anthropic and dispatched to Python.

This module is the boundary. On one side Claude asks for a tool by name with
JSON arguments; on the other are the ordinary functions from Phase 3, unchanged.
Nothing here reimplements a rule: no window arithmetic, no ownership check, no
policy precedence. Those live in `tools/` and stay there.

**The schemas are narrower than the functions.** A tool's Python signature takes
everything it needs to be safe on its own — `customer_id`, `eligibility_token`,
`confirmed`. The JSON schema shown to the model exposes only what the *customer*
decides: which order, which item, why. The rest is injected here from trusted
session state.

That is the whole design of this file. If `confirmed` were a schema field, a
model that hallucinated `confirmed=true` would satisfy `initiate_return`'s
signature; the tool would still be doing its job, and the customer would still
get a return they never asked for. So the model cannot express it. It asks to
open a return; whether the customer agreed is answered by
`agent.confirmation`, and whether a check ever passed is answered by the token
store. Both are read from `SessionState` at call time and passed in as
arguments, exactly as `initiate_return` demands.

The same reasoning applies to identity: `customer_id` is never a schema field.
An unverified session cannot produce one, so the account-scoped tools refuse
before they run rather than trusting the model not to invent one.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from agent.graph import PolicyGraphUnavailableError
from agent.state import PendingReturn, SessionState
from agent.tracing import ToolStatus, summarize
from tools import (
    ReturnBlockedError,
    check_return_eligibility,
    escalate_to_human,
    initiate_return,
    lookup_order,
    search_policy,
    verify_identity,
)

# --- What comes back from a call ----------------------------------------


@dataclass(slots=True)
class ToolOutcome:
    """The result of one dispatched tool call, ready to trace and to return.

    `content` is what Claude sees: the tool's own JSON, or a plain sentence when
    the call could not be made. `payload` is the typed object, kept so the
    orchestrator can update trusted state from it without re-parsing.
    """

    status: ToolStatus
    content: str
    summary: str
    payload: Any = None
    error: str | None = None
    args_used: dict[str, Any] = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        """Whether Claude should be told this was an error.

        A refusal is an error to the model — it means "this did not happen" —
        even when nothing went wrong with the machinery.
        """
        return self.status is not ToolStatus.OK


# --- Schemas -------------------------------------------------------------
#
# Descriptions say what a tool is *for* and when to reach for it. They do not
# restate the rules: nothing here mentions a 30-day window, a regional override,
# or a promotion, because the moment a rule is written in two places one of them
# is wrong. The model is told where the answer comes from, not what it is.

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "verify_identity",
        "description": (
            "Verify the customer by the email address on their Bookly account. Call this "
            "first, before anything account-specific: no other tool will return order or "
            "return information until it succeeds. Returns whether the email matched and, "
            "if it did, how many active orders the account has."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The email address the customer gave, exactly as they typed it.",
                }
            },
            "required": ["email"],
        },
    },
    {
        "name": "lookup_order",
        "description": (
            "Fetch one of the verified customer's orders: its status, dates, and the items "
            "on it. Everything you say about an order must come from here. Requires a "
            "verified customer. Returns nothing if the order id is not theirs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order to fetch, e.g. 'ORD-1001'.",
                }
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "search_policy",
        "description": (
            "Look up Bookly's return policy. Use this for questions about the rules in "
            "general — what the window is, whether ebooks can be returned. It does not look "
            "at an order and does not decide whether a specific return is allowed; use "
            "check_return_eligibility for that. Works without verification."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The customer's question about the rules, in their own words.",
                },
                "product_type": {
                    "type": "string",
                    "enum": ["PhysicalBook", "EBook"],
                    "description": "Set when the question is clearly about one format. Omit if not.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "check_return_eligibility",
        "description": (
            "Decide whether one item on one order can be returned, and get the explanation "
            "to give the customer. This is the only thing that decides eligibility — do not "
            "work it out yourself from dates or policy text. Requires a verified customer. "
            "Call it before ever offering to start a return."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order the item was bought on."},
                "item_id": {"type": "string", "description": "The item the customer wants to return."},
                "reason": {
                    "type": "string",
                    "description": (
                        "The customer's stated reason, in their own words, if they have given "
                        "one. It does not affect the decision — it is recorded for the return."
                    ),
                },
            },
            "required": ["order_id", "item_id"],
        },
    },
    {
        "name": "initiate_return",
        "description": (
            "Open a return. This changes Bookly's records. Only call it after "
            "check_return_eligibility said yes for this exact order and item, and after the "
            "customer has answered yes to a direct question asking them to confirm it. The "
            "tool re-checks both and will refuse if either is missing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order being returned against."},
                "item_id": {"type": "string", "description": "The item being returned."},
                "reason": {
                    "type": "string",
                    "description": "The customer's stated reason, in their own words. Do not paraphrase.",
                },
            },
            "required": ["order_id", "item_id", "reason"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": (
            "Hand the conversation to a human and stop acting. Use it when the customer asks "
            "for a person, when the request is outside what these tools cover, when a tool "
            "has failed twice on the same thing, or when the customer is distressed. Returns "
            "a case reference to read back."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why this needs a person. Context for the colleague picking it up.",
                }
            },
            "required": ["reason"],
        },
    },
]

TOOL_NAMES: frozenset[str] = frozenset(schema["name"] for schema in TOOL_SCHEMAS)

NEEDS_VERIFICATION: frozenset[str] = frozenset(
    {"lookup_order", "check_return_eligibility", "initiate_return"}
)
"""Tools that will not run for an unidentified caller.

`search_policy` is absent because the rules are public. `escalate_to_human` is
absent on purpose: someone who cannot get past verification is exactly who needs
a person.
"""


# --- Dispatch ------------------------------------------------------------


def invoke(
    name: str, args: dict[str, Any], state: SessionState, now: datetime | None = None
) -> ToolOutcome:
    """Run one tool the model asked for.

    Validates the name, injects the trusted arguments, calls the ordinary Python
    function, and packages what came back. Never raises: every failure — unknown
    tool, missing argument, blocked guard, unreachable database, an exception
    nobody predicted — comes back as an outcome the agent can tell the customer
    about. An agent that crashes mid-conversation is worse than one that says it
    could not do something.

    Args:
        name: The tool Claude asked for.
        args: Its arguments, as the model supplied them.
        state: The live session. Read for trusted values; not modified here.
        now: Clock for the return window. Defaults to the real one; tests pass a
            fixed value so the fixture scenarios keep meaning what they say.

    Returns:
        The outcome, including the arguments actually used — which is what gets
        traced, so a trace shows the real call rather than the model's half of it.
    """
    if name not in TOOL_NAMES:
        return ToolOutcome(
            status=ToolStatus.REJECTED,
            content=f"There is no tool called {name!r}.",
            summary=f"unknown tool {name!r}",
            error=f"unknown tool {name!r}",
            args_used=dict(args),
        )

    if name in NEEDS_VERIFICATION and not state.is_verified:
        # The tools enforce this themselves too — they demand a customer_id and
        # check ownership. Refusing here keeps an unverifiable call from being
        # made at all, and gives the model a reason it can act on.
        return ToolOutcome(
            status=ToolStatus.BLOCKED,
            content=(
                "The customer's identity has not been verified yet. Ask for the email "
                "address on their Bookly account and call verify_identity first."
            ),
            summary="blocked: not verified",
            error="identity not verified",
            args_used=dict(args),
        )

    missing = _missing_required(name, args)
    if missing:
        # Checked before dispatch rather than caught after: a handler that
        # reaches for a key the model never sent should not be the thing that
        # discovers the call was malformed.
        return ToolOutcome(
            status=ToolStatus.REJECTED,
            content=(
                f"That call to {name} was missing required argument(s): {', '.join(missing)}. "
                f"Supply them and try again."
            ),
            summary=f"missing arguments: {', '.join(missing)}",
            error=f"missing required argument(s): {', '.join(missing)}",
            args_used=dict(args),
        )

    handler = _HANDLERS[name]
    try:
        return handler(args, state, now)
    except (KeyError, TypeError) as exc:
        # An argument of the wrong shape — present, but not something the tool
        # can use.
        return ToolOutcome(
            status=ToolStatus.REJECTED,
            content=f"That call was malformed: {exc}. Check the tool's arguments and try again.",
            summary="malformed arguments",
            error=f"{type(exc).__name__}: {exc}",
            args_used=dict(args),
        )
    except PolicyGraphUnavailableError as exc:
        # Neo4j is down. There is no fallback and there must not be one: a
        # guessed policy is worse than an admitted outage.
        return ToolOutcome(
            status=ToolStatus.ERROR,
            content=(
                "The policy database is unavailable, so this cannot be answered right now. "
                "Tell the customer you can't confirm the policy at the moment and offer to "
                "hand them to a colleague. Do not state a policy from memory."
            ),
            summary="policy graph unavailable",
            error=str(exc),
            args_used=dict(args),
        )
    except ReturnBlockedError as exc:
        # A guard refused. Nothing was written. The message names the guard.
        return ToolOutcome(
            status=ToolStatus.BLOCKED,
            content=f"The return was not opened: {exc}",
            summary="return blocked",
            error=str(exc),
            args_used=dict(args),
        )
    except Exception as exc:  # noqa: BLE001 - the loop must survive any tool
        return ToolOutcome(
            status=ToolStatus.ERROR,
            content=(
                f"That tool failed: {type(exc).__name__}. Do not assume it succeeded. "
                f"Tell the customer something went wrong and offer to hand them to a colleague."
            ),
            summary=f"error: {type(exc).__name__}",
            error=f"{type(exc).__name__}: {exc}",
            args_used=dict(args),
        )


_REQUIRED_ARGS: dict[str, tuple[str, ...]] = {
    schema["name"]: tuple(schema["input_schema"].get("required", ())) for schema in TOOL_SCHEMAS
}
"""Required arguments per tool, read off the schemas so the two cannot drift."""


def _missing_required(name: str, args: dict[str, Any]) -> list[str]:
    """Required arguments the model left out or left blank."""
    return [key for key in _REQUIRED_ARGS[name] if not args.get(key)]


def invoke_timed(
    name: str, args: dict[str, Any], state: SessionState, now: datetime | None = None
) -> tuple[ToolOutcome, float]:
    """`invoke`, with the wall-clock time it took.

    Returns:
        The outcome and the elapsed milliseconds, for the trace.
    """
    started = time.perf_counter()
    outcome = invoke(name, args, state, now)
    return outcome, (time.perf_counter() - started) * 1000


# --- Handlers ------------------------------------------------------------
#
# One per tool. Each injects the trusted arguments and calls straight through.


def _verify_identity(args: dict, state: SessionState, now: datetime | None) -> ToolOutcome:
    used = {"email": args["email"]}
    result = verify_identity(**used)
    return _ok(result, used)


def _lookup_order(args: dict, state: SessionState, now: datetime | None) -> ToolOutcome:
    used = {"order_id": args["order_id"], "customer_id": state.verified_customer_id}
    result = lookup_order(**used, now=now)
    if result is None:
        # Not found and not-yours are the same answer, deliberately — see
        # `lookup_order`. Saying which would leak whether an order id is real.
        return ToolOutcome(
            status=ToolStatus.OK,
            content=f"No order {args['order_id']} on this customer's account.",
            summary="no matching order",
            payload=None,
            args_used=used,
        )
    return _ok(result, used)


def _search_policy(args: dict, state: SessionState, now: datetime | None) -> ToolOutcome:
    used: dict[str, Any] = {"query": args["query"]}
    if args.get("product_type"):
        used["product_type"] = args["product_type"]
    # Region comes from the verified account, never from the model. A customer's
    # country decides which regional overrides they can be offered, so it is not
    # something to infer from how they phrased a question.
    if state.customer_region:
        used["country"] = state.customer_region
    return _ok(search_policy(**used), used)


def _check_return_eligibility(args: dict, state: SessionState, now: datetime | None) -> ToolOutcome:
    used = {
        "order_id": args["order_id"],
        "item_id": args["item_id"],
        "customer_id": state.verified_customer_id,
    }
    decision = check_return_eligibility(**used, now=now)
    return ToolOutcome(
        status=ToolStatus.OK,
        # The token is withheld from the model. It has no use for one — the
        # orchestrator supplies it to `initiate_return` from session state — and
        # a credential in the transcript is a credential the model could repeat
        # back to the customer. It learns the decision, not the key to it.
        content=decision.model_dump_json(exclude={"eligibility_token"}),
        summary=summarize(decision),
        payload=decision,
        args_used=used,
    )


def _initiate_return(args: dict, state: SessionState, now: datetime | None) -> ToolOutcome:
    # The three arguments the model does not get to supply. Read from state at
    # the moment of the call, so a token cleared by an item switch is gone and a
    # confirmation the customer never gave is False.
    used = {
        "order_id": args["order_id"],
        "item_id": args["item_id"],
        "customer_id": state.verified_customer_id,
        "reason": args["reason"],
        "eligibility_token": state.eligibility_token or "",
        "confirmed": state.confirmed,
    }
    return _ok(initiate_return(**used), used)


def _escalate_to_human(args: dict, state: SessionState, now: datetime | None) -> ToolOutcome:
    used = {
        "reason": args["reason"],
        "customer_id": state.verified_customer_id,
        "order_id": state.active_order_id,
    }
    return _ok(escalate_to_human(**used), used)


_HANDLERS: dict[str, Callable[[dict, SessionState, datetime | None], ToolOutcome]] = {
    "verify_identity": _verify_identity,
    "lookup_order": _lookup_order,
    "search_policy": _search_policy,
    "check_return_eligibility": _check_return_eligibility,
    "initiate_return": _initiate_return,
    "escalate_to_human": _escalate_to_human,
}


def _ok(result: BaseModel, used: dict[str, Any]) -> ToolOutcome:
    """Package a successful call. The model gets the tool's own JSON."""
    return ToolOutcome(
        status=ToolStatus.OK,
        content=result.model_dump_json(),
        summary=summarize(result),
        payload=result,
        args_used=used,
    )


# --- Trusted state updates ----------------------------------------------


def apply_to_state(
    name: str, args: dict[str, Any], outcome: ToolOutcome, state: SessionState
) -> None:
    """Update session state from a tool result — and only from a tool result.

    Called after a successful call, never after a blocked or failed one. This is
    the only place the trusted fields are written outside the confirmation check,
    which keeps the answer to "how did the agent come to believe this?" short:
    a tool said so.

    Nothing here trusts the model's arguments on their own. `verified_customer_id`
    comes from what `verify_identity` returned, not from the email that was
    passed in; `eligibility_token` comes from the decision object, not from a
    field Claude filled in.

    Args:
        name: The tool that ran.
        args: The model's own arguments. Read only for things the tool does not
            take but the conversation established — the customer's stated
            reason. Never for a value a tool result can supply.
        outcome: What it returned.
        state: The session to update, in place.
    """
    if outcome.status is not ToolStatus.OK or outcome.payload is None:
        return

    payload = outcome.payload

    if name == "verify_identity" and payload.verified:
        state.verified_customer_id = payload.customer_id
        state.customer_region = payload.region
        state.active_order_ids = list(payload.active_order_ids)
        # One live order is not ambiguous, so adopt it. Two or more and the
        # agent has to ask — leaving `active_order_id` None is what makes the
        # order-specific tools unusable until it does.
        if len(payload.active_order_ids) == 1:
            state.active_order_id = payload.active_order_ids[0]

    elif name == "lookup_order":
        state.active_order_id = payload.order.order_id

    elif name == "check_return_eligibility":
        order_id = outcome.args_used["order_id"]
        item_id = outcome.args_used["item_id"]

        # A new item means the old attempt is over. Clearing first drops the
        # previous token, decision, and any confirmation, so nothing from the
        # last item can be spent on this one.
        if state.active_item_id not in (None, item_id) or state.active_order_id != order_id:
            state.clear_return_context()

        state.active_order_id = order_id
        state.active_item_id = item_id
        state.eligibility = payload
        state.eligibility_token = payload.eligibility_token
        if args.get("reason"):
            state.return_reason = args["reason"]

        if payload.eligible and payload.eligibility_token:
            # There is now something a "yes" could authorise. It does not count
            # until the agent has actually asked — see `agent.confirmation`.
            state.pending_return = PendingReturn(
                order_id=order_id, item_id=item_id, eligibility_token=payload.eligibility_token
            )
        else:
            state.pending_return = None
            state.confirmed = False

    elif name == "initiate_return":
        # Done either way: `created=False` means a return already existed, and
        # the customer's request is satisfied. Clearing stops a second "yes"
        # later in the conversation from meaning anything.
        state.clear_return_context()

    elif name == "escalate_to_human":
        state.escalated = True
