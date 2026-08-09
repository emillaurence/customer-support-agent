"""The six things the Bookly agent can do, and the boundary Claude reaches them through.

    verify_identity           who the customer is
    lookup_order              the orders they own, or one of them in detail
    search_policy             what the rules are          (body in policy/policy.py)
    check_return_eligibility  whether one item can go back
    initiate_return           the only write
    escalate_to_human         the escape hatch

Ordinary Python functions. Every one works with no model involved, which is what
the tests exercise.

**The schemas are narrower than the functions.** A tool's signature takes
everything it needs to be safe on its own — `customer_id`, `eligibility_token`,
`confirmed` — while the JSON schema exposes only what the *customer* decides:
which order, which item, why. The rest is injected by `invoke_tool` from trusted
session state. If `confirmed` were a schema field, a model that hallucinated
`confirmed=true` would satisfy `initiate_return`'s signature and the customer
would get a return they never asked for. So the model cannot express it — and an
unverified session cannot produce a `customer_id`.
"""

from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent.state import (
    Customer,
    EligibilityDecision,
    Item,
    Order,
    OrderStatus,
    PendingReturn,
    ReturnRecord,
    ReturnStatus,
    SessionState,
    ToolStatus,
    summarize,
)
from policy.graph import PolicyGraphUnavailableError
from policy.policy import (
    PolicyContext,
    ProductType,
    applicable_policies,
    build_rule_path,
    resolve_region,
    search_policy,
)

__all__ = [
    "TOOL_SCHEMAS",
    "CustomerOrders",
    "ItemSummary",
    "OrderSummary",
    "ReturnBlockedError",
    "ToolOutcome",
    "active_order_ids",
    "apply_tool_result",
    "check_return_eligibility",
    "escalate_to_human",
    "initiate_return",
    "invoke_tool",
    "lookup_order",
    "reset_demo",
    "search_policy",
    "verify_identity",
]


# --- The mock transactional data ----------------------------------------
#
# Customers, orders, items, and returns stand in for Bookly's order system;
# policy is not here, it lives in Neo4j. Every call re-reads from disk — the
# store is a few kilobytes, and a write from `initiate_return` is visible to the
# next read with no cache to invalidate.
#
# `DATA_DIR` is a module attribute so tests can point it at a temporary copy and
# exercise the write path without touching the repo's fixtures.

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

RETURNS_FILE = "returns.json"

SEED_SUBDIR = "seed"
"""Pristine copies of the mutable files, for the demo reset to restore.

Only `returns.json` is written at runtime, so only it has a baseline. Kept inside
the data directory so pointing `DATA_DIR` at a temporary copy moves the baseline
with it.
"""


def _read(filename: str) -> list[dict]:
    return json.loads((DATA_DIR / filename).read_text())


def _load_customers() -> list[Customer]:
    return [Customer.model_validate(row) for row in _read("customers.json")]


def _load_items() -> dict[str, Item]:
    """The catalogue, keyed by item id — line items are resolved against it."""
    return {row["item_id"]: Item.model_validate(row) for row in _read("items.json")}


def _load_orders() -> list[Order]:
    return [Order.model_validate(row) for row in _read("orders.json")]


def _load_returns() -> list[ReturnRecord]:
    """Every return on record, including ones opened during this session."""
    return [ReturnRecord.model_validate(row) for row in _read(RETURNS_FILE)]


def _append_return(record: ReturnRecord) -> None:
    """Add one return to the store. Read-modify-write on one small file: not
    concurrency-safe, and not meant to be."""
    path = DATA_DIR / RETURNS_FILE
    rows = json.loads(path.read_text())
    rows.append(json.loads(record.model_dump_json(exclude_none=True)))
    path.write_text(json.dumps(rows, indent=2) + "\n")


def _next_return_id() -> str:
    """Mint the next id in the fixture's RET-5001 series, so it reads like an RMA."""
    numbers = [
        int(record.return_id.removeprefix("RET-"))
        for record in _load_returns()
        if record.return_id.removeprefix("RET-").isdigit()
    ]
    return f"RET-{max(numbers, default=5000) + 1}"


# --- Eligibility tokens --------------------------------------------------
#
# The token carries no meaning of its own: it is an opaque uuid4, and everything
# that matters about it — which customer, order, item, and policy — is held here,
# server-side. A model that invents a plausible-looking token finds nothing on
# the other end; one that replays a real token for a different item finds a grant
# that does not match. No JWT: an unguessable key into a server-side record is
# the simpler equivalent.
#
# In-memory, so tokens do not survive a restart. An eligibility decision is only
# meaningful inside the conversation that produced it.


@dataclass(frozen=True)
class EligibilityGrant:
    """What one token permits: exactly one item, on one order, for one customer."""

    token: str
    customer_id: str
    order_id: str
    item_id: str
    policy_id: str
    issued_at: datetime


_GRANTS: dict[str, EligibilityGrant] = {}


def _create_eligibility_token(
    customer_id: str, order_id: str, item_id: str, policy_id: str
) -> str:
    grant = EligibilityGrant(
        token=str(uuid.uuid4()),
        customer_id=customer_id,
        order_id=order_id,
        item_id=item_id,
        policy_id=policy_id,
        issued_at=datetime.now(UTC),
    )
    _GRANTS[grant.token] = grant
    return grant.token


def _validate_eligibility_token(
    token: str, customer_id: str, order_id: str, item_id: str
) -> None:
    """Raise unless this server issued `token` for exactly this request.

    Raises:
        ReturnBlockedError: If the token is unknown, or was issued for a
            different customer, order, or item.
    """
    grant = _GRANTS.get(token)
    if grant is None:
        # An invented, expired, or copied-from-somewhere token looks exactly like
        # this: nothing on the server was ever issued under it.
        raise ReturnBlockedError(
            "eligibility_token is not a token this server issued. "
            "Run check_return_eligibility and use the token it returns."
        )
    if (grant.customer_id, grant.order_id, grant.item_id) != (customer_id, order_id, item_id):
        raise ReturnBlockedError(
            f"eligibility_token was issued for {grant.customer_id} / {grant.order_id} / "
            f"{grant.item_id}, not {customer_id} / {order_id} / {item_id}. "
            "A token cannot be spent on a different customer, order, or item."
        )
    # A grant exists only on the eligible path of check_return_eligibility, so
    # finding one *is* the proof that a check said yes.


def _clear_eligibility_tokens() -> None:
    """Drop every grant. Used by the demo reset and between tests."""
    _GRANTS.clear()


# --- 1. verify_identity --------------------------------------------------


class VerifyIdentityResult(BaseModel):
    """Who the customer is, and which region's policy applies to them.

    Identity and nothing else. Which orders they have is a question about orders,
    and `lookup_order` answers it — a verification that also listed orders was two
    tools wearing one name, and the caller could not ask either question on its
    own.
    """

    verified: bool
    customer_id: str | None = None
    name: str | None = Field(default=None, description="First name, for addressing them.")
    region: str | None = None
    message: str = ""


def verify_identity(email: str) -> VerifyIdentityResult:
    """Verify a customer by the email address on their account.

    **Mocked, deliberately.** A matching email is not production authentication —
    an email address is a public identifier. Real authentication belongs outside
    the agent. The shape is the part worth keeping: identity is established once,
    by a tool, and every tool that reads customer data demands the resulting
    `customer_id`.
    """
    normalised = email.strip().lower()
    if not normalised:
        return VerifyIdentityResult(
            verified=False, message="I need the email address on the account to look you up."
        )

    customer = next(
        (c for c in _load_customers() if c.email.strip().lower() == normalised), None
    )
    if customer is None:
        # Says only that it did not match. Confirming that an address is *not* a
        # Bookly customer is itself a disclosure.
        return VerifyIdentityResult(
            verified=False,
            message="I can't find an account for that email address. Could you double-check it?",
        )

    first_name = customer.name.split()[0]
    return VerifyIdentityResult(
        verified=True,
        customer_id=customer.customer_id,
        name=first_name,
        region=customer.country,
        message=f"Thanks {first_name} — I've found your account.",
    )


# --- 2. lookup_order -----------------------------------------------------
#
# Two modes, one tool, one ownership rule. Without an order id it lists the
# customer's live orders; with one it reads that order in full. Both start from
# `customer_id`, so neither can answer for someone else's account.


class ItemSummary(BaseModel):
    """One line item, named the way a customer would name it and identified the
    way a tool needs it."""

    item_id: str
    title: str
    product_type: ProductType


class OrderSummary(BaseModel):
    """Just enough of an order to find the one the customer means.

    The id to name it by, the status they can already see, and the items on it
    with their ids — so a title the customer named resolves to something
    actionable without a second call. **Not an order**: no dates, no prices, no
    quantities, no promotion. Anything the agent *says* about an order comes from
    the detailed mode.
    """

    order_id: str
    status: OrderStatus
    items: list[ItemSummary] = Field(default_factory=list, description="In catalogue order.")


class CustomerOrders(BaseModel):
    """Every order the customer could be talking about, newest first."""

    orders: list[OrderSummary] = Field(default_factory=list)


class Shipment(BaseModel):
    """Where the order is. Nothing is estimated: an order with no `delivered_at`
    reports None, not a guessed arrival date."""

    status: OrderStatus
    placed_at: date
    delivered_at: date | None = None
    has_arrived: bool = Field(description="True only when a delivery date is on record.")
    days_since_delivery: int | None = Field(
        default=None, description="None until it has arrived. No window has started before then."
    )


class OrderDetails(BaseModel):
    """An order, its line items resolved to catalogue entries, and its shipment."""

    order: Order
    items: list[Item]
    shipment: Shipment


def lookup_order(
    customer_id: str, order_id: str | None = None, now: datetime | None = None
) -> CustomerOrders | OrderDetails | None:
    """Look up a verified customer's orders, or one of them in detail.

    Omit `order_id` to discover what they have: their live orders, each with the
    items on it, which is what turns "Clean Architecture" into an order id and an
    item id. Pass one to read that order in full.

    `customer_id` is required and comes first, not optional: ownership is enforced
    here rather than trusted to the caller, so a hallucinated or overheard order
    number cannot read out someone else's order. A real order owned by someone
    else returns None — the same answer as a number that does not exist, so the
    response cannot be used to discover whether an id is real.
    """
    if order_id is None:
        return CustomerOrders(orders=_order_summaries(customer_id))

    order = next(
        (
            candidate
            for candidate in _load_orders()
            if candidate.order_id == order_id and candidate.customer_id == customer_id
        ),
        None,
    )
    if order is None:
        return None

    catalogue = _load_items()
    today = (now or datetime.now(UTC)).date()
    arrived = order.delivered_at is not None

    return OrderDetails(
        # `scenario` is a note to whoever reads the JSON. Dropped here so it can
        # never reach the model, let alone the customer.
        order=order.model_copy(update={"scenario": None}),
        items=[catalogue[line.item_id] for line in order.items if line.item_id in catalogue],
        shipment=Shipment(
            status=order.status,
            placed_at=order.placed_at,
            delivered_at=order.delivered_at,
            has_arrived=arrived,
            days_since_delivery=(today - order.delivered_at).days if arrived else None,
        ),
    )


def _order_summaries(customer_id: str) -> list[OrderSummary]:
    """The customer's live (not cancelled) orders, most recently placed first.

    Sorted so the agent has a sensible order to read them in. One pass over the
    catalogue for the whole list, rather than a read per order.
    """
    orders = [
        order
        for order in _load_orders()
        if order.customer_id == customer_id and order.status != OrderStatus.CANCELLED
    ]
    orders.sort(key=lambda order: order.placed_at, reverse=True)

    catalogue = _load_items() if orders else {}
    return [
        OrderSummary(
            order_id=order.order_id,
            status=order.status,
            items=[
                ItemSummary(
                    item_id=item.item_id, title=item.title, product_type=item.product_type
                )
                for line in order.items
                if (item := catalogue.get(line.item_id)) is not None
            ],
        )
        for order in orders
    ]


def active_order_ids(customer_id: str) -> list[str]:
    """The ids of the customer's live orders, most recently placed first."""
    return [summary.order_id for summary in _order_summaries(customer_id)]


def _item_on_order(order: Order, item_id: str) -> bool:
    """Whether an item was actually bought on an order. Asked by both the
    eligibility check and the write, each of which refuses on its own."""
    return any(line.item_id == item_id for line in order.items)


# --- 3. search_policy ----------------------------------------------------
#
# Imported above from `policy/policy.py`. A tool like the other five — schema
# below, handler in the dispatch table — but its body belongs with the policy
# layer, so that it and check_return_eligibility cannot drift into two ideas of
# which policy governs.


# --- 4. check_return_eligibility ----------------------------------------

BLOCKING_RETURN_STATUSES = frozenset(
    {ReturnStatus.REQUESTED, ReturnStatus.APPROVED, ReturnStatus.COMPLETED}
)
"""Statuses that mean a return already exists. A rejected one does not block a retry."""


def check_return_eligibility(
    order_id: str, item_id: str, customer_id: str, now: datetime | None = None
) -> EligibilityDecision:
    """Decide if one item on one order is returnable, and say why.

    Which policy governs comes from `policy.applicable_policies` — the same
    selection `search_policy` uses — so an informational answer about Australia
    and a decision about an Australian customer's order cannot disagree. Returns
    the decision, its policy, a customer-facing explanation, the graph hops behind
    it, and — only when eligible — a token.

    Raises:
        PolicyGraphUnavailableError: If Neo4j is unconfigured or unreachable. Never
            returns a mocked or defaulted decision instead.
    """
    today = (now or datetime.now(UTC)).date()

    customer = next((c for c in _load_customers() if c.customer_id == customer_id), None)
    if customer is None:
        return _refuse("I need to verify your account before I can look at a return.")

    # Ownership is enforced here too, not assumed from the caller.
    details = lookup_order(customer_id, order_id, now=now)
    if details is None:
        return _refuse(f"I can't find order {order_id} on your account.")

    if not _item_on_order(details.order, item_id):
        return _refuse(f"{item_id} isn't on order {order_id}.")

    item = _load_items().get(item_id)
    if item is None:
        return _refuse(f"I can't find {item_id} in the catalogue.")

    applicable = applicable_policies(
        item.product_type.value,
        PolicyContext(
            region=customer.country,
            promotion_code=details.order.promotion_code,
            placed_at=details.order.placed_at,
        ),
    )
    if not applicable:
        return _refuse("I can't find a return policy covering that item.")

    winner = applicable[0]
    policy = winner.policy
    rule_path = build_rule_path(
        item.product_type.value, policy.policy_id, winner.granted_by_region, winner.outranks
    )

    existing = _open_return(order_id, item_id)
    if existing is not None:
        return EligibilityDecision(
            eligible=False,
            policy_id=policy.policy_id,
            explanation=(
                f"There's already a return open for {item.title} on order {order_id} "
                f"({existing.return_id}), so I can't start a second one."
            ),
            rule_path=rule_path,
        )

    # No window is not a window of zero days: it means returns are not offered,
    # and no region or promotion can rescue it.
    if policy.window_days is None:
        return EligibilityDecision(
            eligible=False,
            policy_id=policy.policy_id,
            explanation=(
                f"{item.title} is a digital item, and "
                f"{policy.summary[0].lower()}{policy.summary[1:]}"
            ),
            rule_path=rule_path,
        )

    if details.order.delivered_at is None:
        return EligibilityDecision(
            eligible=False,
            policy_id=policy.policy_id,
            explanation=(
                f"Order {order_id} hasn't arrived yet, so the {policy.window_days}-day "
                f"return window hasn't started. You can return it once it's delivered."
            ),
            rule_path=rule_path,
        )

    days_since_delivery = (today - details.order.delivered_at).days
    if days_since_delivery > policy.window_days:
        return EligibilityDecision(
            eligible=False,
            policy_id=policy.policy_id,
            explanation=(
                f"{item.title} was delivered {days_since_delivery} days ago, and the return "
                f"window for it is {policy.window_days} days, so it's outside the window by "
                f"{days_since_delivery - policy.window_days} days."
            ),
            rule_path=rule_path,
            days_remaining=0,
        )

    days_remaining = policy.window_days - days_since_delivery
    return EligibilityDecision(
        eligible=True,
        policy_id=policy.policy_id,
        explanation=(
            f"{item.title} was delivered {days_since_delivery} days ago and the return window "
            f"is {policy.window_days} days, so it can be returned — you have {days_remaining} "
            f"day{'s' if days_remaining != 1 else ''} left."
        ),
        rule_path=rule_path,
        eligibility_token=_create_eligibility_token(
            customer_id, order_id, item_id, policy.policy_id
        ),
        days_remaining=days_remaining,
    )


def _refuse(explanation: str) -> EligibilityDecision:
    """A refusal that never reached a policy, so it carries no policy and no token."""
    return EligibilityDecision(eligible=False, explanation=explanation)


def _open_return(order_id: str, item_id: str) -> ReturnRecord | None:
    return next(
        (
            record
            for record in _load_returns()
            if record.order_id == order_id
            and record.item_id == item_id
            and record.status in BLOCKING_RETURN_STATUSES
        ),
        None,
    )


# --- 5. initiate_return --------------------------------------------------


class ReturnResult(BaseModel):
    """`created=False` means a return already existed and is being reported back
    rather than duplicated — the customer's request is satisfied either way, and
    Bookly's records are unchanged."""

    created: bool
    return_record: ReturnRecord
    message: str = ""


class ReturnBlockedError(ValueError):
    """A guard failed, so nothing was written. The message names the guard, so the
    failure is debuggable; the agent rephrases it for the customer."""


def initiate_return(
    order_id: str,
    item_id: str,
    customer_id: str,
    reason: str,
    eligibility_token: str,
    confirmed: bool,
) -> ReturnResult:
    """Open a return. The only write in the tool set.

    Its preconditions are part of its signature rather than something the caller
    is trusted to have arranged. It does not trust that the agent checked first —
    the token is required, and is valid only for the exact customer, order, and
    item it was issued against — nor that the agent asked first: `confirmed` must
    be passed in explicitly. `SessionState.confirmed` is loop bookkeeping, and a
    tool that mutates data must not depend on it being correct.

    The guards run in the order below, and every one must pass before anything is
    written. The last is the exception that does not raise: an existing return
    means the customer is asking for something they already have, so the existing
    RMA comes back with `created=False`. Nothing is written either way, so
    repeated requests cannot produce a second RMA.

    Raises:
        ReturnBlockedError: If any guard before that one fails. Nothing is written.
    """
    # Confirmation first: the cheapest check, and the one whose absence is least
    # recoverable, since a return the customer never asked for is visible to them.
    if not confirmed:
        raise ReturnBlockedError(
            "explicit customer confirmation required: initiate_return was called with "
            "confirmed=False. Ask the customer to confirm the return, then call again."
        )

    if not eligibility_token:
        raise ReturnBlockedError(
            "no eligibility_token: run check_return_eligibility first and pass the token it issues."
        )

    customer = next((c for c in _load_customers() if c.customer_id == customer_id), None)
    if customer is None:
        raise ReturnBlockedError(f"unknown customer {customer_id}: identity is not verified.")

    details = lookup_order(customer_id, order_id)
    if details is None:
        raise ReturnBlockedError(
            f"order {order_id} does not belong to {customer_id}, or does not exist."
        )

    if not _item_on_order(details.order, item_id):
        raise ReturnBlockedError(f"item {item_id} is not on order {order_id}.")

    _validate_eligibility_token(eligibility_token, customer_id, order_id, item_id)

    # Checked last, immediately before the write, so the window between the check
    # and the append is as small as it can be.
    existing = _open_return(order_id, item_id)
    if existing is not None:
        return ReturnResult(
            created=False,
            return_record=existing,
            message=(
                f"There's already a return open for that item — reference {existing.return_id}. "
                f"I haven't started a second one."
            ),
        )

    record = ReturnRecord(
        return_id=_next_return_id(),
        order_id=order_id,
        item_id=item_id,
        customer_id=customer_id,
        status=ReturnStatus.REQUESTED,
        reason=reason,
        created_at=datetime.now(UTC),
    )
    _append_return(record)

    return ReturnResult(
        created=True,
        return_record=record,
        message=(
            f"Your return is open — reference {record.return_id}. "
            f"You'll get an email with the next steps."
        ),
    )


# --- 6. escalate_to_human ------------------------------------------------


class EscalationResult(BaseModel):
    """Confirmation that a handoff was queued."""

    case_id: str = Field(description="Reference the customer can quote, e.g. 'CASE-3F2A9C41'.")
    reason: str = Field(description="Context for the human, not for the customer.")
    customer_id: str | None = None
    order_id: str | None = None
    created_at: datetime
    message: str


def escalate_to_human(
    reason: str, customer_id: str | None = None, order_id: str | None = None
) -> EscalationResult:
    """Queue the conversation for a human.

    Mocked. What matters is that the escape hatch exists and is cheap to reach: an
    agent with no way to say "a person will take this" ends up guessing at
    consumer law. Works without identity — whoever failed verification is exactly
    who needs a human.
    """
    case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
    return EscalationResult(
        case_id=case_id,
        reason=reason,
        customer_id=customer_id,
        order_id=order_id,
        created_at=datetime.now(UTC),
        message=(
            f"I'm passing this to a colleague who can help — your reference is {case_id}. "
            f"They'll follow up by email."
        ),
    )


# --- What Claude is shown ------------------------------------------------
#
# Each description answers three questions and stops: when to reach for the tool,
# what it needs, what it gives back. They do not restate the rules — nothing here
# mentions a 30-day window, a regional override, or a promotion, because the
# moment a rule is written in two places one of them is wrong — and they do not
# restate the guards, because the guards are enforced in Python and their refusal
# messages say what went wrong at the moment it matters.
#
# **This list is a literal, and its order is fixed.** Tool definitions render
# first in the request, ahead of the system prompt and the conversation, so a
# schema built per call — or serialized in a different order — would change the
# first bytes of the prompt and invalidate the cache for everything after it.
# Nothing below is computed.

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "verify_identity",
        "description": (
            "Identify the customer by the email address on their Bookly account. Required "
            "before anything account-specific. Returns whether it matched and, if so, their "
            "first name and region. It says nothing about their orders — use lookup_order "
            "for those."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The email address the customer gave, as they typed it.",
                }
            },
            "required": ["email"],
        },
    },
    {
        "name": "lookup_order",
        "description": (
            "The verified customer's orders. Omit order_id to list them all with the items on "
            "each and their item ids — use that to find the book they named. Pass order_id to "
            "read one order in full: dates, delivery, and status. Everything you say about an "
            "order comes from here."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": (
                        "One order to read in detail, e.g. 'ORD-1001'. Omit to list their "
                        "orders instead."
                    ),
                }
            },
            "required": [],
        },
    },
    {
        "name": "search_policy",
        "description": (
            "Look up Bookly's return rules — the window, whether ebooks can be returned, what "
            "applies in a given country. Informational only: it never looks at an order. Pass "
            "the customer's own wording, including any country they named; the region is "
            "resolved for you. Answer from `resolved` and `region_note`, never from your own "
            "knowledge of a country's rules."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The customer's question about the rules, in their own words, "
                        "including any country they named."
                    ),
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
            "Decide whether one item on one order can be returned, and get the explanation to "
            "give the customer. The only thing that decides eligibility — never work it out "
            "from dates or policy text yourself. Call it before offering to start a return."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order the item was bought on."},
                "item_id": {"type": "string", "description": "The item the customer wants to return."},
                "reason": {
                    "type": "string",
                    "description": (
                        "The customer's stated reason, in their words, if they gave one. It "
                        "does not affect the decision; it is recorded on the return."
                    ),
                },
            },
            "required": ["order_id", "item_id"],
        },
    },
    {
        "name": "initiate_return",
        "description": (
            "Open a return. Changes Bookly's records. Call it only after "
            "check_return_eligibility said yes for this exact order and item and the customer "
            "has answered yes to a direct question asking them to confirm. Returns the RMA "
            "reference and its status."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order being returned against."},
                "item_id": {"type": "string", "description": "The item being returned."},
                "reason": {
                    "type": "string",
                    "description": (
                        "The customer's stated reason, in their words, if they gave one. Do "
                        "not paraphrase it, and do not hold the return up to ask for one."
                    ),
                },
            },
            "required": ["order_id", "item_id"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": (
            "Hand the conversation to a colleague and stop acting. Use it when the customer "
            "asks for a person, the request is outside these tools, a tool has failed twice "
            "on the same thing, or the customer is distressed. Returns a case reference to "
            "read back."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why this needs a person. Context for the colleague.",
                }
            },
            "required": ["reason"],
        },
    },
]

TOOL_NAMES: frozenset[str] = frozenset(schema["name"] for schema in TOOL_SCHEMAS)

REQUIRED_ARGS: dict[str, tuple[str, ...]] = {
    schema["name"]: tuple(schema["input_schema"].get("required", ())) for schema in TOOL_SCHEMAS
}
"""Read off the schemas, so the two cannot drift."""

NEEDS_VERIFICATION: frozenset[str] = frozenset(
    {"lookup_order", "check_return_eligibility", "initiate_return"}
)
"""Tools that will not run for an unidentified caller.

`search_policy` is absent because the rules are public. `escalate_to_human` is
absent on purpose: someone who cannot get past verification is exactly who needs
a person.
"""


# --- Dispatch ------------------------------------------------------------


@dataclass(slots=True)
class ToolOutcome:
    """The result of one dispatched call, ready to trace and to return.

    `content` is what Claude sees; `payload` is the typed object, so the loop can
    update trusted state without re-parsing; `args_used` is the call as it was
    actually made, trusted values included, which is what gets traced.
    """

    status: ToolStatus
    content: str
    summary: str
    payload: Any = None
    error: str | None = None
    args_used: dict[str, Any] = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        """A refusal is an error to the model — it means "this did not happen" —
        even when nothing went wrong with the machinery."""
        return self.status is not ToolStatus.OK


def invoke_tool(
    name: str, args: dict[str, Any], state: SessionState, now: datetime | None = None
) -> ToolOutcome:
    """Run one tool the model asked for.

    Validates the name, injects the trusted arguments, calls the function, and
    packages what came back. Never raises: every failure — unknown tool, missing
    argument, blocked guard, unreachable database, an exception nobody predicted —
    comes back as an outcome the agent can tell the customer about. `state` is
    read for trusted values and not modified here.
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
        # The tools enforce this themselves too. Refusing here keeps an
        # unverifiable call from being made at all, and gives the model a reason
        # it can act on.
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

    missing = [key for key in REQUIRED_ARGS[name] if not args.get(key)]
    if missing:
        # Checked before dispatch: a handler reaching for a key the model never
        # sent should not be the thing that discovers the call was malformed.
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

    try:
        return _HANDLERS[name](args, state, now)
    except (KeyError, TypeError) as exc:
        return ToolOutcome(
            status=ToolStatus.REJECTED,
            content=f"That call was malformed: {exc}. Check the tool's arguments and try again.",
            summary="malformed arguments",
            error=f"{type(exc).__name__}: {exc}",
            args_used=dict(args),
        )
    except PolicyGraphUnavailableError as exc:
        # Neo4j is down. There is no fallback and there must not be one: a guessed
        # policy is worse than an admitted outage.
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


def _ok(result: BaseModel, used: dict[str, Any], view: Any | None = None) -> ToolOutcome:
    """Package a successful call.

    Two representations of one result, deliberately. `payload` is the full typed
    object — what Python validates against, updates trusted state from, and shows
    in the trace. `content` is the *model's* view: the same facts with the fields
    Claude has no use for left out.

    That split is a latency and cost decision, not a safety one. A tool result
    goes into the transcript and is re-sent on every subsequent turn of the
    conversation, so a field nobody reads is paid for repeatedly. Nothing is
    summarised or rounded — fields are present or absent.

    Args:
        view: The compact projection. When omitted the whole result is sent,
            which is right for the tools that are already small.
    """
    return ToolOutcome(
        status=ToolStatus.OK,
        content=json.dumps(view, default=str) if view is not None else result.model_dump_json(),
        summary=summarize(result),
        payload=result,
        args_used=used,
    )


# One handler per tool. Each injects the trusted arguments and calls straight
# through — no rule is re-implemented here.


def _handle_verify_identity(args: dict, state: SessionState, now) -> ToolOutcome:
    used = {"email": args["email"]}
    return _ok(verify_identity(**used), used)


def _handle_lookup_order(args: dict, state: SessionState, now) -> ToolOutcome:
    customer_id = state.verified_customer_id

    if not args.get("order_id"):
        # Discovery. Already the compact shape — ids, statuses, and the titles a
        # customer would recognise — so there is nothing to project away.
        orders = lookup_order(customer_id)
        return ToolOutcome(
            status=ToolStatus.OK,
            content=orders.model_dump_json(),
            summary=f"{len(orders.orders)} order(s)",
            payload=orders,
            args_used={"customer_id": customer_id},
        )

    used = {"customer_id": customer_id, "order_id": args["order_id"]}
    result = lookup_order(**used, now=now)
    if result is None:
        # Not found and not-yours are the same answer, deliberately. Saying which
        # would leak whether an order id is real.
        return ToolOutcome(
            status=ToolStatus.OK,
            content=f"No order {args['order_id']} on this customer's account.",
            summary="no matching order",
            args_used=used,
        )
    # What a support conversation is actually about: where the order is, when it
    # arrived, and what is on it. The customer id is the session's, the prices are
    # not in dispute, and `has_arrived` is `delivered_at` asked twice.
    view = {
        "order_id": result.order.order_id,
        "status": result.shipment.status.value,
        "placed_at": result.shipment.placed_at,
        "delivered_at": result.shipment.delivered_at,
        "days_since_delivery": result.shipment.days_since_delivery,
        "items": [
            {"item_id": item.item_id, "title": item.title, "product_type": item.product_type.value}
            for item in result.items
        ],
    }
    return _ok(result, used, view)


def _handle_search_policy(args: dict, state: SessionState, now) -> ToolOutcome:
    used: dict[str, Any] = {"query": args["query"]}
    if args.get("product_type"):
        used["product_type"] = args["product_type"]
    # The region is resolved deterministically, from the customer's own words and
    # from trusted session state — never from a country code the model produced.
    # Treating session state as an *override* rather than a fallback was the
    # Australia bug: a verified GB customer asking about Australian policy had
    # `country=GB` substituted, filtering the AU policy out of an AU question
    # while check_return_eligibility applied the AU window on the same account.
    region = resolve_region(args["query"], state.customer_region)
    if region:
        used["country"] = region

    result = search_policy(**used)
    # The governing policy in full, and the also-rans as one line each. `matches`
    # used to carry every candidate's whole node — properties, precedence,
    # promotion dates, rule path, conditions — which is the largest tool result in
    # the system and answers a question nobody asked: the agent needs the rule
    # that applies and why the others do not.
    return _ok(result, used, _policy_view(result))


def _policy_view(result: Any) -> dict[str, Any]:
    """The model's view of a policy search: the rule that governs, and why."""
    resolved = result.resolved
    return {
        "matched": result.matched,
        "region": result.region,
        "region_policy_found": result.region_policy_found,
        "region_note": result.region_note,
        "resolved": resolved
        and {
            "policy_id": resolved.policy_id,
            "policy_name": resolved.policy_name,
            "category": resolved.category,
            "return_window_days": resolved.return_window_days,
            "window_starts_from": resolved.window_starts_from,
            "rule_path": resolved.rule_path,
            "conditions": resolved.conditions,
            "summary": resolved.summary,
        },
        # One line per candidate that did not govern: enough to say "that one
        # only applies in Australia", without a second copy of the graph.
        "other_policies": [
            {
                "policy_id": candidate.policy.policy_id,
                "category": candidate.category,
                "applies": candidate.applies,
                "return_window_days": candidate.policy.window_days,
                "conditions": candidate.conditions,
            }
            for candidate in result.matches
            if resolved is None or candidate.policy.policy_id != resolved.policy_id
        ],
        "message": result.message,
    }


def _handle_check_return_eligibility(args: dict, state: SessionState, now) -> ToolOutcome:
    used = {
        "order_id": args["order_id"],
        "item_id": args["item_id"],
        "customer_id": state.verified_customer_id,
    }
    decision = check_return_eligibility(**used, now=now)
    return ToolOutcome(
        status=ToolStatus.OK,
        # The token is withheld from the model. It has no use for one — the loop
        # supplies it to initiate_return from session state — and a credential in
        # the transcript is a credential the model could repeat to the customer.
        #
        # So is the rule path: it is the *explanation of the traversal*, shown in
        # the trace and kept on the session, and the prompt forbids mentioning the
        # graph to a customer. `explanation` is the sentence the agent actually
        # says.
        content=json.dumps(
            {
                "eligible": decision.eligible,
                "policy_id": decision.policy_id,
                "explanation": decision.explanation,
                "days_remaining": decision.days_remaining,
            }
        ),
        summary=summarize(decision),
        payload=decision,
        args_used=used,
    )


def _handle_initiate_return(args: dict, state: SessionState, now) -> ToolOutcome:
    # The three arguments the model does not get to supply, read from state at the
    # moment of the call — so a token cleared by an item switch is gone, and a
    # confirmation the customer never gave is False.
    used = {
        "order_id": args["order_id"],
        "item_id": args["item_id"],
        "customer_id": state.verified_customer_id,
        # The reason is not a gate, so it must not be able to stop a confirmed
        # return: fall back to what the session recorded, and to nothing at all.
        "reason": args.get("reason") or state.return_reason or "",
        "eligibility_token": state.eligibility_token or "",
        "confirmed": state.confirmed,
    }
    result = initiate_return(**used)
    # Whether a return was opened or already existed, its reference, and where it
    # stands. The rest of the record is the request the agent just made.
    view = {
        "created": result.created,
        "return_id": result.return_record.return_id,
        "status": result.return_record.status.value,
        "message": result.message,
    }
    return _ok(result, used, view)


def _handle_escalate_to_human(args: dict, state: SessionState, now) -> ToolOutcome:
    used = {
        "reason": args["reason"],
        "customer_id": state.verified_customer_id,
        "order_id": state.active_order_id,
    }
    result = escalate_to_human(**used)
    # The reference to read back, and the sentence to read it back in. The reason
    # is context for the colleague, and the model wrote it.
    return _ok(result, used, {"case_id": result.case_id, "message": result.message})


_HANDLERS: dict[str, Callable[[dict, SessionState, datetime | None], ToolOutcome]] = {
    "verify_identity": _handle_verify_identity,
    "lookup_order": _handle_lookup_order,
    "search_policy": _handle_search_policy,
    "check_return_eligibility": _handle_check_return_eligibility,
    "initiate_return": _handle_initiate_return,
    "escalate_to_human": _handle_escalate_to_human,
}


# --- Trusted state updates ----------------------------------------------


def apply_tool_result(
    name: str,
    args: dict[str, Any],
    outcome: ToolOutcome,
    state: SessionState,
    *,
    adopt_active_order: bool = True,
    adopt_active_item: bool = True,
) -> None:
    """Update session state from a tool result — and only from a tool result.

    Called after a successful call, never a blocked or failed one. With the
    confirmation check in `agent/agent.py`, this is the only place the trusted
    fields are written, which keeps the answer to "how did the agent come to
    believe this?" short: a tool said so. Nothing here trusts the model's
    arguments on their own.

    Args:
        adopt_active_order: Whether a successful `lookup_order` may make its order
            the one under discussion. False when the same turn read several orders
            — that is the agent browsing to build a question, not selecting.
        adopt_active_item: Whether a successful `check_return_eligibility` may
            replace the item under discussion. False when the same turn checks
            several items — that is the agent comparing them, not the customer
            abandoning whichever was already found eligible, so an existing
            pending return and its token are left alone rather than cleared.
    """
    if outcome.status is not ToolStatus.OK or outcome.payload is None:
        return

    payload = outcome.payload

    if name == "verify_identity" and payload.verified:
        state.verified_customer_id = payload.customer_id
        state.customer_region = payload.region

    elif name == "lookup_order" and isinstance(payload, CustomerOrders):
        state.active_order_ids = [order.order_id for order in payload.orders]
        # One live order is not ambiguous, so adopt it — and if that order holds
        # one item, neither is the item. Two or more and the agent has to ask;
        # leaving these None is what makes the order-specific tools unusable until
        # it does. Deterministic either way: the data decides, not the model.
        if len(payload.orders) == 1:
            state.active_order_id = payload.orders[0].order_id
            if len(payload.orders[0].items) == 1:
                state.active_item_id = payload.orders[0].items[0].item_id

    elif name == "lookup_order" and adopt_active_order:
        state.active_order_id = payload.order.order_id

    elif name == "check_return_eligibility":
        order_id = outcome.args_used["order_id"]
        item_id = outcome.args_used["item_id"]
        state.eligibility = payload
        if args.get("reason"):
            state.return_reason = args["reason"]

        switching_item = (
            state.active_item_id not in (None, item_id) or state.active_order_id != order_id
        )

        if switching_item and not adopt_active_item and state.pending_return is not None:
            # The same turn is checking a second item while one is already
            # pending — the agent comparing candidates, not the customer walking
            # away from the first. Leave the existing pending return and its
            # token exactly as they were; this call gets no say over them.
            return

        # A new item means the old attempt is over. Clearing first drops the
        # previous token, decision, and any confirmation, so nothing from the
        # last item can be spent on this one.
        if switching_item:
            state.clear_return_context()
            state.eligibility = payload  # clear_return_context just wiped this

        state.active_order_id = order_id
        state.active_item_id = item_id
        state.eligibility_token = payload.eligibility_token

        if payload.eligible and payload.eligibility_token:
            # There is now something a "yes" could authorise. It does not count
            # until the agent has actually asked.
            state.pending_return = PendingReturn(
                order_id=order_id, item_id=item_id, eligibility_token=payload.eligibility_token
            )
        else:
            state.pending_return = None
            state.confirmed = False

    elif name == "initiate_return":
        # Done either way: `created=False` means a return already existed, and the
        # customer's request is satisfied. Clearing stops a second "yes" later in
        # the conversation from meaning anything.
        state.clear_return_context()

    elif name == "escalate_to_human" and getattr(payload, "case_id", None):
        # Escalation follows a case that was actually created, and nothing else —
        # not a failed policy lookup, not a complex question, not the tier the
        # router picked, and not the agent merely *offering* to pass the customer
        # on. The case id is the evidence.
        state.escalated = True


# --- Putting the demo back ----------------------------------------------


def reset_demo() -> str:
    """Restore the mutable demo data and drop every outstanding eligibility token.

    Not a model-callable tool — it is the one implementation the "Reset demo"
    button and `scripts/reset_demo.py` share, so a rehearsal and a live run cannot
    start from different places. The hero conversation writes a real RMA, which is
    why a second run is not the first unless something restores the file.

    Copies `data/seed/` back over `data/` and empties the process-global token
    store, so a token minted before a reset cannot be spent after one. Idempotent.
    Conversation state is untouched: a caller holding a `SessionState` replaces it.

    Returns:
        One line describing what was restored.
    """
    baseline = DATA_DIR / SEED_SUBDIR
    restored = []
    if baseline.is_dir():
        for source in sorted(baseline.glob("*.json")):
            shutil.copyfile(source, DATA_DIR / source.name)
            restored.append(source.name)

    _clear_eligibility_tokens()
    return (
        f"Demo reset — restored {', '.join(restored) if restored else 'nothing'}; "
        f"eligibility tokens cleared."
    )
