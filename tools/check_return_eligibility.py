"""Tool: decide whether one item on one order can be returned.

The one place the return rules are applied. The agent reports what this returns;
it does not reason about windows, regions, or promotions itself.

Neo4j holds the policies. `tools.policy_rules` reads them, keeps only the ones
whose conditions the order and customer actually satisfy, and lets the highest
precedence survivor decide. That filter-then-rank order is the whole design:
`AU_BOOKLY_EXTENDED_RETURN` has a higher precedence than `STANDARD_30_DAY`, but a
customer in the UK must never be given it, so precedence is only consulted among
policies that already apply.

That selection is *shared* with `search_policy` rather than repeated here, so an
informational answer about Australia and a decision about an Australian
customer's order cannot disagree about which policy governs.

Neo4j is required. With no graph there is no decision to report, and a guess is
worse than an error.
"""

from __future__ import annotations

from datetime import UTC, datetime

from agent.models import EligibilityDecision, Order, Policy, ReturnStatus
from tools import eligibility_tokens, fixtures
from tools.lookup_order import find_line_item, lookup_order
from tools.policy_rules import PolicyContext, build_rule_path, policy_candidates
from tools.policy_rules import policy_applies as _policy_applies_in_context

BLOCKING_RETURN_STATUSES = frozenset(
    {ReturnStatus.REQUESTED, ReturnStatus.APPROVED, ReturnStatus.COMPLETED}
)
"""Statuses that mean a return already exists. A rejected one does not block a retry."""


def check_return_eligibility(
    order_id: str,
    item_id: str,
    customer_id: str,
    now: datetime | None = None,
) -> EligibilityDecision:
    """Decide if one item on one order is returnable, and say why.

    Args:
        order_id: The order the item was bought on.
        item_id: The item in question.
        customer_id: The verified customer, for both ownership and region.
        now: Clock the return window is measured against. Defaults to the current
            UTC time; tests pass a fixed value so the fixtures stay meaningful.

    Returns:
        The decision, the policy behind it, a customer-facing explanation, the
        graph hops that produced it, and — only when eligible — an
        `eligibility_token`.

    Raises:
        PolicyGraphUnavailableError: If Neo4j is unconfigured or unreachable.
            Never returns a mocked or defaulted decision instead.
    """
    today = (now or datetime.now(UTC)).date()

    customer = next(
        (c for c in fixtures.load_customers() if c.customer_id == customer_id), None
    )
    if customer is None:
        return _refuse("I need to verify your account before I can look at a return.")

    # Ownership is enforced here too, not assumed from the caller. An order that
    # is not this customer's is indistinguishable from one that does not exist.
    details = lookup_order(order_id, customer_id, now=now)
    if details is None:
        return _refuse(f"I can't find order {order_id} on your account.")

    if not find_line_item(details.order, item_id):
        return _refuse(f"{item_id} isn't on order {order_id}.")

    item = fixtures.load_items().get(item_id)
    if item is None:
        return _refuse(f"I can't find {item_id} in the catalogue.")

    # --- Which policies actually apply -----------------------------------

    applicable = applicable_policies(
        item.product_type.value, details.order, customer.country
    )
    if not applicable:
        return _refuse("I can't find a return policy covering that item.")

    policy, granted_by_region, outranks = applicable[0]
    rule_path = build_rule_path(item.product_type.value, policy.policy_id, granted_by_region, outranks)

    # --- Then, under that policy, is this return allowed -----------------

    existing = _existing_return(order_id, item_id)
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
            explanation=f"{item.title} is a digital item, and {policy.summary[0].lower()}{policy.summary[1:]}",
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
    grant = eligibility_tokens.issue(customer_id, order_id, item_id, policy.policy_id)
    return EligibilityDecision(
        eligible=True,
        policy_id=policy.policy_id,
        explanation=(
            f"{item.title} was delivered {days_since_delivery} days ago and the return window "
            f"is {policy.window_days} days, so it can be returned — you have {days_remaining} "
            f"day{'s' if days_remaining != 1 else ''} left."
        ),
        rule_path=rule_path,
        eligibility_token=grant.token,
        days_remaining=days_remaining,
    )


def applicable_policies(
    product_type: str, order: Order, customer_region: str
) -> list[tuple[Policy, str | None, list[str]]]:
    """The policies that govern this category *and* apply to this order.

    A thin adapter over `policy_rules.policy_candidates`: the same read, the same
    applicability test, and the same precedence order that `search_policy` uses,
    narrowed to what a decision may rest on. Keeping the selection in one place is
    what makes the informational and the decision path agree.

    Args:
        product_type: The item's category name.
        order: The order, for its promotion code and placement date.
        customer_region: The verified customer's ISO country code.

    Returns:
        Tuples of (policy, region that grants it, policy ids it outranks),
        highest precedence first. Empty when nothing applies.

    Raises:
        PolicyGraphUnavailableError: If Neo4j is unconfigured or unreachable.
    """
    context = PolicyContext.for_order(order, customer_region)
    return [
        (candidate.policy, candidate.granted_by_region, candidate.outranks)
        for candidate in policy_candidates(product_type, context)
        if candidate.applies
    ]


def policy_applies(
    policy: Policy, granted_to_regions: list[str], order: Order, customer_region: str
) -> bool:
    """Whether a policy's conditions are satisfied by this order and customer.

    The rules themselves live in `policy_rules.policy_applies`, which both policy
    tools share; this states them in terms of an order. See that function for why
    a regional policy stays regional and a promotional one stays promotional.

    Args:
        policy: The policy under test.
        granted_to_regions: Region codes with a HAS_OVERRIDE edge to it.
        order: The order in question.
        customer_region: The verified customer's ISO country code.

    Returns:
        True if the policy may be considered for this order.
    """
    return _policy_applies_in_context(
        policy, granted_to_regions, PolicyContext.for_order(order, customer_region)
    )


def _existing_return(order_id: str, item_id: str):
    """The open return for this order and item, if there is one."""
    return next(
        (
            record
            for record in fixtures.load_returns()
            if record.order_id == order_id
            and record.item_id == item_id
            and record.status in BLOCKING_RETURN_STATUSES
        ),
        None,
    )


def _refuse(explanation: str) -> EligibilityDecision:
    """A refusal that never reached a policy, so it carries no policy or token."""
    return EligibilityDecision(eligible=False, explanation=explanation)
