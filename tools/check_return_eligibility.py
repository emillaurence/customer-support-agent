"""Tool: decide whether an item can be returned. Stub only."""

from __future__ import annotations

from agent.models import EligibilityDecision


def check_return_eligibility(order_id: str, item_id: str, customer_id: str) -> EligibilityDecision:
    """Decide if one item on one order is returnable, and say why.

    This is the one place the return rules live. The agent reports the result
    verbatim; it does not reason about windows or exceptions itself.

    The graph resolves which policy applies by walking from the item's category
    to its policies, then applying whichever regional or promotional override
    outranks the default. The hops taken become `rule_path`.

    Args:
        order_id: The order the item was bought on.
        item_id: The item in question.
        customer_id: The verified customer, for both authorisation and region.

    Returns:
        The decision, the policy behind it, a customer-facing explanation, the
        rule path, and — only when eligible — an `eligibility_token`.
    """
    # TODO: 1. lookup_order for delivered_at, promotion_code, and product type.
    # TODO: 2. no delivered_at (in transit) -> not eligible, window has not started.
    # TODO: 3. search_policy(product_type, country, delivered_at) for the winner.
    # TODO: 4. no window on the winning policy (ebooks) -> not eligible, and no
    #          override can rescue it.
    # TODO: 5. compare days-since-delivery against the window.
    # TODO: 6. an open return on this order+item already -> not eligible.
    # TODO: 7. mint eligibility_token only on the eligible path.
    # TODO: 8. build rule_path from the traversal, not from a hardcoded string.
    raise NotImplementedError("check_return_eligibility is a scaffold stub")
