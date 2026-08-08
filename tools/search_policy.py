"""Tool: find the policy text relevant to a customer's question. Stub only."""

from __future__ import annotations

from datetime import date

from agent.models import Policy


def search_policy(
    query: str,
    product_type: str | None = None,
    country: str | None = None,
    delivered_at: date | None = None,
) -> list[Policy]:
    """Find policies that apply to a question.

    Backed by Neo4j, so a query can be narrowed by what was bought, where the
    customer is, and when it arrived — a physical book in Australia during a
    sale matches different rules than an ebook anywhere.

    Args:
        query: Free text, e.g. "can I return a damaged book".
        product_type: Optional 'PhysicalBook' or 'EBook' filter.
        country: Optional ISO country code, for regional overrides.
        delivered_at: Optional delivery date, for promotional windows.

    Returns:
        Matching policies, highest precedence first.

    Raises:
        PolicyGraphUnavailableError: If Neo4j is unconfigured or unreachable.
            Policy text is never served from JSON, so there is nothing to
            return in that case.
    """
    # TODO: get_driver() and run Cypher — the graph is the only policy source.
    # TODO: match (:Category {name: product_type})-[:GOVERNED_BY]->(:Policy).
    # TODO: add (:Region {code: country})-[:HAS_OVERRIDE]->(:Policy).
    # TODO: keep a promotional policy only when delivered_at falls inside
    #       promotion_active_from..promotion_active_to.
    # TODO: order by precedence descending; OVERRIDES edges explain the order.
    raise NotImplementedError("search_policy is a scaffold stub")
