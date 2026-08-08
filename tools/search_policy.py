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

    Backed by the policy graph, so a query can be narrowed by what was bought,
    where the customer is, and when it arrived — a physical book in Australia
    during a sale matches different rules than an ebook anywhere.

    Args:
        query: Free text, e.g. "can I return a damaged book".
        product_type: Optional 'PhysicalBook' or 'EBook' filter.
        country: Optional ISO country code, for regional overrides.
        delivered_at: Optional delivery date, for promotional windows.

    Returns:
        Matching policies, highest precedence first.
    """
    # TODO: read neo4j/policy_graph.json for now; swap for a Cypher query later.
    #       Keep this signature stable across that swap.
    # TODO: filter RegionalPolicy by (:Region)-[:HAS_OVERRIDE]->(:Policy).
    # TODO: filter PromotionalPolicy by the granting promotion's active window
    #       against delivered_at.
    raise NotImplementedError("search_policy is a scaffold stub")
