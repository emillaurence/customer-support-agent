"""Tool: find the policy relevant to a customer's question.

Informational only. This tool answers "what is Bookly's policy on X" — the
standard window, whether ebooks can be returned, what applies in Australia. It
never looks at an order and never decides whether a particular return is allowed.
That is `check_return_eligibility`, and the split is deliberate: a customer
asking about the rules should not need an order, and a customer asking about
their order should get a decision, not a policy leaflet.

Backed by Neo4j, which is the only policy store. There is no local copy to read
if the database is down; the tool raises and the agent tells the customer it
cannot confirm the policy right now.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from pydantic import BaseModel, Field

from agent.graph import fetch_policies_for_category
from agent.models import Policy, ProductType

# Enough keyword routing to answer a free-text question without a model. Once
# the agent is wired up it will usually pass `product_type` and `country`
# explicitly, and these become the fallback for a bare question.
CATEGORY_KEYWORDS: dict[ProductType, tuple[str, ...]] = {
    ProductType.EBOOK: ("ebook", "e-book", "digital", "download", "kindle"),
    ProductType.PHYSICAL_BOOK: ("physical", "paperback", "hardback", "hardcover", "printed", "book"),
}

REGION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "AU": ("australia", "australian", "aussie", " au ", "nsw"),
    "GB": ("uk", "united kingdom", "britain", "british", "england"),
    "US": ("usa", "united states", "america", "american"),
}


class PolicyMatch(BaseModel):
    """One policy, with what makes it reachable.

    `conditions` is the honest part: a policy in the graph is not automatically a
    policy that applies to a given customer. A regional or promotional policy is
    returned with its condition stated, so the agent describes it as conditional
    rather than as the rule.
    """

    policy: Policy
    category: str = Field(description="The product category this policy governs.")
    granted_by_region: str | None = Field(
        default=None,
        description="Set when only customers in this region can reach the policy.",
    )
    outranks: list[str] = Field(
        default_factory=list, description="Policy ids this one displaces, from OVERRIDES edges.",
    )
    conditions: list[str] = Field(
        default_factory=list,
        description="Plain-language conditions on the policy applying at all, e.g. an AU address.",
    )
    rule_path: list[str] = Field(
        default_factory=list,
        description="The graph hops behind the match, e.g. '(PhysicalBook)-[:GOVERNED_BY]->(STANDARD_30_DAY)'.",
    )


class PolicySearchResult(BaseModel):
    """What the graph had to say about a question.

    A no-match is a result, not an exception: `matched=False` with an empty
    `matches` list means the graph holds no policy for what was asked, which the
    agent should report as "I can't confirm that" — not fill in.
    """

    matched: bool
    query: str
    matches: list[PolicyMatch] = Field(default_factory=list)
    message: str = ""
    searched_categories: list[str] = Field(default_factory=list)
    source: str = Field(default="neo4j", description="Always the policy graph. There is no other source.")


def search_policy(
    query: str,
    product_type: str | None = None,
    country: str | None = None,
    as_of: date | None = None,
) -> PolicySearchResult:
    """Find the policies that bear on a question.

    Args:
        query: Free text, e.g. "can I return an ebook".
        product_type: 'PhysicalBook' or 'EBook'. Inferred from `query` when omitted.
        country: ISO 3166-1 alpha-2 code, for regional policies. Inferred from
            `query` when omitted; when neither names a region, regional policies
            are still returned, but marked with the region they require.
        as_of: Date to judge promotional policies against. Defaults to today.

    Returns:
        Matching policies, highest precedence first, each with its conditions and
        the graph hops behind it. `matched=False` when nothing matched.

    Raises:
        PolicyGraphUnavailableError: If Neo4j is unconfigured or unreachable.
            Policy is never served from a local file, so there is nothing to
            return in that case.
    """
    today = as_of or datetime.now(UTC).date()
    categories = _categories_for(query, product_type)
    region = (country or _region_for(query) or "").upper() or None

    matches: list[PolicyMatch] = []
    for category in categories:
        for row in fetch_policies_for_category(category):
            match = _to_match(row, today)
            # A named region filters out other regions' policies. With no region
            # named, everything is returned and the condition is stated instead.
            if region and match.granted_by_region and match.granted_by_region != region:
                continue
            matches.append(match)

    # Precedence only ranks policies competing for the same category, so ordering
    # is by category first — otherwise DIGITAL_NO_RETURN's precedence of 100
    # would head the answer to a question about paperbacks. Within a category, a
    # policy granted by the region the customer actually named comes first.
    matches.sort(
        key=lambda match: (
            categories.index(match.category),
            0 if region and match.granted_by_region == region else 1,
            -match.policy.precedence,
        )
    )

    if not matches:
        return PolicySearchResult(
            matched=False,
            query=query,
            searched_categories=categories,
            message="I couldn't find a Bookly policy covering that.",
        )

    return PolicySearchResult(
        matched=True,
        query=query,
        matches=matches,
        searched_categories=categories,
        message=matches[0].policy.summary,
    )


def _categories_for(query: str, product_type: str | None) -> list[str]:
    """Decide which categories to search.

    An explicit `product_type` wins. Otherwise the query is matched on keywords,
    and a question that names neither searches both — better to return the ebook
    rule alongside the physical one than to pick wrong.
    """
    if product_type:
        return [product_type]

    text = f" {query.lower()} "
    hits = [
        category.value
        for category, keywords in CATEGORY_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    ]
    # "ebook" contains "book", so a specific hit beats the generic one.
    if ProductType.EBOOK.value in hits:
        return [ProductType.EBOOK.value]
    return hits or [t.value for t in ProductType]


def _region_for(query: str) -> str | None:
    """Pull a region out of the question, if it names one."""
    text = f" {query.lower()} "
    for code, keywords in REGION_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return code
    return None


def _to_match(row: dict, today: date) -> PolicyMatch:
    """Turn one graph row into a match, with its conditions spelled out."""
    policy = Policy.model_validate(row["policy"])
    regions = [code for code in row["granted_to_regions"] if code]
    outranks = [policy_id for policy_id in row["outranks"] if policy_id]
    granted_by_region = regions[0] if regions else None

    conditions: list[str] = []
    if granted_by_region:
        conditions.append(f"applies only to customers in {granted_by_region}")
    if policy.promotion_code:
        active = _promotion_covers(policy, today)
        conditions.append(
            f"applies only to orders placed under {policy.promotion_code} "
            f"({policy.promotion_active_from} to {policy.promotion_active_to}"
            f"{'' if active else ', not currently running'})"
        )
    if policy.window_days is None:
        conditions.append("no return window — returns are not offered")

    return PolicyMatch(
        policy=policy,
        category=row["category"],
        granted_by_region=granted_by_region,
        outranks=outranks,
        conditions=conditions,
        rule_path=build_rule_path(row["category"], policy.policy_id, granted_by_region, outranks),
    )


def _promotion_covers(policy: Policy, day: date) -> bool:
    """Whether a promotional policy's window contains a date."""
    if policy.promotion_active_from is None or policy.promotion_active_to is None:
        return False
    return policy.promotion_active_from <= day <= policy.promotion_active_to


def build_rule_path(
    category: str, policy_id: str, granted_by_region: str | None, outranks: list[str]
) -> list[str]:
    """Write the traversal out as readable hops.

    One string per graph hop, in the arrow notation the policy graph README uses,
    so a decision can be explained by the path that produced it rather than by a
    sentence someone wrote by hand.

    Args:
        category: The category the walk started from.
        policy_id: The policy reached.
        granted_by_region: The region that grants it, if it is regional.
        outranks: Policy ids it displaces.

    Returns:
        The hops, category first.
    """
    hops = [f"({category})-[:GOVERNED_BY]->({policy_id})"]
    if granted_by_region:
        hops.append(f"({granted_by_region})-[:HAS_OVERRIDE]->({policy_id})")
    hops.extend(f"({policy_id})-[:OVERRIDES]->({outranked})" for outranked in sorted(outranks))
    return hops
