"""Tool: find the policy relevant to a customer's question.

Informational only. This tool answers "what is Bookly's policy on X" — the
standard window, whether ebooks can be returned, what applies in Australia. It
never looks at an order and never decides whether a particular return is allowed.
That is `check_return_eligibility`, and the split is deliberate: a customer
asking about the rules should not need an order, and a customer asking about
their order should get a decision, not a policy leaflet.

**The two share their policy selection, and only their presentation differs.**
Region applicability, category applicability, precedence, and overrides all come
from `tools.policy_rules`, which is the one mechanism both tools read. That is
what keeps "what is the Australian return window?" and "can I return this?" from
answering differently for the same customer — they did, before, because this tool
had a weaker region filter of its own.

Backed by Neo4j, which is the only policy store. There is no local copy to read
if the database is down; the tool raises and the agent tells the customer it
cannot confirm the policy right now.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from pydantic import BaseModel, Field

from agent.models import Policy, ProductType
from tools.policy_rules import (
    PolicyCandidate,
    PolicyContext,
    build_rule_path,
    normalize_region,
    policy_candidates,
    region_from_text,
)

__all__ = ["PolicyMatch", "PolicySearchResult", "ResolvedPolicy", "build_rule_path", "search_policy"]

# Enough keyword routing to answer a free-text question without a model. The
# agent usually passes `product_type` explicitly, and these are the fallback for
# a bare question.
CATEGORY_KEYWORDS: dict[ProductType, tuple[str, ...]] = {
    ProductType.EBOOK: ("ebook", "e-book", "digital", "download", "kindle"),
    ProductType.PHYSICAL_BOOK: ("physical", "paperback", "hardback", "hardcover", "printed", "book"),
}


class PolicyMatch(BaseModel):
    """One policy, with what makes it reachable.

    `applies` and `conditions` are the honest part: a policy in the graph is not
    automatically a policy that applies to a given customer. A regional or
    promotional policy that the question's region does not reach is returned with
    `applies=False` and its condition stated, so the agent describes it as
    conditional rather than as the rule.
    """

    policy: Policy
    category: str = Field(description="The product category this policy governs.")
    applies: bool = Field(
        default=True,
        description="Whether this policy governs in the region the question resolved to.",
    )
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

    @classmethod
    def from_candidate(cls, candidate: PolicyCandidate) -> PolicyMatch:
        """Present a shared policy candidate as a search match."""
        return cls(
            policy=candidate.policy,
            category=candidate.category,
            applies=candidate.applies,
            granted_by_region=candidate.granted_by_region,
            outranks=candidate.outranks,
            conditions=candidate.conditions,
            rule_path=candidate.rule_path,
        )


class ResolvedPolicy(BaseModel):
    """The one policy that governs the question, spelled out.

    The same policy an eligibility check on the same region and category would
    land on, because both are chosen by `tools.policy_rules`. Flat and explicit so
    the agent has no arithmetic to do and no list to pick from: this is the rule,
    this is the window, this is why it holds.
    """

    policy_id: str
    policy_name: str
    category: str
    return_window_days: int | None = Field(
        default=None,
        description="The window in days. None means returns are not offered at all.",
    )
    window_starts_from: str | None = None
    precedence: int = 0
    region: str | None = Field(
        default=None,
        description="Set when this policy is region-specific — the region that grants it.",
    )
    overrides: list[str] = Field(
        default_factory=list, description="Policy ids this one displaces."
    )
    rule_path: list[str] = Field(
        default_factory=list, description="The graph hops that reached it."
    )
    conditions: list[str] = Field(default_factory=list)
    summary: str = ""

    @classmethod
    def from_match(cls, match: PolicyMatch) -> ResolvedPolicy:
        return cls(
            policy_id=match.policy.policy_id,
            policy_name=match.policy.name,
            category=match.category,
            return_window_days=match.policy.window_days,
            window_starts_from=match.policy.window_starts_from,
            precedence=match.policy.precedence,
            region=match.granted_by_region,
            overrides=match.outranks,
            rule_path=match.rule_path,
            conditions=match.conditions,
            summary=match.policy.summary,
        )


class PolicySearchResult(BaseModel):
    """What the graph had to say about a question.

    Three things are stated rather than left to be inferred, because inferring
    them is how an agent ends up telling an Australian customer there is no
    Australian policy:

    * `region` — the region the question resolved to.
    * `region_policy_found` — whether a policy specific to that region exists.
    * `resolved` — the policy that governs, with its window.

    A no-match is a result, not an exception: `matched=False` with an empty
    `matches` list and no `resolved` policy means the graph holds nothing for what
    was asked, which the agent should report as "I can't confirm that" — not fill
    in.
    """

    matched: bool
    query: str
    region: str | None = Field(
        default=None, description="The region this answer is about. None means global context."
    )
    region_policy_found: bool = Field(
        default=False,
        description="True when the graph holds a policy specific to `region`. Distinguishes "
        "'no regional policy' from 'a regional policy applies'.",
    )
    region_note: str = Field(
        default="", description="One line stating the regional situation in plain language."
    )
    resolved: ResolvedPolicy | None = Field(
        default=None, description="The policy that governs the question. None on a no-match."
    )
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
        country: The region to answer for. Resolved by the orchestrator from the
            question and the verified session — see `policy_rules.resolve_region`
            — and normalized again here, so 'Australia' and 'AU' behave the same.
            Inferred from `query` when omitted; when neither names a region,
            regional policies are still returned, but marked conditional.
        as_of: Date to judge promotional policies against. Defaults to today.

    Returns:
        The governing policy in `resolved`, every candidate in `matches` with its
        conditions and graph hops, and the regional situation stated explicitly.
        `matched=False` when nothing matched.

    Raises:
        PolicyGraphUnavailableError: If Neo4j is unconfigured or unreachable.
            Policy is never served from a local file, so there is nothing to
            return in that case.
    """
    today = as_of or datetime.now(UTC).date()
    categories = _categories_for(query, product_type)
    region = normalize_region(country) or region_from_text(query)
    context = PolicyContext.for_region(region, as_of=today)

    matches: list[PolicyMatch] = []
    region_policy_found = False

    for category in categories:
        for candidate in policy_candidates(category, context):
            if candidate.granted_to_regions:
                # A named region filters out other regions' policies. With no
                # region named, everything is returned and the condition is
                # stated instead.
                if region and region not in candidate.granted_to_regions:
                    continue
                if region:
                    region_policy_found = True
            matches.append(PolicyMatch.from_candidate(candidate))

    # Precedence only ranks policies competing for the same category, so ordering
    # is by category first — otherwise DIGITAL_NO_RETURN's precedence of 100 would
    # head the answer to a question about paperbacks. Within a category, the
    # shared mechanism has already put what applies ahead of what merely exists.
    matches.sort(
        key=lambda match: (
            categories.index(match.category),
            not match.applies,
            -match.policy.precedence,
            match.policy.policy_id,
        )
    )

    if not matches:
        return PolicySearchResult(
            matched=False,
            query=query,
            region=region,
            region_policy_found=False,
            region_note=_region_note(region, found=False, resolved=None),
            searched_categories=categories,
            message="I couldn't find a Bookly policy covering that.",
        )

    # The governing policy is the first that actually applies, which for a region
    # that has an override is that region's policy — the same one an eligibility
    # check would reach. A question whose only matches are conditional resolves to
    # nothing, and the conditions are what the agent reports.
    resolved_match = next((match for match in matches if match.applies), None)
    resolved = ResolvedPolicy.from_match(resolved_match) if resolved_match else None

    return PolicySearchResult(
        matched=True,
        query=query,
        region=region,
        region_policy_found=region_policy_found,
        region_note=_region_note(region, found=region_policy_found, resolved=resolved),
        resolved=resolved,
        matches=matches,
        message=resolved.summary if resolved else matches[0].policy.summary,
        searched_categories=categories,
    )


def _region_note(region: str | None, *, found: bool, resolved: ResolvedPolicy | None) -> str:
    """State the regional situation, so it never has to be inferred from absence.

    "No AU policy was found" and "the AU policy applies, and it is 45 days" are
    different answers, and an agent reading a list of matches cannot reliably tell
    them apart. So the result says which it is.
    """
    if region is None:
        return "No region was specified, so region-specific policies are listed as conditional."
    if found and resolved is not None and resolved.region == region:
        return f"{region} has a region-specific Bookly policy and it applies here."
    if found:
        return f"{region} has a region-specific Bookly policy, but it does not govern this question."
    return f"No {region}-specific policy exists for this; Bookly's standard rules apply."


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
