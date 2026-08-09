"""Bookly's return policy, and the one mechanism that decides which of it applies.

Both policy tools read this module and nothing else. `search_policy` presents the
result as information; `check_return_eligibility` takes the policies that apply
and decides. Before this was shared they each had their own notion of "applies",
and the two answered the same question about Australia differently.

Four things are defined here and nowhere else: **region normalization** (a
deterministic table, never a judgement the model makes), **applicability**
(whether a policy's conditions are satisfied by a region, a promotion, and a
date), **precedence** (filter first, then rank — `AU_BOOKLY_EXTENDED_RETURN`
outranks `STANDARD_30_DAY`, but a UK customer must never be handed it, so
precedence is only consulted among policies that already apply), and **the rule
path** (the graph hops behind a match, so any answer can be explained by the
traversal that produced it).

Neo4j is the only policy store: every read starts at
`graph.fetch_policies_for_category`, and there is no local copy if it is down.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, computed_field

from policy.graph import fetch_policies_for_category


class ProductType(StrEnum):
    """Item category. Matches the `:Category` nodes in the policy graph, and is
    what `Item.product_type` records."""

    PHYSICAL_BOOK = "PhysicalBook"
    EBOOK = "EBook"


class Policy(BaseModel):
    """One `:Policy` node's properties, as they come out of Neo4j.

    Which categories it governs, which regions override into it, and which
    policies it outranks are *edges*, answered by traversal rather than by a list
    on this model. `window_days` of None means returns are not offered at all —
    the absence of a window, not a window of zero length.
    """

    policy_id: str = Field(description="e.g. 'STANDARD_30_DAY', 'AU_BOOKLY_EXTENDED_RETURN'.")
    name: str
    summary: str
    window_days: int | None = None
    window_starts_from: str | None = None
    precedence: int = Field(default=0, description="Higher wins among policies that apply.")
    exceptions: list[str] = Field(default_factory=list)
    promotion_code: str | None = None
    promotion_active_from: date | None = None
    promotion_active_to: date | None = None


# --- Regions -------------------------------------------------------------
#
# Deliberately a table. The model is never asked to produce a country code, and
# a resolution that depends on how a question was phrased has to be reproducible.

REGION_ALIASES: dict[str, str] = {
    "au": "AU", "aus": "AU", "australia": "AU", "australian": "AU", "aussie": "AU",
    "gb": "GB", "uk": "GB", "united kingdom": "GB", "britain": "GB",
    "british": "GB", "england": "GB",
    "us": "US", "usa": "US", "united states": "US", "america": "US", "american": "US",
}

_ALIAS_PATTERN = re.compile(
    r"\b(" + "|".join(sorted((re.escape(a) for a in REGION_ALIASES), key=len, reverse=True)) + r")\b"
)
"""Whole-word alias matcher. Longest alias first, so 'united kingdom' beats 'uk'."""


def normalize_region(value: str | None) -> str | None:
    """Resolve a region name or code to ISO 3166-1 alpha-2.

    An unrecognised two-letter code passes through uppercased, so a region the
    graph knows about but this table does not still filters correctly. Anything
    else resolves to None — better no region than a guessed one.
    """
    if not value:
        return None
    text = value.strip().lower()
    if text in REGION_ALIASES:
        return REGION_ALIASES[text]
    if len(text) == 2 and text.isalpha():
        return text.upper()
    return None


def region_from_text(text: str | None) -> str | None:
    """The region a piece of free text names, if it names one.

    Whole words, earliest mention first: "what about Australia — is it the same as
    the UK?" is a question about Australia.
    """
    if not text:
        return None
    match = _ALIAS_PATTERN.search(text.lower())
    return REGION_ALIASES[match.group(1)] if match else None


def resolve_region(query: str | None, session_region: str | None = None) -> str | None:
    """The region a policy question is about, by fixed precedence: a region named
    in *this* question, then the verified customer's region, then none at all.

    Session state is a fallback, never an override. A verified UK customer asking
    "what is the return policy for Australian customers?" is asking about
    Australia; answering about the United Kingdom answers a different question.
    """
    return region_from_text(query) or normalize_region(session_region)


# --- What a policy is judged against -------------------------------------


@dataclass
class PolicyContext:
    """The facts a policy's conditions are tested against.

    An eligibility check knows all of them, from a real order; an informational
    lookup knows at most a region — which is why a promotional policy comes back
    from a search marked conditional rather than as the rule.
    """

    region: str | None = None
    promotion_code: str | None = None
    placed_at: date | None = None
    as_of: date | None = None
    """Today, for wording whether a promotion is currently running."""

    def __post_init__(self) -> None:
        self.region = normalize_region(self.region) or self.region


class PolicyCandidate(BaseModel):
    """One policy governing a category, with whether it applies and why.

    `applies` is the shared verdict both tools read; `conditions` is what to say
    when it does not, because a conditional policy described flatly is a promise
    Bookly did not make.
    """

    policy: Policy
    category: str
    granted_to_regions: list[str] = Field(
        default_factory=list, description="Region codes with a HAS_OVERRIDE edge to it."
    )
    outranks: list[str] = Field(
        default_factory=list, description="Policy ids it displaces, from OVERRIDES edges."
    )
    applies: bool
    conditions: list[str] = Field(
        default_factory=list, description="Plain-language conditions, e.g. an AU address."
    )

    # Computed rather than stored, and serialized with the rest: the agent reads
    # this JSON, so the traversal has to be in it.

    @computed_field
    @property
    def granted_by_region(self) -> str | None:
        return self.granted_to_regions[0] if self.granted_to_regions else None

    @computed_field
    @property
    def rule_path(self) -> list[str]:
        return build_rule_path(
            self.category, self.policy.policy_id, self.granted_by_region, self.outranks
        )


def policy_candidates(product_type: str, context: PolicyContext) -> list[PolicyCandidate]:
    """Every policy governing a category, judged against one context.

    The single read of the policy graph — both tools go through here, so neither
    can drift into its own idea of applicability, precedence, or overrides.
    Applicable candidates come first and then by precedence, so the head of the
    list is the policy that governs.

    Raises:
        PolicyGraphUnavailableError: If Neo4j is unconfigured or unreachable.
    """
    candidates = []
    for row in fetch_policies_for_category(product_type):
        policy = Policy.model_validate(row["policy"])
        regions = [code for code in row["granted_to_regions"] if code]
        candidates.append(
            PolicyCandidate(
                policy=policy,
                category=row["category"],
                granted_to_regions=regions,
                outranks=[pid for pid in row["outranks"] if pid],
                applies=policy_applies(policy, regions, context),
                conditions=describe_conditions(policy, regions, context),
            )
        )

    # Filter-then-rank, in one sort: applicability is the first key, so
    # precedence can only order policies that already apply.
    candidates.sort(key=lambda c: (not c.applies, -c.policy.precedence, c.policy.policy_id))
    return candidates


def applicable_policies(product_type: str, context: PolicyContext) -> list[PolicyCandidate]:
    """The candidates that actually govern, highest precedence first.

    What an eligibility decision may rest on. Empty when nothing applies.
    """
    return [candidate for candidate in policy_candidates(product_type, context) if candidate.applies]


def policy_applies(policy: Policy, granted_to_regions: list[str], context: PolicyContext) -> bool:
    """Whether a policy's conditions are satisfied by a context.

    Two kinds, both read off the graph rather than hardcoded. A policy reached
    through `(:Region)-[:HAS_OVERRIDE]->(:Policy)` is offered to that region only,
    which is what stops a UK customer being handed `AU_BOOKLY_EXTENDED_RETURN`;
    one carrying a `promotion_code` is offered only to orders placed under that
    promotion, inside its active dates.

    A condition that cannot be shown to be met is not met: absent data is never
    read as permission.
    """
    if granted_to_regions and context.region not in granted_to_regions:
        return False

    if policy.promotion_code:
        if context.promotion_code != policy.promotion_code:
            return False
        if context.placed_at is None or not promotion_covers(policy, context.placed_at):
            return False

    return True


def describe_conditions(
    policy: Policy, granted_to_regions: list[str], context: PolicyContext
) -> list[str]:
    """The conditions on a policy applying at all, in plain language — so the
    informational path can say "45 days, if you are in Australia" without either
    hiding the condition or refusing to state the rule."""
    conditions: list[str] = []

    if granted_to_regions:
        conditions.append(f"applies only to customers in {granted_to_regions[0]}")

    if policy.promotion_code:
        running = context.as_of is not None and promotion_covers(policy, context.as_of)
        conditions.append(
            f"applies only to orders placed under {policy.promotion_code} "
            f"({policy.promotion_active_from} to {policy.promotion_active_to}"
            f"{'' if running else ', not currently running'})"
        )

    if policy.window_days is None:
        conditions.append("no return window — returns are not offered")

    return conditions


def promotion_covers(policy: Policy, day: date) -> bool:
    """Whether a promotional policy's active window contains a date."""
    if policy.promotion_active_from is None or policy.promotion_active_to is None:
        return False
    return policy.promotion_active_from <= day <= policy.promotion_active_to


def build_rule_path(
    category: str, policy_id: str, granted_by_region: str | None, outranks: list[str]
) -> list[str]:
    """The traversal, as readable hops in the arrow notation the graph README uses,
    so a decision is explained by the path that produced it."""
    hops = [f"({category})-[:GOVERNED_BY]->({policy_id})"]
    if granted_by_region:
        hops.append(f"({granted_by_region})-[:HAS_OVERRIDE]->({policy_id})")
    hops.extend(f"({policy_id})-[:OVERRIDES]->({outranked})" for outranked in sorted(outranks))
    return hops


# --- Informational lookup ------------------------------------------------
#
# `search_policy` is one of the six model-callable tools; its body lives here
# because everything it does is policy selection and policy presentation.
# `agent/tools.py` imports it, resolves the region, and lists it in the schemas.

CATEGORY_KEYWORDS: dict[ProductType, tuple[str, ...]] = {
    ProductType.EBOOK: ("ebook", "e-book", "digital", "download", "kindle"),
    ProductType.PHYSICAL_BOOK: ("physical", "paperback", "hardback", "hardcover", "printed", "book"),
}
"""Enough keyword routing to answer a bare question. The agent usually passes
`product_type` explicitly and these are the fallback."""


class ResolvedPolicy(BaseModel):
    """The one policy that governs the question — the same one an eligibility
    check on that region and category would land on. Flat and explicit, so the
    agent has no arithmetic to do and no list to pick from."""

    policy_id: str
    policy_name: str
    category: str
    return_window_days: int | None = Field(
        default=None, description="None means returns are not offered at all."
    )
    window_starts_from: str | None = None
    precedence: int = 0
    region: str | None = Field(
        default=None, description="Set when this policy is region-specific."
    )
    overrides: list[str] = Field(default_factory=list)
    rule_path: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    summary: str = ""

    @classmethod
    def from_candidate(cls, candidate: PolicyCandidate) -> ResolvedPolicy:
        return cls(
            policy_id=candidate.policy.policy_id,
            policy_name=candidate.policy.name,
            category=candidate.category,
            return_window_days=candidate.policy.window_days,
            window_starts_from=candidate.policy.window_starts_from,
            precedence=candidate.policy.precedence,
            region=candidate.granted_by_region,
            overrides=candidate.outranks,
            rule_path=candidate.rule_path,
            conditions=candidate.conditions,
            summary=candidate.policy.summary,
        )


class PolicySearchResult(BaseModel):
    """What the graph had to say about a question.

    The region, whether a policy specific to it exists, and the policy that
    governs are all stated rather than left to be inferred — inferring them is how
    an agent ends up telling an Australian customer there is no Australian policy.
    A no-match is a result, not an exception.
    """

    matched: bool
    query: str
    region: str | None = None
    region_policy_found: bool = False
    region_note: str = Field(default="", description="The regional situation, in one line.")
    resolved: ResolvedPolicy | None = None
    matches: list[PolicyCandidate] = Field(default_factory=list)
    message: str = ""
    searched_categories: list[str] = Field(default_factory=list)
    source: str = Field(default="neo4j", description="Always the graph. There is no other source.")


def search_policy(
    query: str,
    product_type: str | None = None,
    country: str | None = None,
    as_of: date | None = None,
) -> PolicySearchResult:
    """Find the policies that bear on a question about the rules.

    Informational only: it never looks at an order and never decides whether a
    particular return is allowed. That is `check_return_eligibility`, and the
    split is deliberate — a customer asking about the rules should not need an
    order, and one asking about their order should get a decision.

    Args:
        query: Free text, e.g. "can I return an ebook".
        product_type: Inferred from `query` when omitted.
        country: The region to answer for, already resolved by the caller — see
            `resolve_region` — and normalized again here. Inferred from `query`
            when omitted.
        as_of: Date to judge promotional policies against. Defaults to today.

    Raises:
        PolicyGraphUnavailableError: If Neo4j is unconfigured or unreachable.
    """
    today = as_of or datetime.now(UTC).date()
    categories = _categories_for(query, product_type)
    region = normalize_region(country) or region_from_text(query)
    context = PolicyContext(region=region, as_of=today)

    matches: list[PolicyCandidate] = []
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
            matches.append(candidate)

    # Precedence only ranks policies competing for the same category, so ordering
    # is by category first — otherwise DIGITAL_NO_RETURN's precedence of 100 would
    # head the answer to a question about paperbacks.
    matches.sort(
        key=lambda m: (
            categories.index(m.category), not m.applies, -m.policy.precedence, m.policy.policy_id
        )
    )

    if not matches:
        return PolicySearchResult(
            matched=False,
            query=query,
            region=region,
            region_note=_region_note(region, found=False, resolved=None),
            searched_categories=categories,
            message="I couldn't find a Bookly policy covering that.",
        )

    # The governing policy is the first that actually applies — for a region with
    # an override, the same one an eligibility check would reach. A question whose
    # only matches are conditional resolves to nothing, and the conditions are
    # what the agent reports.
    winner = next((m for m in matches if m.applies), None)
    resolved = ResolvedPolicy.from_candidate(winner) if winner else None

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
    different answers that a list of matches does not distinguish.
    """
    if region is None:
        return "No region was specified, so region-specific policies are listed as conditional."
    if found and resolved is not None and resolved.region == region:
        return f"{region} has a region-specific Bookly policy and it applies here."
    if found:
        return f"{region} has a region-specific Bookly policy, but it does not govern this question."
    return f"No {region}-specific policy exists for this; Bookly's standard rules apply."


def _categories_for(query: str, product_type: str | None) -> list[str]:
    """Which categories to search. A question naming neither searches both —
    better to return the ebook rule alongside the physical one than to pick wrong."""
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
