"""The one policy-selection mechanism, shared by both policy tools.

There is exactly one idea of what the graph says and which of it applies, and it
lives here. `search_policy` presents it as information; `check_return_eligibility`
filters it to what applies and decides. Before this module they each had their own
notion of "applies" — the eligibility tool filtered on region, promotion, and
dates, while the search tool only filtered on region — and the two answered the
same question about Australia differently.

Three things are defined here and nowhere else:

* **Region normalization.** `Australia`, `Australian`, `AU` are one region. A
  deterministic table, not a judgement the model makes — see `normalize_region`.
* **Applicability.** Whether a policy's conditions are satisfied by a region, a
  promotion, and a date — see `policy_applies`.
* **The rule path.** The graph hops behind a match, so any answer can be
  explained by the traversal that produced it — see `build_rule_path`.

This is not a second policy engine. Neo4j is still the only policy store: every
function here starts from `fetch_policies_for_category`, and there is no local
copy to read if the graph is down.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, Field

from agent.graph import fetch_policies_for_category
from agent.models import Order, Policy

# --- Regions -------------------------------------------------------------
#
# Deliberately a table. The model is never asked to produce a country code — the
# orchestrator resolves one from the customer's words and from trusted session
# state, and a resolution that depends on how a question was phrased has to be
# reproducible. Extending it is adding a row.

REGION_ALIASES: dict[str, str] = {
    "au": "AU",
    "aus": "AU",
    "australia": "AU",
    "australian": "AU",
    "aussie": "AU",
    "gb": "GB",
    "uk": "GB",
    "united kingdom": "GB",
    "britain": "GB",
    "british": "GB",
    "england": "GB",
    "us": "US",
    "usa": "US",
    "united states": "US",
    "america": "US",
    "american": "US",
}
"""Names and codes that resolve to an ISO 3166-1 alpha-2 region code."""

_ALIAS_PATTERN = re.compile(
    r"\b(" + "|".join(sorted((re.escape(alias) for alias in REGION_ALIASES), key=len, reverse=True)) + r")\b"
)
"""Whole-word alias matcher. Longest alias first, so 'united kingdom' beats 'uk'."""


def normalize_region(value: str | None) -> str | None:
    """Resolve a region name or code to an ISO 3166-1 alpha-2 code.

    Args:
        value: Anything a caller might have — 'Australia', 'australian', 'au',
            'AU', or None.

    Returns:
        The code, or None if `value` is empty or names nothing recognisable. An
        unrecognised two-letter code is passed through uppercased, so a region
        the graph knows about but this table does not still filters correctly.
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

    Matched on whole words, and the *earliest* mention wins: a customer who asks
    "what about Australia — is it the same as the UK?" is asking about Australia.

    Args:
        text: The customer's question, in their own words.

    Returns:
        The region code, or None.
    """
    if not text:
        return None

    match = _ALIAS_PATTERN.search(text.lower())
    return REGION_ALIASES[match.group(1)] if match else None


def resolve_region(query: str | None, session_region: str | None = None) -> str | None:
    """The region a policy question is about, by fixed precedence.

    1. A region the customer named in *this* question.
    2. The verified customer's region, from trusted session state.
    3. No region at all — global policy context.

    Session state is a fallback, never an override. A verified UK customer asking
    "what is the return policy for Australian customers?" is asking about
    Australia, and answering about the United Kingdom is answering a different
    question. That was the bug: the session region was applied unconditionally,
    so the Australian policy was filtered out of an Australian question.

    Args:
        query: The customer's current question.
        session_region: The verified customer's region, if identity is
            established. Trusted, but only used when the question names none.

    Returns:
        The region code, or None.
    """
    return region_from_text(query) or normalize_region(session_region)


# --- What a policy is being judged against -------------------------------


@dataclass(frozen=True)
class PolicyContext:
    """The facts a policy's conditions are tested against.

    An eligibility check knows all of them, from a real order. An informational
    lookup knows at most a region — which is exactly why a promotional policy
    comes back from a search marked conditional rather than as the rule.
    """

    region: str | None = None
    """ISO country code the question or the customer resolves to."""

    promotion_code: str | None = None
    """The promotion an order was placed under, when there is an order."""

    placed_at: date | None = None
    """When the order was placed, for judging a promotion's active window."""

    as_of: date | None = None
    """Today, for wording whether a promotion is currently running."""

    @classmethod
    def for_order(cls, order: Order, customer_region: str) -> PolicyContext:
        """The context of a specific order belonging to a verified customer."""
        return cls(
            region=normalize_region(customer_region) or customer_region,
            promotion_code=order.promotion_code,
            placed_at=order.placed_at,
        )

    @classmethod
    def for_region(cls, region: str | None, as_of: date | None = None) -> PolicyContext:
        """The context of an informational question — a region, and nothing else."""
        return cls(region=normalize_region(region) or region, as_of=as_of)


class PolicyCandidate(BaseModel):
    """One policy governing a category, with whether it applies and why.

    `applies` is the shared verdict both tools read. `conditions` is what to say
    when it does not: a policy in the graph is not automatically a policy that
    governs a given customer, and a conditional policy described flatly is a
    promise Bookly did not make.
    """

    policy: Policy
    category: str = Field(description="The product category this policy governs.")
    granted_to_regions: list[str] = Field(
        default_factory=list, description="Region codes with a HAS_OVERRIDE edge to it."
    )
    outranks: list[str] = Field(
        default_factory=list, description="Policy ids it displaces, from OVERRIDES edges."
    )
    applies: bool = Field(
        description="Whether every condition on the policy is satisfied by the context."
    )
    conditions: list[str] = Field(
        default_factory=list,
        description="Plain-language conditions on the policy applying at all, e.g. an AU address.",
    )

    @property
    def granted_by_region(self) -> str | None:
        """The region that grants it, when it is regional."""
        return self.granted_to_regions[0] if self.granted_to_regions else None

    @property
    def rule_path(self) -> list[str]:
        """The graph hops behind this candidate."""
        return build_rule_path(
            self.category, self.policy.policy_id, self.granted_by_region, self.outranks
        )


def policy_candidates(product_type: str, context: PolicyContext) -> list[PolicyCandidate]:
    """Every policy governing a category, judged against one context.

    The single read of the policy graph. Both tools go through here, so neither
    can drift into its own idea of region applicability, category applicability,
    precedence, or overrides.

    Args:
        product_type: A `:Category` name — 'PhysicalBook' or 'EBook'.
        context: What the conditions are tested against.

    Returns:
        One candidate per policy, applicable ones first and then by precedence,
        so the head of the list is the policy that governs — for an eligibility
        check and for an informational answer alike.

    Raises:
        PolicyGraphUnavailableError: If Neo4j is unconfigured or unreachable.
    """
    candidates: list[PolicyCandidate] = []

    for row in fetch_policies_for_category(product_type):
        policy = Policy.model_validate(row["policy"])
        regions = [code for code in row["granted_to_regions"] if code]
        outranks = [policy_id for policy_id in row["outranks"] if policy_id]

        candidates.append(
            PolicyCandidate(
                policy=policy,
                category=row["category"],
                granted_to_regions=regions,
                outranks=outranks,
                applies=policy_applies(policy, regions, context),
                conditions=describe_conditions(policy, regions, context),
            )
        )

    candidates.sort(key=_precedence_key)
    return candidates


def _precedence_key(candidate: PolicyCandidate) -> tuple:
    """Applicable first, then highest precedence, then a stable tie-break.

    Filter-then-rank, in one sort. `AU_BOOKLY_EXTENDED_RETURN` outranks
    `STANDARD_30_DAY`, but a customer in the UK must never be handed it on the
    strength of its precedence — so applicability is the first key, and
    precedence is only consulted among policies that already apply.
    """
    return (not candidate.applies, -candidate.policy.precedence, candidate.policy.policy_id)


def policy_applies(
    policy: Policy, granted_to_regions: list[str], context: PolicyContext
) -> bool:
    """Whether a policy's conditions are satisfied by a context.

    Two kinds of condition, both read off the graph rather than hardcoded:

    * A policy reached through `(:Region)-[:HAS_OVERRIDE]->(:Policy)` is offered
      to that region only. This is what stops a UK customer being handed
      `AU_BOOKLY_EXTENDED_RETURN`, and what makes an Australian customer — or an
      explicitly Australian question — reach it.
    * A policy carrying a `promotion_code` is offered only to orders placed under
      that promotion, inside its active dates. The extension is not a blanket 60
      days for everyone.

    A policy with neither condition — the default, `STANDARD_30_DAY` — always
    applies to its category.

    Args:
        policy: The policy under test.
        granted_to_regions: Region codes with a HAS_OVERRIDE edge to it.
        context: The region, promotion, and dates to test against.

    Returns:
        True if the policy governs in this context. A condition that cannot be
        shown to be met — an unknown region, a missing promotion date — is not
        met: absent data is never read as permission.
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
    """The conditions on a policy applying at all, in plain language.

    Used by the informational path, which has to be able to say "45 days, if you
    are in Australia" without either hiding the condition or refusing to state
    the rule.

    Args:
        policy: The policy being described.
        granted_to_regions: Region codes with a HAS_OVERRIDE edge to it.
        context: The context it was judged against — `as_of` decides whether a
            promotion is described as currently running.

    Returns:
        Zero or more sentences. Empty for an unconditional policy.
    """
    conditions: list[str] = []

    region = granted_to_regions[0] if granted_to_regions else None
    if region:
        conditions.append(f"applies only to customers in {region}")

    if policy.promotion_code:
        running = (
            context.as_of is not None and promotion_covers(policy, context.as_of)
        )
        conditions.append(
            f"applies only to orders placed under {policy.promotion_code} "
            f"({policy.promotion_active_from} to {policy.promotion_active_to}"
            f"{'' if running else ', not currently running'})"
        )

    if policy.window_days is None:
        conditions.append("no return window — returns are not offered")

    return conditions


def promotion_covers(policy: Policy, day: date) -> bool:
    """Whether a promotional policy's active window contains a date.

    A missing date on either end means the promotion cannot be shown to cover
    anything, so it does not.
    """
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
