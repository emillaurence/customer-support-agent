"""The Neo4j connection every policy decision goes through.

Neo4j is a **required** dependency. Return policy, eligibility rules, policy
overrides, regional overrides, and the explainable rule path are all resolved by
Cypher against the graph. There is no JSON policy engine to fall back to: if the
database is unconfigured or unreachable, `get_driver` raises and the tool fails
loudly rather than answering a customer from a stale fixture.

`neo4j/policy_graph.json` is seed data for `neo4j/ingest.py` only. Nothing here
reads it.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

REQUIRED_ENV = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")

_driver: Driver | None = None


class PolicyGraphUnavailableError(RuntimeError):
    """Neo4j is not configured or not reachable.

    Raised instead of degrading to JSON or returning a mocked policy result.
    Callers should let it surface: a support agent that cannot read the policy
    graph must say so, not guess.
    """


def get_driver() -> Driver:
    """Return the shared driver, opening and verifying it on first use.

    Returns:
        A driver whose connectivity has been confirmed.

    Raises:
        PolicyGraphUnavailableError: If any of NEO4J_URI, NEO4J_USERNAME, or
            NEO4J_PASSWORD is unset, or the server does not answer.
    """
    global _driver
    if _driver is not None:
        return _driver

    load_dotenv()
    config = {name: os.getenv(name) for name in REQUIRED_ENV}
    missing = [name for name, value in config.items() if not value]
    if missing:
        raise PolicyGraphUnavailableError(
            f"Neo4j is required for policy and eligibility decisions, but "
            f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not set. "
            f"Copy .env.example to .env and fill it in."
        )

    driver = GraphDatabase.driver(
        config["NEO4J_URI"], auth=(config["NEO4J_USERNAME"], config["NEO4J_PASSWORD"])
    )
    try:
        driver.verify_connectivity()
    except Exception as exc:  # noqa: BLE001 - surface the driver's own message
        driver.close()
        raise PolicyGraphUnavailableError(
            f"cannot reach Neo4j at {config['NEO4J_URI']}: {exc}. "
            f"Start the database and run `python neo4j/ingest.py`."
        ) from exc

    _driver = driver
    return _driver


def close_driver() -> None:
    """Close the shared driver, if one was opened. Safe to call twice."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


# --- Policy retrieval ---------------------------------------------------
#
# One Cypher query, used by both policy tools. `search_policy` presents the
# result to a customer as information; `check_return_eligibility` filters it by
# applicability and decides. Sharing the read keeps the two from drifting into
# two different ideas of what the graph says, without either owning the other.

CATEGORY_POLICIES = """
MATCH (category:Category {name: $product_type})-[:GOVERNED_BY]->(policy:Policy)
OPTIONAL MATCH (region:Region)-[:HAS_OVERRIDE]->(policy)
OPTIONAL MATCH (policy)-[:OVERRIDES]->(outranked:Policy)
RETURN category.name           AS category,
       properties(policy)      AS policy,
       collect(DISTINCT region.code)        AS granted_to_regions,
       collect(DISTINCT outranked.policy_id) AS outranks
ORDER BY policy.precedence DESC, policy.policy_id
"""


def fetch_policies_for_category(product_type: str) -> list[dict[str, Any]]:
    """Read every policy governing one product category, with its edges.

    The edges come back alongside the properties because they are what makes a
    policy conditional. `granted_to_regions` non-empty means the policy is only
    reachable for customers in those regions; `outranks` is what the policy
    displaces, and is what an explanation is built from.

    Args:
        product_type: A `:Category` name — 'PhysicalBook' or 'EBook'.

    Returns:
        One row per policy, highest precedence first. Empty if the category has
        no policies, or does not exist.

    Raises:
        PolicyGraphUnavailableError: If Neo4j is unconfigured or unreachable.
    """
    records, _, _ = get_driver().execute_query(CATEGORY_POLICIES, product_type=product_type)
    rows = [record.data() for record in records]
    for row in rows:
        row["policy"] = {key: _to_python(value) for key, value in row["policy"].items()}
    return rows


def _to_python(value: Any) -> Any:
    """Flatten a Neo4j temporal value to something Pydantic can parse.

    The seed stores dates as ISO strings, so this is usually a no-op — but a
    property written as a Cypher `date()` would come back as a `neo4j.time.Date`,
    and `Policy` expects a `date`.
    """
    return value.to_native() if hasattr(value, "to_native") else value
