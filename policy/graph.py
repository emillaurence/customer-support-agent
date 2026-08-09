"""The Neo4j connection every policy decision goes through.

Neo4j is a **required** dependency: return policy, regional overrides,
precedence, and the explainable rule path are all resolved by Cypher against the
graph. There is deliberately no JSON policy engine to fall back to — if the
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
"""The one driver for the process.

A Neo4j `Driver` owns a connection pool and is designed to be built once and
shared; every query borrows a short-lived session from it. Opening a driver per
call would pay TCP, TLS, and Bolt handshakes on every policy lookup — the
dominant cost of an otherwise millisecond query.
"""

POOL_SIZE = 20
"""Connections kept in the pool. Comfortably above the concurrency of one
Streamlit process, so a policy read never waits for a free connection."""

ACQUISITION_TIMEOUT_SECONDS = 5.0
"""How long a query waits for a connection before giving up. Bounded, so a wedged
database surfaces as an outage the customer is told about rather than a hang."""

TRANSACTION_RETRY_SECONDS = 5.0
"""The driver's own retry budget for transient cluster errors. Bounded for the
same reason, and left to the driver rather than reimplemented here."""


class PolicyGraphUnavailableError(RuntimeError):
    """Raised instead of degrading to JSON or a mocked result: a support agent
    that cannot read the policy graph must say so, not guess."""


def get_driver() -> Driver:
    """Return the shared driver, opening and verifying it on first use.

    Idempotent: after the first call this is a module-attribute read, so the
    per-query cost is borrowing a pooled connection and nothing else.
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
        config["NEO4J_URI"],
        auth=(config["NEO4J_USERNAME"], config["NEO4J_PASSWORD"]),
        max_connection_pool_size=POOL_SIZE,
        connection_acquisition_timeout=ACQUISITION_TIMEOUT_SECONDS,
        max_transaction_retry_time=TRANSACTION_RETRY_SECONDS,
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
    """Close the shared driver and drop it, so the next call opens a fresh one.

    For process shutdown and for tests. Idempotent.
    """
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


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
    policy conditional: `granted_to_regions` non-empty means it is only reachable
    for customers in those regions, and `outranks` is what it displaces. One row
    per policy, highest precedence first; empty for an unknown category.

    Runs through `execute_query`, which borrows a session from the shared
    driver's pool and returns it — the connection outlives the call, the session
    does not.

    Raises:
        PolicyGraphUnavailableError: If Neo4j is unconfigured or unreachable.
    """
    records, _, _ = get_driver().execute_query(CATEGORY_POLICIES, product_type=product_type)
    rows = [record.data() for record in records]
    for row in rows:
        row["policy"] = {key: _to_python(value) for key, value in row["policy"].items()}
    return rows


def _to_python(value: Any) -> Any:
    """Flatten a Neo4j temporal value so Pydantic can parse it as a `date`."""
    return value.to_native() if hasattr(value, "to_native") else value
