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
