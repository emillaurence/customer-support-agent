"""Shared test fixtures.

Two things make the tool tests deterministic:

* **A fixed clock.** The order fixtures are written relative to 2026-08-08 —
  "day 11", "day 34", "day 67". The tools default to `datetime.now(UTC)`, and
  every test passes `FIXED_NOW` instead, so the scenarios keep meaning what they
  say however long the repo sits.
* **A copy of the data.** `data_dir` redirects the tools at a temporary copy of
  `data/`, so `initiate_return` can be tested for real without a passing test run
  leaving an RMA in the repo.

`seeded_graph` is the offline stand-in for Neo4j. It answers the one policy query
the tools make, built from `neo4j/policy_graph.json` — the same seed that was
ingested into the database — so unit tests exercise the real decision logic
without a database. `tests/test_neo4j_integration.py` runs the same cases against
the live graph, which is what proves the stub is honest.
"""

from __future__ import annotations

import importlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
POLICY_SEED = ROOT / "neo4j" / "policy_graph.json"

FIXED_NOW = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
"""The date the order fixtures are written against."""


@pytest.fixture
def now() -> datetime:
    """The fixed clock, for tests that pass it through to a tool."""
    return FIXED_NOW


@pytest.fixture(autouse=True)
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the tools at a throwaway copy of `data/`.

    Autouse: no test should be able to write to the repo's fixtures, and a test
    that mutates returns.json must not change what the next test reads.
    """
    from tools import fixtures

    copy = tmp_path / "data"
    shutil.copytree(ROOT / "data", copy)
    monkeypatch.setattr(fixtures, "DATA_DIR", copy)
    return copy


@pytest.fixture(autouse=True)
def clean_tokens():
    """Empty the token store around every test.

    The store is process-global, so a token minted by one test would otherwise be
    spendable in another — exactly the confusion the store exists to prevent.
    """
    from tools import eligibility_tokens

    eligibility_tokens.clear()
    yield
    eligibility_tokens.clear()


# --- Offline policy graph -----------------------------------------------


def _seed() -> dict[str, Any]:
    return json.loads(POLICY_SEED.read_text())


def policy_rows_for_category(product_type: str) -> list[dict[str, Any]]:
    """Build what `fetch_policies_for_category` returns, from the seed file.

    Same shape as the Cypher result: the policy's properties, the regions with a
    HAS_OVERRIDE edge to it, and the policy ids it OVERRIDES, ordered by
    precedence descending.

    Args:
        product_type: A category name.

    Returns:
        One row per policy governing that category.
    """
    seed = _seed()
    policies = {policy["policy_id"]: policy for policy in seed["policies"]}
    relationships = seed["relationships"]

    governed = [
        rel["to"]
        for rel in relationships
        if rel["type"] == "GOVERNED_BY" and rel["from"] == product_type
    ]

    rows = [
        {
            "category": product_type,
            # Nulls dropped, because Neo4j does not store null properties: a row
            # off the real graph has the key missing, not set to None. Matching
            # that keeps the stub honest about the shape the tools handle.
            "policy": {
                key: value for key, value in policies[policy_id].items() if value is not None
            },
            "granted_to_regions": [
                rel["from"]
                for rel in relationships
                if rel["type"] == "HAS_OVERRIDE" and rel["to"] == policy_id
            ],
            "outranks": [
                rel["to"]
                for rel in relationships
                if rel["type"] == "OVERRIDES" and rel["from"] == policy_id
            ],
        }
        for policy_id in governed
    ]
    rows.sort(key=lambda row: (-row["policy"]["precedence"], row["policy"]["policy_id"]))
    return rows


def _policy_reading_modules() -> list[Any]:
    """The two modules holding a reference to `fetch_policies_for_category`.

    Fetched from `sys.modules` by name. `tools/__init__` re-exports each tool
    *function* under its module's name, so `tools.search_policy` is the function
    and a plain `import ... as` would hand back the wrong object.
    """
    return [
        importlib.import_module(f"tools.{name}")
        for name in ("check_return_eligibility", "search_policy")
    ]


def break_policy_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every policy read raise, as an unreachable Neo4j would.

    A plain function rather than a fixture, so a test can call it after
    `seeded_graph` has been applied and be sure it wins.

    Args:
        monkeypatch: The calling test's monkeypatch.
    """
    from agent.graph import PolicyGraphUnavailableError

    def unavailable(_product_type: str):
        raise PolicyGraphUnavailableError("cannot reach Neo4j at bolt://localhost:7687")

    for module in _policy_reading_modules():
        monkeypatch.setattr(module, "fetch_policies_for_category", unavailable)


@pytest.fixture
def seeded_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer policy queries from the seed file instead of Neo4j.

    Patched on the two tools that read the graph, at the name each of them
    imported, so nothing reaches a driver. Requested explicitly rather than
    autouse — the integration tests must not get it.
    """
    for module in _policy_reading_modules():
        monkeypatch.setattr(module, "fetch_policies_for_category", policy_rows_for_category)
