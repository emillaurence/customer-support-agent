"""Ingest the Bookly policy graph into Neo4j.

Reads neo4j/policy_graph.json — the single source of the policy data — and
MERGEs it into Neo4j. Nothing about the graph is hardcoded here; this file only
knows the *shape* (which sections exist, which labels and relationship types are
allowed), never the values.

Idempotent: constraints are IF NOT EXISTS, and every node and relationship is
MERGEd on its natural key. Running it twice changes nothing.

Usage:
    python neo4j/ingest.py

Requires NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD in .env. Neo4j is a
required dependency of this repo: policy_graph.json is seed data for this
script, not a runtime policy store, so the app has nothing to read until this
has run. There is no fallback path — if the database is unreachable, ingestion
fails and says why.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

FIXTURE = Path(__file__).resolve().parent / "policy_graph.json"

# Each section maps to one label, keyed on one property. Adding a node type is a
# one-line change here plus a constraint below.
NODE_SECTIONS: dict[str, tuple[str, str]] = {
    "categories": ("Category", "name"),
    "policies": ("Policy", "policy_id"),
    "regions": ("Region", "code"),
}

# Allowed relationship types, and which labels they may join. Anything outside
# this table is a hard failure — a typo in the fixture must not silently
# produce a graph that is missing an edge.
RELATIONSHIP_TYPES: dict[str, tuple[str, str]] = {
    "GOVERNED_BY": ("Category", "Policy"),
    "OVERRIDES": ("Policy", "Policy"),
    "HAS_OVERRIDE": ("Region", "Policy"),
}

KEY_BY_LABEL: dict[str, str] = {label: key for label, key in NODE_SECTIONS.values()}

CONSTRAINTS: list[str] = [
    "CREATE CONSTRAINT category_name IF NOT EXISTS "
    "FOR (c:Category) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT policy_id IF NOT EXISTS "
    "FOR (p:Policy) REQUIRE p.policy_id IS UNIQUE",
    "CREATE CONSTRAINT region_code IF NOT EXISTS "
    "FOR (r:Region) REQUIRE r.code IS UNIQUE",
]


class IngestError(RuntimeError):
    """The fixture or the environment is not usable. Reported, never swallowed."""


def load_fixture(path: Path = FIXTURE) -> dict[str, Any]:
    """Read and sanity-check policy_graph.json.

    Args:
        path: Location of the fixture.

    Returns:
        The parsed fixture.

    Raises:
        IngestError: If the file is missing, unparseable, or a section is absent.
    """
    if not path.exists():
        raise IngestError(f"fixture not found: {path}")
    try:
        fixture = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise IngestError(f"{path.name} is not valid JSON: {exc}") from exc

    for section in (*NODE_SECTIONS, "relationships"):
        if section not in fixture:
            raise IngestError(f"{path.name} is missing the '{section}' section")
    return fixture


def build_index(fixture: dict[str, Any]) -> dict[str, str]:
    """Map every node's natural key to its label.

    Relationship endpoints in the fixture are bare keys — 'PhysicalBook', 'AU',
    'STANDARD_30_DAY' — so they have to be resolved to a label before they can
    be matched.

    Args:
        fixture: The parsed fixture.

    Returns:
        Natural key -> label.

    Raises:
        IngestError: If a node is missing its key property, or two nodes share
            a key across sections (which would make endpoints ambiguous).
    """
    index: dict[str, str] = {}
    for section, (label, key_property) in NODE_SECTIONS.items():
        for node in fixture[section]:
            key = node.get(key_property)
            if not key:
                raise IngestError(f"{section}: a node has no '{key_property}'")
            if key in index:
                raise IngestError(f"duplicate node key '{key}' ({index[key]} and {label})")
            index[key] = label
    return index


def validate_relationships(fixture: dict[str, Any], index: dict[str, str]) -> None:
    """Check every relationship before touching the database.

    Validated up front so ingestion is all-or-nothing: a bad edge fails the run
    rather than leaving a half-populated graph.

    Args:
        fixture: The parsed fixture.
        index: Natural key -> label, from `build_index`.

    Raises:
        IngestError: On an unknown relationship type, an unknown endpoint, or
            endpoints whose labels the relationship type does not allow.
    """
    for rel in fixture["relationships"]:
        rel_type = rel.get("type")
        source, target = rel.get("from"), rel.get("to")

        if rel_type not in RELATIONSHIP_TYPES:
            raise IngestError(
                f"unknown relationship type {rel_type!r} "
                f"({source} -> {target}). Known types: {', '.join(RELATIONSHIP_TYPES)}"
            )
        for endpoint in (source, target):
            if endpoint not in index:
                raise IngestError(
                    f"{rel_type}: endpoint {endpoint!r} matches no category, policy, or region"
                )

        expected_source, expected_target = RELATIONSHIP_TYPES[rel_type]
        if index[source] != expected_source or index[target] != expected_target:
            raise IngestError(
                f"{rel_type} must go ({expected_source})->({expected_target}), "
                f"but {source} is a {index[source]} and {target} is a {index[target]}"
            )


def connect() -> Driver:
    """Open a driver from the environment and confirm the server answers.

    Returns:
        A connected driver. The caller closes it.

    Raises:
        IngestError: If NEO4J_URI, NEO4J_USERNAME, or NEO4J_PASSWORD is unset.
        Exception: The driver's own error if the server cannot be reached or
            the credentials are refused. Reported by `main`, never swallowed.
    """
    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    missing = [
        name
        for name, value in [
            ("NEO4J_URI", uri),
            ("NEO4J_USERNAME", username),
            ("NEO4J_PASSWORD", password),
        ]
        if not value
    ]
    if missing:
        raise IngestError(f"missing in .env: {', '.join(missing)}")

    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()
    print(f"connected to {uri} as {username}")
    return driver


def create_constraints(driver: Driver) -> None:
    """Apply the uniqueness constraints the MERGEs rely on."""
    for statement in CONSTRAINTS:
        driver.execute_query(statement)
    print(f"constraints ready ({len(CONSTRAINTS)})")


def ingest_nodes(driver: Driver, fixture: dict[str, Any]) -> int:
    """MERGE every node, then overwrite its properties.

    MERGE on the key alone and `SET n += props` afterwards, so re-running picks
    up edited property values instead of creating a second node.

    Returns:
        How many nodes were written.
    """
    total = 0
    for section, (label, key_property) in NODE_SECTIONS.items():
        nodes = fixture[section]
        # The label and key property come from NODE_SECTIONS, never from the
        # fixture, so this interpolation cannot be influenced by the data.
        query = (
            f"UNWIND $nodes AS node "
            f"MERGE (n:{label} {{{key_property}: node.`{key_property}`}}) "
            f"SET n += node"
        )
        driver.execute_query(query, nodes=nodes)
        print(f"  {label:<9} {len(nodes)} node(s)")
        total += len(nodes)
    return total


def ingest_relationships(driver: Driver, fixture: dict[str, Any]) -> int:
    """MERGE every relationship, grouped by type.

    Cypher cannot parameterise a relationship type, so it is interpolated — but
    only after `validate_relationships` has confirmed it is one of the three
    known types.

    Returns:
        How many relationships were written.
    """
    by_type: dict[str, list[dict[str, str]]] = {}
    for rel in fixture["relationships"]:
        by_type.setdefault(rel["type"], []).append(rel)

    total = 0
    for rel_type, rels in by_type.items():
        source_label, target_label = RELATIONSHIP_TYPES[rel_type]
        source_key = KEY_BY_LABEL[source_label]
        target_key = KEY_BY_LABEL[target_label]
        query = (
            f"UNWIND $rels AS rel "
            f"MATCH (a:{source_label} {{{source_key}: rel.from}}) "
            f"MATCH (b:{target_label} {{{target_key}: rel.to}}) "
            f"MERGE (a)-[:{rel_type}]->(b)"
        )
        driver.execute_query(query, rels=rels)
        print(f"  {rel_type:<13} {len(rels)} relationship(s)")
        total += len(rels)
    return total


def main() -> int:
    """Load the fixture, validate it, and ingest it.

    Returns:
        A process exit code: 0 on success, 1 on a reported failure.
    """
    try:
        fixture = load_fixture()
        index = build_index(fixture)
        validate_relationships(fixture, index)
        print(f"fixture ok: {len(index)} nodes, {len(fixture['relationships'])} relationships")
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    driver = None
    try:
        driver = connect()
        create_constraints(driver)
        print("ingesting nodes...")
        nodes = ingest_nodes(driver, fixture)
        print("ingesting relationships...")
        relationships = ingest_relationships(driver, fixture)
        print(f"done: {nodes} nodes, {relationships} relationships (idempotent)")
        return 0
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - surface the driver's own message
        print(f"error: Neo4j ingestion failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if driver is not None:
            driver.close()
            print("driver closed")


if __name__ == "__main__":
    raise SystemExit(main())
