"""The ingestion script's fixture validation.

These tests DO run, and they never touch Neo4j — only the pure functions that
check the fixture before any connection is opened. The point is that a typo in
policy_graph.json fails loudly here rather than producing a graph with a
silently missing edge.

`ingest.py` is a script in neo4j/, not part of an installed package, so it is
loaded by path. That also avoids putting the repo root on sys.path, where the
neo4j/ directory would shadow the neo4j driver package.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
INGEST_PATH = ROOT / "neo4j" / "ingest.py"


def _load_ingest():
    spec = importlib.util.spec_from_file_location("bookly_ingest", INGEST_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ingest = _load_ingest()


@pytest.fixture()
def fixture() -> dict[str, Any]:
    return json.loads((ROOT / "neo4j" / "policy_graph.json").read_text())


# --- The real fixture is valid ------------------------------------------


def test_real_fixture_loads() -> None:
    loaded = ingest.load_fixture()
    assert loaded["policies"] and loaded["relationships"]


def test_real_fixture_validates() -> None:
    loaded = ingest.load_fixture()
    index = ingest.build_index(loaded)
    ingest.validate_relationships(loaded, index)


def test_index_labels_every_node(fixture: dict[str, Any]) -> None:
    index = ingest.build_index(fixture)
    assert index["PhysicalBook"] == "Category"
    assert index["STANDARD_30_DAY"] == "Policy"
    assert index["AU"] == "Region"


def test_every_declared_node_type_is_present(fixture: dict[str, Any]) -> None:
    for section in ingest.NODE_SECTIONS:
        assert fixture[section], section


# --- Bad fixtures fail clearly ------------------------------------------


def test_missing_section_is_rejected(tmp_path: Path, fixture: dict[str, Any]) -> None:
    del fixture["regions"]
    path = tmp_path / "policy_graph.json"
    path.write_text(json.dumps(fixture))
    with pytest.raises(ingest.IngestError, match="missing the 'regions' section"):
        ingest.load_fixture(path)


def test_unparseable_fixture_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "policy_graph.json"
    path.write_text("{not json")
    with pytest.raises(ingest.IngestError, match="not valid JSON"):
        ingest.load_fixture(path)


def test_missing_fixture_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ingest.IngestError, match="not found"):
        ingest.load_fixture(tmp_path / "nope.json")


def test_unknown_relationship_type_is_rejected(fixture: dict[str, Any]) -> None:
    """The requirement that matters: no silent skipping."""
    fixture["relationships"].append(
        {"type": "HAS_WINDOW", "from": "STANDARD_30_DAY", "to": "AU"}
    )
    index = ingest.build_index(fixture)
    with pytest.raises(ingest.IngestError, match="unknown relationship type 'HAS_WINDOW'"):
        ingest.validate_relationships(fixture, index)


def test_unknown_endpoint_is_rejected(fixture: dict[str, Any]) -> None:
    fixture["relationships"].append(
        {"type": "OVERRIDES", "from": "STANDARD_30_DAY", "to": "TYPO_POLICY"}
    )
    index = ingest.build_index(fixture)
    with pytest.raises(ingest.IngestError, match="matches no category, policy, or region"):
        ingest.validate_relationships(fixture, index)


def test_wrong_endpoint_labels_are_rejected(fixture: dict[str, Any]) -> None:
    """GOVERNED_BY must go Category -> Policy, not Region -> Policy."""
    fixture["relationships"].append(
        {"type": "GOVERNED_BY", "from": "AU", "to": "STANDARD_30_DAY"}
    )
    index = ingest.build_index(fixture)
    with pytest.raises(ingest.IngestError, match=r"must go \(Category\)->\(Policy\)"):
        ingest.validate_relationships(fixture, index)


def test_duplicate_node_key_is_rejected(fixture: dict[str, Any]) -> None:
    """Endpoints are bare keys, so a key shared across sections is ambiguous."""
    fixture["regions"].append({"code": "PhysicalBook", "name": "Nonsense"})
    with pytest.raises(ingest.IngestError, match="duplicate node key"):
        ingest.build_index(fixture)


def test_node_without_its_key_property_is_rejected(fixture: dict[str, Any]) -> None:
    fixture["policies"].append({"name": "No id"})
    with pytest.raises(ingest.IngestError, match="no 'policy_id'"):
        ingest.build_index(fixture)
