"""Fixture integrity: the mock data and the policy graph must agree.

These tests DO run. They assert nothing about product behaviour — only that
every JSON file parses, validates against the models, and cross-references
something that exists. Broken fixtures would otherwise show up much later as
confusing agent behaviour.

Nothing here touches Neo4j.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.models import Customer, Item, Order, Policy, ProductType, ReturnRecord

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
GRAPH = ROOT / "neo4j" / "policy_graph.json"


def _load(path: Path) -> list | dict:
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def customers() -> list[Customer]:
    return [Customer.model_validate(c) for c in _load(DATA / "customers.json")]


@pytest.fixture(scope="module")
def items() -> list[Item]:
    return [Item.model_validate(i) for i in _load(DATA / "items.json")]


@pytest.fixture(scope="module")
def orders() -> list[Order]:
    return [Order.model_validate(o) for o in _load(DATA / "orders.json")]


@pytest.fixture(scope="module")
def policies() -> list[Policy]:
    return [Policy.model_validate(p) for p in _load(DATA / "policies.json")]


@pytest.fixture(scope="module")
def returns() -> list[ReturnRecord]:
    return [ReturnRecord.model_validate(r) for r in _load(DATA / "returns.json")]


@pytest.fixture(scope="module")
def graph() -> dict:
    return _load(GRAPH)


# --- Every file parses and validates ------------------------------------


@pytest.mark.parametrize(
    "filename",
    ["customers.json", "items.json", "orders.json", "policies.json", "returns.json"],
)
def test_data_file_parses(filename: str) -> None:
    assert isinstance(_load(DATA / filename), list)


def test_policy_graph_parses(graph: dict) -> None:
    assert graph["nodes"] and graph["relationships"]


# --- Ids are unique -----------------------------------------------------


def test_ids_are_unique(
    customers: list[Customer],
    items: list[Item],
    orders: list[Order],
    policies: list[Policy],
    returns: list[ReturnRecord],
) -> None:
    for label, ids in [
        ("customer", [c.customer_id for c in customers]),
        ("item", [i.item_id for i in items]),
        ("order", [o.order_id for o in orders]),
        ("policy", [p.policy_id for p in policies]),
        ("return", [r.return_id for r in returns]),
    ]:
        assert len(ids) == len(set(ids)), f"duplicate {label} id"


# --- Cross-references resolve -------------------------------------------


def test_orders_reference_real_customers_and_items(
    orders: list[Order], customers: list[Customer], items: list[Item]
) -> None:
    customer_ids = {c.customer_id for c in customers}
    item_ids = {i.item_id for i in items}
    for order in orders:
        assert order.customer_id in customer_ids, order.order_id
        assert order.items, f"{order.order_id} has no line items"
        for line in order.items:
            assert line.item_id in item_ids, f"{order.order_id} -> {line.item_id}"


def test_returns_reference_real_orders_and_items(
    returns: list[ReturnRecord], orders: list[Order]
) -> None:
    by_id = {o.order_id: o for o in orders}
    for record in returns:
        order = by_id.get(record.order_id)
        assert order is not None, record.return_id
        assert record.customer_id == order.customer_id
        assert record.item_id in {line.item_id for line in order.items}


def test_delivered_orders_have_a_delivery_date(orders: list[Order]) -> None:
    for order in orders:
        if order.status == "delivered":
            assert order.delivered_at is not None, order.order_id
            assert order.delivered_at >= order.placed_at, order.order_id
        else:
            assert order.delivered_at is None, order.order_id


# --- policies.json and the graph agree ----------------------------------


def _graph_nodes(graph: dict, label: str) -> dict[str, dict]:
    return {n["key"]: n for n in graph["nodes"] if label in n["labels"]}


def test_policy_ids_match_the_graph(policies: list[Policy], graph: dict) -> None:
    assert {p.policy_id for p in policies} == set(_graph_nodes(graph, "Policy"))


def test_policy_properties_match_the_graph(policies: list[Policy], graph: dict) -> None:
    nodes = _graph_nodes(graph, "Policy")
    for policy in policies:
        props = nodes[policy.policy_id]["properties"]
        assert policy.name == props["name"]
        assert policy.summary == props["summary"]
        assert policy.precedence == props["precedence"]


def test_policy_windows_match_the_graph(policies: list[Policy], graph: dict) -> None:
    """window_days in JSON must equal the ReturnWindow the graph points at."""
    windows = {k: n["properties"]["days"] for k, n in _graph_nodes(graph, "ReturnWindow").items()}
    has_window = {
        r["from"]: r["to"] for r in graph["relationships"] if r["type"] == "HAS_WINDOW"
    }
    for policy in policies:
        target = has_window.get(policy.policy_id)
        expected = windows[target] if target else None
        assert policy.window_days == expected, policy.policy_id


def test_digital_policy_has_no_window_and_no_overrides(graph: dict) -> None:
    """The absence of these edges is the rule that ebooks cannot be rescued."""
    rels = graph["relationships"]
    assert not [r for r in rels if r["type"] == "HAS_WINDOW" and r["from"] == "DIGITAL_NO_RETURN"]
    assert not [r for r in rels if r["type"] == "OVERRIDES" and r["to"] == "DIGITAL_NO_RETURN"]
    assert not [r for r in rels if r["type"] == "WAIVES" and r["to"] == "DIGITAL_NO_RETURN"]


def test_graph_relationship_endpoints_exist(graph: dict) -> None:
    keys = {n["key"] for n in graph["nodes"]}
    for rel in graph["relationships"]:
        assert rel["from"] in keys, rel
        assert rel["to"] in keys, rel


def test_expected_policy_relationships_are_present(graph: dict) -> None:
    """The four edges the demo turns on."""
    rels = {(r["type"], r["from"], r["to"]) for r in graph["relationships"]}
    assert ("GOVERNED_BY", "PhysicalBook", "STANDARD_30_DAY") in rels
    assert ("GOVERNED_BY", "EBook", "DIGITAL_NO_RETURN") in rels
    assert ("OVERRIDES", "HOLIDAY_EXTENDED_RETURN", "STANDARD_30_DAY") in rels
    assert ("HAS_OVERRIDE", "AU", "AU_BOOKLY_EXTENDED_RETURN") in rels


def test_product_types_match_the_enum(items: list[Item], graph: dict) -> None:
    graph_types = {n["properties"]["name"] for n in _graph_nodes(graph, "ProductType").values()}
    assert graph_types == {t.value for t in ProductType}
    assert {i.product_type.value for i in items} <= graph_types


# --- Regions and promotions ---------------------------------------------


def test_customer_countries_exist_as_regions(customers: list[Customer], graph: dict) -> None:
    codes = {n["properties"]["code"] for n in _graph_nodes(graph, "Region").values()}
    for customer in customers:
        assert customer.country in codes, customer.customer_id


def test_order_promotions_exist_in_the_graph(orders: list[Order], graph: dict) -> None:
    codes = {n["properties"]["code"] for n in _graph_nodes(graph, "Promotion").values()}
    for order in orders:
        if order.promotion_code:
            assert order.promotion_code in codes, order.order_id


def test_promotional_order_was_delivered_inside_the_promotion(orders: list[Order], graph: dict) -> None:
    """A promotional extension is only coherent if the dates line up."""
    promos = {
        n["properties"]["code"]: n["properties"]
        for n in _graph_nodes(graph, "Promotion").values()
    }
    for order in orders:
        if not order.promotion_code:
            continue
        promo = promos[order.promotion_code]
        assert promo["active_from"] <= order.placed_at.isoformat() <= promo["active_to"], order.order_id


def test_seed_cypher_mentions_every_policy(policies: list[Policy]) -> None:
    """Cheap consistency check between the fixture and the Cypher seed."""
    cypher = (ROOT / "neo4j" / "seed.cypher").read_text()
    for policy in policies:
        assert policy.policy_id in cypher, policy.policy_id
