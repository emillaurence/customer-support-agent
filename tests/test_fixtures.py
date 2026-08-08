"""Fixture integrity: the mock data and the policy graph seed must agree.

These tests DO run. They assert nothing about product behaviour — only that
every JSON file parses, validates against the models, and cross-references
something that exists. Broken fixtures would otherwise show up much later as
confusing agent behaviour.

`data/` is mock transactional data only. Policy lives in Neo4j;
`neo4j/policy_graph.json` is the seed ingestion reads, and it is checked here as
seed data, not as a runtime policy source. Nothing here connects to Neo4j.
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
def returns() -> list[ReturnRecord]:
    return [ReturnRecord.model_validate(r) for r in _load(DATA / "returns.json")]


@pytest.fixture(scope="module")
def graph() -> dict:
    return _load(GRAPH)


@pytest.fixture(scope="module")
def policies(graph: dict) -> list[Policy]:
    """Policies come from the graph seed — there is no policy JSON in data/."""
    return [Policy.model_validate(p) for p in graph["policies"]]


# --- Every file parses and validates ------------------------------------


@pytest.mark.parametrize(
    "filename",
    ["customers.json", "items.json", "orders.json", "returns.json"],
)
def test_data_file_parses(filename: str) -> None:
    assert isinstance(_load(DATA / filename), list)


def test_policy_graph_has_every_section(graph: dict) -> None:
    for section in ("categories", "policies", "regions", "relationships"):
        assert graph[section], section


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


# --- The policy graph seed is well formed -------------------------------


def test_data_dir_holds_no_policy_json() -> None:
    """Policy is Neo4j's job. A policy file in data/ would invite a fallback."""
    assert sorted(p.name for p in DATA.glob("*.json")) == [
        "customers.json",
        "items.json",
        "orders.json",
        "returns.json",
    ]


def test_every_policy_node_validates(policies: list[Policy]) -> None:
    """The seed's Policy nodes match the model the tools will read them back as."""
    assert {p.policy_id for p in policies} == {
        "STANDARD_30_DAY",
        "DIGITAL_NO_RETURN",
        "HOLIDAY_EXTENDED_RETURN",
        "AU_BOOKLY_EXTENDED_RETURN",
    }
    for policy in policies:
        assert policy.name and policy.summary
        # A window without a start date, or a start date without a window,
        # would be undecidable at eligibility time.
        assert (policy.window_days is None) == (policy.window_starts_from is None)


def test_promotional_policy_has_an_active_window(policies: list[Policy]) -> None:
    for policy in policies:
        if policy.promotion_code:
            assert policy.promotion_active_from and policy.promotion_active_to
            assert policy.promotion_active_from <= policy.promotion_active_to


def test_digital_policy_has_no_window_and_no_overrides(graph: dict) -> None:
    """The absence of these is the rule that ebooks cannot be rescued."""
    digital = next(p for p in graph["policies"] if p["policy_id"] == "DIGITAL_NO_RETURN")
    assert digital["window_days"] is None
    assert digital["exceptions"] == []
    assert not [r for r in graph["relationships"] if r["to"] == "DIGITAL_NO_RETURN" and r["type"] != "GOVERNED_BY"]


def test_expected_policy_relationships_are_present(graph: dict) -> None:
    """The four edges the demo turns on."""
    rels = {(r["type"], r["from"], r["to"]) for r in graph["relationships"]}
    assert ("GOVERNED_BY", "PhysicalBook", "STANDARD_30_DAY") in rels
    assert ("GOVERNED_BY", "EBook", "DIGITAL_NO_RETURN") in rels
    assert ("OVERRIDES", "HOLIDAY_EXTENDED_RETURN", "STANDARD_30_DAY") in rels
    assert ("HAS_OVERRIDE", "AU", "AU_BOOKLY_EXTENDED_RETURN") in rels


def test_categories_match_the_product_type_enum(items: list[Item], graph: dict) -> None:
    names = {c["name"] for c in graph["categories"]}
    assert names == {t.value for t in ProductType}
    assert {i.product_type.value for i in items} <= names


def test_customer_countries_exist_as_regions(customers: list[Customer], graph: dict) -> None:
    codes = {r["code"] for r in graph["regions"]}
    for customer in customers:
        assert customer.country in codes, customer.customer_id


def test_order_promotions_exist_in_the_graph(orders: list[Order], graph: dict) -> None:
    codes = {p["promotion_code"] for p in graph["policies"] if p.get("promotion_code")}
    for order in orders:
        if order.promotion_code:
            assert order.promotion_code in codes, order.order_id


def test_promotional_order_was_placed_inside_the_promotion(orders: list[Order], graph: dict) -> None:
    """A promotional extension is only coherent if the dates line up."""
    promos = {
        p["promotion_code"]: p for p in graph["policies"] if p.get("promotion_code")
    }
    for order in orders:
        if not order.promotion_code:
            continue
        promo = promos[order.promotion_code]
        placed = order.placed_at.isoformat()
        assert promo["promotion_active_from"] <= placed <= promo["promotion_active_to"], order.order_id


def test_seed_cypher_mentions_every_policy(policies: list[Policy]) -> None:
    """Cheap consistency check between the fixture and the reference Cypher."""
    cypher = (ROOT / "neo4j" / "seed.cypher").read_text()
    for policy in policies:
        assert policy.policy_id in cypher, policy.policy_id


def test_seed_cypher_mentions_every_category_and_region(graph: dict) -> None:
    cypher = (ROOT / "neo4j" / "seed.cypher").read_text()
    for category in graph["categories"]:
        assert category["name"] in cypher
    for region in graph["regions"]:
        assert f"'{region['code']}'" in cypher
