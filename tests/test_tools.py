"""The tools as ordinary Python: identity, orders, escalation, and the one write.

No model is involved in any of this. Each tool takes plain arguments, reads its
own data, and returns a typed result — which is exactly how the agent calls them,
minus the dispatch.

The behaviours worth locking down are the refusals: an unknown email, an order
number that does not exist, and an order that exists but belongs to someone else.
The guards on the write are in `test_guardrails.py`; what is here is that the
write works and that it writes once.

The last section checks the mock data itself. Broken fixtures would otherwise
show up much later as confusing agent behaviour.
"""

from __future__ import annotations

import inspect
import json
from datetime import datetime
from pathlib import Path

import pytest

from agent.state import (
    Customer,
    Item,
    Order,
    OrderStatus,
    ReturnRecord,
    SessionState,
    ToolStatus,
)
from agent.tools import (
    TOOL_SCHEMAS,
    _load_returns,
    active_order_ids,
    apply_tool_result,
    check_return_eligibility,
    escalate_to_human,
    initiate_return,
    invoke_tool,
    lookup_order,
    verify_identity,
)
from tests.conftest import HERO_CUSTOMER, IN_WINDOW_ITEM, IN_WINDOW_ORDER

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def test_all_six_tools_are_offered_to_the_model() -> None:
    """The six capabilities, and nothing else. A seventh would need a schema."""
    assert [schema["name"] for schema in TOOL_SCHEMAS] == [
        "verify_identity",
        "lookup_order",
        "search_policy",
        "check_return_eligibility",
        "initiate_return",
        "escalate_to_human",
    ]


# --- verify_identity ----------------------------------------------------


def test_known_email_verifies() -> None:
    result = verify_identity("ada@example.com")
    assert result.verified is True
    assert result.customer_id == HERO_CUSTOMER
    assert result.region == "GB"


def test_result_carries_customer_and_region() -> None:
    """The two things the agent needs to carry forward, plus a name to use."""
    result = verify_identity("bruce@example.com")
    assert (result.customer_id, result.name, result.region) == ("CUST-002", "Bruce", "AU")


def test_unknown_email_is_rejected() -> None:
    result = verify_identity("nobody@example.com")
    assert result.verified is False
    assert result.customer_id is None
    assert result.region is None


@pytest.mark.parametrize("email", ["", "   ", "ada@", "ada"])
def test_invalid_email_fails_cleanly(email: str) -> None:
    """No exception, no partial result — just an unverified answer."""
    result = verify_identity(email)
    assert result.verified is False
    assert result.message


def test_email_match_ignores_case_and_whitespace() -> None:
    """Customers type their address; they do not paste it."""
    result = verify_identity("  Ada@Example.COM  ")
    assert result.verified is True
    assert result.customer_id == HERO_CUSTOMER


def test_verify_identity_answers_only_the_identity_question() -> None:
    """Who they are and which region's rules apply. Nothing about their orders.

    Order ids, titles, statuses, dates, prices, and promotions are lookup_order's
    to give out — verification that also listed orders was two tools under one
    name, and neither question could be asked on its own.
    """
    dumped = verify_identity("ada@example.com").model_dump()

    assert set(dumped) == {"verified", "customer_id", "name", "region", "message"}
    assert "ORD-" not in json.dumps(dumped)


def test_rejection_does_not_say_whether_the_address_is_known() -> None:
    """Confirming an address is not a Bookly customer is itself a disclosure."""
    message = verify_identity("nobody@example.com").message.lower()
    assert "cust-" not in message
    assert "ord-" not in message


def test_active_orders_are_newest_first() -> None:
    """ORD-1005 was placed 2026-08-05, ORD-1004 on 2026-08-01."""
    assert active_order_ids("CUST-003") == ["ORD-1005", "ORD-1004"]


def test_every_fixture_customer_has_more_than_one_active_order() -> None:
    """The fixtures make the ambiguous case the default, deliberately.

    Guessing which order a customer means is the failure the whole flow is built
    to avoid, so the data does not let the agent get away with it once.
    """
    for customer_id in ("CUST-001", "CUST-002", "CUST-003", "CUST-004"):
        assert len(active_order_ids(customer_id)) > 1


# --- lookup_order: discovery --------------------------------------------
#
# One tool, two modes. Without an order id it lists what the customer has, which
# is how a title becomes an order id and an item id; with one it reads that order
# in full. Ownership is enforced the same way in both.


def test_lookup_order_lists_the_customers_orders() -> None:
    """Bruce's two orders, newest first, with the books on each."""
    result = lookup_order("CUST-002")

    assert [order.order_id for order in result.orders] == ["ORD-1007", "ORD-1003"]
    assert [item.title for item in result.orders[1].items] == [
        "A Short History of Nearly Everything",
        "Clean Architecture (ebook)",
    ]


def test_listed_orders_carry_the_item_ids_needed_to_act() -> None:
    """The point of the list mode: a title the customer named resolves to an
    order and an item without a second lookup."""
    result = lookup_order("CUST-002")

    match = [
        (order.order_id, item.item_id)
        for order in result.orders
        for item in order.items
        if item.title.startswith("Clean Architecture")
    ]
    assert match == [("ORD-1003", "ITEM-201")]


def test_listed_orders_carry_no_order_detail() -> None:
    """Enough to find the right order, and no more: dates, prices, quantities,
    and promotions belong to the detailed mode."""
    dumped = lookup_order("CUST-002").model_dump()

    assert set(dumped["orders"][0]) == {"order_id", "status", "items"}
    assert set(dumped["orders"][0]["items"][0]) == {"item_id", "title", "product_type"}


def test_listing_someone_elses_orders_is_impossible() -> None:
    """The list is built from the customer id, so there is nothing to spoof."""
    listed = {order.order_id for order in lookup_order(HERO_CUSTOMER).orders}

    assert listed == {IN_WINDOW_ORDER, "ORD-1002"}
    assert "ORD-1003" not in listed  # Bruce's


def test_listing_an_unknown_customer_is_empty_not_an_error() -> None:
    assert lookup_order("CUST-999").orders == []


# --- lookup_order: one order --------------------------------------------


def test_lookup_order_returns_items(now: datetime) -> None:
    details = lookup_order("CUST-002", "ORD-1003", now=now)
    assert details is not None
    assert [item.item_id for item in details.items] == ["ITEM-102", "ITEM-201"]
    assert details.items[0].title == "A Short History of Nearly Everything"


def test_lookup_order_unknown_id_returns_none(now: datetime) -> None:
    """An order number that does not exist yields None, not an error."""
    assert lookup_order(HERO_CUSTOMER, "ORD-9999", now=now) is None


def test_lookup_order_rejects_other_customers_order(now: datetime) -> None:
    """ORD-1008 is CUST-004's. CUST-001 quoting the real number gets nothing."""
    assert lookup_order(HERO_CUSTOMER, "ORD-1008", now=now) is None


def test_missing_and_forbidden_orders_are_indistinguishable(now: datetime) -> None:
    """Same answer either way, so a response cannot confirm an order id is real."""
    assert lookup_order(HERO_CUSTOMER, "ORD-1008", now=now) == lookup_order(
        HERO_CUSTOMER, "ORD-9999", now=now
    )


def test_order_details_need_a_customer_id() -> None:
    """There is no way to call lookup_order without saying who is asking — in
    either mode. The order id is the optional one."""
    parameters = inspect.signature(lookup_order).parameters

    assert parameters["customer_id"].default is inspect.Parameter.empty
    assert parameters["order_id"].default is None


def test_lookup_order_does_not_leak_fixture_annotations(now: datetime) -> None:
    """`scenario` is a note to whoever reads the JSON, not data for the model."""
    details = lookup_order("CUST-002", "ORD-1007", now=now)
    assert details is not None
    assert details.order.scenario is None
    assert "RET-5001" not in details.model_dump_json()


def test_shipment_reports_days_since_delivery(now: datetime) -> None:
    """ORD-1001 was delivered 2026-07-28 — 11 days before the fixed clock."""
    details = lookup_order(HERO_CUSTOMER, IN_WINDOW_ORDER, now=now)
    assert details is not None
    assert details.shipment.has_arrived is True
    assert details.shipment.days_since_delivery == 11
    assert details.shipment.status == OrderStatus.DELIVERED


def test_in_transit_order_has_no_delivery_date(now: datetime) -> None:
    """ORD-1005 is still in transit, so no return clock has started."""
    details = lookup_order("CUST-003", "ORD-1005", now=now)
    assert details is not None
    assert details.order.delivered_at is None
    assert details.order.status == OrderStatus.IN_TRANSIT
    assert details.shipment.has_arrived is False
    # Not estimated, not zero. Absent.
    assert details.shipment.days_since_delivery is None


def test_promotional_order_keeps_its_promotion_code(now: datetime) -> None:
    """Eligibility needs it to decide whether the holiday policy applies at all."""
    details = lookup_order("CUST-004", "ORD-1006", now=now)
    assert details is not None
    assert details.order.promotion_code == "MIDYEAR_HOLIDAY_SALE_2026"


# --- escalate_to_human --------------------------------------------------
#
# Mocked, so there is little to assert about integration. What matters is that it
# never refuses: an agent that can only escalate for verified customers cannot
# escalate the person who failed verification.


def test_escalation_returns_a_distinct_case_id() -> None:
    result = escalate_to_human("customer asked for a person")
    assert result.case_id.startswith("CASE-")
    assert result.case_id in result.message

    ids = {escalate_to_human("asked for a person").case_id for _ in range(10)}
    assert len(ids) == 10


def test_escalation_records_the_context_a_human_needs() -> None:
    result = escalate_to_human(
        "asked whether Australian consumer law overrides the return window",
        customer_id="CUST-002",
        order_id="ORD-1003",
    )
    assert result.customer_id == "CUST-002"
    assert result.order_id == "ORD-1003"
    assert "consumer law" in result.reason
    assert result.created_at.tzinfo is not None


def test_escalation_works_without_identity() -> None:
    """Whoever could not be verified is exactly who needs a human."""
    result = escalate_to_human("could not verify identity, customer is upset")
    assert result.case_id
    assert result.customer_id is None
    assert result.order_id is None


def test_customer_facing_message_promises_nothing() -> None:
    """It says a person will pick it up, not what they will decide."""
    message = escalate_to_human("wants a refund outside the window").message.lower()
    assert "refund" not in message
    for promise in ("will approve", "guarantee", "you'll get your money"):
        assert promise not in message


# --- initiate_return: the success path ----------------------------------


@pytest.fixture
def token(seeded_graph, now: datetime) -> str:
    """A real token from a real eligible decision — ORD-1001 / ITEM-100 at day 11."""
    decision = check_return_eligibility(IN_WINDOW_ORDER, IN_WINDOW_ITEM, HERO_CUSTOMER, now=now)
    assert decision.eligible and decision.eligibility_token
    return decision.eligibility_token


def test_valid_confirmed_return_creates_exactly_one_rma(token: str) -> None:
    result = initiate_return(
        order_id=IN_WINDOW_ORDER,
        item_id=IN_WINDOW_ITEM,
        customer_id=HERO_CUSTOMER,
        reason="Cover was creased on arrival.",
        eligibility_token=token,
        confirmed=True,
    )

    assert result.created is True
    assert result.return_record.status == "requested"
    assert result.return_record.reason == "Cover was creased on arrival."
    assert result.return_record.return_id.startswith("RET-")
    assert len([r for r in _load_returns() if r.order_id == IN_WINDOW_ORDER]) == 1


def test_written_record_is_readable_back(token: str) -> None:
    """The store is the record. What was written is what a later read returns."""
    result = initiate_return(
        order_id=IN_WINDOW_ORDER,
        item_id=IN_WINDOW_ITEM,
        customer_id=HERO_CUSTOMER,
        reason="Wrong edition.",
        eligibility_token=token,
        confirmed=True,
    )

    stored = next(r for r in _load_returns() if r.return_id == result.return_record.return_id)
    assert stored.customer_id == HERO_CUSTOMER
    assert stored.reason == "Wrong edition."
    assert stored.created_at.tzinfo is not None, "timestamps are timezone-aware"


def test_a_new_return_makes_the_item_ineligible(token: str, seeded_graph, now: datetime) -> None:
    """The two tools agree: once opened, eligibility stops saying yes."""
    initiate_return(
        order_id=IN_WINDOW_ORDER,
        item_id=IN_WINDOW_ITEM,
        customer_id=HERO_CUSTOMER,
        reason="Wrong edition.",
        eligibility_token=token,
        confirmed=True,
    )

    again = check_return_eligibility(IN_WINDOW_ORDER, IN_WINDOW_ITEM, HERO_CUSTOMER, now=now)
    assert again.eligible is False
    assert again.eligibility_token is None


# --- What the model is sent ----------------------------------------------
#
# A tool result goes into the transcript and is re-sent on every later turn, so a
# field nobody reads is paid for again and again. `ToolOutcome.content` is the
# model's view; `ToolOutcome.payload` is the full typed result Python keeps.
#
# Two properties, on both sides. The content carries everything the agent needs
# to answer the customer and make the next call, and nothing bulky it does not.
# The payload is untouched, so trusted state and the trace still see it all.


@pytest.fixture
def verified() -> SessionState:
    """A session with identity settled, so the account tools will run."""
    return SessionState(
        verified_customer_id=HERO_CUSTOMER,
        customer_region="GB",
        active_order_ids=[IN_WINDOW_ORDER],
        active_order_id=IN_WINDOW_ORDER,
    )


def test_order_list_sends_ids_titles_and_statuses(verified, now: datetime) -> None:
    """The discovery mode: what they have, in the words they would use for it."""
    outcome = invoke_tool("lookup_order", {}, verified, now=now)
    sent = json.loads(outcome.content)

    assert set(sent) == {"orders"}
    assert {order["order_id"] for order in sent["orders"]} == {IN_WINDOW_ORDER, "ORD-1002"}
    assert set(sent["orders"][0]["items"][0]) == {"item_id", "title", "product_type"}
    # Not the detailed mode's fields, which nobody asked for.
    for absent in ("placed_at", "delivered_at", "price", "promotion_code", "customer_id"):
        assert absent not in outcome.content


def test_order_list_is_scoped_to_the_verified_customer(now: datetime) -> None:
    """The model supplies no customer id, and cannot: it is injected from the
    session, so there is nothing to point at another account."""
    bruce = SessionState(verified_customer_id="CUST-002", customer_region="AU")
    outcome = invoke_tool("lookup_order", {}, bruce, now=now)

    assert {order["order_id"] for order in json.loads(outcome.content)["orders"]} == {
        "ORD-1003",
        "ORD-1007",
    }
    assert outcome.args_used == {"customer_id": "CUST-002"}


def test_order_lookup_sends_the_conversational_fields(verified, now: datetime) -> None:
    """Where it is, when it arrived, and what is on it with the ids to act on."""
    outcome = invoke_tool("lookup_order", {"order_id": IN_WINDOW_ORDER}, verified, now=now)
    sent = json.loads(outcome.content)

    assert set(sent) == {
        "order_id", "status", "placed_at", "delivered_at", "days_since_delivery", "items"
    }
    assert sent["order_id"] == IN_WINDOW_ORDER
    assert sent["status"] == "delivered"
    assert set(sent["items"][0]) == {"item_id", "title", "product_type"}
    assert sent["items"][0]["item_id"] == IN_WINDOW_ITEM


def test_order_lookup_withholds_the_rest_of_the_record(verified, now: datetime) -> None:
    """Prices, quantities, promotions, and the customer id the session already
    holds. None of it changes what the agent says, and all of it is re-sent every
    turn once it is in the transcript."""
    outcome = invoke_tool("lookup_order", {"order_id": IN_WINDOW_ORDER}, verified, now=now)

    for absent in ("price", "currency", "unit_price", "quantity", "promotion_code", "customer_id"):
        assert absent not in outcome.content

    # The full order is still right there for Python.
    assert outcome.payload.order.customer_id == HERO_CUSTOMER
    assert outcome.payload.items[0].price > 0


def test_eligibility_sends_the_decision_and_the_explanation(
    verified, seeded_graph, now: datetime
) -> None:
    """The verdict, the policy behind it, the sentence to say, and the days left."""
    outcome = invoke_tool(
        "check_return_eligibility",
        {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM},
        verified,
        now=now,
    )
    sent = json.loads(outcome.content)

    assert set(sent) == {"eligible", "policy_id", "explanation", "days_remaining"}
    assert sent["eligible"] is True
    assert sent["policy_id"] == "STANDARD_30_DAY"
    assert sent["explanation"]


def test_eligibility_withholds_the_token_and_the_traversal(
    verified, seeded_graph, now: datetime
) -> None:
    """The token is a credential the loop supplies itself, and the rule path is
    for the trace — the prompt forbids mentioning the graph to a customer. Both
    survive on the payload, which is what the session and the UI read."""
    outcome = invoke_tool(
        "check_return_eligibility",
        {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM},
        verified,
        now=now,
    )

    assert "eligibility_token" not in outcome.content
    assert "GOVERNED_BY" not in outcome.content
    assert outcome.payload.eligibility_token
    assert outcome.payload.rule_path


def test_policy_search_sends_the_governing_rule_not_the_whole_graph(verified) -> None:
    """One resolved policy in full, the others as a line each. `matches` used to
    carry every candidate node whole — the largest result in the system."""
    outcome = invoke_tool(
        "search_policy", {"query": "how long do I have to return a paperback"}, verified
    )
    sent = json.loads(outcome.content)

    assert set(sent) == {
        "matched", "region", "region_policy_found", "region_note",
        "resolved", "other_policies", "message",
    }
    assert sent["resolved"]["policy_id"] == "STANDARD_30_DAY"
    assert sent["resolved"]["return_window_days"] == 30
    assert sent["resolved"]["rule_path"]
    assert all(set(other) == {
        "policy_id", "category", "applies", "return_window_days", "conditions"
    } for other in sent["other_policies"])

    # Still the whole search on the payload.
    assert outcome.payload.matches


def test_policy_search_is_smaller_than_the_full_result(verified) -> None:
    """The reduction is the point, so it is asserted rather than assumed."""
    outcome = invoke_tool("search_policy", {"query": "return policy"}, verified)

    assert len(outcome.content) < len(outcome.payload.model_dump_json()) / 2


def test_return_creation_sends_the_reference_and_the_status(
    verified, seeded_graph, now: datetime
) -> None:
    """Whether it was opened or already existed, the RMA, and where it stands."""
    invoke_tool(
        "check_return_eligibility",
        {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM},
        verified,
        now=now,
    )
    apply_tool_result(
        "check_return_eligibility",
        {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM},
        invoke_tool(
            "check_return_eligibility",
            {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM},
            verified,
            now=now,
        ),
        verified,
    )
    verified.confirmed = True

    outcome = invoke_tool(
        "initiate_return",
        {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM},
        verified,
        now=now,
    )
    sent = json.loads(outcome.content)

    assert set(sent) == {"created", "return_id", "status", "message"}
    assert sent["created"] is True
    assert sent["return_id"].startswith("RET-")
    assert sent["status"] == "requested"
    assert HERO_CUSTOMER not in outcome.content

    # The full record is still on the payload, which is what a trace would read.
    assert outcome.payload.return_record.customer_id == HERO_CUSTOMER


def test_escalation_sends_the_case_and_the_sentence(verified) -> None:
    """The reference to read back, and the line to read it back in."""
    outcome = invoke_tool("escalate_to_human", {"reason": "wants a person"}, verified)
    sent = json.loads(outcome.content)

    assert set(sent) == {"case_id", "message"}
    assert sent["case_id"].startswith("CASE-")
    assert outcome.payload.reason == "wants a person"


def test_compaction_does_not_disturb_trusted_state(verified, seeded_graph, now: datetime) -> None:
    """State is updated from the payload, never from what the model was shown, so
    a smaller content block changes nothing about what the session believes."""
    state = SessionState()

    identity = invoke_tool("verify_identity", {"email": "ada@example.com"}, state)
    apply_tool_result("verify_identity", {}, identity, state)
    assert state.verified_customer_id == HERO_CUSTOMER
    assert state.customer_region == "GB"

    listed = invoke_tool("lookup_order", {}, state, now=now)
    apply_tool_result("lookup_order", {}, listed, state)
    assert state.active_order_ids == [IN_WINDOW_ORDER, "ORD-1002"]

    order = invoke_tool("lookup_order", {"order_id": IN_WINDOW_ORDER}, state, now=now)
    apply_tool_result("lookup_order", {}, order, state)
    assert state.active_order_id == IN_WINDOW_ORDER

    decision = invoke_tool(
        "check_return_eligibility",
        {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM},
        state,
        now=now,
    )
    apply_tool_result(
        "check_return_eligibility",
        {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM},
        decision,
        state,
    )
    assert state.eligibility_token
    assert state.pending_return is not None
    assert state.eligibility is not None and state.eligibility.rule_path


def test_comparing_two_items_in_one_turn_keeps_the_first_grant(
    verified, seeded_graph, now: datetime
) -> None:
    """Checking a second item does not spend a token already granted for the first.

    Reproduces a real session: a customer's order carries a physical book (in
    the AU extended window) and an ebook (never returnable). Both get checked in
    the same turn — "can I return either of these?" — and the ebook coming back
    ineligible must not clear the pending return the paperback's check already
    created. `adopt_active_item=False` is what one model turn checking several
    items looks like; the loop computes it from the batch of tool calls.
    """
    bruce = SessionState(verified_customer_id="CUST-002", customer_region="AU")
    order_id, physical_item, ebook_item = "ORD-1003", "ITEM-102", "ITEM-201"

    eligible = invoke_tool(
        "check_return_eligibility", {"order_id": order_id, "item_id": physical_item}, bruce, now=now
    )
    apply_tool_result(
        "check_return_eligibility",
        {"order_id": order_id, "item_id": physical_item},
        eligible,
        bruce,
        adopt_active_item=False,
    )
    assert bruce.pending_return is not None
    token_after_first_check = bruce.eligibility_token
    assert token_after_first_check

    ineligible = invoke_tool(
        "check_return_eligibility", {"order_id": order_id, "item_id": ebook_item}, bruce, now=now
    )
    apply_tool_result(
        "check_return_eligibility",
        {"order_id": order_id, "item_id": ebook_item},
        ineligible,
        bruce,
        adopt_active_item=False,
    )

    # The ebook's own refusal is not lost...
    assert ineligible.payload.eligible is False
    # ...but the paperback's grant, from earlier in the same turn, survives.
    assert bruce.pending_return is not None
    assert bruce.pending_return.item_id == physical_item
    assert bruce.eligibility_token == token_after_first_check

    # And the surviving token actually spends: this is the return that was
    # wrongly blocked in the session that surfaced the bug.
    bruce.confirmed = True
    opened = invoke_tool(
        "initiate_return", {"order_id": order_id, "item_id": physical_item}, bruce, now=now
    )
    assert opened.status is ToolStatus.OK
    assert opened.payload.created is True


def test_checking_a_different_item_on_its_own_turn_still_clears_the_old_one(
    verified, seeded_graph, now: datetime
) -> None:
    """The clearing this guards against is still real: a genuine switch — one
    item checked, then later a different one, each its own turn — drops the
    first item's token exactly as before. Only *comparing* items in one turn is
    protected."""
    bruce = SessionState(verified_customer_id="CUST-002", customer_region="AU")
    order_id, physical_item, ebook_item = "ORD-1003", "ITEM-102", "ITEM-201"

    eligible = invoke_tool(
        "check_return_eligibility", {"order_id": order_id, "item_id": physical_item}, bruce, now=now
    )
    apply_tool_result(
        "check_return_eligibility",
        {"order_id": order_id, "item_id": physical_item},
        eligible,
        bruce,
    )
    assert bruce.pending_return is not None

    ineligible = invoke_tool(
        "check_return_eligibility", {"order_id": order_id, "item_id": ebook_item}, bruce, now=now
    )
    apply_tool_result(
        "check_return_eligibility",
        {"order_id": order_id, "item_id": ebook_item},
        ineligible,
        bruce,
    )

    assert bruce.pending_return is None
    assert bruce.eligibility_token is None


# --- The mock data ------------------------------------------------------
#
# `data/` is mock transactional data only. Nothing here connects to Neo4j; the
# policy seed is checked in test_policy.py.


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


@pytest.mark.parametrize(
    "filename", ["customers.json", "items.json", "orders.json", "returns.json"]
)
def test_data_file_parses(filename: str) -> None:
    assert isinstance(_load(DATA / filename), list)


def test_data_dir_holds_no_policy_json() -> None:
    """Policy is Neo4j's job. A policy file in data/ would invite a fallback."""
    assert sorted(p.name for p in DATA.glob("*.json")) == [
        "customers.json",
        "items.json",
        "orders.json",
        "returns.json",
    ]


def test_ids_are_unique(
    customers: list[Customer],
    items: list[Item],
    orders: list[Order],
    returns: list[ReturnRecord],
) -> None:
    for label, ids in [
        ("customer", [c.customer_id for c in customers]),
        ("item", [i.item_id for i in items]),
        ("order", [o.order_id for o in orders]),
        ("return", [r.return_id for r in returns]),
    ]:
        assert len(ids) == len(set(ids)), f"duplicate {label} id"


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


def test_baseline_exists_for_every_mutable_file(data_dir: Path) -> None:
    """`returns.json` is the only file written at runtime, and it has a baseline.

    If a later change adds another mutable fixture without a seed copy, the reset
    would silently stop covering it. This is the reminder.
    """
    assert (data_dir / "seed" / "returns.json").is_file()


def test_baseline_matches_the_shipped_data(data_dir: Path) -> None:
    """A fresh checkout starts where the baseline says it does."""
    live = json.loads((data_dir / "returns.json").read_text())
    baseline = json.loads((data_dir / "seed" / "returns.json").read_text())
    assert live == baseline
