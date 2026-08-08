"""The demo reset: putting the mutable data back so the hero flow can run again.

The hero conversation writes a real RMA. That is what makes it worth showing,
and it is also why a second rehearsal is not the same conversation as the first
unless something restores the file in between.

These tests run against the temporary copy of `data/` that `conftest.data_dir`
installs, which includes `data/seed/` — so a reset here restores the copy, and
the repo's fixtures are never touched.
"""

from __future__ import annotations

import json

from agent.demo import fresh_session, reset_demo
from agent.models import Role
from agent.state import SessionState
from agent.tracing import ToolStatus
from tests.conftest import text, tool_call
from tests.test_hero_flow import (
    CONFIRM_QUESTION,
    HERO_CUSTOMER,
    IN_WINDOW_ITEM,
    IN_WINDOW_ORDER,
    hero_script,  # noqa: F401 - imported so pytest registers the fixture here too
    returns_in,
    run_hero_flow,
)
from tools import eligibility_tokens, fixtures


def test_baseline_exists_for_every_mutable_file(data_dir) -> None:
    """`returns.json` is the only file written at runtime, and it has a baseline.

    If a later phase adds another mutable fixture without a seed copy, the reset
    would silently stop covering it. This is the reminder.
    """
    assert (data_dir / "seed" / "returns.json").is_file()


def test_baseline_matches_the_shipped_data(data_dir) -> None:
    """A fresh checkout starts where the baseline says it does."""
    live = json.loads((data_dir / "returns.json").read_text())
    baseline = json.loads((data_dir / "seed" / "returns.json").read_text())
    assert live == baseline


def test_reset_restores_returns_after_a_write(make_agent, seeded_graph, data_dir) -> None:
    """The RMA the hero flow created is gone; the seeded one is back."""
    before = returns_in(data_dir)

    agent, _ = make_agent(
        tool_call(
            "check_return_eligibility",
            {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM, "reason": "Not for me."},
        ),
        text(CONFIRM_QUESTION),
        tool_call(
            "initiate_return",
            {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM, "reason": "Not for me."},
            block_id="toolu_write",
        ),
        text("Your return is open."),
    )
    state = SessionState(
        verified_customer_id=HERO_CUSTOMER,
        customer_region="GB",
        active_order_ids=[IN_WINDOW_ORDER],
        active_order_id=IN_WINDOW_ORDER,
    )
    agent.respond(state, "I'd like to return it")
    agent.respond(state, "yes please")
    assert len(returns_in(data_dir)) == len(before) + 1

    result = reset_demo()

    assert result.restored_files == ["returns.json"]
    assert returns_in(data_dir) == before


def test_reset_is_idempotent(data_dir) -> None:
    """Running it twice leaves the same bytes as running it once."""
    reset_demo()
    once = (data_dir / "returns.json").read_bytes()
    reset_demo()
    assert (data_dir / "returns.json").read_bytes() == once


def test_reset_clears_outstanding_eligibility_tokens() -> None:
    """A token minted before the reset cannot be spent after it.

    The store is in-memory and process-global. A Streamlit process that reset
    the data but kept its tokens would let a token issued against the deleted
    RMA's order still satisfy `initiate_return`.
    """
    grant = eligibility_tokens.issue(HERO_CUSTOMER, IN_WINDOW_ORDER, IN_WINDOW_ITEM, "STANDARD_30_DAY")
    assert eligibility_tokens.lookup(grant.token) is not None

    reset_demo()

    assert eligibility_tokens.lookup(grant.token) is None


def test_fresh_session_carries_nothing_over() -> None:
    """The session the UI installs after a reset knows nothing about the last run."""
    used = SessionState(
        verified_customer_id=HERO_CUSTOMER,
        customer_region="GB",
        active_order_id=IN_WINDOW_ORDER,
        active_item_id=IN_WINDOW_ITEM,
        eligibility_token="tok",
        confirmed=True,
        escalated=True,
    )
    used.add_message(Role.USER, "hello")

    clean = fresh_session()

    assert clean.session_id != used.session_id
    assert clean.messages == []
    assert clean.transcript == []
    assert clean.tool_traces == []
    assert clean.model_turns == []
    assert clean.verified_customer_id is None
    assert clean.active_order_id is None
    assert clean.eligibility_token is None
    assert clean.confirmed is False
    assert clean.escalated is False
    assert clean.may_mutate is False


def test_hero_flow_runs_again_after_a_reset(make_agent, seeded_graph, hero_script, data_dir) -> None:
    """The whole point: two identical rehearsals, each creating a real RMA.

    Without the reset the second pass would find the first pass's return and
    correctly refuse to duplicate it — right behaviour, wrong demo.
    """
    agent, _ = make_agent(*hero_script)
    first = fresh_session()
    run_hero_flow(agent, first)
    first_rma = [r for r in returns_in(data_dir) if r["order_id"] == IN_WINDOW_ORDER]
    assert len(first_rma) == 1

    reset_demo()

    agent2, _ = make_agent(*hero_script)
    second = fresh_session()
    run_hero_flow(agent2, second)

    second_rma = [r for r in returns_in(data_dir) if r["order_id"] == IN_WINDOW_ORDER]
    assert len(second_rma) == 1
    assert second_rma[0]["return_id"] == first_rma[0]["return_id"]
    assert second.tool_traces[-1].tool_name == "initiate_return"
    assert second.tool_traces[-1].status is ToolStatus.OK
    assert "created=True" in second.tool_traces[-1].result_summary


def test_reset_leaves_static_fixtures_alone(data_dir) -> None:
    """Customers, orders, and items are never written, so they are never restored."""
    orders_before = (data_dir / "orders.json").read_bytes()
    customers_before = (data_dir / "customers.json").read_bytes()

    reset_demo()

    assert (data_dir / "orders.json").read_bytes() == orders_before
    assert (data_dir / "customers.json").read_bytes() == customers_before


def test_reset_without_a_baseline_directory_is_not_an_error(tmp_path, monkeypatch) -> None:
    """Nothing mutable means nothing to restore — not a crash."""
    empty = tmp_path / "empty-data"
    empty.mkdir()
    monkeypatch.setattr(fixtures, "DATA_DIR", empty)

    result = reset_demo()

    assert result.restored_files == []


def test_the_script_and_the_button_share_one_implementation() -> None:
    """`scripts/reset_demo.py` holds no reset logic of its own.

    Two copies of "put it back" is how a rehearsal and a live run end up
    starting from different states.
    """
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "scripts" / "reset_demo.py").read_text()
    assert "from agent.demo import reset_demo" in source

    # Nothing in the script may touch files or JSON itself — if it did, it would
    # be a second implementation of the reset, free to drift from the shared one.
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not imported & {"json", "shutil", "os"}
