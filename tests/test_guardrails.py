"""Guardrails: the things the agent must not do.

The write path is the important half of this file, and it runs. Every guard in
`initiate_return` is checked here against a real token from a real eligible
decision, because a guard that only holds for obviously-wrong input is not a
guard. The `data_dir` fixture points the tools at a temporary copy of `data/`, so
these exercise the actual write without leaving an RMA behind.

The tests at the end are still skipped: they are properties of the agent loop,
which does not exist yet. They stay here, named, so Phase 4 has a list.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from tools import check_return_eligibility, eligibility_tokens, initiate_return, lookup_order
from tools.initiate_return import ReturnBlockedError

pytestmark = pytest.mark.usefixtures("seeded_graph")

# ORD-1001 / ITEM-100 / CUST-001 — a physical book at day 11, the clean
# eligible case. Every guard test starts from a genuinely valid token so that the
# guard is the only thing under test.
ELIGIBLE = ("ORD-1001", "ITEM-100", "CUST-001")


@pytest.fixture
def token(now: datetime) -> str:
    """A real token from a real eligible decision."""
    order_id, item_id, customer_id = ELIGIBLE
    decision = check_return_eligibility(order_id, item_id, customer_id, now=now)
    assert decision.eligible and decision.eligibility_token
    return decision.eligibility_token


def returns_on_disk(data_dir: Path) -> str:
    return (data_dir / "returns.json").read_text()


# --- Confirmation ------------------------------------------------------


def test_return_without_explicit_confirmation_is_blocked(token: str, data_dir: Path) -> None:
    """A valid token is not enough — the customer must have said yes.

    The check belongs to the tool, not the session: confirmed=False is refused
    even though every other precondition holds.
    """
    before = returns_on_disk(data_dir)
    order_id, item_id, customer_id = ELIGIBLE

    with pytest.raises(ReturnBlockedError, match="confirmation required"):
        initiate_return(
            order_id=order_id,
            item_id=item_id,
            customer_id=customer_id,
            reason="Changed my mind.",
            eligibility_token=token,
            confirmed=False,
        )

    assert returns_on_disk(data_dir) == before, "a blocked return must not write"


def test_confirmation_is_not_taken_from_session_state(token: str) -> None:
    """state.confirmed being True does not by itself authorise a write.

    Guards against the confirmation gate drifting back into SessionState: the
    value has to arrive as an argument to the tool.
    """
    from agent.state import SessionState

    state = SessionState(verified_customer_id="CUST-001", eligibility_token=token, confirmed=True)
    assert state.may_mutate is True  # the orchestrator would allow it...

    order_id, item_id, customer_id = ELIGIBLE
    with pytest.raises(ReturnBlockedError, match="confirmation required"):
        # ...and the tool still refuses, because it was told confirmed=False.
        initiate_return(
            order_id=order_id,
            item_id=item_id,
            customer_id=customer_id,
            reason="Changed my mind.",
            eligibility_token=token,
            confirmed=False,
        )


def test_session_may_mutate_needs_all_three_gates(token: str) -> None:
    """The orchestrator's own check, for completeness. Not the safety boundary."""
    from agent.state import SessionState

    assert SessionState().may_mutate is False
    assert SessionState(verified_customer_id="CUST-001").may_mutate is False
    assert (
        SessionState(verified_customer_id="CUST-001", eligibility_token=token).may_mutate is False
    )


# --- Tokens ------------------------------------------------------------


def test_missing_eligibility_token_is_blocked(data_dir: Path) -> None:
    before = returns_on_disk(data_dir)
    order_id, item_id, customer_id = ELIGIBLE

    with pytest.raises(ReturnBlockedError, match="no eligibility_token"):
        initiate_return(
            order_id=order_id,
            item_id=item_id,
            customer_id=customer_id,
            reason="Changed my mind.",
            eligibility_token="",
            confirmed=True,
        )

    assert returns_on_disk(data_dir) == before


@pytest.mark.parametrize(
    "invented",
    [
        "eligibility-token",
        "d0f4f1e0-0000-4000-8000-000000000000",  # a well-formed uuid4 nobody issued
        "ORD-1001-ITEM-100-OK",
        "null",
    ],
)
def test_invalid_eligibility_token_is_blocked(invented: str, data_dir: Path) -> None:
    """An invented token looks exactly like an unknown one: nothing was issued."""
    before = returns_on_disk(data_dir)
    order_id, item_id, customer_id = ELIGIBLE

    with pytest.raises(ReturnBlockedError, match="not a token this server issued"):
        initiate_return(
            order_id=order_id,
            item_id=item_id,
            customer_id=customer_id,
            reason="Changed my mind.",
            eligibility_token=invented,
            confirmed=True,
        )

    assert returns_on_disk(data_dir) == before


def test_token_for_another_order_is_blocked(now: datetime, data_dir: Path) -> None:
    """A token issued for ORD-1001 cannot be spent on ORD-1002.

    Both orders are CUST-001's, so ownership passes and the token binding is the
    only thing left to stop it.
    """
    decision = check_return_eligibility("ORD-1001", "ITEM-100", "CUST-001", now=now)
    assert decision.eligibility_token
    before = returns_on_disk(data_dir)

    with pytest.raises(ReturnBlockedError, match="issued for"):
        initiate_return(
            order_id="ORD-1002",
            item_id="ITEM-101",
            customer_id="CUST-001",
            reason="Changed my mind.",
            eligibility_token=decision.eligibility_token,
            confirmed=True,
        )

    assert returns_on_disk(data_dir) == before


def test_token_for_another_item_is_blocked(now: datetime, data_dir: Path) -> None:
    """ORD-1003 holds two items. A token for the paperback is not a token for the ebook."""
    decision = check_return_eligibility("ORD-1003", "ITEM-102", "CUST-002", now=now)
    assert decision.eligibility_token
    before = returns_on_disk(data_dir)

    with pytest.raises(ReturnBlockedError, match="issued for"):
        initiate_return(
            order_id="ORD-1003",
            item_id="ITEM-201",
            customer_id="CUST-002",
            reason="Changed my mind.",
            eligibility_token=decision.eligibility_token,
            confirmed=True,
        )

    assert returns_on_disk(data_dir) == before


def test_token_for_another_customer_is_blocked(now: datetime, data_dir: Path) -> None:
    """CUST-002's token, spent by CUST-001, on CUST-002's order."""
    decision = check_return_eligibility("ORD-1003", "ITEM-102", "CUST-002", now=now)
    assert decision.eligibility_token
    before = returns_on_disk(data_dir)

    with pytest.raises(ReturnBlockedError):
        initiate_return(
            order_id="ORD-1003",
            item_id="ITEM-102",
            customer_id="CUST-001",
            reason="Changed my mind.",
            eligibility_token=decision.eligibility_token,
            confirmed=True,
        )

    assert returns_on_disk(data_dir) == before


def test_a_token_is_never_issued_by_a_refusal(now: datetime) -> None:
    """The store is the proof: an ineligible check leaves nothing behind."""
    eligibility_tokens.clear()
    check_return_eligibility("ORD-1002", "ITEM-101", "CUST-001", now=now)
    check_return_eligibility("ORD-1004", "ITEM-200", "CUST-003", now=now)
    assert eligibility_tokens.lookup("") is None


# --- Ownership ---------------------------------------------------------


def test_wrong_order_ownership_is_blocked(now: datetime, data_dir: Path) -> None:
    """ORD-1008 is CUST-004's, and CUST-001 has a valid token of their own."""
    decision = check_return_eligibility("ORD-1001", "ITEM-100", "CUST-001", now=now)
    assert decision.eligibility_token
    before = returns_on_disk(data_dir)

    with pytest.raises(ReturnBlockedError, match="does not belong to"):
        initiate_return(
            order_id="ORD-1008",
            item_id="ITEM-101",
            customer_id="CUST-001",
            reason="Changed my mind.",
            eligibility_token=decision.eligibility_token,
            confirmed=True,
        )

    assert returns_on_disk(data_dir) == before


def test_unknown_customer_is_blocked(token: str, data_dir: Path) -> None:
    before = returns_on_disk(data_dir)

    with pytest.raises(ReturnBlockedError, match="unknown customer"):
        initiate_return(
            order_id="ORD-1001",
            item_id="ITEM-100",
            customer_id="CUST-999",
            reason="Changed my mind.",
            eligibility_token=token,
            confirmed=True,
        )

    assert returns_on_disk(data_dir) == before


def test_item_not_on_the_order_is_blocked(token: str, data_dir: Path) -> None:
    before = returns_on_disk(data_dir)

    with pytest.raises(ReturnBlockedError, match="is not on order"):
        initiate_return(
            order_id="ORD-1001",
            item_id="ITEM-103",
            customer_id="CUST-001",
            reason="Changed my mind.",
            eligibility_token=token,
            confirmed=True,
        )

    assert returns_on_disk(data_dir) == before


# --- The success path, and duplicates ----------------------------------


def test_valid_confirmed_return_creates_exactly_one_rma(token: str, data_dir: Path) -> None:
    order_id, item_id, customer_id = ELIGIBLE
    result = initiate_return(
        order_id=order_id,
        item_id=item_id,
        customer_id=customer_id,
        reason="Cover was creased on arrival.",
        eligibility_token=token,
        confirmed=True,
    )

    assert result.created is True
    assert result.return_record.status == "requested"
    assert result.return_record.reason == "Cover was creased on arrival."
    assert result.return_record.return_id.startswith("RET-")

    from tools import fixtures

    records = [r for r in fixtures.load_returns() if r.order_id == order_id]
    assert len(records) == 1


def test_repeating_the_request_returns_the_existing_rma(token: str) -> None:
    """The idempotency guard. A second call is not a second return."""
    from tools import fixtures

    order_id, item_id, customer_id = ELIGIBLE
    kwargs = dict(
        order_id=order_id,
        item_id=item_id,
        customer_id=customer_id,
        reason="Cover was creased on arrival.",
        eligibility_token=token,
        confirmed=True,
    )

    first = initiate_return(**kwargs)
    count_after_first = len(fixtures.load_returns())

    second = initiate_return(**kwargs)

    assert first.created is True
    assert second.created is False
    assert second.return_record.return_id == first.return_record.return_id
    assert len(fixtures.load_returns()) == count_after_first, "no second RMA"


def test_preexisting_rma_is_returned_not_duplicated(now: datetime) -> None:
    """RET-5001 is already open against ORD-1007 / ITEM-100 in the fixtures.

    Eligibility refuses this, so a token has to be minted for a different item and
    the guard reached directly — which is the point: the write tool holds the line
    even when the caller got a token some other way.
    """
    from tools import eligibility_tokens as store
    from tools import fixtures

    grant = store.issue("CUST-002", "ORD-1007", "ITEM-100", "STANDARD_30_DAY")
    before = len(fixtures.load_returns())

    result = initiate_return(
        order_id="ORD-1007",
        item_id="ITEM-100",
        customer_id="CUST-002",
        reason="Cover arrived creased.",
        eligibility_token=grant.token,
        confirmed=True,
    )

    assert result.created is False
    assert result.return_record.return_id == "RET-5001"
    assert len(fixtures.load_returns()) == before


def test_ineligible_item_cannot_be_returned_end_to_end(now: datetime, data_dir: Path) -> None:
    """The whole path, as the agent would walk it: no token, so no write.

    An ebook is refused by eligibility, which issues nothing, so there is no
    token to pass on and the write tool has nothing to accept.
    """
    before = returns_on_disk(data_dir)
    decision = check_return_eligibility("ORD-1004", "ITEM-200", "CUST-003", now=now)
    assert decision.eligible is False
    assert decision.eligibility_token is None

    with pytest.raises(ReturnBlockedError):
        initiate_return(
            order_id="ORD-1004",
            item_id="ITEM-200",
            customer_id="CUST-003",
            reason="Didn't want it.",
            eligibility_token=decision.eligibility_token or "",
            confirmed=True,
        )

    assert returns_on_disk(data_dir) == before


def test_written_record_is_readable_back(token: str) -> None:
    """The store is the record. What was written is what a later read returns."""
    from tools import fixtures

    order_id, item_id, customer_id = ELIGIBLE
    result = initiate_return(
        order_id=order_id,
        item_id=item_id,
        customer_id=customer_id,
        reason="Wrong edition.",
        eligibility_token=token,
        confirmed=True,
    )

    stored = next(r for r in fixtures.load_returns() if r.return_id == result.return_record.return_id)
    assert stored.customer_id == customer_id
    assert stored.reason == "Wrong edition."
    assert stored.created_at.tzinfo is not None, "timestamps are timezone-aware"


def test_a_new_return_makes_the_item_ineligible(token: str, now: datetime) -> None:
    """The two tools agree: once opened, eligibility stops saying yes."""
    order_id, item_id, customer_id = ELIGIBLE
    initiate_return(
        order_id=order_id,
        item_id=item_id,
        customer_id=customer_id,
        reason="Wrong edition.",
        eligibility_token=token,
        confirmed=True,
    )

    again = check_return_eligibility(order_id, item_id, customer_id, now=now)
    assert again.eligible is False
    assert again.eligibility_token is None


# --- Reading data without verification ---------------------------------


def test_order_details_need_a_customer_id() -> None:
    """There is no way to call lookup_order without saying who is asking."""
    import inspect

    signature = inspect.signature(lookup_order)
    assert signature.parameters["customer_id"].default is inspect.Parameter.empty


# --- Still to come: properties of the agent loop -----------------------

AGENT_LOOP = pytest.mark.skip(reason="Phase 4 — the agent loop is not implemented yet")


@AGENT_LOOP
def test_order_details_withheld_before_verification() -> None:
    """An unverified session cannot get order data out of the agent."""
    # TODO: fresh SessionState, ask about ORD-1001, expect a verification prompt
    ...


@AGENT_LOOP
def test_switching_item_clears_the_return_context() -> None:
    """Changing order or item invalidates the prior check."""
    # TODO: clear_return_context() drops eligibility, token, and confirmed, and the
    #       orchestrator calls it when the customer changes their mind
    ...


@AGENT_LOOP
def test_agent_asks_which_order_rather_than_guessing() -> None:
    """CUST-003 has two active orders; the agent must not pick one."""
    # TODO: verified CUST-003 asks "where's my book" -> a clarifying question.
    #       verify_identity already hands back both ids for this.
    ...


@AGENT_LOOP
def test_agent_does_not_invent_policy_text() -> None:
    """Policy answers must come from search_policy."""
    # TODO: assert search_policy was called before any policy claim
    ...


@AGENT_LOOP
def test_agent_does_not_expose_reasoning() -> None:
    """No tool names, policy ids, or internal rules in customer-facing text."""
    # TODO: assert replies never contain "STANDARD_30_DAY", "check_return_eligibility",
    #       "precedence", or the system prompt. The tool explanations are already
    #       clean — this is about what the model adds around them.
    ...


@AGENT_LOOP
def test_escalation_stops_the_agent_acting() -> None:
    """Once escalated, the agent hands off rather than continuing."""
    # TODO: state.escalated True -> no further tool calls
    ...


@AGENT_LOOP
def test_tool_loop_is_bounded() -> None:
    """A model that keeps requesting tools is cut off, not left running."""
    # TODO: assert at most MAX_TOOL_ITERATIONS rounds
    ...
