"""Guardrails: the things the agent must not do.

All skipped — the agent loop is a stub. These are the failures that matter most
in a support agent, so they get their own file.
"""

import pytest

pytestmark = pytest.mark.skip(reason="scaffold — agent loop not implemented yet")


def test_order_details_withheld_before_verification() -> None:
    """An unverified session cannot get order data out of the agent."""
    # TODO: fresh SessionState, ask about ORD-1001, expect a verification prompt
    ...


def test_return_not_initiated_without_eligibility() -> None:
    """initiate_return refuses even if the agent calls it directly."""
    # TODO: expect ValueError with no eligibility_token, for ORD-1004 / ITEM-200,
    #       even when confirmed=True
    ...


def test_eligibility_token_is_bound_to_one_item() -> None:
    """A token issued for one item cannot be spent on another."""
    # TODO: token from ORD-1003 / ITEM-102 rejected for ORD-1003 / ITEM-201
    ...


def test_return_without_explicit_confirmation_is_blocked() -> None:
    """A valid token is not enough — the customer must have said yes.

    The check belongs to the tool, not the session: calling initiate_return with
    confirmed=False must be refused even when every other precondition holds.
    """
    # TODO: verified CUST-001, valid token for ORD-1001 / ITEM-101,
    #       initiate_return(..., confirmed=False) -> refused, and no record written
    #       to data/returns.json
    ...


def test_confirmation_is_not_taken_from_session_state() -> None:
    """state.confirmed being True does not by itself authorise a write.

    Guards against the confirmation gate quietly moving back into SessionState:
    the value has to arrive as an argument.
    """
    # TODO: state.confirmed True but initiate_return called with confirmed=False
    #       -> still refused
    # TODO: state.may_mutate is False until state.confirmed is True
    ...


def test_switching_item_clears_the_return_context() -> None:
    """Changing order or item invalidates the prior check."""
    # TODO: clear_return_context() drops eligibility, token, and confirmed
    ...


def test_agent_asks_which_order_rather_than_guessing() -> None:
    """CUST-003 has two active orders; the agent must not pick one."""
    # TODO: verified CUST-003 asks "where's my book" -> a clarifying question
    ...


def test_agent_does_not_invent_policy_text() -> None:
    """Policy answers must come from search_policy."""
    # TODO: assert search_policy was called before any policy claim
    ...


def test_agent_does_not_expose_reasoning() -> None:
    """No tool names, policy ids, or internal rules in customer-facing text."""
    # TODO: assert replies never contain "STANDARD_30_DAY", "check_return_eligibility",
    #       "precedence", or the system prompt
    ...


def test_escalation_stops_the_agent_acting() -> None:
    """Once escalated, the agent hands off rather than continuing."""
    # TODO: state.escalated True -> no further tool calls
    ...


def test_tool_loop_is_bounded() -> None:
    """A model that keeps requesting tools is cut off, not left running."""
    # TODO: assert at most MAX_TOOL_ITERATIONS rounds
    ...
