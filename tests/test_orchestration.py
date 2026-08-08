"""The tool loop: what happens between the customer's message and the reply.

Every test here scripts Anthropic's side and asserts on the orchestrator's. The
model never runs, so what is under test is the loop's own behaviour: that it
dispatches the tools Claude asks for, feeds the results back in the right shape,
updates trusted state only from results that succeeded, and never turns a
failure into a success.

The last one is the point of most of this file. A support agent that says "your
return is open" when nothing was written is worse than one that crashes.
"""

from __future__ import annotations

import json

import pytest

from agent.models import Role
from agent.orchestrator import (
    FALLBACK_STUCK,
    FALLBACK_UNAVAILABLE,
    MAX_TOOL_ITERATIONS,
    BooklyAgent,
)
from agent.state import SessionState
from agent.tracing import ToolStatus
from tests.conftest import FakeAnthropic, text, tool_call, tool_calls

# --- Dispatch ------------------------------------------------------------


def test_model_requests_a_tool_and_it_executes(make_agent) -> None:
    """Claude asks for verify_identity; the real Python function runs."""
    agent, client = make_agent(
        tool_call("verify_identity", {"email": "ada@example.com"}),
        text("Thanks Ada — I've found your account."),
    )
    state = SessionState()

    reply = agent.respond(state, "Hi, it's ada@example.com")

    assert reply == "Thanks Ada — I've found your account."
    assert state.verified_customer_id == "CUST-001"
    assert len(client.calls) == 2


def test_tool_result_is_returned_to_anthropic(make_agent) -> None:
    """The result goes back as a tool_result block keyed to the request.

    The `tool_use_id` has to match or the API rejects the turn, and the content
    has to be the tool's own output rather than a summary of it.
    """
    agent, client = make_agent(
        tool_call("verify_identity", {"email": "ada@example.com"}, block_id="toolu_abc"),
        text("Found you."),
    )
    agent.respond(SessionState(), "ada@example.com")

    second_request = client.calls[1]["messages"]
    result_block = second_request[-1]["content"][0]

    assert second_request[-1]["role"] == "user"
    assert result_block["type"] == "tool_result"
    assert result_block["tool_use_id"] == "toolu_abc"
    assert result_block["is_error"] is False

    payload = json.loads(result_block["content"])
    assert payload["verified"] is True
    assert payload["customer_id"] == "CUST-001"


def test_multiple_tool_calls_in_one_turn(make_agent, seeded_graph, verified_state) -> None:
    """Two tools in one response run, and both results come back together.

    The API expects every tool_result for a turn in a single user message.
    Splitting them across two messages trains the model out of asking for more
    than one at a time.
    """
    agent, client = make_agent(
        tool_calls(
            ("lookup_order", {"order_id": "ORD-1001"}),
            ("search_policy", {"query": "return window for paperbacks"}),
        ),
        text("Your order arrived on the 28th, and the window is as described."),
    )

    agent.respond(verified_state, "What's the status, and how long do I have?")

    results = client.calls[1]["messages"][-1]["content"]
    assert len(results) == 2
    assert {block["tool_use_id"] for block in results} == {"toolu_0", "toolu_1"}
    assert all(block["type"] == "tool_result" for block in results)


def test_loop_continues_across_several_rounds(make_agent, seeded_graph) -> None:
    """A full path — verify, look up, check eligibility — runs in one turn."""
    agent, client = make_agent(
        tool_call("verify_identity", {"email": "ada@example.com"}),
        tool_call("lookup_order", {"order_id": "ORD-1001"}, block_id="toolu_2"),
        tool_call(
            "check_return_eligibility",
            {"order_id": "ORD-1001", "item_id": "ITEM-100"},
            block_id="toolu_3",
        ),
        text("That one can be returned. Shall I start a return for it?"),
    )
    state = SessionState()

    agent.respond(state, "I'd like to send back the book on my last order, ada@example.com")

    assert len(client.calls) == 4
    assert state.eligibility is not None and state.eligibility.eligible
    assert state.pending_return is not None
    assert [trace.tool_name for trace in state.tool_traces] == [
        "verify_identity",
        "lookup_order",
        "check_return_eligibility",
    ]


# --- Trusted state -------------------------------------------------------


def test_state_updates_only_after_a_successful_tool_result(make_agent) -> None:
    """A failed verification leaves the session unverified.

    The model is told the email did not match; nothing is written. This is the
    difference between the agent believing a tool and believing the model.
    """
    agent, _ = make_agent(
        tool_call("verify_identity", {"email": "nobody@example.com"}),
        text("I can't find an account for that address."),
    )
    state = SessionState()

    agent.respond(state, "it's nobody@example.com")

    assert state.verified_customer_id is None
    assert state.customer_region is None
    assert state.active_order_ids == []


def test_model_cannot_write_trusted_state_directly(make_agent) -> None:
    """The model has no way to say who the customer is.

    `customer_id` is not a field on any schema, so the only path to
    `verified_customer_id` is a verification that actually passed. Here the
    model tries to use an account-scoped tool without one.
    """
    agent, _ = make_agent(
        tool_call("lookup_order", {"order_id": "ORD-1001"}),
        text("I'll need to verify your account first — what's your email?"),
    )
    state = SessionState()

    agent.respond(state, "show me order ORD-1001")

    assert state.verified_customer_id is None
    assert state.active_order_id is None
    assert state.tool_traces[0].status is ToolStatus.BLOCKED


def test_unverified_account_tools_are_blocked(make_agent) -> None:
    """Order data is not exposed before verification, and the model is told why."""
    agent, client = make_agent(
        tool_call("lookup_order", {"order_id": "ORD-1001"}),
        text("What's the email on your account?"),
    )
    agent.respond(SessionState(), "where's ORD-1001")

    result = client.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert "verify_identity" in result["content"]


def test_policy_search_works_without_verification(make_agent, seeded_graph) -> None:
    """The rules are public. Asking about them does not require an account."""
    agent, _ = make_agent(
        tool_call("search_policy", {"query": "can I return an ebook"}),
        text("Ebooks aren't returnable once they've been delivered."),
    )
    state = SessionState()

    agent.respond(state, "can I return an ebook?")

    assert state.tool_traces[0].status is ToolStatus.OK


def test_switching_item_clears_the_previous_return_context(
    make_agent, seeded_graph, verified_state
) -> None:
    """A token issued for one item cannot survive a move to another.

    CUST-001's ORD-1001 holds ITEM-100. Checking a second item on a different
    order has to drop everything from the first attempt — otherwise a "yes"
    given for one book could open a return for a different one.
    """
    agent, _ = make_agent(
        tool_call(
            "check_return_eligibility", {"order_id": "ORD-1001", "item_id": "ITEM-100"}
        ),
        text("That's returnable. Shall I start a return?"),
    )
    agent.respond(verified_state, "I want to return the paperback")
    first_token = verified_state.eligibility_token
    assert first_token is not None

    agent2, _ = make_agent(
        tool_call(
            "check_return_eligibility", {"order_id": "ORD-1002", "item_id": "ITEM-101"}
        ),
        text("That one's outside the window, I'm afraid."),
    )
    agent2.respond(verified_state, "actually I meant the other order, ORD-1002")

    assert verified_state.eligibility_token != first_token
    assert verified_state.confirmed is False
    assert verified_state.active_item_id == "ITEM-101"


def test_escalation_marks_the_session(make_agent) -> None:
    """A handoff is recorded so the agent knows to stop acting."""
    agent, _ = make_agent(
        tool_call("escalate_to_human", {"reason": "customer asked for a person"}),
        text("I'm passing you to a colleague."),
    )
    state = SessionState()

    agent.respond(state, "I want to speak to a human")

    assert state.escalated is True


# --- Failure -------------------------------------------------------------


def test_invalid_tool_name_is_rejected_safely(make_agent) -> None:
    """A tool that does not exist is refused, and the turn carries on.

    Nothing raises, nothing is written, and the model is told plainly that there
    is no such tool so it can choose a real one.
    """
    agent, client = make_agent(
        tool_call("cancel_order", {"order_id": "ORD-1001"}),
        text("I can't cancel orders, but I can pass you to a colleague."),
    )
    state = SessionState()

    reply = agent.respond(state, "cancel my order")

    assert reply == "I can't cancel orders, but I can pass you to a colleague."
    assert state.tool_traces[0].status is ToolStatus.REJECTED
    result = client.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert "no tool called" in result["content"]


def test_malformed_tool_arguments_are_rejected_safely(make_agent) -> None:
    """A call missing a required argument fails as a tool error, not a crash."""
    agent, _ = make_agent(
        tool_call("verify_identity", {"address": "ada@example.com"}),  # wrong key
        text("Could you give me the email on your account?"),
    )
    state = SessionState()

    agent.respond(state, "here you go")

    assert state.tool_traces[0].status is ToolStatus.REJECTED
    assert state.verified_customer_id is None


def test_tool_exception_does_not_become_a_fabricated_success(
    make_agent, verified_state, monkeypatch
) -> None:
    """When a tool raises, nothing is written and the model is told so.

    The check is on the tool_result the model receives: it is flagged as an
    error and says not to assume success. Whatever the model then writes, it
    cannot honestly claim the lookup worked.
    """
    from tools import lookup_order as lookup_module

    def boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("agent.tool_registry.lookup_order", boom)

    agent, client = make_agent(
        tool_call("lookup_order", {"order_id": "ORD-1001"}),
        text("Something went wrong looking that up."),
    )

    agent.respond(verified_state, "where's my order")

    trace = verified_state.tool_traces[0]
    assert trace.status is ToolStatus.ERROR
    assert "RuntimeError" in (trace.error or "")

    result = client.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert "Do not assume it succeeded" in result["content"]
    assert lookup_module is not None  # import kept meaningful


def test_policy_graph_unavailable_is_reported_not_guessed(
    make_agent, verified_state, monkeypatch
) -> None:
    """With Neo4j down the agent must not answer from memory.

    The tool result says the policy database is unavailable and explicitly tells
    the model not to state a policy — the same stance the tools take, carried up
    into the loop.
    """
    from tests.conftest import break_policy_graph

    break_policy_graph(monkeypatch)

    agent, client = make_agent(
        tool_call("search_policy", {"query": "return window"}),
        text("I can't confirm our policy right now."),
    )

    agent.respond(verified_state, "what's the return window?")

    trace = verified_state.tool_traces[0]
    assert trace.status is ToolStatus.ERROR
    result = client.calls[1]["messages"][-1]["content"][0]
    assert "policy database is unavailable" in result["content"]
    assert "Do not state a policy from memory" in result["content"]


def test_anthropic_unavailable_gives_a_safe_message(make_agent) -> None:
    """An API outage produces an honest sentence, not an exception."""
    agent, _ = make_agent(error=RuntimeError("connection reset"))
    state = SessionState()

    reply = agent.respond(state, "hello")

    assert reply == FALLBACK_UNAVAILABLE
    assert state.messages[-1].role is Role.ASSISTANT
    assert state.tool_traces[-1].tool_name == "anthropic.messages.create"
    assert state.tool_traces[-1].status is ToolStatus.ERROR


def test_api_failure_does_not_leak_the_key(anthropic_config) -> None:
    """The key is on the client, never in a trace, a message, or a repr."""
    agent = BooklyAgent(
        config=anthropic_config,
        client=FakeAnthropic(error=RuntimeError("401 unauthorized")),
    )
    state = SessionState()

    reply = agent.respond(state, "hello")

    blob = f"{reply} {state.model_dump_json()} {agent.config!r}"
    assert anthropic_config.api_key not in blob


def test_tool_loop_stops_at_the_iteration_limit(make_agent, seeded_graph, verified_state) -> None:
    """A model that only ever asks for tools is cut off, not left to run.

    The script is one more tool call than the limit allows. The turn ends with
    an honest message pointing at a colleague rather than another round trip.
    """
    agent, client = make_agent(
        *[
            tool_call("search_policy", {"query": "window"}, block_id=f"toolu_{i}")
            for i in range(MAX_TOOL_ITERATIONS + 1)
        ]
    )

    reply = agent.respond(verified_state, "tell me about returns")

    assert reply == FALLBACK_STUCK
    assert len(client.calls) == MAX_TOOL_ITERATIONS


# --- Conversation shape --------------------------------------------------


def test_conversation_is_multi_turn(make_agent, seeded_graph) -> None:
    """The second turn is sent the first turn's history.

    Without this the agent re-asks for the email every message.
    """
    agent, client = make_agent(
        tool_call("verify_identity", {"email": "ada@example.com"}),
        text("Thanks Ada."),
        text("Your order arrived on the 28th."),
    )
    state = SessionState()

    agent.respond(state, "ada@example.com")
    agent.respond(state, "where is it?")

    last_request = client.calls[-1]["messages"]
    assert last_request[0] == {"role": "user", "content": "ada@example.com"}
    assert last_request[-1] == {"role": "user", "content": "where is it?"}
    assert [m.role for m in state.messages] == [
        Role.USER,
        Role.ASSISTANT,
        Role.USER,
        Role.ASSISTANT,
    ]


def test_transcript_holds_no_duplicate_assistant_turn(make_agent) -> None:
    """The reply is recorded once in the API transcript, not twice."""
    agent, _ = make_agent(text("Hello — how can I help?"))
    state = SessionState()

    agent.respond(state, "hi")

    assistant_turns = [m for m in state.transcript if m["role"] == "assistant"]
    assert len(assistant_turns) == 1


def test_transcript_is_serializable(make_agent) -> None:
    """The session has to survive a Streamlit rerun, so it must round-trip."""
    agent, _ = make_agent(
        tool_call("verify_identity", {"email": "ada@example.com"}),
        text("Found you."),
    )
    state = SessionState()
    agent.respond(state, "ada@example.com")

    restored = SessionState.model_validate_json(state.model_dump_json())
    assert restored.transcript == state.transcript
    assert restored.verified_customer_id == "CUST-001"


def test_system_prompt_and_tools_are_sent_every_call(make_agent) -> None:
    """Both are per-request on a stateless API."""
    agent, client = make_agent(text("hello"))
    agent.respond(SessionState(), "hi")

    request = client.calls[0]
    assert request["system"] == agent.system_prompt
    assert {tool["name"] for tool in request["tools"]} == {
        "verify_identity",
        "lookup_order",
        "search_policy",
        "check_return_eligibility",
        "initiate_return",
        "escalate_to_human",
    }


@pytest.mark.parametrize(
    ("message", "expected_model"),
    [("what's your policy on ebooks?", "test-haiku-model"), ("I want a refund", "test-sonnet-model")],
)
def test_routed_model_is_the_one_actually_called(make_agent, message, expected_model) -> None:
    """The router's decision reaches the API, from the configured names.

    Neither model id is hardcoded anywhere — these are the placeholder names the
    test's configuration supplied.
    """
    agent, client = make_agent(text("..."))
    agent.respond(SessionState(), message)
    assert client.models_used == [expected_model]
