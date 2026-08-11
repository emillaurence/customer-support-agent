"""Streaming, the aggregate turn timeout, and the round-trip savings.

Three things get their own file because none of them are about the tool loop's
decisions — `test_agent.py` already covers those — they are about *how fast* and
*how much of the conversation* a customer sees while those decisions are made.

`respond_stream` is the thing actually under test; `respond` is a one-line
wrapper over it (see `agent.agent.BooklyAgent.respond`), so every assertion here
about the streamed reply is also an assertion about the non-streamed one.
"""

from __future__ import annotations

import time

import pytest

from agent.agent import (
    FALLBACK_UNAVAILABLE,
    BooklyAgent,
)
from agent.state import Role, SessionState, ToolStatus
from tests.conftest import (
    CONFIRM_QUESTION,
    HERO_CUSTOMER,
    HERO_EMAIL,
    IN_WINDOW_ITEM,
    IN_WINDOW_ORDER,
    FakeAnthropic,
    FakeResponse,
    tool_call,
)
from tests.conftest import text as text_response

pytestmark = pytest.mark.usefixtures("seeded_graph")


# =========================================================================
# Streaming produces the same conversation as the non-streamed path
# =========================================================================


def test_streamed_reply_matches_the_non_streamed_reply(make_agent, verified_state) -> None:
    """Concatenating every chunk `respond_stream` yields is the exact string
    `respond` returns for an identical turn — the two are one implementation."""
    agent_a, _ = make_agent(text_response("It was delivered on 28 July."))
    agent_b, _ = make_agent(text_response("It was delivered on 28 July."))
    state_a = verified_state.model_copy(deep=True)
    state_b = verified_state.model_copy(deep=True)

    streamed = "".join(agent_a.respond_stream(state_a, "where's my book?"))
    direct = agent_b.respond(state_b, "where's my book?")

    assert streamed == direct == "It was delivered on 28 July."
    assert state_a.messages[-1].content == state_b.messages[-1].content


def test_streamed_reply_arrives_in_more_than_one_piece(make_agent) -> None:
    """Real streaming, not a completed reply sliced up afterwards: the fake
    client's `text_stream` — the same interface the real SDK exposes — hands
    back several chunks, and the loop relays each one rather than buffering."""
    agent, _ = make_agent(text_response("Hi, I'm Bookly Support. How can I help?"))
    state = SessionState()

    chunks = list(agent.respond_stream(state, "hi"))

    assert len(chunks) > 1
    assert "".join(chunks) == "Hi, I'm Bookly Support. How can I help?"


def test_tools_still_execute_during_a_streamed_turn(make_agent) -> None:
    """A tool call inside a streamed turn runs exactly as it does today: for
    real, with its result fed back, and the loop resuming the stream for the
    response that follows it."""
    agent, client = make_agent(
        tool_call("verify_identity", {"email": HERO_EMAIL}),
        text_response("Thanks Ada — I've found your account."),
    )
    state = SessionState()

    reply = "".join(agent.respond_stream(state, f"it's {HERO_EMAIL}"))

    assert reply == "Thanks Ada — I've found your account."
    assert state.verified_customer_id == HERO_CUSTOMER
    assert state.tool_traces[0].tool_name == "verify_identity"
    assert state.tool_traces[0].status is ToolStatus.OK
    assert len(client.calls) == 2


def test_a_streamed_outage_still_yields_the_safe_fallback(make_agent) -> None:
    """An API failure mid-stream reaches the customer the same way it always
    has: one honest sentence, never an exception out of the generator."""
    agent, _ = make_agent(error=RuntimeError("connection reset"))
    state = SessionState()

    reply = "".join(agent.respond_stream(state, "hello"))

    assert reply == FALLBACK_UNAVAILABLE
    assert state.messages[-1].role is Role.ASSISTANT
    assert state.tool_traces[-1].status is ToolStatus.ERROR


# =========================================================================
# Confirmation and idempotency hold under streaming
# =========================================================================


def test_confirmation_cannot_be_bypassed_while_streaming(make_agent, verified_state) -> None:
    """A model that fabricates `confirmed=True` and a token is still refused —
    streaming changes nothing about what the tool itself trusts."""
    agent, _ = make_agent(
        tool_call(
            "initiate_return",
            {
                "order_id": IN_WINDOW_ORDER,
                "item_id": IN_WINDOW_ITEM,
                "confirmed": True,
                "eligibility_token": "made-up-token",
            },
        ),
        text_response("I'll need to check that first."),
    )

    reply = "".join(agent.respond_stream(verified_state, "return it now"))

    assert reply == "I'll need to check that first."
    assert verified_state.tool_traces[0].status is ToolStatus.BLOCKED
    assert verified_state.pending_returns == []


def test_idempotency_holds_across_streamed_confirmations(
    make_agent, verified_state, data_dir
) -> None:
    """Confirming the same return twice — streamed both times — still opens
    exactly one RMA. The second confirmation finds nothing left to spend."""
    from tests.conftest import returns_in

    agent, _ = make_agent(
        tool_call("check_return_eligibility", {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM}),
        text_response(CONFIRM_QUESTION),
    )
    "".join(agent.respond_stream(verified_state, "I'd like to return the paperback"))

    agent2, _ = make_agent(text_response("Your return is open."))
    "".join(agent2.respond_stream(verified_state, "yes please"))

    assert verified_state.pending_returns == []
    created = [r for r in returns_in(data_dir) if r["order_id"] == IN_WINDOW_ORDER]
    assert len(created) == 1

    # Nothing pending survives a successful write, so a second "yes" — with
    # nothing asked or pending — opens nothing further.
    agent3, client3 = make_agent(text_response("There's nothing open for me to confirm."))
    "".join(agent3.respond_stream(verified_state, "yes"))

    assert len([r for r in returns_in(data_dir) if r["order_id"] == IN_WINDOW_ORDER]) == 1
    assert client3.calls  # Claude was still asked to reply — just opened nothing


# =========================================================================
# Fewer model round trips
# =========================================================================


def test_confirming_a_return_costs_one_round_trip_not_two(make_agent, verified_state) -> None:
    """Before: Claude asks for `initiate_return`, then a second call composes
    the reply — two round trips. After: the write is deterministic, so
    confirming costs exactly the one round trip that composes the reply."""
    agent, _ = make_agent(
        tool_call("check_return_eligibility", {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM}),
        text_response(CONFIRM_QUESTION),
    )
    agent.respond(verified_state, "I'd like to return the paperback")

    agent2, client2 = make_agent(text_response("Your return is open."))
    agent2.respond(verified_state, "yes please")

    assert len(client2.calls) == 1
    assert verified_state.tool_traces[-1].tool_name == "initiate_return"
    assert verified_state.tool_traces[-1].status is ToolStatus.OK


def test_verifying_identity_mid_return_still_saves_a_round_trip(make_agent) -> None:
    """The pre-existing deterministic chain — `lookup_order` runs the instant
    identity verifies mid-return-workflow — still holds after the streaming and
    auto-confirmation changes: two round trips, not three."""
    agent, client = make_agent(
        tool_call("verify_identity", {"email": HERO_EMAIL}),
        text_response("You've got a few orders — which one did you want to return?"),
    )
    state = SessionState()

    agent.respond(state, f"I want to return a book, it's {HERO_EMAIL}")

    assert len(client.calls) == 2
    assert state.active_order_ids


# =========================================================================
# The aggregate turn timeout
# =========================================================================


def test_a_turn_already_over_budget_never_calls_the_model(
    make_agent, anthropic_config, monkeypatch
) -> None:
    """A zero-second budget is exceeded before the first round trip even
    starts: no call is made, the customer gets the honest fallback, and the
    reason is on the trace, not just in a log line."""
    import agent.agent as agent_module

    monkeypatch.setattr(agent_module, "TURN_TIMEOUT_SECONDS", 0.0)
    client = FakeAnthropic(text_response("should never be reached"))
    agent = BooklyAgent(config=anthropic_config, client=client)
    state = SessionState()

    reply = agent.respond(state, "hello")

    assert reply == FALLBACK_UNAVAILABLE
    assert client.calls == []
    timeout_trace = state.tool_traces[-1]
    assert timeout_trace.tool_name == "turn_timeout"
    assert timeout_trace.status is ToolStatus.ERROR
    assert "budget" in (timeout_trace.error or "")
    assert state.model_turns[-1].timed_out is True


def test_the_timeout_does_not_undo_a_write_already_committed(
    make_agent, anthropic_config, monkeypatch, verified_state, data_dir
) -> None:
    """A confirmed return is written before Claude is asked anything this turn
    (see `_auto_initiate_confirmed_returns`). If the budget is already spent by
    the time the loop would ask Claude to compose the reply, the turn still
    ends in the safe fallback — but the write that already committed stands,
    never rolled back and never repeated."""
    import agent.agent as agent_module
    from tests.conftest import returns_in

    agent, _ = make_agent(
        tool_call("check_return_eligibility", {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM}),
        text_response(CONFIRM_QUESTION),
    )
    agent.respond(verified_state, "I'd like to return the paperback")

    monkeypatch.setattr(agent_module, "TURN_TIMEOUT_SECONDS", 0.0)
    client = FakeAnthropic(text_response("should never be reached"))
    agent2 = BooklyAgent(config=anthropic_config, client=client)

    reply = agent2.respond(verified_state, "yes please")

    assert reply == FALLBACK_UNAVAILABLE
    assert client.calls == []  # the budget was spent before Claude was asked to reply
    write_trace = next(t for t in verified_state.tool_traces if t.tool_name == "initiate_return")
    assert write_trace.status is ToolStatus.OK  # the write itself was never touched
    assert len([r for r in returns_in(data_dir) if r["order_id"] == IN_WINDOW_ORDER]) == 1
    assert verified_state.pending_returns == []


def test_the_timeout_fires_mid_loop_not_just_at_the_start(
    make_agent, anthropic_config, monkeypatch
) -> None:
    """A budget that runs out partway through a multi-round-trip turn stops the
    *next* round trip — the one already in flight when the clock was read is
    unaffected, since the check only ever runs between iterations."""
    import agent.agent as agent_module

    monkeypatch.setattr(agent_module, "TURN_TIMEOUT_SECONDS", 0.05)

    class SlowStream:
        """A fake stream whose text arrives fast, but whose iteration — like a
        real slow tool call — burns most of the budget before returning."""

        def __init__(self, response: FakeResponse) -> None:
            self._response = response

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        @property
        def text_stream(self):
            time.sleep(0.12)
            for block in self._response.content:
                if block.type == "text" and block.text:
                    yield block.text

        def get_final_message(self):
            return self._response

    class SlowMessages:
        def __init__(self, responses):
            self._responses = list(responses)
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return self._responses.pop(0)

        def stream(self, **kwargs):
            return SlowStream(self.create(**kwargs))

    class SlowClient:
        def __init__(self, *responses):
            self.messages = SlowMessages(responses)

        @property
        def calls(self):
            return self.messages.calls

    client = SlowClient(
        tool_call("search_policy", {"query": "return window"}),
        text_response("This should never be reached: the budget is gone by now."),
    )
    agent = BooklyAgent(config=anthropic_config, client=client)
    state = SessionState()

    reply = agent.respond(state, "what's the return window?")

    assert reply == FALLBACK_UNAVAILABLE
    assert len(client.calls) == 1  # the slow first call ran; the second never started
    assert state.model_turns[-1].timed_out is True


# =========================================================================
# Agent Trace stays correct, and no reasoning leaks into it
# =========================================================================


def test_the_trace_carries_the_new_latency_metrics(make_agent, verified_state) -> None:
    """Time to first text, total latency, round trips, and the model/tool split
    are all on the turn — not estimated by the UI, read off what the loop
    actually measured."""
    agent, _ = make_agent(
        tool_call("check_return_eligibility", {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM}),
        text_response(CONFIRM_QUESTION),
    )

    agent.respond(verified_state, "can I return the paperback?")

    turn = verified_state.model_turns[-1]
    assert turn.iterations == 2
    assert turn.time_to_first_token_ms is not None
    assert turn.total_latency_ms >= turn.time_to_first_token_ms
    assert turn.model_latency_ms >= 0
    assert turn.tool_latency_ms >= 0
    assert turn.timed_out is False


def test_no_internal_reasoning_ever_reaches_the_call_or_the_reply(make_agent, verified_state) -> None:
    """No `thinking` block is requested, none could be relayed even if the SDK
    sent one — only `text_stream` deltas are ever yielded — and a tool's
    arguments, sanitized or not, never appear in what the customer is shown."""
    agent, client = make_agent(
        tool_call("check_return_eligibility", {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM}),
        text_response(CONFIRM_QUESTION),
    )

    reply = "".join(agent.respond_stream(verified_state, "can I return the paperback?"))

    assert all("thinking" not in call for call in client.calls)
    assert reply == CONFIRM_QUESTION  # exactly what Claude sent — nothing appended, nothing narrated
    assert "eligibility_token" not in reply
    assert "policy_id" not in reply
    assert "rule_path" not in reply
