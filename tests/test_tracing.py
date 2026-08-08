"""Tool tracing: every call observable, and nothing secret written down.

Two halves. The first is that the trace exists and is complete — a Phase 6 UI
can only render what was captured, so every field it will need has to be here
now: which model, which tool, what status, how long.

The second is what a trace must *not* contain. It records observable execution:
what was called and what came back. Not the model's reasoning, not the API key,
not the Neo4j password, and not the customer's email address in the clear.
"""

from __future__ import annotations

from agent.state import SessionState
from agent.tracing import (
    ModelTurn,
    ToolStatus,
    ToolTrace,
    mask_email,
    sanitize_args,
    summarize,
)
from tests.conftest import text, tool_call

# --- The trace is created and complete -----------------------------------


def test_a_tool_call_creates_a_trace(make_agent) -> None:
    """One call in, one trace out."""
    agent, _ = make_agent(
        tool_call("verify_identity", {"email": "ada@example.com"}),
        text("Found you."),
    )
    state = SessionState()

    agent.respond(state, "ada@example.com")

    assert len(state.tool_traces) == 1
    assert state.tool_traces[0].tool_name == "verify_identity"


def test_trace_records_the_selected_model(make_agent, seeded_graph, verified_state) -> None:
    """A trace names both the tier and the model id it resolved to.

    This is what lets the demo show routing: a policy question traced against
    Haiku, a return traced against Sonnet, side by side.
    """
    agent, _ = make_agent(
        tool_call("search_policy", {"query": "ebooks"}),
        text("Ebooks aren't returnable."),
    )
    agent.respond(verified_state, "can I return an ebook")  # return intent → Sonnet

    trace = verified_state.tool_traces[0]
    assert trace.model_tier == "sonnet"
    assert trace.model == "test-sonnet-model"


def test_trace_contains_latency_and_status(make_agent) -> None:
    """Both are required for the trace to be worth rendering."""
    agent, _ = make_agent(
        tool_call("verify_identity", {"email": "ada@example.com"}),
        text("Found you."),
    )
    state = SessionState()
    agent.respond(state, "ada@example.com")

    trace = state.tool_traces[0]
    assert trace.status is ToolStatus.OK
    assert trace.latency_ms >= 0
    assert trace.result_summary
    assert trace.error is None


def test_trace_carries_identity_and_time(make_agent) -> None:
    """Session, trace id, and timestamp — enough to order and group traces."""
    agent, _ = make_agent(
        tool_call("verify_identity", {"email": "ada@example.com"}),
        text("Found you."),
    )
    state = SessionState()
    agent.respond(state, "ada@example.com")

    trace = state.tool_traces[0]
    assert trace.session_id == state.session_id
    assert trace.trace_id.startswith("TRC-")
    assert trace.timestamp is not None


def test_failed_calls_are_traced_too(make_agent) -> None:
    """A refused or unknown call is exactly what someone reads a trace for."""
    agent, _ = make_agent(
        tool_call("delete_account", {"why": "because"}),
        text("I can't do that."),
    )
    state = SessionState()
    agent.respond(state, "delete my account")

    trace = state.tool_traces[0]
    assert trace.status is ToolStatus.REJECTED
    assert trace.error is not None


def test_traces_accumulate_in_order(make_agent, seeded_graph) -> None:
    """The list is the sequence of what happened, oldest first."""
    agent, _ = make_agent(
        tool_call("verify_identity", {"email": "ada@example.com"}),
        tool_call("lookup_order", {"order_id": "ORD-1001"}, block_id="toolu_2"),
        text("It arrived on the 28th."),
    )
    state = SessionState()
    agent.respond(state, "ada@example.com, where's ORD-1001?")

    assert [t.tool_name for t in state.tool_traces] == ["verify_identity", "lookup_order"]


# --- Model turns ---------------------------------------------------------


def test_every_turn_records_its_model(make_agent) -> None:
    """Recorded even when no tool runs, so routing is visible on any turn."""
    agent, _ = make_agent(text("Hello!"), text("Of course."))
    state = SessionState()

    agent.respond(state, "hi")
    agent.respond(state, "I need a refund")

    assert [turn.model_tier for turn in state.model_turns] == ["haiku", "sonnet"]
    assert [turn.model for turn in state.model_turns] == [
        "test-haiku-model",
        "test-sonnet-model",
    ]


def test_model_turn_records_the_routing_reason_and_counts(make_agent) -> None:
    """Why the tier was chosen, and how much work the turn took."""
    agent, _ = make_agent(
        tool_call("verify_identity", {"email": "ada@example.com"}),
        text("Found you."),
    )
    state = SessionState()
    agent.respond(state, "ada@example.com")

    turn = state.model_turns[0]
    assert turn.routing_reason
    assert turn.iterations == 2
    assert turn.tool_calls == 1


def test_model_call_failure_is_traced(make_agent) -> None:
    """An outage is part of the execution record too."""
    agent, _ = make_agent(error=RuntimeError("529 overloaded"))
    state = SessionState()
    agent.respond(state, "hello")

    trace = state.tool_traces[-1]
    assert trace.tool_name == "anthropic.messages.create"
    assert trace.status is ToolStatus.ERROR
    assert "529" in (trace.error or "")


# --- Nothing secret ------------------------------------------------------


def test_email_addresses_are_masked() -> None:
    """The domain is kept because it helps when reading; the local part is not."""
    assert mask_email("ada@example.com") == "a***@example.com"
    assert sanitize_args({"email": "ada@example.com"})["email"] == "a***@example.com"


def test_sensitive_keys_are_redacted() -> None:
    """Anything credential-shaped is replaced whole, whatever it holds."""
    clean = sanitize_args(
        {
            "api_key": "sk-ant-real-key",
            "password": "hunter2",
            "eligibility_token": "9f2c-real-token",
            "order_id": "ORD-1001",
        }
    )
    assert clean["api_key"] == "***"
    assert clean["password"] == "***"
    assert clean["eligibility_token"] == "***"
    assert clean["order_id"] == "ORD-1001"


def test_traced_call_does_not_record_the_eligibility_token(
    make_agent, seeded_graph, verified_state
) -> None:
    """The token is injected into the real call but never written to a trace.

    `initiate_return` is passed a live token from session state. A trace that
    recorded it would put a spendable credential into a log and, in Phase 6,
    onto a screen.
    """
    agent, _ = make_agent(
        tool_call("check_return_eligibility", {"order_id": "ORD-1001", "item_id": "ITEM-100"}),
        text("Shall I start a return for that?"),
    )
    agent.respond(verified_state, "I'd like to return the paperback")
    token = verified_state.eligibility_token
    assert token

    agent2, _ = make_agent(
        tool_call(
            "initiate_return",
            {"order_id": "ORD-1001", "item_id": "ITEM-100", "reason": "damaged"},
        ),
        text("Your return is open."),
    )
    agent2.respond(verified_state, "yes please")

    traced = "".join(trace.model_dump_json() for trace in verified_state.tool_traces)
    assert token not in traced
    assert verified_state.tool_traces[-1].tool_args["eligibility_token"] == "***"

    # Nor does the model ever see it: the eligibility result sent back to
    # Anthropic has the token stripped out.
    transcript = str(verified_state.transcript)
    assert token not in transcript


def test_no_secret_reaches_the_traces(make_agent, anthropic_config, monkeypatch) -> None:
    """Neither the Anthropic key nor the Neo4j password is anywhere near a trace.

    Both are set in the environment for this test, so a leak would show up
    rather than being absent by luck.

    The assertions are scoped to the traces, not the whole session. The customer
    typed their email and the conversation legitimately contains it — masking
    the transcript would break the agent's memory of what was said. The trace is
    the artefact that gets rendered and kept, so the trace is what is masked.
    """
    monkeypatch.setenv("NEO4J_PASSWORD", "neo4j-secret-password")

    agent, _ = make_agent(
        tool_call("verify_identity", {"email": "ada@example.com"}),
        text("Found you."),
    )
    state = SessionState()
    agent.respond(state, "ada@example.com")

    traced = "".join(trace.model_dump_json() for trace in state.tool_traces)
    assert anthropic_config.api_key not in traced
    assert "neo4j-secret-password" not in traced
    assert "ada@example.com" not in traced
    assert "a***@example.com" in traced


def test_traces_hold_no_reasoning(make_agent) -> None:
    """A trace records execution, not thinking.

    The trace model has no field for reasoning, and the loop drops any block
    that is not text or a tool call — so there is nowhere for chain-of-thought
    to land even if a response carried it.
    """
    assert "thinking" not in ToolTrace.model_fields
    assert "reasoning" not in ToolTrace.model_fields
    assert "thinking" not in ModelTurn.model_fields

    agent, _ = make_agent(
        tool_call("verify_identity", {"email": "ada@example.com"}),
        text("Found you."),
    )
    state = SessionState()
    agent.respond(state, "ada@example.com")

    for message in state.transcript:
        content = message["content"]
        if isinstance(content, list):
            assert all(block["type"] in {"text", "tool_use", "tool_result"} for block in content)


def test_summaries_are_short() -> None:
    """A summary is a line to scan, not the payload."""
    assert len(summarize("x" * 5_000)) <= 200
    assert summarize(None) == "no result"
