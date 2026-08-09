"""The presentation layer, and the Streamlit shell around it.

Two halves. The first is pure functions — how a latency is written, how a graph
path reads, what an argument is allowed to show, and which assistant reply a tool
call belongs to. That is what a reviewer's trust in the trace actually rests on,
and it is testable without a browser.

The second drives `app.py` through Streamlit's own `AppTest` — in process, no
browser, no server — typing the hero flow into the chat box and asserting on what
appears under each reply, on both reset buttons, and on the absence of an email
address or a token anywhere on the page.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import ui
from agent.agent import AnthropicConfigError, _policy_decision
from agent.state import (
    EligibilityDecision,
    ModelTurn,
    Role,
    SessionState,
    ToolStatus,
    ToolTrace,
    mask_email,
    sanitize_args,
    summarize,
)
from agent.tools import _create_eligibility_token, _GRANTS, reset_demo
from policy.graph import PolicyGraphUnavailableError
from tests.conftest import (
    EXPIRED_ITEM,
    EXPIRED_ORDER,
    HERO_CUSTOMER,
    HERO_EMAIL,
    IN_WINDOW_ITEM,
    IN_WINDOW_ORDER,
    returns_in,
    run_hero_flow,
    text,
    tool_call,
    tool_calls,
)

APP = str(Path(__file__).resolve().parent.parent / "app.py")
"""Absolute, because `AppTest.from_file` resolves a relative path against *this* file."""

TIMEOUT = 30
"""Generous: the whole hero flow runs in one `run()` on some paths."""

STANDARD_PATH = ["(PhysicalBook)-[:GOVERNED_BY]->(STANDARD_30_DAY)"]
"""The hero decision's path: one hop, category to policy."""

REGIONAL_PATH = [
    "(PhysicalBook)-[:GOVERNED_BY]->(AU_BOOKLY_EXTENDED_RETURN)",
    "(AU)-[:HAS_OVERRIDE]->(AU_BOOKLY_EXTENDED_RETURN)",
    "(AU_BOOKLY_EXTENDED_RETURN)-[:OVERRIDES]->(STANDARD_30_DAY)",
]
"""A regional override: what granted it, and what it displaced."""


def trace(**overrides) -> ToolTrace:
    """A trace with the required fields filled in, for the helper tests."""
    fields = {
        "session_id": "SESS-TEST",
        "model": "test-haiku-model",
        "model_tier": "haiku",
        "tool_name": "lookup_order",
        "tool_args": {"order_id": IN_WINDOW_ORDER},
        "status": ToolStatus.OK,
        "latency_ms": 12.0,
        "result_summary": "OrderDetails",
    }
    return ToolTrace(**(fields | overrides))


# =========================================================================
# Formatting
# =========================================================================


@pytest.mark.parametrize(
    ("latency_ms", "expected"),
    [
        (0.0, "0.0 ms"), (0.42, "0.4 ms"), (9.94, "9.9 ms"), (31.0, "31 ms"),
        (120.4, "120 ms"), (999.6, "1000 ms"), (1200.0, "1.2 s"), (12500.0, "12.5 s"),
    ],
)
def test_latency_is_readable_at_every_scale(latency_ms: float, expected: str) -> None:
    """Short enough to scan, and never rounded to a zero it did not measure.

    The deterministic tools answer in a fraction of a millisecond, so a formatter
    that rounded to whole milliseconds would report "0 ms" and read as a broken
    clock rather than a fast lookup. Nothing here measures anything: what goes in
    is what comes out.
    """
    assert ui.format_latency(latency_ms) == expected


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (ToolStatus.OK, "Success"),
        (ToolStatus.BLOCKED, "Blocked"),
        (ToolStatus.ERROR, "Failed"),
        (ToolStatus.REJECTED, "Rejected"),
    ],
)
def test_every_status_has_its_own_word(status: ToolStatus, expected: str) -> None:
    """A refused guard and a broken database must not read the same. "Blocked" is
    the system working as designed; "Failed" is not."""
    assert ui.status_label(status) == expected


def test_status_labels_and_glyphs_cover_the_enum() -> None:
    """A status added later shows up as itself rather than as a blank. The mark is
    for scanning; the word is what actually says what happened."""
    assert {ui.status_label(s) for s in ToolStatus} == {"Success", "Blocked", "Failed", "Rejected"}
    assert ui.status_mark(ToolStatus.OK) == "✓"
    assert {ui.status_mark(s) for s in ToolStatus} == {"✓", "⊘", "✕", "–"}


@pytest.mark.parametrize(("tier", "expected"), [("haiku", "Haiku"), ("sonnet", "Sonnet"), ("", "—")])
def test_tier_labels_name_the_model_the_reviewer_is_looking_for(tier, expected) -> None:
    assert ui.tier_label(tier) == expected


def test_decision_reads_as_a_decision() -> None:
    assert ui.decision_label(True) == "Eligible"
    assert ui.decision_label(False) == "Not eligible"


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("informational policy lookup", "Policy lookup"),
        ("return or refund workflow", "Return workflow"),
        ("a return workflow is open", "Return workflow"),
        ("a return is awaiting confirmation", "Return confirmation"),
        ("escalation intent", "Escalation"),
        ("request to depart from policy", "Policy exception"),
        ("ambiguous reference to resolve", "Clarification"),
        ("multi-turn context (6 turns)", "Extended conversation"),
    ],
)
def test_the_metadata_line_names_the_turn_without_quoting_the_router(reason, expected) -> None:
    """Short, friendly, and none of the router's own phrasing. The reason itself
    is not lost — it is printed verbatim inside the trace."""
    assert ui.activity_label(reason) == expected


@pytest.mark.parametrize(
    ("tools", "expected"),
    [
        (["lookup_order"], "Order lookup"),
        (["verify_identity", "lookup_order", "lookup_order"], "Order lookup"),
        (["search_policy"], "Policy lookup"),
        (["verify_identity"], "Identity check"),
        (["lookup_order", "check_return_eligibility"], "Eligibility check"),
        ([], "Support question"),
    ],
)
def test_a_reason_that_says_nothing_is_labelled_by_what_ran(tools, expected) -> None:
    """"Simple read-only request" describes the routing, not the turn. So for the
    Haiku default the label comes from the tools, most decisive first."""
    assert ui.activity_label("simple read-only request", tools) == expected


def test_an_unrecognised_reason_falls_back_rather_than_leaking() -> None:
    """A reason added to the router later reads as a turn, not as a raw string."""
    assert ui.activity_label("some new rule fired", []) == "Support question"
    assert ui.activity_label("some new rule fired", ["lookup_order"]) == "Order lookup"


@pytest.mark.parametrize(("count", "expected"), [(0, ""), (1, "1 tool"), (2, "2 tools")])
def test_the_tool_count_is_compact_and_absent_when_nothing_ran(count, expected) -> None:
    """"0 tools" is a segment saying nothing, so the line drops it."""
    assert ui.tool_count_label(count) == expected


# =========================================================================
# Sanitization
#
# A trace records observable execution: what was called and what came back. Not
# the model's reasoning, not the API key, not the Neo4j password, and not the
# customer's email address in the clear.
# =========================================================================


def test_email_addresses_are_masked() -> None:
    """The domain is kept because it helps when reading; the local part is not."""
    assert mask_email("ada@example.com") == "a***@example.com"
    assert sanitize_args({"email": HERO_EMAIL})["email"] == "a***@example.com"


def test_sensitive_keys_are_redacted() -> None:
    """Anything credential-shaped is replaced whole, whatever it holds."""
    clean = sanitize_args(
        {
            "api_key": "sk-ant-real-key",
            "password": "hunter2",
            "eligibility_token": "9f2c-real-token",
            "order_id": IN_WINDOW_ORDER,
        }
    )
    assert clean["api_key"] == "***"
    assert clean["password"] == "***"
    assert clean["eligibility_token"] == "***"
    assert clean["order_id"] == IN_WINDOW_ORDER


def test_summaries_are_short() -> None:
    """A summary is a line to scan, not the payload."""
    assert len(summarize("x" * 5_000)) <= 200
    assert summarize(None) == "no result"


def test_display_sanitizes_a_second_time_on_the_way_out() -> None:
    """The loop sanitizes before it stores; this is the last gate before a screen.
    A UI that trusted its input to be clean would leak the first time a trace was
    built anywhere else."""
    shown = ui.display_args(
        {
            "email": HERO_EMAIL,
            "eligibility_token": "elig-abc123",
            "api_key": "sk-ant-real-key",
            "password": "neo4j-password",
            "order_id": IN_WINDOW_ORDER,
        }
    )

    assert shown["email"] == "a***@example.com"
    assert shown["eligibility_token"] == "***"
    assert shown["api_key"] == "***"
    assert shown["password"] == "***"
    assert shown["order_id"] == IN_WINDOW_ORDER


def test_display_drops_empty_arguments() -> None:
    """A blank row says nothing about the call, so it is not shown."""
    assert ui.display_args({"reason": "", "item_id": None, "order_id": IN_WINDOW_ORDER}) == {
        "order_id": IN_WINDOW_ORDER
    }


def test_arguments_render_as_one_line() -> None:
    assert ui.format_args({"order_id": "ORD-1001", "item_id": "ITEM-100"}) == (
        "order_id=ORD-1001, item_id=ITEM-100"
    )
    assert ui.format_args({}) == "—"


def test_no_secret_reaches_the_traces(make_agent, anthropic_config, monkeypatch) -> None:
    """Neither the Anthropic key nor the Neo4j password is anywhere near a trace.

    Both are set in the environment here, so a leak would show up rather than
    being absent by luck. The assertions are scoped to the traces, not the whole
    session: the customer typed their email and the conversation legitimately
    contains it — masking the transcript would break the agent's memory of what
    was said. The trace is the artefact that gets rendered and kept.
    """
    monkeypatch.setenv("NEO4J_PASSWORD", "neo4j-secret-password")

    agent, _ = make_agent(
        tool_call("verify_identity", {"email": HERO_EMAIL}), text("Found you.")
    )
    state = SessionState()
    agent.respond(state, HERO_EMAIL)

    traced = "".join(t.model_dump_json() for t in state.tool_traces)
    assert anthropic_config.api_key not in traced
    assert "neo4j-secret-password" not in traced
    assert HERO_EMAIL not in traced
    assert "a***@example.com" in traced


# =========================================================================
# The policy rule path
# =========================================================================


def test_a_simple_path_reads_from_category_to_policy() -> None:
    """The hero decision: one hop, two nodes, in the order they were walked."""
    assert ui.rule_path_nodes(STANDARD_PATH) == ["PhysicalBook", "STANDARD_30_DAY"]
    assert ui.format_rule_path(STANDARD_PATH) == "PhysicalBook → STANDARD_30_DAY"


def test_a_regional_path_shows_what_granted_it_and_what_it_displaced() -> None:
    """Every node of the traversal appears once, in the order it was reached.

    This is the whole point of showing the path: the answer came from a walk over
    the graph rather than from the model's opinion.
    """
    assert ui.rule_path_nodes(REGIONAL_PATH) == [
        "PhysicalBook", "AU_BOOKLY_EXTENDED_RETURN", "AU", "STANDARD_30_DAY"
    ]


def test_an_unrecognised_hop_is_shown_rather_than_dropped() -> None:
    """A path with a hop missing would misdescribe the decision, so a hop in a
    shape the formatter does not parse is displayed verbatim."""
    assert ui.rule_path_nodes(["(EBook)-[:GOVERNED_BY]->(NO_RETURN)", "something else"]) == [
        "EBook", "NO_RETURN", "something else"
    ]


def test_no_path_renders_as_nothing() -> None:
    """A refusal that never reached a policy has no traversal to show."""
    assert ui.rule_path_nodes([]) == []
    assert ui.format_rule_path([]) == ""


def test_the_path_comes_off_a_real_decision(seeded_graph, now) -> None:
    """Formatted from what the tool produced, not from a string written by hand."""
    from agent.tools import check_return_eligibility

    decision = check_return_eligibility(IN_WINDOW_ORDER, IN_WINDOW_ITEM, HERO_CUSTOMER, now=now)

    assert ui.format_rule_path(decision.rule_path) == "PhysicalBook → STANDARD_30_DAY"


# =========================================================================
# Traces belong to the turn that caused them
# =========================================================================


def test_capture_records_the_model_that_handled_the_turn() -> None:
    """The badge under a reply is read off the turn the router actually recorded."""
    state = SessionState()
    state.model_turns.append(
        ModelTurn(
            session_id=state.session_id,
            model_tier="sonnet",
            model="test-sonnet-model",
            routing_reason="return or refund workflow",
        )
    )

    turn = ui.capture_turn(state, "Shall I start that return?", trace_offset=0)

    assert turn.model_tier == "sonnet"
    assert turn.model == "test-sonnet-model"
    assert turn.routing_reason == "return or refund workflow"
    assert turn.reply == "Shall I start that return?"


def test_capture_takes_only_the_traces_from_this_turn() -> None:
    """The offset is what keeps an earlier turn's tools out of a later trace."""
    state = SessionState()
    state.tool_traces.append(trace(tool_name="verify_identity"))
    offset = len(state.tool_traces)
    state.tool_traces.append(trace(tool_name="lookup_order"))
    state.tool_traces.append(trace(tool_name="check_return_eligibility"))

    turn = ui.capture_turn(state, "Here you go.", trace_offset=offset)

    assert turn.tool_names == ["lookup_order", "check_return_eligibility"]


def test_capture_keeps_tools_in_execution_order() -> None:
    """Two orders read in one turn are shown in the order they were read."""
    state = SessionState()
    state.tool_traces.append(trace(tool_args={"order_id": IN_WINDOW_ORDER}))
    state.tool_traces.append(trace(tool_args={"order_id": EXPIRED_ORDER}))

    turn = ui.capture_turn(state, "Which of these did you mean?", trace_offset=0)

    assert [t.tool_args["order_id"] for t in turn.tool_traces] == [IN_WINDOW_ORDER, EXPIRED_ORDER]


def test_a_turn_that_ran_no_tools_captures_none() -> None:
    """Asking for an email address is a turn too, and it has a model badge."""
    state = SessionState()
    state.model_turns.append(
        ModelTurn(
            session_id=state.session_id,
            model_tier="haiku",
            model="test-haiku-model",
            routing_reason="simple read-only request",
        )
    )

    turn = ui.capture_turn(state, "What's the email on your account?", trace_offset=0)

    assert turn.tool_traces == []
    assert turn.model_tier == "haiku"


def test_a_blocked_write_is_visible_on_the_turn() -> None:
    """A blocked write is not a success, and the turn carries the status."""
    state = SessionState()
    state.tool_traces.append(
        trace(
            tool_name="initiate_return",
            status=ToolStatus.BLOCKED,
            result_summary="return blocked",
            error="no eligibility token",
        )
    )

    turn = ui.capture_turn(state, "I can't open that.", trace_offset=0)
    assert ui.status_label(turn.tool_traces[0].status) == "Blocked"
    assert turn.tool_traces[0].error


# --- The eligibility decision beside its check ---------------------------


def test_a_decision_travels_on_the_call_that_made_it() -> None:
    """The renderable decision is on the trace, so the tool call explains itself."""
    decision = _policy_decision(
        EligibilityDecision(
            eligible=True,
            policy_id="STANDARD_30_DAY",
            explanation="Inside the window.",
            rule_path=STANDARD_PATH,
            eligibility_token="elig-secret",
            days_remaining=12,
        )
    )

    assert decision == {
        "eligible": True,
        "policy_id": "STANDARD_30_DAY",
        "rule_path": STANDARD_PATH,
        "days_remaining": 12,
    }
    # The token is a credential, and the explanation is already the reply.
    assert "eligibility_token" not in decision
    assert "explanation" not in decision


def test_a_call_that_is_not_an_eligibility_check_carries_no_decision() -> None:
    """Every other tool leaves the field empty, so nothing borrows a policy."""
    assert _policy_decision(None) is None
    assert _policy_decision("ORD-1001 · 2 items") is None
    assert trace(tool_name="lookup_order").policy_decision is None


# --- Pairing the transcript with the traces ------------------------------


def turn_named(reply: str, tier: str = "haiku") -> ui.AssistantTurn:
    return ui.AssistantTurn(reply=reply, model_tier=tier, model=f"test-{tier}-model")


def test_each_assistant_message_is_paired_with_its_own_turn() -> None:
    """Matched by position, so two identical replies are still two turns."""
    state = SessionState()
    state.add_message(Role.USER, "Where's my book?")
    state.add_message(Role.ASSISTANT, "What's your email?")
    state.add_message(Role.USER, HERO_EMAIL)
    state.add_message(Role.ASSISTANT, "Which order did you mean?")

    paired = ui.pair_turns(
        state.messages, [turn_named("What's your email?"), turn_named("Which order did you mean?")]
    )

    assert [message.content for message, _ in paired] == [
        "Where's my book?", "What's your email?", HERO_EMAIL, "Which order did you mean?"
    ]
    assert [turn.reply if turn else None for _, turn in paired] == [
        None, "What's your email?", None, "Which order did you mean?"
    ]


def test_a_user_message_never_gets_a_trace() -> None:
    """The trace goes under the reply, because the reply is what it explains."""
    state = SessionState()
    state.add_message(Role.USER, "Yes please")

    assert ui.pair_turns(state.messages, [turn_named("Your return is open.")]) == [
        (state.messages[0], None)
    ]


def test_an_assistant_message_with_no_record_renders_without_one() -> None:
    """Happens to a session restored without its traces. Borrowing the following
    turn's trace would attribute tool calls to a reply that did not make them."""
    state = SessionState()
    state.add_message(Role.ASSISTANT, "Hello.")
    state.add_message(Role.ASSISTANT, "Still here.")

    paired = ui.pair_turns(state.messages, [turn_named("Hello.")])

    assert paired[0][1] is not None
    assert paired[1][1] is None


# --- The hero flow, as the UI would slice it -----------------------------


def capture_flow(agent, state: SessionState, *prompts: str) -> list[ui.AssistantTurn]:
    """Drive a conversation the way `app.run_turn` does, keeping the turns."""
    turns = []
    for prompt in prompts:
        offset = len(state.tool_traces)
        reply = agent.respond(state, prompt)
        turns.append(ui.capture_turn(state, reply, trace_offset=offset))
    return turns


HERO_PROMPTS = (
    "Where's my book?",
    HERO_EMAIL,
    "The Pragmatic Programmer one",
    "Actually, I want to return it.",
    "Yes please",
)


def test_the_hero_flow_traces_line_up_with_the_replies(make_agent, seeded_graph, hero_script) -> None:
    """The property the whole trace UI rests on: the reviewer can point at a reply
    and see what produced it. Nothing here is arranged for the demo."""
    agent, _ = make_agent(*hero_script)
    state = SessionState()

    turns = capture_flow(agent, state, *HERO_PROMPTS)

    assert [turn.tool_names for turn in turns] == [
        [],
        ["verify_identity", "lookup_order"],
        ["lookup_order"],
        ["check_return_eligibility"],
        ["initiate_return"],
    ]
    assert [turn.model_tier for turn in turns] == ["haiku", "haiku", "haiku", "sonnet", "sonnet"]
    assert all(t.status is ToolStatus.OK for turn in turns for t in turn.tool_traces)


def test_the_eligibility_turn_shows_the_policy_and_the_path(
    make_agent, seeded_graph, hero_script
) -> None:
    """The turn that decided the return is the turn that explains the decision."""
    agent, _ = make_agent(*hero_script)
    state = SessionState()

    turns = capture_flow(agent, state, *HERO_PROMPTS[:4])

    decision = turns[-1].tool_traces[0].policy_decision
    assert decision is not None
    assert decision["policy_id"] == "STANDARD_30_DAY"
    assert decision["eligible"] is True
    assert ui.format_rule_path(decision["rule_path"]) == "PhysicalBook → STANDARD_30_DAY"
    # The earlier turns explain nothing about policy, and must not claim to.
    assert all(t.policy_decision is None for turn in turns[:-1] for t in turn.tool_traces)


def test_the_hero_flow_display_shows_no_email_and_no_token(
    make_agent, seeded_graph, hero_script
) -> None:
    """The loop already sanitized these. The assertion is on the display path,
    because that is the one a reviewer looks at."""
    agent, _ = make_agent(*hero_script)
    state = SessionState()

    turns = capture_flow(agent, state, *HERO_PROMPTS)

    lines = [ui.format_args(t.tool_args) for turn in turns for t in turn.tool_traces]
    assert any("a***@example.com" in line for line in lines)
    assert not any(HERO_EMAIL in line for line in lines)
    assert not any("elig-" in line for line in lines)


def test_the_outside_window_turn_shows_a_refusal_with_its_policy(
    make_agent, seeded_graph, hero_verified
) -> None:
    """The edge case needs no special display: it is the same trace, saying no."""
    agent, _ = make_agent(
        tool_call(
            "check_return_eligibility",
            {"order_id": EXPIRED_ORDER, "item_id": EXPIRED_ITEM, "reason": "Didn't want it."},
        ),
        text("That one was delivered too long ago to return, I'm afraid."),
    )

    turn = capture_flow(agent, hero_verified, "Can I send this back?")[0]

    assert turn.model_tier == "sonnet"
    assert turn.tool_names == ["check_return_eligibility"]
    decision = turn.tool_traces[0].policy_decision
    assert decision is not None
    assert ui.decision_label(decision["eligible"]) == "Not eligible"
    assert decision["policy_id"] == "STANDARD_30_DAY"
    assert ui.format_rule_path(decision["rule_path"]) == "PhysicalBook → STANDARD_30_DAY"


def test_two_checks_in_one_turn_keep_their_own_decisions(make_agent, seeded_graph) -> None:
    """The regression: one turn, two items, two different policies.

    The session only ever holds the *last* decision, so a trace that read it would
    show the ebook's refusal against the physical book's check.
    """
    agent, _ = make_agent(
        tool_calls(
            ("check_return_eligibility", {"order_id": "ORD-1003", "item_id": "ITEM-102"}),
            ("check_return_eligibility", {"order_id": "ORD-1003", "item_id": "ITEM-201"}),
        ),
        text("The paperback can go back; the ebook can't."),
    )
    state = SessionState(
        verified_customer_id="CUST-002", customer_region="AU", active_order_ids=["ORD-1003"]
    )

    book, ebook = capture_flow(agent, state, "Which one am I eligible for?")[0].tool_traces

    assert book.policy_decision["policy_id"] == "AU_BOOKLY_EXTENDED_RETURN"
    assert ui.decision_label(book.policy_decision["eligible"]) == "Eligible"
    assert ui.format_rule_path(book.policy_decision["rule_path"]).startswith(
        "PhysicalBook → AU_BOOKLY_EXTENDED_RETURN"
    )

    assert ebook.policy_decision["policy_id"] == "DIGITAL_NO_RETURN"
    assert ui.decision_label(ebook.policy_decision["eligible"]) == "Not eligible"
    assert ui.format_rule_path(ebook.policy_decision["rule_path"]) == "EBook → DIGITAL_NO_RETURN"

    # The turn-level decision is the ebook's, and it did not overwrite the first.
    assert state.eligibility.policy_id == "DIGITAL_NO_RETURN"


def test_an_anthropic_outage_is_traced_and_answered(make_agent) -> None:
    """The customer gets a sentence; the trace says which call failed. The reply
    carries no exception type, and neither transcript carries a stack."""
    agent, _ = make_agent(error=RuntimeError("connection reset"))
    state = SessionState()

    turn = capture_flow(agent, state, "Where's my book?")[0]

    assert turn.tool_names == ["anthropic.messages.create"]
    assert ui.status_label(turn.tool_traces[0].status) == "Failed"
    assert "connection reset" in (turn.tool_traces[0].error or "")
    assert "RuntimeError" not in turn.reply
    assert "trouble" in turn.reply


# =========================================================================
# The demo reset
#
# The hero conversation writes a real RMA. That is what makes it worth showing,
# and it is also why a second rehearsal is not the same conversation as the first
# unless something restores the file in between.
# =========================================================================


def test_reset_restores_returns_after_a_write(
    make_agent, seeded_graph, hero_script, data_dir
) -> None:
    """The RMA the hero flow created is gone; the seeded one is back."""
    before = returns_in(data_dir)

    agent, _ = make_agent(*hero_script)
    run_hero_flow(agent, SessionState())
    assert len(returns_in(data_dir)) == len(before) + 1

    summary = reset_demo()

    assert "returns.json" in summary
    assert returns_in(data_dir) == before


def test_reset_is_idempotent(data_dir) -> None:
    """Running it twice leaves the same bytes as running it once."""
    reset_demo()
    once = (data_dir / "returns.json").read_bytes()
    reset_demo()
    assert (data_dir / "returns.json").read_bytes() == once


def test_reset_clears_outstanding_eligibility_tokens() -> None:
    """A token minted before the reset cannot be spent after it.

    The store is in-memory and process-global. A Streamlit process that reset the
    data but kept its tokens would let a token issued against the deleted RMA's
    order still satisfy `initiate_return`.
    """
    token = _create_eligibility_token(HERO_CUSTOMER, IN_WINDOW_ORDER, IN_WINDOW_ITEM, "STANDARD_30_DAY")
    assert token in _GRANTS

    reset_demo()

    assert _GRANTS == {}


def test_reset_leaves_static_fixtures_alone(data_dir) -> None:
    """Customers, orders, and items are never written, so they are never restored."""
    orders_before = (data_dir / "orders.json").read_bytes()
    customers_before = (data_dir / "customers.json").read_bytes()

    reset_demo()

    assert (data_dir / "orders.json").read_bytes() == orders_before
    assert (data_dir / "customers.json").read_bytes() == customers_before


def test_reset_without_a_baseline_directory_is_not_an_error(tmp_path, monkeypatch) -> None:
    """Nothing mutable means nothing to restore — not a crash."""
    from agent import tools

    empty = tmp_path / "empty-data"
    empty.mkdir()
    monkeypatch.setattr(tools, "DATA_DIR", empty)

    assert "nothing" in reset_demo()


def test_a_new_session_carries_nothing_over() -> None:
    """The session the UI installs after a reset knows nothing about the last run.

    A new object rather than a cleared one: `SessionState` grows fields as the
    agent does, and a reset that clears them one by one silently stops being
    complete the next time one is added.
    """
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

    clean = SessionState()

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


def test_the_hero_flow_runs_again_after_a_reset(
    make_agent, seeded_graph, hero_script, data_dir
) -> None:
    """Two identical rehearsals, each creating a real RMA.

    Without the reset the second pass would find the first pass's return and
    correctly refuse to duplicate it — right behaviour, wrong demo.
    """
    agent, _ = make_agent(*hero_script)
    run_hero_flow(agent, SessionState())
    first = [r for r in returns_in(data_dir) if r["order_id"] == IN_WINDOW_ORDER]
    assert len(first) == 1

    reset_demo()

    agent2, _ = make_agent(*hero_script)
    second_state = SessionState()
    run_hero_flow(agent2, second_state)

    second = [r for r in returns_in(data_dir) if r["order_id"] == IN_WINDOW_ORDER]
    assert len(second) == 1
    assert second[0]["return_id"] == first[0]["return_id"]
    assert second_state.tool_traces[-1].tool_name == "initiate_return"
    assert second_state.tool_traces[-1].status is ToolStatus.OK


def test_the_script_and_the_button_share_one_implementation() -> None:
    """`scripts/reset_demo.py` holds no reset logic of its own.

    Two copies of "put it back" is how a rehearsal and a live run end up starting
    from different states, so nothing in the script may touch files or JSON.
    """

    source = (Path(__file__).resolve().parent.parent / "scripts" / "reset_demo.py").read_text()
    assert "from agent.tools import reset_demo" in source
    assert not _imported_modules(source) & {"json", "shutil"}


def _imported_modules(source: str) -> set[str]:
    import ast

    tree = ast.parse(source)
    return {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }


# =========================================================================
# The Streamlit shell
# =========================================================================


@pytest.fixture
def app(make_agent, seeded_graph):
    """An `AppTest` for `app.py`, wired to a scripted Anthropic client.

    `app.get_agent` builds an agent only when the session does not already hold
    one, so a test can supply one without the app knowing.
    """

    def start(*responses: Any) -> AppTest:
        agent, _ = make_agent(*responses)
        at = AppTest.from_file(APP, default_timeout=TIMEOUT)
        at.session_state["bookly_agent"] = agent
        return at.run()

    return start


def say(at: AppTest, message: str) -> AppTest:
    """Send one customer message, as typing into the chat box does."""
    at.chat_input[0].set_value(message).run()
    return at


def traces(at: AppTest) -> list[Any]:
    """The per-turn trace expanders in the conversation, in transcript order."""
    return [block for block in at.expander if block.label == ui.TRACE_LABEL]


def body(block: Any) -> str:
    """Everything written inside one block, as one searchable string."""
    return "\n".join(e.value for e in block.markdown) + "\n".join(e.value for e in block.caption)


def page_text(at: AppTest) -> str:
    """Every markdown and caption on the page, as one searchable string."""
    return "\n".join(e.value for e in list(at.main.markdown) + list(at.main.caption))


def run_hero_flow_in_ui(at: AppTest) -> AppTest:
    """The five customer turns of the demo, typed into the chat box."""
    for message in HERO_PROMPTS:
        say(at, message)
    return at


def press(at: AppTest, label: str) -> AppTest:
    """Press a sidebar button by its label."""
    next(button for button in at.sidebar.button if button.label == label).click().run()
    return at


def test_the_app_opens_on_a_welcome_and_a_chat_box(app) -> None:
    """A short greeting, an input, and nothing else preloaded. Specifically not
    the hero conversation: the demo is typed live."""
    at = app(text("hello"))

    assert at.chat_input
    assert "Bookly Support" in page_text(at)
    assert "orders, returns, refunds" in page_text(at)
    assert traces(at) == []
    assert not at.error


def test_the_developer_state_is_available_and_separate(app) -> None:
    """The debug view is in the sidebar, collapsed, away from the conversation."""
    at = app(text("hello"))

    developer = [b for b in at.sidebar.expander if b.label == ui.DEVELOPER_LABEL]
    assert len(developer) == 1
    assert "Verified customer" in body(developer[0])


def test_the_page_carries_no_implementation_notes(app) -> None:
    """The customer's screen is the conversation, not the design of the shell."""
    at = app(text("hello"))
    page = page_text(at)

    for phrase in ("What the customer sees", "agent.agent", "agent.tools", "stack trace"):
        assert phrase not in page


def test_the_script_renders_nothing_it_did_not_ask_to() -> None:
    """Streamlit's magic writes any bare expression at the top level to the page.

    Which makes a module-level string a rendered paragraph, not a note — how a
    docstring describing the shell's error handling once ended up above the Bookly
    header. Implementation notes in `app.py` are comments for that reason.
    """
    import ast

    body_nodes = ast.parse(Path(APP).read_text()).body
    stray = [
        node.lineno
        for node in body_nodes[1:]  # the module docstring is not magic
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    ]
    assert stray == []


def test_a_missing_anthropic_configuration_is_an_explanation_not_a_crash(
    monkeypatch, seeded_graph
) -> None:
    """An unconfigured deployment stops with a message and no chat box.

    The agent is constructed at startup for exactly this reason, so the failure
    lands here rather than on the customer's first message.
    """
    from agent import agent as agent_module

    def unconfigured(*_args: Any, **_kwargs: Any):
        raise AnthropicConfigError("ANTHROPIC_API_KEY is not set")

    monkeypatch.setattr(agent_module, "BooklyAgent", unconfigured)

    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()

    assert "ANTHROPIC_API_KEY is not set" in at.error[0].value
    assert not at.chat_input


def test_neo4j_is_warmed_once_however_often_the_script_reruns(app, monkeypatch) -> None:
    """Streamlit reruns the whole script per interaction; the pool is warmed once.

    Read at the driver, so this also asserts the warm-up and the policy tools go
    through the same shared driver rather than a second one.
    """
    queries: list[str] = []

    class WarmDriver:
        def execute_query(self, query: str, **_: Any):
            queries.append(query)
            return [], None, None

    monkeypatch.setattr("policy.graph.get_driver", WarmDriver)
    st.cache_resource.clear()

    at = app(text("Hello!"), text("Still here."))
    say(at, "Where's my book?")
    say(at, "Thanks!")

    assert queries == ["RETURN 1"]


def test_the_shell_still_opens_when_neo4j_is_down(app, monkeypatch) -> None:
    """Best-effort: a cold database costs a slow first query, not a broken page."""

    def refuse() -> None:
        raise PolicyGraphUnavailableError("cannot reach Neo4j at bolt://nowhere")

    monkeypatch.setattr("policy.graph.get_driver", refuse)
    st.cache_resource.clear()

    at = app(text("Hello!"))

    assert not at.exception
    assert at.chat_input
    assert "Neo4j" not in page_text(at)


def test_a_reply_carries_its_own_trace(app) -> None:
    """The tool that ran appears under the reply that used it, with a latency."""
    at = app(
        tool_call("lookup_order", {"order_id": IN_WINDOW_ORDER}),
        text("It was delivered on 28 July."),
    )
    at.session_state["bookly_state"] = SessionState(
        verified_customer_id=HERO_CUSTOMER,
        customer_region="GB",
        active_order_ids=[IN_WINDOW_ORDER],
        active_order_id=IN_WINDOW_ORDER,
    )
    at.run()

    say(at, "Where's my book?")

    assert "It was delivered on 28 July." in page_text(at)
    assert len(traces(at)) == 1
    shown = body(traces(at)[0])
    assert "lookup_order" in shown
    assert "ms" in shown or " s" in shown
    assert "Success" in shown
    # The router's own wording stays in the trace, where it explains the badge.
    assert "Routing · simple read-only request" in shown


def test_the_model_badge_says_which_tier_handled_the_turn(app) -> None:
    """Routing is visible on the page without opening anything: the tier as a
    badge, then what the turn was. The router's own reason is in the trace."""
    at = app(text("What's the email address on your Bookly account?"))

    say(at, "Where's my book?")

    assert "Haiku" in page_text(at)
    assert "Support question" in page_text(at)
    assert "simple read-only request" not in page_text(at)


def test_a_turn_with_no_tools_has_no_trace_to_open(app) -> None:
    """An expander promising a trace of nothing is a click for no reason."""
    at = app(text("What's the email address on your Bookly account?"))

    say(at, "Where's my book?")

    assert traces(at) == []
    assert "Haiku" in page_text(at)


def test_a_shell_failure_reaches_the_customer_as_a_sentence(app) -> None:
    """No stack trace, ever, and no module names — one sentence and a way out."""
    at = app(text("hello"))

    class Broken:
        def respond(self, *_args: Any, **_kwargs: Any) -> str:
            raise RuntimeError("something unforeseen")

    at.session_state["bookly_agent"] = Broken()
    at.run()

    say(at, "Where's my book?")

    assert "Something went wrong" in at.error[0].value
    assert not at.exception
    assert "RuntimeError" not in page_text(at)
    assert "agent.agent" not in page_text(at)


def test_the_hero_flow_renders_a_trace_per_acting_turn(app, hero_script, data_dir) -> None:
    """Five replies, four of which did something, and one real RMA at the end."""
    at = run_hero_flow_in_ui(app(*hero_script))

    assert len(traces(at)) == 4
    assert [t.tool_name for t in at.session_state["bookly_state"].tool_traces] == [
        "verify_identity",
        "lookup_order",  # discovery: which orders are there to choose between
        "lookup_order",  # the one she chose, in detail
        "check_return_eligibility",
        "initiate_return",
    ]
    assert len([r for r in returns_in(data_dir) if r["order_id"] == IN_WINDOW_ORDER]) == 1


def test_the_hero_flow_shows_both_tiers_where_the_router_chose_them(app, hero_script) -> None:
    """Haiku on the lookups, Sonnet from the return onwards. Nothing forced."""
    at = run_hero_flow_in_ui(app(*hero_script))

    page = page_text(at)
    assert "Haiku" in page
    assert "Sonnet" in page
    assert [turn.model_tier for turn in at.session_state["bookly_turns"]] == [
        "haiku", "haiku", "haiku", "sonnet", "sonnet"
    ]


def test_the_eligibility_trace_shows_the_policy_and_the_graph_path(app, hero_script) -> None:
    """The decision is explained by the traversal that produced it."""
    at = run_hero_flow_in_ui(app(*hero_script))

    eligibility = next(b for b in traces(at) if "check_return_eligibility" in body(b))
    shown = body(eligibility)
    assert "STANDARD_30_DAY" in shown
    assert "Eligible" in shown
    assert "PhysicalBook" in shown
    assert "→ STANDARD_30_DAY" in shown


def test_the_traces_stay_attached_to_the_right_replies(app, hero_script) -> None:
    """Turn two verified and listed her orders to ask which; turn three read the
    one she picked."""
    at = run_hero_flow_in_ui(app(*hero_script))

    shown = [body(block) for block in traces(at)]
    assert "verify_identity" in shown[0]
    assert shown[0].count("lookup_order") == 1
    assert shown[1].count("lookup_order") == 1
    assert "check_return_eligibility" in shown[2]
    assert "initiate_return" in shown[3]


def test_no_email_or_token_reaches_the_trace(app, hero_script) -> None:
    """The customer's own message still says their address, because they typed it
    — masking a customer's view of their own conversation would be theatre. What
    matters is that the *trace* records the attempt without the value."""
    at = run_hero_flow_in_ui(app(*hero_script))

    shown = "\n".join(body(block) for block in traces(at))
    assert HERO_EMAIL not in shown
    assert "a***@example.com" in shown
    assert "elig-" not in shown + page_text(at)


def test_the_developer_state_reports_a_token_without_showing_it(app, hero_script) -> None:
    """Presence is the useful part; the value is a credential."""
    at = app(*hero_script)
    for message in HERO_PROMPTS[:4]:
        say(at, message)

    developer = next(b for b in at.sidebar.expander if b.label == ui.DEVELOPER_LABEL)
    shown = body(developer)
    token = at.session_state["bookly_state"].eligibility_token
    assert token
    assert "held" in shown
    assert token not in shown


def test_the_outside_window_case_needs_no_special_display(app) -> None:
    """Same trace, saying no: policy named, path shown, nothing written."""
    at = app(
        tool_call(
            "check_return_eligibility",
            {"order_id": EXPIRED_ORDER, "item_id": EXPIRED_ITEM, "reason": "Didn't want it."},
        ),
        text("That one was delivered too long ago to return, I'm afraid."),
    )
    at.session_state["bookly_state"] = SessionState(
        verified_customer_id=HERO_CUSTOMER,
        customer_region="GB",
        active_order_ids=[IN_WINDOW_ORDER, EXPIRED_ORDER],
    )
    at.run()

    say(at, "Can I send this back?")

    shown = body(traces(at)[0])
    assert "Sonnet" in page_text(at)
    assert "Not eligible" in shown
    assert "STANDARD_30_DAY" in shown
    assert "→ STANDARD_30_DAY" in shown
    assert at.session_state["bookly_state"].eligibility_token is None


def test_reset_demo_clears_the_conversation_the_traces_and_the_data(
    app, hero_script, data_dir
) -> None:
    """One button, and the demo is rehearsable again.

    The reset itself is `agent.tools.reset_demo` — the same function the command
    line calls. What this checks is that the UI's own state goes with it.
    """
    at = run_hero_flow_in_ui(app(*hero_script))
    before = at.session_state["bookly_state"].session_id
    assert at.session_state["bookly_turns"]

    press(at, "Reset demo")

    assert at.session_state["bookly_state"].messages == []
    assert at.session_state["bookly_state"].tool_traces == []
    assert at.session_state["bookly_state"].session_id != before
    assert at.session_state["bookly_turns"] == []
    assert traces(at) == []
    assert not [r for r in returns_in(data_dir) if r["order_id"] == IN_WINDOW_ORDER]
    assert "Demo reset" in at.sidebar.success[0].value
    # The agent survives, so a reset does not re-read the configuration — and a
    # rehearsal cannot be stopped by a key that was fine a moment ago.
    assert "bookly_agent" in at.session_state


def test_reset_conversation_leaves_the_data_alone(app, hero_script, data_dir) -> None:
    """Two resets, deliberately different: this one forgets, it does not restore.
    The RMA the conversation created is a real record."""
    at = run_hero_flow_in_ui(app(*hero_script))
    created = [r for r in returns_in(data_dir) if r["order_id"] == IN_WINDOW_ORDER]
    assert len(created) == 1

    press(at, "Reset conversation")

    assert at.session_state["bookly_state"].messages == []
    assert traces(at) == []
    assert [r for r in returns_in(data_dir) if r["order_id"] == IN_WINDOW_ORDER] == created


def test_the_hero_flow_can_be_run_twice_through_the_ui(
    app, make_agent, hero_script, data_dir
) -> None:
    """The rehearsal property, from the reviewer's side of the screen."""
    at = run_hero_flow_in_ui(app(*hero_script))
    first = [r for r in returns_in(data_dir) if r["order_id"] == IN_WINDOW_ORDER]

    press(at, "Reset demo")

    # A fresh agent for the second pass, because the first one's *script* was
    # consumed. The reset itself keeps the agent — it carries no conversation
    # state, and rebuilding it would only re-read the same configuration.
    second_agent, _ = make_agent(*hero_script)
    at.session_state["bookly_agent"] = second_agent
    at.run()
    run_hero_flow_in_ui(at)

    second = [r for r in returns_in(data_dir) if r["order_id"] == IN_WINDOW_ORDER]
    assert len(second) == 1
    assert second[0]["return_id"] == first[0]["return_id"]
    assert len(traces(at)) == 4
    assert at.session_state["bookly_state"].tool_traces[-1].tool_name == "initiate_return"


def test_the_shell_opens_a_rotating_log_file(monkeypatch) -> None:
    """`logs/bookly.log`, created on the way up if it is not there.

    Rotation is the stdlib's, at a megabyte and three backups, so a long-running
    demo cannot fill a disk and nothing here implements a rotation scheme. The
    `log_file` fixture already pointed `BOOKLY_LOG_FILE` at a throwaway path;
    reimporting picks it up, since `LOG_FILE` is read once at module load.
    """
    import importlib
    import logging
    from logging.handlers import RotatingFileHandler

    import app

    app = importlib.reload(app)
    app.setup_logging()
    app.LOG.info("tool name=verify_identity status=success latency_ms=1.2")

    assert app.LOG_FILE.exists()
    written = app.LOG_FILE.read_text()
    assert "INFO tool name=verify_identity status=success latency_ms=1.2" in written

    handler = next(h for h in app.LOG.handlers if isinstance(h, RotatingFileHandler))
    assert (handler.maxBytes, handler.backupCount) == (1_000_000, 3)
    assert app.LOG.level == logging.INFO


def test_an_unwritable_log_file_does_not_stop_the_app(tmp_path, monkeypatch) -> None:
    """A log nobody can write is a log nobody gets, not a shell that will not
    start."""
    import importlib

    import app

    blocked = tmp_path / "not-a-directory"
    blocked.write_text("")
    monkeypatch.setenv("BOOKLY_LOG_FILE", str(blocked / "logs" / "bookly.log"))

    app = importlib.reload(app)
    handlers_before = list(app.LOG.handlers)
    app.setup_logging()  # raises nothing

    assert app.LOG.handlers == handlers_before


def test_the_ui_reset_holds_no_reset_logic_of_its_own() -> None:
    """`app.py` restores nothing itself — it calls the shared implementation."""
    source = Path(APP).read_text()
    assert "from agent.tools import reset_demo" in source
    # No reading, copying, or deleting of the demo data here. `os` and `pathlib`
    # are allowed: they only place the log file, and neither restores anything.
    assert not _imported_modules(source) & {"json", "shutil"}


def test_the_ui_layer_holds_no_business_logic() -> None:
    """`ui.py` calls no tool, evaluates no policy, and chooses no model.

    It reads the records the loop wrote and turns them into something to look at.
    Importing a tool or the policy layer here would be the first step towards a
    second answer to a question the agent already answered.
    """
    source = (Path(__file__).resolve().parent.parent / "ui.py").read_text()
    imported = _imported_modules(source)
    assert "policy" not in imported
    assert "agent.tools" not in source
