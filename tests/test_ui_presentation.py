"""The presentation layer: what the trace says, and which turn it says it about.

Nothing here renders a page. The interesting parts of the UI are pure functions —
how a latency is written, how a graph path reads, what an argument is allowed to
show, and which assistant reply a tool call belongs to — and those are what a
reviewer's trust in the trace actually rests on. `tests/test_ui_app.py` drives the
Streamlit script itself.
"""

from __future__ import annotations

import pytest

from agent.models import EligibilityDecision, Role
from agent.state import SessionState
from agent.tracing import ModelTurn, ToolStatus, ToolTrace
from tests.conftest import text, tool_call
from tests.test_hero_flow import (
    EXPIRED_ITEM,
    EXPIRED_ORDER,
    HERO_CUSTOMER,
    IN_WINDOW_ITEM,
    IN_WINDOW_ORDER,
    hero_script,  # noqa: F401 - imported so pytest registers the fixture here too
)
from ui.format import (
    decision_label,
    display_args,
    format_args,
    format_latency,
    format_rule_path,
    rule_path_nodes,
    status_label,
    tier_label,
)
from ui.turns import AssistantTurn, capture_turn, eligibility_for, pair_turns

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


# --- Latency -------------------------------------------------------------


@pytest.mark.parametrize(
    ("latency_ms", "expected"),
    [
        (0.0, "0.0 ms"),
        (0.42, "0.4 ms"),
        (9.94, "9.9 ms"),
        (31.0, "31 ms"),
        (120.4, "120 ms"),
        (999.6, "1000 ms"),
        (1200.0, "1.2 s"),
        (12500.0, "12.5 s"),
    ],
)
def test_latency_is_readable_at_every_scale(latency_ms: float, expected: str) -> None:
    """Short enough to scan, and never rounded to a zero it did not measure.

    The deterministic tools answer in a fraction of a millisecond, so a formatter
    that rounded to whole milliseconds would report "0 ms" and read as a broken
    clock rather than a fast lookup.
    """
    assert format_latency(latency_ms) == expected


def test_latency_is_never_invented() -> None:
    """The formatter measures nothing — what goes in is what comes out.

    Guards the property the trace's credibility rests on: every number shown was
    recorded by `invoke_timed`, not produced here.
    """
    assert format_latency(31.0) == "31 ms"
    assert format_latency(0.0) == "0.0 ms"


# --- Labels --------------------------------------------------------------


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
    """A refused guard and a broken database must not read the same.

    "Blocked" is the system working as designed; "Failed" is not, and a reviewer
    reading the trace should not have to work out which one they are looking at.
    """
    assert status_label(status) == expected


def test_status_labels_cover_the_enum() -> None:
    """A status added later shows up as itself rather than as a blank."""
    assert {status_label(status) for status in ToolStatus} == {
        "Success",
        "Blocked",
        "Failed",
        "Rejected",
    }


@pytest.mark.parametrize(
    ("tier", "expected"), [("haiku", "Haiku"), ("sonnet", "Sonnet"), ("", "—")]
)
def test_tier_labels_name_the_model_the_reviewer_is_looking_for(
    tier: str, expected: str
) -> None:
    """Haiku and Sonnet, written the way the badge shows them."""
    assert tier_label(tier) == expected


def test_decision_reads_as_a_decision() -> None:
    assert decision_label(True) == "Eligible"
    assert decision_label(False) == "Not eligible"


# --- Sanitization --------------------------------------------------------


def test_display_masks_the_customers_email() -> None:
    """A trace shows that verification was attempted, not who by."""
    assert display_args({"email": "ada@example.com"}) == {"email": "a***@example.com"}


def test_display_redacts_anything_credential_shaped() -> None:
    """Every sensitive key, whatever its value, and regardless of how it arrived.

    The orchestrator sanitizes before it stores; this is the second pass, on the
    way to the screen. A UI that trusted its input to already be clean would leak
    the first time a trace was built anywhere else.
    """
    shown = display_args(
        {
            "eligibility_token": "elig-abc123",
            "api_key": "sk-ant-real-key",
            "password": "neo4j-password",
            "order_id": IN_WINDOW_ORDER,
        }
    )

    assert shown["eligibility_token"] == "***"
    assert shown["api_key"] == "***"
    assert shown["password"] == "***"
    assert shown["order_id"] == IN_WINDOW_ORDER


def test_display_drops_empty_arguments() -> None:
    """A blank row says nothing about the call, so it is not shown."""
    assert display_args({"reason": "", "item_id": None, "order_id": IN_WINDOW_ORDER}) == {
        "order_id": IN_WINDOW_ORDER
    }


def test_arguments_render_as_one_line() -> None:
    assert format_args({"order_id": "ORD-1001", "item_id": "ITEM-100"}) == (
        "order_id=ORD-1001, item_id=ITEM-100"
    )
    assert format_args({}) == "—"


# --- The policy rule path ------------------------------------------------


def test_a_simple_path_reads_from_category_to_policy() -> None:
    """The hero decision: one hop, two nodes, in the order they were walked."""
    assert rule_path_nodes(STANDARD_PATH) == ["PhysicalBook", "STANDARD_30_DAY"]
    assert format_rule_path(STANDARD_PATH) == "PhysicalBook\n→ STANDARD_30_DAY"


def test_a_regional_path_shows_what_granted_it_and_what_it_displaced() -> None:
    """Every node of the traversal appears once, in the order it was reached.

    This is the whole point of showing the path: the answer came from a walk over
    the graph — category, the region that unlocked the override, the policy that
    won, and the one it outranked — rather than from the model's opinion.
    """
    assert rule_path_nodes(REGIONAL_PATH) == [
        "PhysicalBook",
        "AU_BOOKLY_EXTENDED_RETURN",
        "AU",
        "STANDARD_30_DAY",
    ]


def test_an_unrecognised_hop_is_shown_rather_than_dropped() -> None:
    """A path with a hop missing would misdescribe the decision.

    So a hop in a shape the formatter does not parse is displayed verbatim. Worse
    to read than the arrow notation; better than a shorter path than the one that
    was actually walked.
    """
    assert rule_path_nodes(["(EBook)-[:GOVERNED_BY]->(NO_RETURN)", "something else"]) == [
        "EBook",
        "NO_RETURN",
        "something else",
    ]


def test_no_path_renders_as_nothing() -> None:
    """A refusal that never reached a policy has no traversal to show."""
    assert rule_path_nodes([]) == []
    assert format_rule_path([]) == ""


def test_the_path_comes_off_a_real_decision(seeded_graph, now) -> None:
    """Formatted from what the tool produced, not from a string written by hand."""
    from tools.check_return_eligibility import check_return_eligibility

    decision = check_return_eligibility(IN_WINDOW_ORDER, IN_WINDOW_ITEM, HERO_CUSTOMER, now=now)

    assert format_rule_path(decision.rule_path) == "PhysicalBook\n→ STANDARD_30_DAY"


# --- Traces belong to the turn that caused them --------------------------


def test_capture_records_the_model_that_handled_the_turn() -> None:
    """The badge under a reply is read off the turn the router actually recorded."""
    state = SessionState()
    state.model_turns.append(
        ModelTurn(
            session_id=state.session_id,
            model_tier="sonnet",
            model="test-sonnet-model",
            routing_reason="return or refund intent",
        )
    )

    turn = capture_turn(state, "Shall I start that return?", trace_offset=0)

    assert turn.model_tier == "sonnet"
    assert turn.model == "test-sonnet-model"
    assert turn.routing_reason == "return or refund intent"
    assert turn.reply == "Shall I start that return?"


def test_capture_takes_only_the_traces_from_this_turn() -> None:
    """The offset is what keeps an earlier turn's tools out of a later trace."""
    state = SessionState()
    state.tool_traces.append(trace(tool_name="verify_identity"))
    offset = len(state.tool_traces)
    state.tool_traces.append(trace(tool_name="lookup_order"))
    state.tool_traces.append(trace(tool_name="check_return_eligibility"))

    turn = capture_turn(state, "Here you go.", trace_offset=offset)

    assert turn.tool_names == ["lookup_order", "check_return_eligibility"]


def test_capture_keeps_tools_in_execution_order() -> None:
    """Two orders read in one turn are shown in the order they were read."""
    state = SessionState()
    state.tool_traces.append(trace(tool_args={"order_id": IN_WINDOW_ORDER}))
    state.tool_traces.append(trace(tool_args={"order_id": EXPIRED_ORDER}))

    turn = capture_turn(state, "Which of these did you mean?", trace_offset=0)

    assert [t.tool_args["order_id"] for t in turn.tool_traces] == [
        IN_WINDOW_ORDER,
        EXPIRED_ORDER,
    ]


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

    turn = capture_turn(state, "What's the email on your account?", trace_offset=0)

    assert turn.tool_traces == []
    assert turn.had_failure is False
    assert turn.model_tier == "haiku"


def test_a_failure_is_visible_on_the_turn() -> None:
    """A blocked write is not a success, and the turn says so."""
    state = SessionState()
    state.tool_traces.append(
        trace(
            tool_name="initiate_return",
            status=ToolStatus.BLOCKED,
            result_summary="return blocked",
            error="no eligibility token",
        )
    )

    assert capture_turn(state, "I can't open that.", trace_offset=0).had_failure is True


# --- The eligibility decision beside its check ---------------------------


def eligible_state() -> SessionState:
    """A session holding a decision for the hero order and item."""
    return SessionState(
        verified_customer_id=HERO_CUSTOMER,
        active_order_id=IN_WINDOW_ORDER,
        active_item_id=IN_WINDOW_ITEM,
        eligibility=EligibilityDecision(
            eligible=True,
            policy_id="STANDARD_30_DAY",
            explanation="Inside the window.",
            rule_path=STANDARD_PATH,
        ),
    )


def eligibility_trace(order_id: str = IN_WINDOW_ORDER, item_id: str = IN_WINDOW_ITEM) -> ToolTrace:
    return trace(
        tool_name="check_return_eligibility",
        tool_args={"order_id": order_id, "item_id": item_id},
        result_summary="eligible=True, policy_id=STANDARD_30_DAY",
    )


def test_the_decision_is_shown_against_the_check_that_made_it() -> None:
    """Policy and path reach the turn whose eligibility check produced them."""
    state = eligible_state()

    decision = eligibility_for(state, [eligibility_trace()])

    assert decision is not None
    assert decision.policy_id == "STANDARD_30_DAY"
    assert format_rule_path(decision.rule_path) == "PhysicalBook\n→ STANDARD_30_DAY"


def test_a_turn_without_a_check_gets_no_decision() -> None:
    """A held decision is not attached to a turn that did not check anything."""
    assert eligibility_for(eligible_state(), [trace(tool_name="lookup_order")]) is None


def test_a_decision_for_another_item_is_not_shown() -> None:
    """A path shown against the wrong item is worse than no path at all.

    The session holds one decision at a time. If the customer switched item, the
    check in an older turn is not what the session now holds, so that turn shows
    the call and not a decision that was never its own.
    """
    stale = eligibility_trace(order_id=EXPIRED_ORDER, item_id=EXPIRED_ITEM)

    assert eligibility_for(eligible_state(), [stale]) is None


def test_a_failed_check_carries_no_decision() -> None:
    """Nothing can be concluded from a check that did not run."""
    broken = eligibility_trace()
    broken.status = ToolStatus.ERROR

    assert eligibility_for(eligible_state(), [broken]) is None


def test_a_cleared_decision_is_not_borrowed() -> None:
    """After the write clears the return context there is nothing to display."""
    state = eligible_state()
    state.clear_return_context()

    assert eligibility_for(state, [eligibility_trace()]) is None


# --- Pairing the transcript with the traces ------------------------------


def turn_named(reply: str, tier: str = "haiku") -> AssistantTurn:
    return AssistantTurn(reply=reply, model_tier=tier, model=f"test-{tier}-model")


def test_each_assistant_message_is_paired_with_its_own_turn() -> None:
    """Matched by position, so two identical replies are still two turns."""
    state = SessionState()
    state.add_message(Role.USER, "Where's my book?")
    state.add_message(Role.ASSISTANT, "What's your email?")
    state.add_message(Role.USER, "ada@example.com")
    state.add_message(Role.ASSISTANT, "Which order did you mean?")

    paired = pair_turns(
        state.messages, [turn_named("What's your email?"), turn_named("Which order did you mean?")]
    )

    assert [message.content for message, _ in paired] == [
        "Where's my book?",
        "What's your email?",
        "ada@example.com",
        "Which order did you mean?",
    ]
    assert [turn.reply if turn else None for _, turn in paired] == [
        None,
        "What's your email?",
        None,
        "Which order did you mean?",
    ]


def test_a_user_message_never_gets_a_trace() -> None:
    """The trace goes under the reply, because the reply is what it explains."""
    state = SessionState()
    state.add_message(Role.USER, "Yes please")

    assert pair_turns(state.messages, [turn_named("Your return is open.")]) == [
        (state.messages[0], None)
    ]


def test_an_assistant_message_with_no_record_renders_without_one() -> None:
    """A reply the UI has no turn for shows no trace rather than the next one's.

    Happens to a session restored without its traces. Borrowing the following
    turn's trace would attribute tool calls to a reply that did not make them.
    """
    state = SessionState()
    state.add_message(Role.ASSISTANT, "Hello.")
    state.add_message(Role.ASSISTANT, "Still here.")

    paired = pair_turns(state.messages, [turn_named("Hello.")])

    assert paired[0][1] is not None
    assert paired[1][1] is None


# --- The hero flow, as the UI would show it ------------------------------


def capture_flow(agent, state: SessionState, *prompts: str) -> list[AssistantTurn]:
    """Drive a conversation the way `app.run_turn` does, keeping the turns."""
    turns = []
    for prompt in prompts:
        offset = len(state.tool_traces)
        reply = agent.respond(state, prompt)
        turns.append(capture_turn(state, reply, trace_offset=offset))
    return turns


def test_the_hero_flow_traces_line_up_with_the_replies(
    make_agent, seeded_graph, hero_script
) -> None:
    """Each turn of the demo carries exactly the tools that turn ran.

    This is the property the whole trace UI rests on: the reviewer can point at a
    reply and see what produced it. Nothing here is arranged for the demo — the
    tiers are the router's and the tools are the ones the loop dispatched.
    """
    agent, _ = make_agent(*hero_script)
    state = SessionState()

    turns = capture_flow(
        agent,
        state,
        "Where's my book?",
        "ada@example.com",
        "The Pragmatic Programmer one",
        "Actually, I want to return it.",
        "Yes please",
    )

    assert [turn.tool_names for turn in turns] == [
        [],
        ["verify_identity", "lookup_order", "lookup_order"],
        ["lookup_order"],
        ["check_return_eligibility"],
        ["initiate_return"],
    ]
    assert [turn.model_tier for turn in turns] == [
        "haiku",
        "haiku",
        "haiku",
        "sonnet",
        "sonnet",
    ]
    assert not any(turn.had_failure for turn in turns)


def test_the_eligibility_turn_shows_the_policy_and_the_path(
    make_agent, seeded_graph, hero_script
) -> None:
    """The turn that decided the return is the turn that explains the decision."""
    agent, _ = make_agent(*hero_script)
    state = SessionState()

    turns = capture_flow(
        agent,
        state,
        "Where's my book?",
        "ada@example.com",
        "The Pragmatic Programmer one",
        "Actually, I want to return it.",
    )

    eligibility_turn = turns[-1]
    assert eligibility_turn.eligibility is not None
    assert eligibility_turn.eligibility.policy_id == "STANDARD_30_DAY"
    assert eligibility_turn.eligibility.eligible is True
    assert format_rule_path(eligibility_turn.eligibility.rule_path) == (
        "PhysicalBook\n→ STANDARD_30_DAY"
    )
    # The earlier turns explain nothing about policy, and must not claim to.
    assert all(turn.eligibility is None for turn in turns[:-1])


def test_the_hero_flow_trace_shows_no_email_and_no_token(
    make_agent, seeded_graph, hero_script
) -> None:
    """What reaches the screen is what a trace is allowed to say.

    The orchestrator already sanitized these. The assertion is on the display
    path, because that is the one a reviewer looks at.
    """
    agent, _ = make_agent(*hero_script)
    state = SessionState()

    turns = capture_flow(
        agent,
        state,
        "Where's my book?",
        "ada@example.com",
        "The Pragmatic Programmer one",
        "Actually, I want to return it.",
        "Yes please",
    )

    lines = [
        format_args(trace.tool_args) for turn in turns for trace in turn.tool_traces
    ]
    assert any("a***@example.com" in line for line in lines)
    assert not any("ada@example.com" in line for line in lines)
    assert not any("elig-" in line for line in lines)


def test_the_outside_window_turn_shows_a_refusal_with_its_policy(
    make_agent, seeded_graph
) -> None:
    """The edge case needs no special display: it is the same trace, saying no.

    The check ran, the policy is named, the path is there, and the decision is
    "Not eligible" — which is what makes it clear the answer was evaluated rather
    than improvised.
    """
    agent, _ = make_agent(
        tool_call(
            "check_return_eligibility",
            {"order_id": EXPIRED_ORDER, "item_id": EXPIRED_ITEM, "reason": "Didn't want it."},
        ),
        text("That one was delivered too long ago to return, I'm afraid."),
    )
    state = SessionState(
        verified_customer_id=HERO_CUSTOMER,
        customer_region="GB",
        active_order_ids=[IN_WINDOW_ORDER, EXPIRED_ORDER],
    )

    turn = capture_flow(agent, state, "Can I send this back?")[0]

    assert turn.model_tier == "sonnet"
    assert turn.tool_names == ["check_return_eligibility"]
    assert turn.eligibility is not None
    assert decision_label(turn.eligibility.eligible) == "Not eligible"
    assert turn.eligibility.policy_id == "STANDARD_30_DAY"
    assert format_rule_path(turn.eligibility.rule_path) == "PhysicalBook\n→ STANDARD_30_DAY"


def test_a_blocked_write_is_visible_in_the_turn_that_attempted_it(
    make_agent, seeded_graph
) -> None:
    """The guard refusing is shown as Blocked, on the turn that tried it."""
    agent, _ = make_agent(
        tool_call(
            "initiate_return",
            {"order_id": EXPIRED_ORDER, "item_id": EXPIRED_ITEM, "reason": "Didn't want it."},
        ),
        text("I can't open that return — it's outside the window."),
    )
    state = SessionState(
        verified_customer_id=HERO_CUSTOMER,
        customer_region="GB",
        active_order_ids=[EXPIRED_ORDER],
        active_order_id=EXPIRED_ORDER,
    )

    turn = capture_flow(agent, state, "Yes, go ahead")[0]

    assert turn.had_failure is True
    assert status_label(turn.tool_traces[0].status) == "Blocked"
    assert turn.tool_traces[0].error


def test_an_anthropic_outage_is_traced_and_answered(make_agent, seeded_graph) -> None:
    """The customer gets a sentence; the trace says which call failed.

    Technical detail belongs in the trace. The reply the customer read does not
    carry an exception type, and neither transcript carries a stack.
    """
    agent, _ = make_agent(error=RuntimeError("connection reset"))
    state = SessionState()

    turn = capture_flow(agent, state, "Where's my book?")[0]

    assert turn.tool_names == ["anthropic.messages.create"]
    assert status_label(turn.tool_traces[0].status) == "Failed"
    assert "connection reset" in (turn.tool_traces[0].error or "")
    assert "RuntimeError" not in turn.reply
    assert "trouble" in turn.reply
