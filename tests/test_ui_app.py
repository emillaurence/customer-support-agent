"""The Streamlit script itself, driven by Streamlit's own test harness.

`AppTest` runs `app.py` in process, with no browser and no server, so the shell
can be exercised the way a reviewer will use it: type a message, look at what
appears under the reply, press Reset demo.

Anthropic is the same scripted stand-in the rest of the suite uses, installed
into `st.session_state` before the first run — `app.get_agent` builds an agent
only when the session does not already hold one, so a test can supply one without
the app knowing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from agent.config import AnthropicConfigError
from agent.demo import fresh_session
from tests.conftest import text, tool_call
from tests.test_hero_flow import (
    EXPIRED_ITEM,
    EXPIRED_ORDER,
    HERO_CUSTOMER,
    HERO_EMAIL,
    IN_WINDOW_ORDER,
    hero_script,  # noqa: F401 - imported so pytest registers the fixture here too
    returns_in,
)
from ui.render import DEVELOPER_LABEL, TRACE_LABEL

APP = str(Path(__file__).resolve().parent.parent / "app.py")
"""Absolute, because `AppTest.from_file` resolves a relative path against *this* file."""

TIMEOUT = 30
"""Generous: the whole hero flow runs in one `run()` on some paths."""


@pytest.fixture
def app(make_agent, seeded_graph):
    """An `AppTest` for `app.py`, wired to a scripted Anthropic client.

    Args:
        make_agent: The suite's agent factory — scripted client, fixed clock.
        seeded_graph: Policy answers from the seed file rather than Neo4j.

    Returns:
        A callable taking the scripted responses and returning a started AppTest.
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
    return [block for block in at.expander if block.label == TRACE_LABEL]


def body(block: Any) -> str:
    """Everything written inside one block, as one searchable string."""
    return "\n".join(element.value for element in block.markdown) + "\n".join(
        element.value for element in block.caption
    )


def page_text(at: AppTest) -> str:
    """Every markdown and caption on the page, as one searchable string."""
    return "\n".join(
        element.value for element in list(at.main.markdown) + list(at.main.caption)
    )


# --- Opening the app -----------------------------------------------------


def test_the_app_opens_on_a_welcome_and_a_chat_box(app) -> None:
    """A short greeting, an input, and nothing else preloaded.

    Specifically not the hero conversation: the demo is typed live.
    """
    at = app(text("hello"))

    assert at.chat_input
    assert "Bookly Support" in page_text(at)
    assert "orders, returns, refunds" in page_text(at)
    assert traces(at) == []
    assert not at.error


def test_the_developer_state_is_available_and_separate(app) -> None:
    """The debug view is in the sidebar, collapsed, away from the conversation."""
    at = app(text("hello"))

    developer = [block for block in at.sidebar.expander if block.label == DEVELOPER_LABEL]
    assert len(developer) == 1
    assert "Verified customer" in body(developer[0])


def test_a_missing_anthropic_configuration_is_an_explanation_not_a_crash(
    monkeypatch, seeded_graph
) -> None:
    """An unconfigured deployment stops with a message and no chat box.

    The agent is constructed at startup for exactly this reason, so the failure
    lands here rather than on the customer's first message.
    """
    from agent import orchestrator

    def unconfigured(*_args: Any, **_kwargs: Any):
        raise AnthropicConfigError("ANTHROPIC_API_KEY is not set")

    monkeypatch.setattr(orchestrator, "BooklyAgent", unconfigured)

    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()

    assert "ANTHROPIC_API_KEY is not set" in at.error[0].value
    assert not at.chat_input


# --- One turn ------------------------------------------------------------


def test_a_reply_carries_its_own_trace(app) -> None:
    """The tool that ran appears under the reply that used it, with a latency."""
    at = app(
        tool_call("lookup_order", {"order_id": IN_WINDOW_ORDER}),
        text("It was delivered on 28 July."),
    )
    at.session_state["bookly_state"] = fresh_session().model_copy(
        update={
            "verified_customer_id": HERO_CUSTOMER,
            "customer_region": "GB",
            "active_order_ids": [IN_WINDOW_ORDER],
            "active_order_id": IN_WINDOW_ORDER,
        }
    )
    at.run()

    say(at, "Where's my book?")

    assert "It was delivered on 28 July." in page_text(at)
    assert len(traces(at)) == 1
    trace = body(traces(at)[0])
    assert "lookup_order" in trace
    assert "ms" in trace or " s" in trace
    assert "Success" in trace


def test_the_model_badge_says_which_tier_handled_the_turn(app) -> None:
    """Routing is visible on the page without opening anything."""
    at = app(text("What's the email address on your Bookly account?"))

    say(at, "Where's my book?")

    assert "Model: Haiku" in page_text(at)
    assert "simple read-only request" in page_text(at)


def test_a_turn_with_no_tools_has_no_trace_to_open(app) -> None:
    """An expander promising a trace of nothing is a click for no reason."""
    at = app(text("What's the email address on your Bookly account?"))

    say(at, "Where's my book?")

    assert traces(at) == []
    assert "Model: Haiku" in page_text(at)


def test_a_shell_failure_reaches_the_customer_as_a_sentence(app, monkeypatch) -> None:
    """No stack trace, ever — the technical detail is not the customer's problem."""
    at = app(text("hello"))

    class Broken:
        def respond(self, *_args: Any, **_kwargs: Any) -> str:
            raise RuntimeError("something unforeseen")

    at.session_state["bookly_agent"] = Broken()
    at.run()

    say(at, "Where's my book?")

    assert "trouble accessing support" in at.error[0].value
    assert not at.exception
    assert "RuntimeError" not in page_text(at)


# --- The hero flow, in the UI -------------------------------------------


def run_hero_flow_in_ui(at: AppTest) -> AppTest:
    """The five customer turns of the demo, typed into the chat box."""
    for message in (
        "Where's my book?",
        HERO_EMAIL,
        "The Pragmatic Programmer one",
        "Actually, I want to return it.",
        "Yes please",
    ):
        say(at, message)
    return at


def test_the_hero_flow_renders_a_trace_per_acting_turn(app, hero_script, data_dir) -> None:
    """Five replies, four of which did something, and one real RMA at the end."""
    at = run_hero_flow_in_ui(app(*hero_script))

    assert len(traces(at)) == 4
    assert [t.tool_name for t in at.session_state["bookly_state"].tool_traces] == [
        "verify_identity",
        "lookup_order",
        "lookup_order",
        "lookup_order",
        "check_return_eligibility",
        "initiate_return",
    ]
    assert len([r for r in returns_in(data_dir) if r["order_id"] == IN_WINDOW_ORDER]) == 1


def test_the_hero_flow_shows_both_tiers_where_the_router_chose_them(
    app, hero_script
) -> None:
    """Haiku on the lookups, Sonnet from the return onwards. Nothing forced."""
    at = run_hero_flow_in_ui(app(*hero_script))

    page = page_text(at)
    assert "Model: Haiku" in page
    assert "Model: Sonnet" in page
    assert [turn.model_tier for turn in at.session_state["bookly_turns"]] == [
        "haiku",
        "haiku",
        "haiku",
        "sonnet",
        "sonnet",
    ]


def test_the_eligibility_trace_shows_the_policy_and_the_graph_path(
    app, hero_script
) -> None:
    """The decision is explained by the traversal that produced it."""
    at = run_hero_flow_in_ui(app(*hero_script))

    eligibility = next(
        block for block in traces(at) if "check_return_eligibility" in body(block)
    )
    shown = body(eligibility)
    assert "STANDARD_30_DAY" in shown
    assert "Eligible" in shown
    assert "PhysicalBook" in shown
    assert "→ STANDARD_30_DAY" in shown


def test_the_traces_stay_attached_to_the_right_replies(app, hero_script) -> None:
    """Turn three looked up one order; turn two looked up two. In that order."""
    at = run_hero_flow_in_ui(app(*hero_script))

    shown = [body(block) for block in traces(at)]
    assert "verify_identity" in shown[0]
    assert shown[0].count("lookup_order") == 2
    assert shown[1].count("lookup_order") == 1
    assert "check_return_eligibility" in shown[2]
    assert "initiate_return" in shown[3]


def test_no_email_or_token_reaches_the_trace(app, hero_script) -> None:
    """What the trace displays is what a trace is allowed to say.

    The customer's own message still says their address, because they typed it —
    masking a customer's view of their own conversation would be theatre. What
    matters is that the *trace* records the attempt without the value.
    """
    at = run_hero_flow_in_ui(app(*hero_script))

    shown = "\n".join(body(block) for block in traces(at))
    assert HERO_EMAIL not in shown
    assert "a***@example.com" in shown
    assert "elig-" not in shown + page_text(at)


def test_the_developer_state_reports_a_token_without_showing_it(app, hero_script) -> None:
    """Presence is the useful part; the value is a credential."""
    at = app(*hero_script)
    for message in ("Where's my book?", HERO_EMAIL, "The Pragmatic Programmer one", "Actually, I want to return it."):
        say(at, message)

    developer = next(
        block for block in at.sidebar.expander if block.label == DEVELOPER_LABEL
    )
    shown = body(developer)
    token = at.session_state["bookly_state"].eligibility_token
    assert token
    assert "held" in shown
    assert token not in shown


# --- The outside-window case --------------------------------------------


def test_the_outside_window_case_needs_no_special_display(app) -> None:
    """Same trace, saying no: policy named, path shown, nothing written."""
    at = app(
        tool_call(
            "check_return_eligibility",
            {"order_id": EXPIRED_ORDER, "item_id": EXPIRED_ITEM, "reason": "Didn't want it."},
        ),
        text("That one was delivered too long ago to return, I'm afraid."),
    )
    at.session_state["bookly_state"] = fresh_session().model_copy(
        update={
            "verified_customer_id": HERO_CUSTOMER,
            "customer_region": "GB",
            "active_order_ids": [IN_WINDOW_ORDER, EXPIRED_ORDER],
        }
    )
    at.run()

    say(at, "Can I send this back?")

    shown = body(traces(at)[0])
    assert "Model: Sonnet" in page_text(at)
    assert "Not eligible" in shown
    assert "STANDARD_30_DAY" in shown
    assert "→ STANDARD_30_DAY" in shown
    assert at.session_state["bookly_state"].eligibility_token is None


# --- Reset ---------------------------------------------------------------


def press(at: AppTest, label: str) -> AppTest:
    """Press a sidebar button by its label."""
    next(button for button in at.sidebar.button if button.label == label).click().run()
    return at


def test_reset_demo_clears_the_conversation_the_traces_and_the_data(
    app, hero_script, data_dir
) -> None:
    """One button, and the demo is rehearsable again.

    The reset itself is `agent.demo.reset_demo` — the same function
    `scripts/reset_demo.py` calls. What this checks is that the UI's own state
    goes with it: the transcript, the captured traces, and the session.
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

    The RMA the conversation created is a real record, and forgetting the
    conversation is not a reason to delete it.
    """
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


def test_the_ui_reset_holds_no_reset_logic_of_its_own() -> None:
    """`app.py` restores nothing itself — it calls the shared implementation.

    Two copies of "put it back" is how a rehearsal and a live run end up starting
    from different states, which is the same reason `scripts/reset_demo.py` has
    none either.
    """
    import ast

    source = Path(APP).read_text()
    assert "from agent.demo import fresh_session, reset_demo" in source

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
    assert not imported & {"json", "shutil", "os", "pathlib"}
