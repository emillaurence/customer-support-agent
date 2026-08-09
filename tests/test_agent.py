"""The agent: configuration, model routing, the tool loop, and multi-turn state.

Every test here scripts Anthropic's side and asserts on the agent's. The model
never runs, so what is under test is the loop's own behaviour: that it routes
deterministically, dispatches the tools Claude asks for, feeds the results back
in the right shape, updates trusted state only from results that succeeded, and
never turns a failure into a success.

That last one is the point of most of this file. A support agent that says "your
return is open" when nothing was written is worse than one that crashes.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pytest

from agent.agent import (
    COMPLEX_TURN_COUNT,
    FALLBACK_STUCK,
    FALLBACK_UNAVAILABLE,
    MAX_TOKENS,
    MAX_TOOL_ITERATIONS,
    REQUIRED_ENV,
    TEMPERATURE_ENV,
    AnthropicConfigError,
    BooklyAgent,
    ModelDecision,
    ModelTier,
    load_anthropic_config,
    select_model,
)
from agent.state import (
    EligibilityDecision,
    ModelTurn,
    PendingReturn,
    Role,
    SessionState,
    ToolStatus,
    ToolTrace,
)
from agent.tools import TOOL_SCHEMAS
from tests.conftest import (
    CONFIRM_QUESTION,
    EXPIRED_ITEM,
    EXPIRED_ORDER,
    HERO_CUSTOMER,
    HERO_EMAIL,
    IN_WINDOW_ITEM,
    IN_WINDOW_ORDER,
    FakeAnthropic,
    FakeBlock,
    FakeResponse,
    break_policy_graph,
    returns_in,
    run_hero_flow,
    text,
    tiers,
    tool_call,
    tool_calls,
    tool_names,
)

ROOT = Path(__file__).resolve().parent.parent


# =========================================================================
# Configuration
#
# The behaviour worth pinning down is that the *application* has no opinion
# about which Claude models it runs on. The router picks a tier; the environment
# decides what that tier is.
# =========================================================================


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch):
    """Start from no Anthropic configuration at all.

    `load_dotenv` would otherwise read a developer's real `.env` and the test
    would pass or fail depending on whose machine it ran on.
    """
    monkeypatch.setattr("agent.agent.load_dotenv", lambda *a, **k: None)
    for name in (*REQUIRED_ENV, TEMPERATURE_ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def configured(monkeypatch):
    """The three required variables, leaving the temperature to the test."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("ANTHROPIC_MODEL_HAIKU", "fast")
    monkeypatch.setenv("ANTHROPIC_MODEL_SONNET", "capable")


def test_complete_configuration_loads(clean_env, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("ANTHROPIC_MODEL_HAIKU", "some-fast-model")
    monkeypatch.setenv("ANTHROPIC_MODEL_SONNET", "some-capable-model")

    config = load_anthropic_config()

    assert config.haiku_model == "some-fast-model"
    assert config.sonnet_model == "some-capable-model"


@pytest.mark.parametrize("missing", REQUIRED_ENV)
def test_any_missing_variable_fails_clearly(clean_env, monkeypatch, missing: str) -> None:
    """Both model names included: an agent configured with only the cheap model
    cannot route, and would silently do the consequential turns on the wrong one."""
    for name in REQUIRED_ENV:
        monkeypatch.setenv(name, "value")
    monkeypatch.delenv(missing)

    with pytest.raises(AnthropicConfigError, match=missing):
        load_anthropic_config()


def test_blank_is_treated_as_missing(clean_env, monkeypatch) -> None:
    """`ANTHROPIC_MODEL_SONNET=` in a half-filled `.env` is not configuration."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("ANTHROPIC_MODEL_HAIKU", "fast")
    monkeypatch.setenv("ANTHROPIC_MODEL_SONNET", "   ")

    with pytest.raises(AnthropicConfigError, match="ANTHROPIC_MODEL_SONNET"):
        load_anthropic_config()


def test_error_lists_every_missing_variable_at_once(clean_env) -> None:
    """One error, one fix — not three restarts."""
    with pytest.raises(AnthropicConfigError) as caught:
        load_anthropic_config()

    for name in REQUIRED_ENV:
        assert name in str(caught.value)


def test_agent_construction_fails_when_unconfigured(clean_env) -> None:
    """The failure happens at startup, not on the customer's first message."""
    with pytest.raises(AnthropicConfigError):
        BooklyAgent()


def test_api_key_is_not_in_the_repr(anthropic_config) -> None:
    """A config object can end up in a traceback or a log line."""
    assert anthropic_config.api_key not in repr(anthropic_config)
    assert anthropic_config.api_key not in str(anthropic_config)


def test_temperature_is_unset_by_default(clean_env, configured) -> None:
    """No variable, no temperature. The model manages its own sampling."""
    assert load_anthropic_config().temperature is None


def test_temperature_is_read_from_the_environment(clean_env, configured, monkeypatch) -> None:
    monkeypatch.setenv(TEMPERATURE_ENV, "0.7")
    assert load_anthropic_config().temperature == 0.7


def test_zero_is_a_setting_not_an_absence(clean_env, configured, monkeypatch) -> None:
    """`0` must survive as 0 — the falsiest value the setting can hold."""
    monkeypatch.setenv(TEMPERATURE_ENV, "0")
    assert load_anthropic_config().temperature == 0.0


def test_blank_temperature_is_treated_as_unset(clean_env, configured, monkeypatch) -> None:
    monkeypatch.setenv(TEMPERATURE_ENV, "   ")
    assert load_anthropic_config().temperature is None


def test_unparseable_temperature_fails_loudly(clean_env, configured, monkeypatch) -> None:
    """A typo is a startup error, not a silent fallback that looks deliberate."""
    monkeypatch.setenv(TEMPERATURE_ENV, "warm")

    with pytest.raises(AnthropicConfigError, match=TEMPERATURE_ENV):
        load_anthropic_config()


def test_no_temperature_is_sent_when_none_is_configured(make_agent) -> None:
    """The parameter is absent from the request, not sent as a default.

    A model that manages its own sampling rejects an explicit temperature
    outright, so sending 0 would fail every turn rather than quietly doing nothing.
    """
    agent, client = make_agent(text("Ebooks aren't returnable."), text("Let me check that."))
    state = SessionState()

    agent.respond(state, "what's your policy on ebooks?")
    agent.respond(state, "I'd like a refund")

    assert client.models_used == ["test-haiku-model", "test-sonnet-model"]
    assert all("temperature" not in call for call in client.calls)


def test_both_tiers_are_sent_the_same_temperature(anthropic_config, make_agent) -> None:
    """One value, both models, every call. There is one exit from the loop to
    Anthropic, so there is nowhere for a turn to be sampled differently."""
    agent, client = make_agent(text("Ebooks aren't returnable."), text("Let me check that."))
    agent.config = anthropic_config.model_copy(update={"temperature": 0.4})
    state = SessionState()

    agent.respond(state, "what's your policy on ebooks?")
    agent.respond(state, "I'd like a refund")

    assert client.models_used == ["test-haiku-model", "test-sonnet-model"]
    assert [call["temperature"] for call in client.calls] == [0.4, 0.4]


def test_a_configured_zero_is_actually_sent(anthropic_config, make_agent) -> None:
    """0 is a value, not an absence — it must not be dropped as falsy."""
    agent, client = make_agent(text("hello"))
    agent.config = anthropic_config.model_copy(update={"temperature": 0.0})

    agent.respond(SessionState(), "hi")

    assert client.calls[0]["temperature"] == 0.0


def test_temperature_is_not_hardcoded_in_the_loop() -> None:
    """The loop reads it from the config, and nowhere else."""
    assert "self.config.temperature" in (ROOT / "agent" / "agent.py").read_text()


@pytest.mark.parametrize(
    ("tier", "expected"),
    [(ModelTier.HAIKU, "test-haiku-model"), (ModelTier.SONNET, "test-sonnet-model")],
)
def test_both_tiers_resolve_to_their_configured_model(anthropic_config, tier, expected) -> None:
    agent = BooklyAgent(config=anthropic_config, client=object())
    assert agent._model_id(ModelDecision(tier=tier, reason="test")) == expected


def test_no_model_id_is_hardcoded_in_application_code() -> None:
    """Which Claude models Bookly runs on is a deployment decision.

    The only places a model id may appear are `.env` and the deployment that fills
    it in — not a constant in the source, and not a default in the router.
    """
    looks_like_a_model_id = re.compile(r"claude[-_][a-z0-9][a-z0-9.\-]*", re.IGNORECASE)

    offenders = []
    shipped = (*(ROOT / "agent").glob("*.py"), *(ROOT / "policy").glob("*.py"),
               ROOT / "app.py", ROOT / "ui.py")
    for path in sorted(shipped):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if looks_like_a_model_id.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")

    assert not offenders, "model ids belong in .env, not in code:\n" + "\n".join(offenders)


def test_env_example_documents_every_variable() -> None:
    """`.env.example` is the contract — it has to name every variable."""
    example = (ROOT / ".env.example").read_text()
    for name in (*REQUIRED_ENV, TEMPERATURE_ENV, "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"):
        assert f"{name}=" in example


def test_env_example_ships_no_values() -> None:
    """Every key is blank. A committed `.env.example` must never carry a secret,
    and leaving the temperature unset is the working configuration."""
    for line in (ROOT / ".env.example").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            assert line.split("=", 1)[1] == "", f"{line} has a value committed"


# =========================================================================
# Model routing
#
# The obvious property is that simple turns go to Haiku and consequential ones
# to Sonnet. The one that matters more is that the decision is *deterministic* —
# a router that sometimes picks Sonnet is a router nobody can demo or debug.
# =========================================================================


def route(message: str, state: SessionState | None = None) -> ModelTier:
    return select_model(state or SessionState(), message).tier


@pytest.mark.parametrize(
    "message",
    [
        "What's Bookly's policy on ebooks?",
        "How long do I have to send something back",  # 'send something back' is not a keyword
        "Do you ship to Ireland?",
        "Where is my order ORD-1001?",
        "Has my order been delivered yet?",
        "hi",
    ],
)
def test_simple_requests_use_haiku(message: str) -> None:
    """A policy question or an order-status check is retrieval, not reasoning."""
    assert route(message) is ModelTier.HAIKU


def test_order_status_on_a_verified_session_stays_on_haiku(verified_state) -> None:
    """Being verified is not complexity. Reading out a status is still a lookup."""
    assert route("Where's my order?", verified_state) is ModelTier.HAIKU


@pytest.mark.parametrize(
    "message",
    [
        "What is the return policy?",
        "What is the return policy for Australian customers?",
        "Can ebooks be returned?",
        "How long is the holiday return window?",
        "How long do I have to return a physical book?",
        "What are your refund rules?",
        "Do you allow returns on audiobooks?",
    ],
)
def test_informational_policy_questions_use_haiku(message: str) -> None:
    """"Return" in a question about the rules is a topic, not an intent.

    Every one of these is a read of the policy graph with no customer in it.
    Matching the keyword alone would send them all to Sonnet.
    """
    assert route(message) is ModelTier.HAIKU


def test_the_australian_policy_question_is_a_lookup_not_a_workflow() -> None:
    """The case the routing fix exists for, asserted on the reason as well."""
    decision = select_model(SessionState(), "What is the return policy for Australian customers?")

    assert decision.tier is ModelTier.HAIKU
    assert decision.reason == "informational policy lookup"
    assert decision.return_intent is False  # nothing to remember for the next turn


def test_a_policy_question_on_a_verified_session_is_still_a_lookup(verified_state) -> None:
    """Knowing who is asking does not turn the question into their transaction."""
    assert route("What's the return policy for ebooks?", verified_state) is ModelTier.HAIKU


@pytest.mark.parametrize(
    "message",
    [
        "I want to return this book",
        "Can I get a refund?",
        "The book arrived damaged",
        "I'd like my money back please",
        "This is faulty, can I exchange it",
    ],
)
def test_return_intent_uses_sonnet(message: str) -> None:
    """Anything that could become a return is reasoned about, not looked up."""
    assert route(message) is ModelTier.SONNET


@pytest.mark.parametrize(
    "message",
    [
        "Can I return my order?",
        "Am I eligible for a return?",
        "I want to return ORD-1003",
        "Start the return",
        "Can I get a refund for ORD-1003?",
        "Why was my return rejected?",
        "Can you make an exception?",
        "Am I eligible to return my book?",
    ],
)
def test_customer_specific_return_questions_use_sonnet(message: str) -> None:
    """The same topic, asked about the customer's own record, is the workflow.

    First person, a named order, or an instruction to act — each is enough on its
    own, and none may be demoted by informational phrasing.
    """
    assert route(message) is ModelTier.SONNET


def test_the_eligibility_question_reads_as_a_workflow_in_the_trace() -> None:
    """The reason has to name the workflow, so the trace distinguishes the two."""
    decision = select_model(SessionState(), "Am I eligible to return my book?")

    assert decision.tier is ModelTier.SONNET
    assert decision.reason == "return or refund workflow"
    assert decision.return_intent is True


def test_informational_phrasing_does_not_rescue_a_specific_order() -> None:
    """The order id is a veto: uncertainty resolves towards the stronger model."""
    assert route("What is the return policy for my order ORD-1003?") is ModelTier.SONNET


@pytest.mark.parametrize(
    "message",
    [
        "I want to speak to a human",
        "Let me talk to your manager",
        "I'm going to raise a chargeback",
        "What are my rights under consumer law?",
    ],
)
def test_escalation_intent_uses_sonnet(message: str) -> None:
    assert route(message) is ModelTier.SONNET


@pytest.mark.parametrize(
    "message",
    ["Which one was the paperback?", "I'm not sure which order it was", "actually, the other one"],
)
def test_ambiguity_uses_sonnet(message: str) -> None:
    """A reference the agent has to resolve is exactly where guessing costs."""
    assert route(message) is ModelTier.SONNET


def test_active_return_workflow_keeps_sonnet(verified_state) -> None:
    """A bare "ok" with no keyword in it. Without the state check this would drop
    to Haiku halfway through a return."""
    verified_state.active_item_id = IN_WINDOW_ITEM
    assert route("ok", verified_state) is ModelTier.SONNET


def test_eligibility_decision_keeps_sonnet(verified_state) -> None:
    """A decision on file means the workflow is live, eligible or not."""
    verified_state.eligibility = EligibilityDecision(eligible=False, explanation="outside window")
    assert route("I see", verified_state) is ModelTier.SONNET


def test_pending_confirmation_uses_sonnet(verified_state) -> None:
    """The turn that might authorise a write is the last one to run cheaply."""
    verified_state.pending_return = PendingReturn(
        order_id=IN_WINDOW_ORDER, item_id=IN_WINDOW_ITEM, eligibility_token="tok", asked=True
    )
    assert route("yes please", verified_state) is ModelTier.SONNET
    assert route("Yes, go ahead", verified_state) is ModelTier.SONNET


def test_a_generic_follow_up_mid_workflow_stays_on_sonnet(verified_state) -> None:
    """"What happens next?" has informational phrasing and nothing
    customer-specific in it, so on its own text it would demote. The state check
    runs first, so it cannot."""
    verified_state.return_intent_expressed = True
    decision = select_model(verified_state, "What happens next?")

    assert decision.tier is ModelTier.SONNET
    assert decision.reason == "a return workflow is open"


def test_a_policy_question_mid_workflow_stays_on_sonnet(verified_state) -> None:
    """Mid-return, even the policy question belongs to the return."""
    verified_state.active_item_id = IN_WINDOW_ITEM
    assert route("What is the return policy for ebooks?", verified_state) is ModelTier.SONNET


def test_confirmed_return_uses_sonnet(verified_state) -> None:
    verified_state.confirmed = True
    verified_state.eligibility_token = "tok"
    assert route("thanks", verified_state) is ModelTier.SONNET


def test_escalated_conversation_stays_on_sonnet(verified_state) -> None:
    """After a handoff the agent must stop acting — not the moment to economise."""
    verified_state.escalated = True
    assert route("ok thanks", verified_state) is ModelTier.SONNET


def test_long_conversation_promotes_to_sonnet() -> None:
    state = SessionState()
    for _ in range(COMPLEX_TURN_COUNT):
        state.add_message(Role.USER, "and?")
        state.add_message(Role.ASSISTANT, "...")
    assert route("what about that", state) is ModelTier.SONNET


def test_routing_is_deterministic(verified_state) -> None:
    """The property the demo depends on: nothing samples, and nothing depends on
    a clock or a model."""
    for message in ("what's your return policy", "I want a refund", "ok"):
        decisions = {select_model(verified_state, message).tier for _ in range(20)}
        assert len(decisions) == 1


def test_every_decision_carries_a_reason() -> None:
    """A tier with no stated reason cannot be shown in a trace."""
    for message in ("hello", "refund please", "I want a human"):
        assert select_model(SessionState(), message).reason


def test_promotion_is_one_way_within_a_workflow(verified_state) -> None:
    """Walks the workflow with messages that would each route to Haiku alone."""
    for item_id, token, message in [(IN_WINDOW_ITEM, None, "ok"), (IN_WINDOW_ITEM, "tok", "right")]:
        verified_state.active_item_id = item_id
        verified_state.eligibility_token = token
        assert route(message, verified_state) is ModelTier.SONNET


@pytest.mark.parametrize(
    ("message", "expected_model"),
    [("what's your policy on ebooks?", "test-haiku-model"), ("I want a refund", "test-sonnet-model")],
)
def test_routed_model_is_the_one_actually_called(make_agent, message, expected_model) -> None:
    """The router's decision reaches the API, from the configured names."""
    agent, client = make_agent(text("..."))
    agent.respond(SessionState(), message)
    assert client.models_used == [expected_model]


# =========================================================================
# Prompt caching
#
# A support conversation re-sends its whole prefix on every turn — six tool
# schemas, the system prompt, and everything said so far — so the prefix is the
# bulk of what is paid for, and it is identical each time. These check that the
# caching is asked for and that nothing in the prefix wobbles, since caching is a
# prefix match and one changed byte gives it all back.
# =========================================================================


def test_cache_control_is_set_on_every_request(make_agent, seeded_graph) -> None:
    """Including the follow-up calls inside one turn, which carry the longest
    prefixes and are exactly where the cache pays."""
    agent, client = make_agent(
        tool_call("verify_identity", {"email": HERO_EMAIL}),
        text("Thanks Ada."),
    )
    agent.respond(SessionState(), HERO_EMAIL)

    assert len(client.calls) == 2
    assert all(call["cache_control"] == {"type": "ephemeral"} for call in client.calls)


def test_caching_does_not_disturb_the_rest_of_the_request(make_agent) -> None:
    """Everything the loop sent before is still sent, unchanged."""
    agent, client = make_agent(text("hello"))
    agent.respond(SessionState(), "hi")

    request = client.calls[0]
    assert request["model"] == "test-haiku-model"
    assert request["max_tokens"] == MAX_TOKENS
    assert request["system"] == agent.system_prompt
    assert len(request["tools"]) == 6
    assert request["messages"][-1] == {"role": "user", "content": "hi"}
    assert "temperature" not in request  # unset by default, so never sent


def test_the_cached_prefix_is_byte_identical_across_turns(make_agent) -> None:
    """Tools and system prompt render ahead of the conversation, so a difference
    in either — a rebuilt schema, an interpolated date — would invalidate the
    cache for the whole transcript behind it."""
    agent, client = make_agent(text("one"), text("two"), text("three"))
    state = SessionState()

    for message in ("hi", "still there?", "thanks"):
        agent.respond(state, message)

    prefixes = {
        (json.dumps(call["tools"], sort_keys=False), call["system"]) for call in client.calls
    }
    assert len(prefixes) == 1


def test_tool_schemas_are_a_stable_shared_object(make_agent) -> None:
    """Not rebuilt per request, and not reordered: the same list, in the same
    order, is handed to every call."""
    agent, client = make_agent(text("one"), text("two"))
    state = SessionState()
    agent.respond(state, "hi")
    agent.respond(state, "hello again")

    assert [call["tools"] for call in client.calls] == [TOOL_SCHEMAS, TOOL_SCHEMAS]
    assert [schema["name"] for schema in TOOL_SCHEMAS] == [
        "verify_identity",
        "lookup_order",
        "search_policy",
        "check_return_eligibility",
        "initiate_return",
        "escalate_to_human",
    ]


def test_all_six_tools_stay_registered_and_callable() -> None:
    """The schema list, the dispatch table, and the required-argument map are
    three views of one tool set. A tool the model can ask for and nothing can run
    is a turn that fails at dispatch."""
    from agent.tools import _HANDLERS, REQUIRED_ARGS, TOOL_NAMES

    assert len(TOOL_SCHEMAS) == 6
    assert TOOL_NAMES == set(_HANDLERS) == set(REQUIRED_ARGS)
    for schema in TOOL_SCHEMAS:
        assert schema["input_schema"]["type"] == "object"
        assert schema["description"].strip()
        assert callable(_HANDLERS[schema["name"]])


def test_cache_usage_is_recorded_on_the_turn(make_agent) -> None:
    """Reported by the response, summed over the turn's round trips, and kept in
    the internal trace — not shown to the customer."""
    from tests.conftest import FakeUsage

    agent, _ = make_agent(
        FakeResponse(
            content=[FakeBlock(type="tool_use", id="toolu_1", name="verify_identity",
                               input={"email": HERO_EMAIL})],
            usage=FakeUsage(cache_creation_input_tokens=1500, cache_read_input_tokens=0),
        ),
        FakeResponse(
            content=[FakeBlock(type="text", text="Thanks Ada.")],
            usage=FakeUsage(cache_creation_input_tokens=0, cache_read_input_tokens=1500),
        ),
    )
    state = SessionState()

    agent.respond(state, HERO_EMAIL)

    turn = state.model_turns[-1]
    assert turn.cache_creation_input_tokens == 1500
    assert turn.cache_read_input_tokens == 1500


def test_a_response_without_usage_is_not_an_error(make_agent) -> None:
    """Missing counters are zero. A turn must not fail over its own accounting."""
    agent, _ = make_agent(text("hello"))
    state = SessionState()

    agent.respond(state, "hi")

    assert state.model_turns[-1].cache_creation_input_tokens == 0
    assert state.model_turns[-1].cache_read_input_tokens == 0


def test_timeout_and_retries_are_bounded_on_the_real_client(anthropic_config) -> None:
    """The SDK's own configuration, rather than a retry loop in the agent: one
    call cannot hang, and a flaky connection cannot be retried forever."""
    from agent.agent import MAX_RETRIES, REQUEST_TIMEOUT_SECONDS

    agent = BooklyAgent(config=anthropic_config)

    assert agent.client.timeout == REQUEST_TIMEOUT_SECONDS
    assert agent.client.max_retries == MAX_RETRIES
    assert 0 < REQUEST_TIMEOUT_SECONDS <= 60
    assert 0 <= MAX_RETRIES <= 3


# =========================================================================
# Dispatch
# =========================================================================


def test_model_requests_a_tool_and_it_executes(make_agent) -> None:
    """Claude asks for verify_identity; the real Python function runs."""
    agent, client = make_agent(
        tool_call("verify_identity", {"email": HERO_EMAIL}),
        text("Thanks Ada — I've found your account."),
    )
    state = SessionState()

    reply = agent.respond(state, "Hi, it's ada@example.com")

    assert reply == "Thanks Ada — I've found your account."
    assert state.verified_customer_id == HERO_CUSTOMER
    assert len(client.calls) == 2


def test_tool_result_is_returned_to_anthropic(make_agent) -> None:
    """The `tool_use_id` has to match or the API rejects the turn, and the content
    has to be the tool's own output rather than a summary of it."""
    agent, client = make_agent(
        tool_call("verify_identity", {"email": HERO_EMAIL}, block_id="toolu_abc"),
        text("Found you."),
    )
    agent.respond(SessionState(), HERO_EMAIL)

    second_request = client.calls[1]["messages"]
    result_block = second_request[-1]["content"][0]

    assert second_request[-1]["role"] == "user"
    assert result_block["type"] == "tool_result"
    assert result_block["tool_use_id"] == "toolu_abc"
    assert result_block["is_error"] is False

    payload = json.loads(result_block["content"])
    assert payload["verified"] is True
    assert payload["customer_id"] == HERO_CUSTOMER


def test_multiple_tool_calls_in_one_turn(make_agent, seeded_graph, verified_state) -> None:
    """The API expects every tool_result for a turn in a single user message.
    Splitting them trains the model out of asking for more than one at a time."""
    agent, client = make_agent(
        tool_calls(
            ("lookup_order", {"order_id": IN_WINDOW_ORDER}),
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
        tool_call("verify_identity", {"email": HERO_EMAIL}),
        tool_call("lookup_order", {"order_id": IN_WINDOW_ORDER}, block_id="toolu_2"),
        tool_call(
            "check_return_eligibility",
            {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM},
            block_id="toolu_3",
        ),
        text("That one can be returned. Shall I start a return for it?"),
    )
    state = SessionState()

    agent.respond(state, "I'd like to send back the book on my last order, ada@example.com")

    assert len(client.calls) == 4
    assert state.eligibility is not None and state.eligibility.eligible
    assert state.pending_return is not None
    assert tool_names(state) == ["verify_identity", "lookup_order", "check_return_eligibility"]


def test_system_prompt_and_tools_are_sent_every_call(make_agent) -> None:
    """Both are per-request on a stateless API, and all six tools are offered."""
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


# =========================================================================
# Trusted state
# =========================================================================


def test_state_updates_only_after_a_successful_tool_result(make_agent) -> None:
    """A failed verification leaves the session unverified.

    The difference between the agent believing a tool and believing the model.
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
    """`customer_id` is not a field on any schema, so the only path to
    `verified_customer_id` is a verification that actually passed."""
    agent, _ = make_agent(
        tool_call("lookup_order", {"order_id": IN_WINDOW_ORDER}),
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
        tool_call("lookup_order", {"order_id": IN_WINDOW_ORDER}),
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


def test_single_active_order_is_adopted(make_agent) -> None:
    """One live order is not ambiguous, so there is nothing to ask about.

    No fixture customer has only one — the data makes the ambiguous case the
    default — so this is exercised through a customer who does.
    """
    from agent.tools import OrderSummary, ToolOutcome, VerifyIdentityResult, apply_tool_result

    state = SessionState()
    result = VerifyIdentityResult(
        verified=True,
        customer_id="CUST-009",
        region="GB",
        active_orders=[
            OrderSummary(order_id="ORD-2001", status="delivered", items=["Some Book"])
        ],
    )
    apply_tool_result(
        "verify_identity",
        {"email": "solo@example.com"},
        ToolOutcome(status=ToolStatus.OK, content="{}", summary="verified=True", payload=result),
        state,
    )

    assert state.active_order_id == "ORD-2001"


def test_two_active_orders_are_not_guessed(make_agent) -> None:
    """CUST-003 has two live orders, so the agent is left with nothing to guess with.

    Verification reports both ids and does not set `active_order_id`. The
    account-scoped tools need one, so the only way forward is to ask — which the
    model does here, in its own words. No `request_clarification` tool exists.
    """
    agent, client = make_agent(
        tool_call("verify_identity", {"email": "sofia@example.com"}),
        text("Thanks Sofia. I can see two active orders — ORD-1005 and ORD-1004. Which one?"),
    )
    state = SessionState()

    reply = agent.respond(state, "hi, it's sofia@example.com, I have a question about my order")

    assert state.verified_customer_id == "CUST-003"
    assert len(state.active_order_ids) == 2
    assert state.active_order_id is None
    assert "which one" in reply.lower()

    # The model was given both ids and no default — the clarification is grounded
    # in what the tool returned.
    result = client.calls[1]["messages"][-1]["content"][0]
    assert "ORD-1004" in result["content"] and "ORD-1005" in result["content"]


def test_verification_alone_is_enough_to_ask_which_order(make_agent) -> None:
    """No order is read before the customer has chosen one.

    The clarifying question needs the titles, not the orders. `verify_identity`
    carries both ids and both books, so the turn that asks costs one tool call —
    it used to cost three, because the agent read every live order to name them.
    """
    agent, client = make_agent(
        tool_call("verify_identity", {"email": "sofia@example.com"}),
        text("Two orders — one with Refactoring, one with Domain-Driven Design. Which one?"),
    )
    state = SessionState()

    agent.respond(state, "hi, it's sofia@example.com")

    assert tool_names(state) == ["verify_identity"]

    # Everything the question needs came back from the one call that was made.
    payload = json.loads(client.calls[1]["messages"][-1]["content"][0]["content"])
    assert {order["order_id"] for order in payload["active_orders"]} == {"ORD-1004", "ORD-1005"}
    assert all(order["items"] for order in payload["active_orders"])
    assert state.active_order_id is None


def test_only_the_chosen_order_is_looked_up(make_agent, seeded_graph) -> None:
    """After the customer picks, exactly one order is read — theirs."""
    agent, _ = make_agent(
        tool_call("verify_identity", {"email": "sofia@example.com"}),
        text("Two orders. Which one did you mean?"),
        tool_call("lookup_order", {"order_id": "ORD-1005"}, block_id="toolu_pick"),
        text("That one was delivered on the 6th."),
    )
    state = SessionState()

    agent.respond(state, "hi, it's sofia@example.com")
    agent.respond(state, "the Refactoring one")

    assert tool_names(state) == ["verify_identity", "lookup_order"]
    assert state.active_order_id == "ORD-1005"


def test_switching_item_clears_the_previous_return_context(
    make_agent, seeded_graph, verified_state
) -> None:
    """A token issued for one item cannot survive a move to another — otherwise a
    "yes" given for one book could open a return for a different one."""
    agent, _ = make_agent(
        tool_call("check_return_eligibility", {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM}),
        text("That's returnable. Shall I start a return?"),
    )
    agent.respond(verified_state, "I want to return the paperback")
    first_token = verified_state.eligibility_token
    assert first_token is not None

    agent2, _ = make_agent(
        tool_call("check_return_eligibility", {"order_id": EXPIRED_ORDER, "item_id": EXPIRED_ITEM}),
        text("That one's outside the window, I'm afraid."),
    )
    agent2.respond(verified_state, "actually I meant the other order, ORD-1002")

    assert verified_state.eligibility_token != first_token
    assert verified_state.confirmed is False
    assert verified_state.active_item_id == EXPIRED_ITEM


# =========================================================================
# Confirmation, through the loop
#
# The phrase-level rules are in test_guardrails.py. What is here is the loop's
# part: reading a "yes" against what was actually pending, before the model sees
# the message.
# =========================================================================


def test_yes_with_no_pending_action_does_not_confirm(make_agent) -> None:
    """The one that matters. A bare "yes" authorises nothing.

    No eligibility check has run, so there is nothing pending and `confirmed` must
    stay False — whatever the conversation looked like.
    """
    agent, _ = make_agent(text("Sure — what can I help with?"))
    state = SessionState()
    state.add_message(Role.ASSISTANT, "Can I help with anything else?")

    agent.respond(state, "yes")

    assert state.confirmed is False
    assert state.pending_return is None
    assert state.may_mutate is False


def test_yes_after_a_statement_does_not_confirm(make_agent, seeded_graph, verified_state) -> None:
    """A pending return is not enough — the agent has to have asked.

    Here it reported eligibility without asking anything. The customer's "yes"
    agrees with the news; it does not authorise the return.
    """
    agent, _ = make_agent(
        tool_call("check_return_eligibility", {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM}),
        text("Good news — that one is eligible, with 19 days left."),
    )
    agent.respond(verified_state, "can I return the paperback?")
    assert verified_state.pending_return is not None

    agent2, _ = make_agent(text("Would you like me to start it?"))
    agent2.respond(verified_state, "yes")

    assert verified_state.confirmed is False
    assert verified_state.pending_return.asked is False


def test_yes_after_an_explicit_request_confirms(make_agent, seeded_graph, verified_state) -> None:
    """The path that is allowed to work: eligibility passed, the agent asked a
    direct question, the customer said yes — and only for this order and item."""
    agent, _ = make_agent(
        tool_call("check_return_eligibility", {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM}),
        text(CONFIRM_QUESTION),
    )
    agent.respond(verified_state, "I'd like to return the paperback")

    agent2, _ = make_agent(text("Done — your return is open."))
    agent2.respond(verified_state, "yes please")

    assert verified_state.confirmed is True
    assert verified_state.may_mutate is True
    assert verified_state.pending_return.order_id == IN_WINDOW_ORDER


def test_confirmed_return_can_be_opened(make_agent, seeded_graph, verified_state) -> None:
    """With all three gates satisfied, the write goes through.

    The model never supplies the token or the confirmation — it asks to open a
    return for an order and item, and the loop passes the trusted values in.
    """
    agent, _ = make_agent(
        tool_call("check_return_eligibility", {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM}),
        text(CONFIRM_QUESTION),
    )
    agent.respond(verified_state, "I want to return the paperback, it arrived damaged")

    agent2, _ = make_agent(
        tool_call(
            "initiate_return",
            {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM, "reason": "arrived damaged"},
        ),
        text("Your return is open — you'll get an email with the next steps."),
    )
    agent2.respond(verified_state, "yes please")

    trace = verified_state.tool_traces[-1]
    assert trace.tool_name == "initiate_return"
    assert trace.status is ToolStatus.OK
    assert "created=True" in trace.result_summary
    # The workflow is over: a later "yes" cannot re-open anything.
    assert verified_state.confirmed is False
    assert verified_state.eligibility_token is None


def test_confirmation_does_not_survive_an_item_switch(
    make_agent, seeded_graph, verified_state
) -> None:
    """A yes given for one book cannot open a return for another."""
    agent, _ = make_agent(
        tool_call("check_return_eligibility", {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM}),
        text(CONFIRM_QUESTION),
    )
    agent.respond(verified_state, "return the paperback please")

    agent2, _ = make_agent(text("Which one did you mean?"))
    agent2.respond(verified_state, "yes")
    assert verified_state.confirmed is True

    agent3, _ = make_agent(
        tool_call("check_return_eligibility", {"order_id": EXPIRED_ORDER, "item_id": EXPIRED_ITEM}),
        text("That one's outside the return window, I'm afraid."),
    )
    agent3.respond(verified_state, "actually I meant the one from ORD-1002")

    assert verified_state.confirmed is False
    assert verified_state.eligibility_token is None
    assert verified_state.pending_return is None


# =========================================================================
# Failure
# =========================================================================


def test_invalid_tool_name_is_rejected_safely(make_agent) -> None:
    """Nothing raises, nothing is written, and the model is told plainly that
    there is no such tool so it can choose a real one."""
    agent, client = make_agent(
        tool_call("cancel_order", {"order_id": IN_WINDOW_ORDER}),
        text("I can't cancel orders, but I can pass you to a colleague."),
    )
    state = SessionState()

    reply = agent.respond(state, "cancel my order")

    assert reply == "I can't cancel orders, but I can pass you to a colleague."
    assert state.tool_traces[0].status is ToolStatus.REJECTED
    assert state.tool_traces[0].error is not None
    result = client.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert "no tool called" in result["content"]


def test_malformed_tool_arguments_are_rejected_safely(make_agent) -> None:
    """A call missing a required argument fails as a tool error, not a crash."""
    agent, _ = make_agent(
        tool_call("verify_identity", {"address": HERO_EMAIL}),  # wrong key
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

    Whatever the model then writes, it cannot honestly claim the lookup worked.
    """

    def boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("agent.tools.lookup_order", boom)

    agent, client = make_agent(
        tool_call("lookup_order", {"order_id": IN_WINDOW_ORDER}),
        text("Something went wrong looking that up."),
    )

    agent.respond(verified_state, "where's my order")

    trace = verified_state.tool_traces[0]
    assert trace.status is ToolStatus.ERROR
    assert "RuntimeError" in (trace.error or "")

    result = client.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert "Do not assume it succeeded" in result["content"]


def test_policy_graph_unavailable_is_reported_not_guessed(
    make_agent, verified_state, monkeypatch
) -> None:
    """With Neo4j down the agent must not answer from memory.

    The tool result says the policy database is unavailable and explicitly tells
    the model not to state a policy — the same stance the tools take.
    """
    break_policy_graph(monkeypatch)

    agent, client = make_agent(
        tool_call("search_policy", {"query": "return window"}),
        text("I can't confirm our policy right now."),
    )

    agent.respond(verified_state, "what's the return window?")

    assert verified_state.tool_traces[0].status is ToolStatus.ERROR
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
    assert "connection reset" in (state.tool_traces[-1].error or "")


def test_api_failure_does_not_leak_the_key(anthropic_config) -> None:
    """The key is on the client, never in a trace, a message, or a repr."""
    agent = BooklyAgent(
        config=anthropic_config, client=FakeAnthropic(error=RuntimeError("401 unauthorized"))
    )
    state = SessionState()

    reply = agent.respond(state, "hello")

    blob = f"{reply} {state.model_dump_json()} {agent.config!r}"
    assert anthropic_config.api_key not in blob


def test_tool_loop_stops_at_the_iteration_limit(make_agent, seeded_graph, verified_state) -> None:
    """A model that only ever asks for tools is cut off, not left to run.

    The script is one more tool call than the limit allows. The turn ends with an
    honest message pointing at a colleague rather than another round trip.
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


# =========================================================================
# Conversation shape
# =========================================================================


def test_conversation_is_multi_turn(make_agent, seeded_graph) -> None:
    """The second turn is sent the first turn's history. Without this the agent
    re-asks for the email every message."""
    agent, client = make_agent(
        tool_call("verify_identity", {"email": HERO_EMAIL}),
        text("Thanks Ada."),
        text("Your order arrived on the 28th."),
    )
    state = SessionState()

    agent.respond(state, HERO_EMAIL)
    agent.respond(state, "where is it?")

    last_request = client.calls[-1]["messages"]
    assert last_request[0] == {"role": "user", "content": HERO_EMAIL}
    assert last_request[-1] == {"role": "user", "content": "where is it?"}
    assert [m.role for m in state.messages] == [
        Role.USER, Role.ASSISTANT, Role.USER, Role.ASSISTANT
    ]


def test_transcript_holds_no_duplicate_assistant_turn(make_agent) -> None:
    """The reply is recorded once in the API transcript, not twice."""
    agent, _ = make_agent(text("Hello — how can I help?"))
    state = SessionState()

    agent.respond(state, "hi")

    assert len([m for m in state.transcript if m["role"] == "assistant"]) == 1


def test_transcript_is_serializable(make_agent) -> None:
    """The session has to survive a Streamlit rerun, so it must round-trip."""
    agent, _ = make_agent(
        tool_call("verify_identity", {"email": HERO_EMAIL}), text("Found you.")
    )
    state = SessionState()
    agent.respond(state, HERO_EMAIL)

    restored = SessionState.model_validate_json(state.model_dump_json())
    assert restored.transcript == state.transcript
    assert restored.verified_customer_id == HERO_CUSTOMER


# =========================================================================
# Tracing
#
# A trace is created for every call, complete enough to render, and carries
# execution rather than reasoning. What it must *not* contain is asserted in
# test_ui.py, where the display path is.
# =========================================================================


def test_a_tool_call_creates_a_complete_trace(make_agent) -> None:
    """One call in, one trace out — with everything the UI needs to render it."""
    agent, _ = make_agent(
        tool_call("verify_identity", {"email": HERO_EMAIL}), text("Found you.")
    )
    state = SessionState()

    agent.respond(state, HERO_EMAIL)

    assert len(state.tool_traces) == 1
    trace = state.tool_traces[0]
    assert trace.tool_name == "verify_identity"
    assert trace.status is ToolStatus.OK
    assert trace.latency_ms >= 0
    assert trace.result_summary
    assert trace.error is None
    assert trace.session_id == state.session_id
    assert trace.trace_id.startswith("TRC-")
    assert trace.timestamp is not None


def test_trace_records_the_selected_model(make_agent, seeded_graph, verified_state) -> None:
    """A trace names both the tier and the model id it resolved to, which is what
    lets the demo show routing."""
    agent, _ = make_agent(
        tool_call("search_policy", {"query": "ebooks"}), text("Ebooks aren't returnable.")
    )
    agent.respond(verified_state, "can I return an ebook")  # return intent → Sonnet

    trace = verified_state.tool_traces[0]
    assert trace.model_tier == "sonnet"
    assert trace.model == "test-sonnet-model"


def test_failed_calls_are_traced_too(make_agent) -> None:
    """A refused or unknown call is exactly what someone reads a trace for."""
    agent, _ = make_agent(tool_call("delete_account", {"why": "because"}), text("I can't do that."))
    state = SessionState()
    agent.respond(state, "delete my account")

    assert state.tool_traces[0].status is ToolStatus.REJECTED
    assert state.tool_traces[0].error is not None


def test_traces_accumulate_in_order(make_agent, seeded_graph) -> None:
    """The list is the sequence of what happened, oldest first."""
    agent, _ = make_agent(
        tool_call("verify_identity", {"email": HERO_EMAIL}),
        tool_call("lookup_order", {"order_id": IN_WINDOW_ORDER}, block_id="toolu_2"),
        text("It arrived on the 28th."),
    )
    state = SessionState()
    agent.respond(state, "ada@example.com, where's ORD-1001?")

    assert tool_names(state) == ["verify_identity", "lookup_order"]


def test_every_turn_records_its_model(make_agent) -> None:
    """Recorded even when no tool runs, so routing is visible on any turn."""
    agent, _ = make_agent(text("Hello!"), text("Of course."))
    state = SessionState()

    agent.respond(state, "hi")
    agent.respond(state, "I need a refund")

    assert tiers(state) == ["haiku", "sonnet"]
    assert [turn.model for turn in state.model_turns] == ["test-haiku-model", "test-sonnet-model"]


def test_model_turn_records_the_routing_reason_and_counts(make_agent) -> None:
    """Why the tier was chosen, and how much work the turn took."""
    agent, _ = make_agent(
        tool_call("verify_identity", {"email": HERO_EMAIL}), text("Found you.")
    )
    state = SessionState()
    agent.respond(state, HERO_EMAIL)

    turn = state.model_turns[0]
    assert turn.routing_reason
    assert turn.iterations == 2
    assert turn.tool_calls == 1


def test_traces_hold_no_reasoning(make_agent) -> None:
    """A trace records execution, not thinking.

    The models have no field for it, and the loop drops any block that is not text
    or a tool call — so there is nowhere for chain-of-thought to land even if a
    response carried it.
    """
    assert "thinking" not in ToolTrace.model_fields
    assert "reasoning" not in ToolTrace.model_fields
    assert "thinking" not in ModelTurn.model_fields

    agent, _ = make_agent(
        tool_call("verify_identity", {"email": HERO_EMAIL}), text("Found you.")
    )
    state = SessionState()
    agent.respond(state, HERO_EMAIL)

    for message in state.transcript:
        content = message["content"]
        if isinstance(content, list):
            assert all(block["type"] in {"text", "tool_use", "tool_result"} for block in content)


# =========================================================================
# The hero journey, end to end
#
#     order status → verify → clarify between two orders → look one up
#     → switch to a return → eligibility → confirm → a real RMA
# =========================================================================


def test_hero_customer_really_has_two_active_orders() -> None:
    """The clarification step is a property of the fixtures, not of the script."""
    from agent.tools import active_order_ids

    assert len(active_order_ids(HERO_CUSTOMER)) >= 2


def test_hero_flow_end_to_end(make_agent, seeded_graph, hero_script, data_dir) -> None:
    """The whole journey, in one session, ending in a real RMA."""
    agent, _ = make_agent(*hero_script)
    state = SessionState()

    run_hero_flow(agent, state)

    assert tool_names(state) == [
        "verify_identity",
        "lookup_order",
        "check_return_eligibility",
        "initiate_return",
    ]
    assert all(trace.status is ToolStatus.OK for trace in state.tool_traces)

    created = [row for row in returns_in(data_dir) if row["order_id"] == IN_WINDOW_ORDER]
    assert len(created) == 1
    assert created[0]["item_id"] == IN_WINDOW_ITEM
    assert created[0]["customer_id"] == HERO_CUSTOMER
    assert created[0]["reason"] == "Not what I expected."


def test_two_active_orders_are_not_resolved_by_guessing(make_agent, seeded_graph, hero_script) -> None:
    """Reading two orders out is the agent building its clarifying question. If
    the second silently became the active order, the agent would be answering the
    question it is in the middle of asking."""
    agent, _ = make_agent(*hero_script)
    state = SessionState()

    agent.respond(state, "Where's my book?")
    agent.respond(state, HERO_EMAIL)

    assert state.verified_customer_id == HERO_CUSTOMER
    assert len(state.active_order_ids) == 2
    assert state.active_order_id is None


def test_customer_selection_sets_the_active_order(make_agent, seeded_graph, hero_script) -> None:
    """A single lookup after the customer chose is what records the choice."""
    agent, _ = make_agent(*hero_script)
    state = SessionState()

    agent.respond(state, "Where's my book?")
    agent.respond(state, HERO_EMAIL)
    agent.respond(state, "The Pragmatic Programmer one")

    assert state.active_order_id == IN_WINDOW_ORDER


def test_status_to_return_transition_reuses_trusted_state(
    make_agent, seeded_graph, hero_script
) -> None:
    """Switching to a return re-asks for nothing the session already holds."""
    agent, _ = make_agent(*hero_script)
    state = SessionState()

    run_hero_flow(agent, state)

    assert tool_names(state).count("verify_identity") == 1
    assert state.verified_customer_id == HERO_CUSTOMER
    assert state.customer_region == "GB"

    check = next(t for t in state.tool_traces if t.tool_name == "check_return_eligibility")
    assert check.tool_args["order_id"] == IN_WINDOW_ORDER


def test_eligible_return_leaves_something_to_confirm(make_agent, seeded_graph, hero_script) -> None:
    """A passing check mints a token and puts a specific return in the balance."""
    agent, _ = make_agent(*hero_script)
    state = SessionState()

    agent.respond(state, "Where's my book?")
    agent.respond(state, HERO_EMAIL)
    agent.respond(state, "The Pragmatic Programmer one")
    agent.respond(state, "Actually, I want to return it.")

    assert state.eligibility is not None and state.eligibility.eligible
    assert state.eligibility_token is not None
    assert state.pending_return is not None
    assert state.pending_return.order_id == IN_WINDOW_ORDER
    assert state.pending_return.item_id == IN_WINDOW_ITEM
    assert state.confirmed is False  # asked, but not yet answered
    assert state.active_item_id == IN_WINDOW_ITEM
    assert state.return_reason == "Not what I expected."


def test_return_workflow_state_is_cleared_after_the_write(
    make_agent, seeded_graph, hero_script
) -> None:
    """Identity survives — the customer is still the customer — but the token, the
    pending action, and the confirmation do not."""
    agent, _ = make_agent(*hero_script)
    state = SessionState()

    run_hero_flow(agent, state)

    assert state.verified_customer_id == HERO_CUSTOMER
    assert state.eligibility_token is None
    assert state.pending_return is None
    assert state.confirmed is False


def test_hero_flow_routing_is_the_generic_router(make_agent, seeded_graph, hero_script) -> None:
    """Status turns stay on Haiku; the return and the confirmation are Sonnet.

    Nothing here forces a tier — these are the decisions the router makes from the
    message and the session, and the demo inherits them.
    """
    agent, _ = make_agent(*hero_script)
    state = SessionState()

    run_hero_flow(agent, state)

    assert tiers(state) == [
        ModelTier.HAIKU,  # "Where's my book?"
        ModelTier.HAIKU,  # the email
        ModelTier.HAIKU,  # picking the order
        ModelTier.SONNET,  # "Actually, I want to return it."
        ModelTier.SONNET,  # confirming, with a return pending
    ]
    assert state.model_turns[3].routing_reason == "return or refund workflow"
    assert state.model_turns[4].routing_reason == "a return workflow is open"


def test_return_intent_keeps_the_following_turns_on_sonnet(
    make_agent, seeded_graph, hero_verified
) -> None:
    """Naming a book is not a return keyword, but it is still part of a return.

    That second turn is where the eligibility check runs, and on its own text it
    looks like a simple lookup — so without the session remembering the intent it
    would drop to the cheaper model mid-workflow.
    """
    agent, _ = make_agent(
        text("Which one — The Pragmatic Programmer, or Designing Data-Intensive Applications?"),
        tool_call("check_return_eligibility", {"order_id": EXPIRED_ORDER, "item_id": EXPIRED_ITEM}),
        text("That one's outside the return window, I'm afraid."),
    )

    agent.respond(hero_verified, "I want to return a book")
    agent.respond(hero_verified, "Designing Data-Intensive Applications")

    assert tiers(hero_verified) == [ModelTier.SONNET, ModelTier.SONNET]
    assert hero_verified.tool_traces[-1].model_tier == ModelTier.SONNET


def test_hero_flow_trace_is_readable_and_leaks_nothing(
    make_agent, seeded_graph, hero_script
) -> None:
    """The trace tells the story, without the customer's address or a token."""
    agent, _ = make_agent(*hero_script)
    state = SessionState()

    run_hero_flow(agent, state)

    for trace in state.tool_traces:
        assert trace.model
        assert trace.model_tier in {"haiku", "sonnet"}
        assert trace.latency_ms >= 0
        assert trace.result_summary
        assert HERO_EMAIL not in json.dumps(trace.tool_args)
        assert "eligibility_token" not in trace.result_summary

    verification = state.tool_traces[0]
    assert verification.tool_args["email"] == "a***@example.com"
    assert "verified=True" in verification.result_summary
    assert "created=True" in state.tool_traces[-1].result_summary


def test_a_confirmed_return_is_not_held_up_for_a_reason(
    make_agent, seeded_graph, verified_state, data_dir
) -> None:
    """A customer who never says why still gets their return.

    `reason` is recorded on the RMA, not checked by any guard, so it must not be
    able to block a confirmed, eligible return.
    """
    agent, _ = make_agent(
        tool_call("check_return_eligibility", {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM}),
        text(CONFIRM_QUESTION),
        tool_call(
            "initiate_return",
            {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM},
            block_id="toolu_write",
        ),
        text("Your return is open."),
    )

    agent.respond(verified_state, "I'd like to send this one back")
    agent.respond(verified_state, "Go ahead")

    assert verified_state.tool_traces[-1].status is ToolStatus.OK
    created = [r for r in returns_in(data_dir) if r["order_id"] == IN_WINDOW_ORDER]
    assert len(created) == 1
    assert created[0]["reason"] == ""


def test_a_reason_given_earlier_is_not_asked_for_again(
    make_agent, seeded_graph, verified_state, data_dir
) -> None:
    """The model omits the reason on the write. The session already holds it, so
    it is filled in from there rather than lost or re-requested."""
    agent, _ = make_agent(
        tool_call(
            "check_return_eligibility",
            {
                "order_id": IN_WINDOW_ORDER,
                "item_id": IN_WINDOW_ITEM,
                "reason": "The spine was cracked.",
            },
        ),
        text(CONFIRM_QUESTION),
        tool_call(
            "initiate_return",
            {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM},
            block_id="toolu_write",
        ),
        text("Your return is open."),
    )

    agent.respond(verified_state, "The spine was cracked, I want to return it")
    agent.respond(verified_state, "Yes")

    created = [r for r in returns_in(data_dir) if r["order_id"] == IN_WINDOW_ORDER]
    assert created[0]["reason"] == "The spine was cracked."


def test_repeating_the_hero_return_does_not_create_a_second_rma(
    make_agent, seeded_graph, hero_script, data_dir, verified_state
) -> None:
    """Idempotency, from the customer's side: asking twice yields one RMA.

    Eligibility refuses the second time — a return is already open — so the write
    is never reached.
    """
    agent, _ = make_agent(*hero_script)
    run_hero_flow(agent, SessionState())

    agent2, _ = make_agent(
        tool_call(
            "check_return_eligibility",
            {
                "order_id": IN_WINDOW_ORDER,
                "item_id": IN_WINDOW_ITEM,
                "reason": "Changed my mind.",
            },
        ),
        text("There's already a return open for that one."),
    )
    agent2.respond(verified_state, "I want to return the Pragmatic Programmer")

    assert verified_state.eligibility is not None
    assert verified_state.eligibility.eligible is False
    assert verified_state.eligibility_token is None
    assert len([r for r in returns_in(data_dir) if r["order_id"] == IN_WINDOW_ORDER]) == 1


# --- The wording is not load-bearing ------------------------------------


@pytest.mark.parametrize(
    "phrasing",
    [
        "Where's my book?",
        "Can you check my delivery?",
        "What's happening with my order?",
        "I haven't received my book yet.",
    ],
)
def test_status_phrasings_all_reach_an_order_lookup(
    make_agent, seeded_graph, verified_state, phrasing
) -> None:
    """Four ways of asking the same thing, none special-cased anywhere."""
    agent, _ = make_agent(
        tool_call("lookup_order", {"order_id": IN_WINDOW_ORDER}),
        text("It was delivered on 28 July."),
    )

    agent.respond(verified_state, phrasing)

    assert tool_names(verified_state) == ["lookup_order"]
    assert verified_state.model_turns[0].model_tier == ModelTier.HAIKU


@pytest.mark.parametrize(
    "phrasing",
    [
        "Actually, I want to return it.",
        "Can I send this back?",
        "I don't want it anymore.",
        "Can I get a refund?",
    ],
)
def test_return_phrasings_all_reach_the_eligibility_check(
    make_agent, seeded_graph, hero_verified, phrasing
) -> None:
    """Four ways of changing intent, all promoted and all deciding the same way.

    "I don't want it anymore" carries no return keyword and reaches Sonnet the
    other way — as a message the agent has to resolve. Which route it takes is the
    router's business; that it does not land on the cheap model is the property.
    """
    agent, _ = make_agent(
        tool_call(
            "check_return_eligibility",
            {"order_id": IN_WINDOW_ORDER, "item_id": IN_WINDOW_ITEM, "reason": phrasing},
        ),
        text(CONFIRM_QUESTION),
    )

    agent.respond(hero_verified, phrasing)

    assert tool_names(hero_verified) == ["check_return_eligibility"]
    assert hero_verified.pending_return is not None
    assert hero_verified.model_turns[0].model_tier == ModelTier.SONNET


@pytest.mark.parametrize(
    "phrasing", ["Yes", "yes please", "Go ahead", "Please do", "Confirm it", "proceed"]
)
def test_confirmation_phrasings_work_in_a_pending_context(
    make_agent, seeded_graph, hero_verified, phrasing, data_dir
) -> None:
    """Six ways of agreeing, all of which open the return that was asked about."""
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
    agent.respond(hero_verified, "I'd like to return it")
    agent.respond(hero_verified, phrasing)

    write = hero_verified.tool_traces[-1]
    assert write.tool_name == "initiate_return"
    assert write.status is ToolStatus.OK
    assert len([r for r in returns_in(data_dir) if r["order_id"] == IN_WINDOW_ORDER]) == 1


def test_a_fixed_clock_keeps_the_scenarios_meaning_what_they_say(now: datetime) -> None:
    """The suite measures every date against 2026-08-08, which is what the order
    fixtures were written for."""
    assert now.date().isoformat() == "2026-08-08"
