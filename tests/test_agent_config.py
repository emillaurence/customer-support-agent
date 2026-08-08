"""Anthropic configuration: both model names required, and no id in the code.

The behaviour worth pinning down is that the *application* has no opinion about
which Claude models it runs on. The router picks a tier; the environment decides
what that tier is. Swapping models is an `.env` edit, and a missing one is a
startup error rather than a surprise on the first customer message.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent.config import (
    REQUIRED_ENV,
    TEMPERATURE_ENV,
    AnthropicConfigError,
    load_anthropic_config,
)
from agent.orchestrator import BooklyAgent
from agent.routing import ModelDecision, ModelTier
from agent.state import SessionState
from tests.conftest import text

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch):
    """Start from no Anthropic configuration at all.

    `load_dotenv` would otherwise read a developer's real `.env` and the test
    would pass or fail depending on whose machine it ran on.
    """
    monkeypatch.setattr("agent.config.load_dotenv", lambda *a, **k: None)
    for name in (*REQUIRED_ENV, TEMPERATURE_ENV):
        monkeypatch.delenv(name, raising=False)


# --- Loading -------------------------------------------------------------


def test_complete_configuration_loads(clean_env, monkeypatch) -> None:
    """All three present is all it takes."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("ANTHROPIC_MODEL_HAIKU", "some-fast-model")
    monkeypatch.setenv("ANTHROPIC_MODEL_SONNET", "some-capable-model")

    config = load_anthropic_config()

    assert config.haiku_model == "some-fast-model"
    assert config.sonnet_model == "some-capable-model"


@pytest.mark.parametrize("missing", REQUIRED_ENV)
def test_any_missing_variable_fails_clearly(clean_env, monkeypatch, missing: str) -> None:
    """Each one is required, and the error names the one that is absent.

    Both model names included: an agent configured with only the cheap model
    cannot route, and would silently do the consequential turns on the wrong
    one.
    """
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


# --- Temperature ---------------------------------------------------------


@pytest.fixture
def configured(monkeypatch):
    """Set the three required variables, leaving the temperature to the test."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("ANTHROPIC_MODEL_HAIKU", "fast")
    monkeypatch.setenv("ANTHROPIC_MODEL_SONNET", "capable")


def test_temperature_is_unset_by_default(clean_env, configured) -> None:
    """No variable, no temperature. The model manages its own sampling."""
    assert load_anthropic_config().temperature is None


def test_temperature_is_read_from_the_environment(clean_env, configured, monkeypatch) -> None:
    """A deployment on a model that accepts one sets it in `.env`."""
    monkeypatch.setenv(TEMPERATURE_ENV, "0.7")
    assert load_anthropic_config().temperature == 0.7


def test_zero_is_a_setting_not_an_absence(clean_env, configured, monkeypatch) -> None:
    """`0` must survive as 0 — the falsiest value the setting can hold."""
    monkeypatch.setenv(TEMPERATURE_ENV, "0")
    assert load_anthropic_config().temperature == 0.0


def test_blank_temperature_is_treated_as_unset(clean_env, configured, monkeypatch) -> None:
    """`ANTHROPIC_TEMPERATURE=` in a half-filled `.env` is not a setting."""
    monkeypatch.setenv(TEMPERATURE_ENV, "   ")
    assert load_anthropic_config().temperature is None


def test_unparseable_temperature_fails_loudly(clean_env, configured, monkeypatch) -> None:
    """A typo is a startup error, not a silent fallback that looks deliberate."""
    monkeypatch.setenv(TEMPERATURE_ENV, "warm")

    with pytest.raises(AnthropicConfigError, match=TEMPERATURE_ENV):
        load_anthropic_config()


def test_no_temperature_is_sent_when_none_is_configured(make_agent) -> None:
    """The parameter is absent from the request, not sent as a default.

    This is the whole point of the None: a model that manages its own sampling
    rejects an explicit temperature outright, so sending 0 would fail every turn
    rather than quietly doing nothing.
    """
    agent, client = make_agent(text("Ebooks aren't returnable."), text("Let me check that."))
    state = SessionState()

    agent.respond(state, "what's your policy on ebooks?")
    agent.respond(state, "I'd like a refund")

    assert client.models_used == ["test-haiku-model", "test-sonnet-model"]
    assert all("temperature" not in call for call in client.calls)


def test_both_tiers_are_sent_the_same_temperature(anthropic_config, make_agent) -> None:
    """When one is configured: one value, both models, every call.

    A Haiku turn and a Sonnet turn in one conversation. There is one exit from
    the loop to Anthropic, so there is nowhere for a turn to be sampled
    differently — this pins that down.
    """
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
    """The orchestrator reads it from the config, and nowhere else."""
    source = (ROOT / "agent" / "orchestrator.py").read_text()
    assert "self.config.temperature" in source


# --- Both paths work -----------------------------------------------------


@pytest.mark.parametrize(
    ("tier", "expected"),
    [(ModelTier.HAIKU, "test-haiku-model"), (ModelTier.SONNET, "test-sonnet-model")],
)
def test_both_tiers_resolve_to_their_configured_model(
    anthropic_config, tier: ModelTier, expected: str
) -> None:
    """Each tier resolves to the name the environment gave it, and only that."""
    agent = BooklyAgent(config=anthropic_config, client=object())
    resolved = agent._model_id(ModelDecision(tier=tier, reason="test"))
    assert resolved == expected


def test_both_configuration_paths_reach_the_api(make_agent) -> None:
    """A Haiku turn and a Sonnet turn in one conversation, each on its own model."""
    agent, client = make_agent(text("Ebooks aren't returnable."), text("Let me check that."))
    state = SessionState()

    agent.respond(state, "what's your policy on ebooks?")
    agent.respond(state, "I'd like a refund")

    assert client.models_used == ["test-haiku-model", "test-sonnet-model"]


# --- No hardcoded model ids ---------------------------------------------


def test_no_model_id_is_hardcoded_in_application_code() -> None:
    """Which Claude models Bookly runs on is a deployment decision.

    Scans the shipped modules for anything shaped like a Claude model id. The
    only places one may appear are `.env` and the deployment that fills it in —
    not a constant in the source, and not a default in the router.
    """
    looks_like_a_model_id = re.compile(r"claude[-_][a-z0-9][a-z0-9.\-]*", re.IGNORECASE)

    offenders = []
    for path in sorted((*(ROOT / "agent").glob("*.py"), *(ROOT / "tools").glob("*.py"), ROOT / "app.py")):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if looks_like_a_model_id.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")

    assert not offenders, "model ids belong in .env, not in code:\n" + "\n".join(offenders)


def test_env_example_documents_both_models() -> None:
    """`.env.example` is the contract — it has to name every variable."""
    example = (ROOT / ".env.example").read_text()
    for name in (
        *REQUIRED_ENV,
        TEMPERATURE_ENV,
        "NEO4J_URI",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD",
    ):
        assert f"{name}=" in example


def test_env_example_ships_no_values() -> None:
    """Every key is blank. A committed `.env.example` must never carry a secret.

    The temperature is blank too, and not because it is a secret: leaving it
    unset is the working configuration, so a copied `.env` starts correct.
    """
    for line in (ROOT / ".env.example").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            assert line.split("=", 1)[1] == "", f"{line} has a value committed"
