"""Anthropic configuration, read from the environment and validated up front.

Two model names, not one. The orchestrator routes simple, read-only turns to the
cheaper model and anything that reasons about eligibility or changes Bookly's
records to the stronger one, so both have to be configured before the agent can
run.

No model id is hardcoded anywhere in the application. Which Claude models Bookly
runs on is a deployment decision — it belongs in `.env`, next to the API key, not
in a constant someone has to find and edit. The router picks a *tier*; the
environment decides what that tier resolves to.

Missing configuration fails here, loudly, at construction time. An agent that
starts up and only discovers it has no API key on the customer's first message
has already wasted the customer's time.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

REQUIRED_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL_HAIKU", "ANTHROPIC_MODEL_SONNET")

TEMPERATURE_ENV = "ANTHROPIC_TEMPERATURE"
"""Optional sampling temperature, used for both tiers — one value, never one per model.

**Unset by default, and that is the right default.** This agent wants
consistent, repeatable behaviour: the same question should get the same answer,
and a demo should behave the same way twice. Current Claude models manage their
own sampling internally and reject an explicit temperature outright, so the way
to get that here is to send nothing and let the model do it — not to pin a
number the API will refuse.

The setting stays because it is a deployment choice, not a code one: a
deployment running a model that does accept a temperature can set it in `.env`
without touching the loop. When it is unset, the parameter is not sent at all.

Either way it only reaches tone. Business truth and every state-changing action
stay with the deterministic tools, so sampling cannot decide an eligibility or
open a return.
"""


class AnthropicConfigError(RuntimeError):
    """Anthropic is not configured. Raised at startup, never mid-conversation."""


class AnthropicConfig(BaseModel):
    """The three values the orchestrator needs to talk to Anthropic.

    `api_key` is here because the client needs it. It is never logged, never
    traced, and never included in a `repr` — see `__repr__` below.
    """

    api_key: str
    haiku_model: str = Field(description="Model id for simple, read-only turns.")
    sonnet_model: str = Field(description="Model id for reasoning and state-changing turns.")
    temperature: float | None = Field(
        default=None,
        description=(
            "Sampling temperature for both tiers. None means the parameter is not sent "
            "at all, which is what current models want."
        ),
    )

    def __repr__(self) -> str:
        """Redact the key. A config object can end up in a traceback."""
        return (
            f"AnthropicConfig(api_key='***', haiku_model={self.haiku_model!r}, "
            f"sonnet_model={self.sonnet_model!r}, temperature={self.temperature!r})"
        )

    __str__ = __repr__


def load_anthropic_config() -> AnthropicConfig:
    """Read and validate the Anthropic configuration.

    Returns:
        The configuration, with all three required values present and non-empty,
        and the temperature parsed or defaulted.

    Raises:
        AnthropicConfigError: If any of ANTHROPIC_API_KEY, ANTHROPIC_MODEL_HAIKU,
            or ANTHROPIC_MODEL_SONNET is unset or blank. The message names every
            missing variable at once, so the fix is one edit rather than three
            restarts. Also raised if ANTHROPIC_TEMPERATURE is set to something
            that is not a number.
    """
    load_dotenv()
    values = {name: (os.getenv(name) or "").strip() for name in REQUIRED_ENV}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise AnthropicConfigError(
            f"Anthropic is required to run the agent, but "
            f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not set. "
            f"Copy .env.example to .env and fill it in. Both model names are "
            f"required: the agent routes between them."
        )

    return AnthropicConfig(
        api_key=values["ANTHROPIC_API_KEY"],
        haiku_model=values["ANTHROPIC_MODEL_HAIKU"],
        sonnet_model=values["ANTHROPIC_MODEL_SONNET"],
        temperature=load_temperature(),
    )


def load_temperature() -> float | None:
    """Read `ANTHROPIC_TEMPERATURE`, if a deployment set one.

    Returns:
        The configured temperature, or None when the variable is unset or blank —
        which means the orchestrator sends no temperature at all.

    Raises:
        AnthropicConfigError: If it is set to something that is not a number.
            Silently ignoring a typo would hide it behind behaviour that looks
            deliberate.
    """
    raw = (os.getenv(TEMPERATURE_ENV) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        raise AnthropicConfigError(
            f"{TEMPERATURE_ENV} must be a number, but it is set to {raw!r}. "
            f"Leave it unset to let the model manage its own sampling."
        ) from None
