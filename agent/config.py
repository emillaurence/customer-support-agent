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

    def __repr__(self) -> str:
        """Redact the key. A config object can end up in a traceback."""
        return (
            f"AnthropicConfig(api_key='***', haiku_model={self.haiku_model!r}, "
            f"sonnet_model={self.sonnet_model!r})"
        )

    __str__ = __repr__


def load_anthropic_config() -> AnthropicConfig:
    """Read and validate the Anthropic configuration.

    Returns:
        The configuration, with all three values present and non-empty.

    Raises:
        AnthropicConfigError: If any of ANTHROPIC_API_KEY, ANTHROPIC_MODEL_HAIKU,
            or ANTHROPIC_MODEL_SONNET is unset or blank. The message names every
            missing variable at once, so the fix is one edit rather than three
            restarts.
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
    )
