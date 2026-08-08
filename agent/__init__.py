"""Bookly support agent: orchestrator, model routing, prompt, session state,
tracing, domain models, and the required Neo4j connection behind every policy
decision.

`BooklyAgent` is imported lazily, and has to be. The dependency runs in both
directions: the tools read policy from `agent.graph`, and the orchestrator
dispatches to the tools. Importing the orchestrator here eagerly would mean
`import tools` initialises this package, which imports the orchestrator, which
imports `tools` — while `tools` is still half-built.

PEP 562 breaks it. `from agent import BooklyAgent` still works and still
type-checks; the orchestrator module is simply not touched until something asks
for it, by which point `tools` has finished importing.
"""

from typing import TYPE_CHECKING

from agent.config import AnthropicConfig, AnthropicConfigError, load_anthropic_config
from agent.graph import (
    PolicyGraphUnavailableError,
    close_driver,
    fetch_policies_for_category,
    get_driver,
)
from agent.routing import ModelDecision, ModelTier, select_model
from agent.state import SessionState
from agent.tracing import ModelTurn, ToolStatus, ToolTrace

if TYPE_CHECKING:  # pragma: no cover - for type checkers only, never at runtime
    from agent.orchestrator import BooklyAgent


def __getattr__(name: str):
    """Resolve `BooklyAgent` on first use, after `tools` has finished importing."""
    if name == "BooklyAgent":
        from agent.orchestrator import BooklyAgent

        return BooklyAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnthropicConfig",
    "AnthropicConfigError",
    "BooklyAgent",
    "ModelDecision",
    "ModelTier",
    "ModelTurn",
    "PolicyGraphUnavailableError",
    "SessionState",
    "ToolStatus",
    "ToolTrace",
    "close_driver",
    "fetch_policies_for_category",
    "get_driver",
    "load_anthropic_config",
    "select_model",
]
