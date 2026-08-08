"""Observable execution records: what the agent actually did, and how long it took.

Two record types, both plain data:

* `ToolTrace` — one deterministic tool call: what was asked, what came back,
  whether it worked, how long it took.
* `ModelTurn` — one conversational turn: which model tier handled it, why the
  router chose that tier, and how many model calls it took to finish.

Both are held on `SessionState` so the Streamlit UI can render them in Phase 6
without the orchestrator knowing a UI exists.

**This traces execution, not reasoning.** A trace records that
`check_return_eligibility` was called with an order and an item and returned
`eligible=False` in 4ms. It does not record what the model was thinking, and no
`thinking` content is ever captured here — chain-of-thought is not an observable
event, and putting it in a trace turns a debugging tool into a leak.

Nothing secret is stored. Tool arguments are sanitized before they are recorded:
email addresses are masked, and anything that looks like a credential is
redacted outright. The Anthropic key and the Neo4j password are never near this
module — they live in `agent.config` and `agent.graph`, and neither is passed to
a tool.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

MAX_SUMMARY_CHARS = 200
"""Result summaries are for reading, not for reconstructing the payload."""

SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "anthropic_api_key",
        "password",
        "neo4j_password",
        "token",
        "eligibility_token",
        "secret",
        "authorization",
    }
)
"""Argument names that are redacted whole, whatever their value.

`eligibility_token` is on the list even though the model never sees one — it is
injected by the orchestrator from session state. Redacting it means a trace
stays safe if that ever changes.
"""

EMAIL_PATTERN = re.compile(r"([^@\s]+)@([^@\s]+)")


class ToolStatus(StrEnum):
    """How a tool call ended."""

    OK = "ok"
    """The tool ran and returned a result. Says nothing about whether the answer
    was yes — a refused eligibility check is a successful tool call."""

    BLOCKED = "blocked"
    """A guard refused the call. `initiate_return` without confirmation lands here."""

    ERROR = "error"
    """The tool raised, or a dependency was unavailable. Nothing can be concluded."""

    REJECTED = "rejected"
    """The model asked for a tool that does not exist, or with unusable arguments."""


class ToolTrace(BaseModel):
    """One tool call, as an observer would see it."""

    trace_id: str = Field(default_factory=lambda: f"TRC-{uuid.uuid4().hex[:8].upper()}")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    session_id: str
    model: str = Field(description="The model id that requested the call.")
    model_tier: str = Field(description="'haiku' or 'sonnet' — which tier that id was.")
    tool_name: str
    tool_args: dict[str, Any] = Field(
        default_factory=dict, description="Sanitized. Never the raw arguments."
    )
    status: ToolStatus
    latency_ms: float
    result_summary: str = Field(
        default="", description="A short, readable line. Not the full payload."
    )
    error: str | None = Field(
        default=None, description="Set on BLOCKED, ERROR, and REJECTED. The message, never a stack."
    )

    @property
    def ok(self) -> bool:
        return self.status is ToolStatus.OK


class ModelTurn(BaseModel):
    """One user turn, and which model handled it.

    Recorded whether or not any tool ran, so the demo can show the router's
    decision on a plain policy question as clearly as on a return.
    """

    turn_id: str = Field(default_factory=lambda: f"TURN-{uuid.uuid4().hex[:8].upper()}")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    session_id: str
    model_tier: str = Field(description="'haiku' or 'sonnet'.")
    model: str = Field(description="The model id the tier resolved to.")
    routing_reason: str = Field(description="The one rule that decided the tier.")
    tool_calls: int = Field(default=0, description="Tools run during this turn.")
    iterations: int = Field(default=0, description="Round trips to Anthropic this turn.")


def sanitize_args(args: dict[str, Any]) -> dict[str, Any]:
    """Make tool arguments safe to record.

    Three rules:

    * A key on `SENSITIVE_KEYS` is replaced with `'***'`, whatever it holds.
    * An email address is masked to its first character and domain, so a trace
      shows that verification was attempted for `e***@example.com` without
      writing the customer's address into a log.
    * Anything long is truncated. A return reason is a sentence, not an essay,
      and a trace is not the place to store one anyway.

    Args:
        args: The arguments as they were passed to the tool.

    Returns:
        A new dict. The input is not modified.
    """
    clean: dict[str, Any] = {}
    for key, value in args.items():
        if key.lower() in SENSITIVE_KEYS:
            clean[key] = "***"
        elif isinstance(value, str):
            clean[key] = _truncate(mask_email(value))
        else:
            clean[key] = value
    return clean


def mask_email(value: str) -> str:
    """Mask any email address in a string.

    `ada@bookly.test` becomes `a***@bookly.test`. The domain is kept because it
    is useful when reading a trace and is not personal; the local part is not.
    """
    return EMAIL_PATTERN.sub(lambda m: f"{m.group(1)[:1]}***@{m.group(2)}", value)


def summarize(value: Any) -> str:
    """Turn a tool result into one readable line.

    Pulls out the fields that say what happened — verified, eligible, created —
    rather than dumping the model. A summary is for a human scanning the trace
    list; the tool result itself is what the agent reasons over.

    Args:
        value: Whatever the tool returned.

    Returns:
        A short line, always safe to display.
    """
    if value is None:
        return "no result"

    if isinstance(value, BaseModel):
        fields = value.model_dump()
        interesting = {
            key: fields[key]
            for key in ("verified", "eligible", "matched", "created", "policy_id", "case_id", "requires_human")
            if key in fields and fields[key] is not None
        }
        if interesting:
            summary = ", ".join(f"{key}={val}" for key, val in interesting.items())
        else:
            summary = type(value).__name__
        return _truncate(mask_email(summary))

    return _truncate(mask_email(str(value)))


def _truncate(text: str) -> str:
    """Cap a string at `MAX_SUMMARY_CHARS`, with an ellipsis when it was cut."""
    if len(text) <= MAX_SUMMARY_CHARS:
        return text
    return f"{text[: MAX_SUMMARY_CHARS - 1]}…"
