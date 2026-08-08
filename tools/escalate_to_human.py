"""Tool: hand the conversation to a human agent.

Mocked, and deliberately so: there is no ticketing system behind this. It mints a
case id, records the context a human would need, and returns a line the agent can
read out. Wiring it to a real helpdesk is an integration, not a design question.

What matters is that the escape hatch exists and is cheap to reach. An agent with
no way to say "a person will take this" ends up guessing at consumer law or a
payment dispute instead.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field


class EscalationResult(BaseModel):
    """Confirmation that a handoff was queued."""

    case_id: str = Field(description="Reference the customer can quote, e.g. 'CASE-3F2A9C41'.")
    reason: str = Field(description="Why the handoff happened — context for the human, not the customer.")
    customer_id: str | None = None
    order_id: str | None = None
    created_at: datetime
    message: str
    """Customer-facing line. Says a human will pick it up, not what they'll decide."""


def escalate_to_human(
    reason: str, customer_id: str | None = None, order_id: str | None = None
) -> EscalationResult:
    """Queue the conversation for a human, and stop the agent acting.

    Called when the customer asks for a person, when a request falls outside the
    tools, when a tool has failed twice on the same thing, or when a customer
    raises consumer-law or payment-dispute questions the agent must not answer.

    Works without identity: someone who cannot get past verification is exactly
    who needs a human, so `customer_id` and `order_id` are optional context.

    Args:
        reason: Why the handoff is happening — goes to the human as context.
        customer_id: Verified customer, if identity was established.
        order_id: The order under discussion, if there is one.

    Returns:
        The case id, the context recorded against it, and a line to read back to
        the customer.
    """
    case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"

    # The orchestrator sets state.escalated after this returns. The tool stays
    # stateless — it records a handoff, it does not police the conversation.
    return EscalationResult(
        case_id=case_id,
        reason=reason,
        customer_id=customer_id,
        order_id=order_id,
        created_at=datetime.now(UTC),
        message=(
            f"I'm passing this to a colleague who can help — your reference is {case_id}. "
            f"They'll follow up by email."
        ),
    )
