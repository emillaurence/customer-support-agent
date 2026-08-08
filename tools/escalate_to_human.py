"""Tool: hand the conversation to a human agent. Stub only."""

from __future__ import annotations

from pydantic import BaseModel


class EscalationResult(BaseModel):
    """Confirmation that a handoff was queued."""

    ticket_id: str
    message: str
    """Customer-facing line. Says a human will pick it up, not what they'll decide."""


def escalate_to_human(reason: str, customer_id: str | None = None, order_id: str | None = None) -> EscalationResult:
    """Queue the conversation for a human, and stop the agent acting.

    The escape hatch. Called when the customer asks for a person, when a request
    falls outside the tools, when a tool has failed twice on the same thing, or
    when a customer raises consumer-law or payment-dispute questions the agent
    must not answer.

    Args:
        reason: Why the handoff is happening — goes to the human as context.
        customer_id: Verified customer, if identity was established.
        order_id: The order under discussion, if there is one.

    Returns:
        The ticket id and a line to read back to the customer.
    """
    # TODO: generate a ticket id and attach the transcript.
    # TODO: the orchestrator sets state.escalated after this returns; the tool
    #       stays stateless.
    raise NotImplementedError("escalate_to_human is a scaffold stub")
