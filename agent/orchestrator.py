"""The Bookly agent loop.

Scaffold only. This will eventually: send the transcript plus the tool schemas
to Claude, dispatch any tool calls, feed the results back, and repeat until the
model produces a plain reply.

No Anthropic client here yet — `respond` returns a canned string.
"""

from __future__ import annotations

from agent.models import Role
from agent.prompts import SYSTEM_PROMPT
from agent.state import SessionState

MAX_TOOL_ITERATIONS = 5
"""Safety stop so a misbehaving loop cannot call tools forever."""


class BooklyAgent:
    """A single support agent with a flat set of tools.

    Deliberately not a framework: one class, one loop, one prompt.
    """

    def __init__(self, system_prompt: str = SYSTEM_PROMPT) -> None:
        self.system_prompt = system_prompt
        # TODO: construct the Anthropic client here from ANTHROPIC_API_KEY.
        # TODO: build the tool registry (name -> callable + JSON schema).

    def respond(self, state: SessionState, user_message: str) -> str:
        """Take one user turn and produce one assistant reply.

        Mutates `state` in place: both turns are appended to the transcript.

        Args:
            state: The live session, carried across turns.
            user_message: What the customer just said.

        Returns:
            The assistant's reply text.
        """
        state.add_message(Role.USER, user_message)

        # TODO: replace with the real loop:
        #   1. call the model with system_prompt + state.messages + tool schemas
        #   2. if it asks for tools, run them and append the results
        #   3. repeat, up to MAX_TOOL_ITERATIONS
        #   4. return the final text
        reply = (
            "Bookly agent scaffold — the model loop is not wired up yet. "
            f"You said: {user_message!r}"
        )

        state.add_message(Role.ASSISTANT, reply)
        return reply
