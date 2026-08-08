"""The Bookly agent loop: one user turn in, one reply out.

A hand-written tool loop against the Anthropic Messages API. No framework, no
second agent, no service layer — one class, one loop, one prompt. The loop is
short enough to read in one sitting, which is the point: everything that decides
whether a customer's records change is visible in it.

The division of labour is the whole design.

**Claude owns language.** Understanding what the customer means, asking the
right clarifying question, choosing which tool fits, and writing the reply.

**Python owns truth.** Which model handles the turn, whether identity is
verified, what the policy says, whether a return is eligible, whether the
customer actually agreed, and whether anything is written. Claude can ask for a
tool; it cannot decide the answer, and it cannot set a trusted field. Every
value in `SessionState` that a guard depends on is written here from a tool
result that succeeded, or by `agent.confirmation` from the conversation.

One turn, in order:

1. Read the customer's message and, before anything else, decide whether it
   confirms a pending return.
2. Route to Haiku or Sonnet, deterministically, and record which.
3. Call Anthropic with the transcript, the system prompt, and the tool schemas.
4. If Claude asked for tools, run them, trace them, update trusted state from
   the results, feed the results back, and go again — up to a small limit.
5. When Claude replies in plain text, return it.

Nothing in the loop invents a result. Every failure — a refused guard, an
unreachable database, an unknown tool, an API outage — reaches the customer as
what it was.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import anthropic

from agent import confirmation
from agent.config import AnthropicConfig, load_anthropic_config
from agent.models import Role
from agent.prompts import SYSTEM_PROMPT
from agent.routing import ModelDecision, ModelTier, select_model
from agent.state import SessionState
from agent.tool_registry import TOOL_SCHEMAS, apply_to_state, invoke_timed
from agent.tracing import ModelTurn, ToolStatus, ToolTrace, sanitize_args

MAX_TOOL_ITERATIONS = 6
"""Safety stop so a misbehaving loop cannot call tools forever.

Six is comfortably more than the longest real path — verify, look up the order,
check eligibility, open the return — with room for a retry and a policy lookup.
Hitting it means something is wrong, so the turn ends with an honest message
rather than another round trip.
"""

MAX_TOKENS = 1024
"""Support replies are short. A cap this size also bounds a runaway turn."""

FALLBACK_UNAVAILABLE = (
    "I'm having trouble reaching our systems right now, so I can't look that up. "
    "Please try again in a moment, or reply and I'll pass you to a colleague."
)

FALLBACK_STUCK = (
    "I'm going in circles on this one rather than getting you an answer. "
    "Let me hand you to a colleague who can pick it up."
)

FALLBACK_EMPTY = (
    "Sorry — I didn't manage to put that into words. Could you say that again?"
)


class BooklyAgent:
    """A single support agent with a flat set of tools.

    Deliberately not a framework: one class, one loop, one prompt.

    Stateless between turns. Everything that persists is on the `SessionState`
    passed to `respond`, so one agent can serve many conversations and a
    conversation survives being handed to a new agent instance.
    """

    def __init__(
        self,
        config: AnthropicConfig | None = None,
        client: Any | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        clock: datetime | None = None,
    ) -> None:
        """Build an agent, failing now if Anthropic is not configured.

        Args:
            config: Anthropic configuration. Read from the environment when
                omitted, which raises `AnthropicConfigError` if it is missing —
                at construction, not on the customer's first message.
            client: An Anthropic client. Constructed from `config` when omitted.
                Tests pass a stand-in so the loop can be exercised offline.
            system_prompt: Override for the system prompt.
            clock: Fixed time for the return window. Tests pass one so the
                fixture scenarios keep meaning what they say; production leaves
                it None and the tools use the real clock.
        """
        self.config = config or load_anthropic_config()
        self.client = client or anthropic.Anthropic(api_key=self.config.api_key)
        self.system_prompt = system_prompt
        self.clock = clock

    # --- One turn --------------------------------------------------------

    def respond(self, state: SessionState, user_message: str) -> str:
        """Take one user turn and produce one assistant reply.

        Mutates `state` in place: both transcripts, any trusted fields the tools
        established, and the traces.

        Args:
            state: The live session, carried across turns.
            user_message: What the customer just said.

        Returns:
            The assistant's reply text. Always something sayable — an outage or
            a stuck loop produces an honest message, never an exception and
            never a fabricated success.
        """
        # Confirmation is read before the model sees the message, and from the
        # state as it was when the agent asked its question. Doing it here means
        # a "yes" is judged against what was actually pending, not against
        # whatever the model does with the turn afterwards.
        self._update_confirmation(state, user_message)

        state.add_message(Role.USER, user_message)
        state.transcript.append({"role": "user", "content": user_message})

        decision = select_model(state, user_message)
        if decision.return_intent:
            # The router found return intent in this message. Remember it, so the
            # turns spent working out which book they mean stay on Sonnet — the
            # eligibility check usually happens on one of them.
            state.return_intent_expressed = True
        model_id = self._model_id(decision)
        turn = ModelTurn(
            session_id=state.session_id,
            model_tier=decision.tier.value,
            model=model_id,
            routing_reason=decision.reason,
        )
        state.model_turns.append(turn)

        # The loop owns `state.transcript` — every assistant turn it produces,
        # including a fallback, is written there by the loop itself. Here we only
        # mirror the reply into the visible transcript.
        reply = self._run_tool_loop(state, decision, model_id, turn)
        state.add_message(Role.ASSISTANT, reply)
        return reply

    def _run_tool_loop(
        self, state: SessionState, decision: ModelDecision, model_id: str, turn: ModelTurn
    ) -> str:
        """Call Anthropic, run whatever tools it asks for, repeat until it replies.

        Args:
            state: The live session.
            decision: The routing decision for this turn.
            model_id: The model the tier resolved to.
            turn: The turn record, updated in place with counts.

        Returns:
            The final assistant text.
        """
        for _ in range(MAX_TOOL_ITERATIONS):
            turn.iterations += 1

            try:
                response = self.client.messages.create(
                    model=model_id,
                    max_tokens=MAX_TOKENS,
                    system=self.system_prompt,
                    tools=TOOL_SCHEMAS,
                    messages=state.transcript,
                    # One temperature, both tiers — the loop has a single exit to
                    # Anthropic, so there is nowhere for a turn to be sampled
                    # differently. Omitted entirely unless a deployment set one:
                    # this agent wants consistent, repeatable behaviour, and
                    # current models get there by managing their own sampling.
                    # Either way it only reaches tone — business truth and every
                    # state-changing action stay with the deterministic tools.
                    **self._sampling(),
                )
            except Exception as exc:  # noqa: BLE001 - any API failure ends the turn safely
                # Rate limit, outage, bad key, dropped connection. The customer
                # gets one honest sentence; the transcript is left as it was so
                # the next turn is not built on a half-finished exchange.
                self._trace_api_failure(state, decision, model_id, exc)
                _append_assistant_text(state, FALLBACK_UNAVAILABLE)
                return FALLBACK_UNAVAILABLE

            state.transcript.append(
                {"role": "assistant", "content": _assistant_blocks(response)}
            )

            tool_uses = [block for block in response.content if _block_type(block) == "tool_use"]
            if not tool_uses:
                text = _text_of(response)
                if text:
                    return text
                # A turn with neither text nor a tool call. Rather than leave an
                # empty assistant turn in the transcript for the next turn to
                # build on, replace it with what the customer will actually see.
                state.transcript[-1] = _assistant_text_message(FALLBACK_EMPTY)
                return FALLBACK_EMPTY

            # Every tool Claude asked for in this response runs, and all the
            # results go back in one user message — that is what the Messages
            # API expects, and splitting them teaches the model not to ask for
            # more than one at a time.
            # Several orders looked up together is the agent building a "which
            # of these did you mean?" question, so none of them becomes the
            # active order — see `_browsing_orders`.
            adopt_active_order = not _browsing_orders(tool_uses)

            results = []
            for block in tool_uses:
                turn.tool_calls += 1
                results.append(
                    self._run_one_tool(
                        state, decision, model_id, block, adopt_active_order=adopt_active_order
                    )
                )
            state.transcript.append({"role": "user", "content": results})

        # Out of iterations. Something is looping; say so rather than trying
        # again, and point at the escape hatch.
        _append_assistant_text(state, FALLBACK_STUCK)
        return FALLBACK_STUCK

    def _run_one_tool(
        self,
        state: SessionState,
        decision: ModelDecision,
        model_id: str,
        block: Any,
        *,
        adopt_active_order: bool = True,
    ) -> dict[str, Any]:
        """Run one requested tool, trace it, and update state from what it returned.

        Args:
            state: The live session.
            decision: This turn's routing decision, recorded on the trace.
            model_id: The model that asked for the call.
            block: The `tool_use` block from the response.
            adopt_active_order: Passed through to `apply_to_state`. False when
                this turn is reading out several orders rather than opening one.

        Returns:
            The `tool_result` block to send back to Anthropic.
        """
        name = getattr(block, "name", "")
        args = dict(getattr(block, "input", None) or {})

        outcome, latency_ms = invoke_timed(name, args, state, now=self.clock)

        # State is updated only from a tool that actually succeeded. A blocked
        # guard or a failed call leaves the session exactly as it was.
        if outcome.status is ToolStatus.OK:
            apply_to_state(name, args, outcome, state, adopt_active_order=adopt_active_order)

        state.tool_traces.append(
            ToolTrace(
                session_id=state.session_id,
                model=model_id,
                model_tier=decision.tier.value,
                tool_name=name or "<unnamed>",
                # The arguments as they were actually used — trusted values
                # included, so the trace shows the real call — with anything
                # sensitive masked or redacted first.
                tool_args=sanitize_args(outcome.args_used or args),
                status=outcome.status,
                latency_ms=round(latency_ms, 2),
                result_summary=outcome.summary,
                error=outcome.error,
            )
        )

        return {
            "type": "tool_result",
            "tool_use_id": getattr(block, "id", ""),
            "content": outcome.content,
            "is_error": outcome.is_error,
        }

    # --- Trusted-state helpers -------------------------------------------

    def _update_confirmation(self, state: SessionState, user_message: str) -> None:
        """Decide whether this message confirms the pending return.

        Three things must hold, and all three are checked here rather than
        anywhere the model can reach: a return is pending from a real
        eligibility check, the agent's last message asked about it, and the
        customer's reply is an affirmative. See `agent.confirmation` for why
        each one is necessary.

        A bare "yes" with nothing pending changes nothing at all — no field is
        written, and the model is left to work out what the customer meant.
        """
        pending = state.pending_return
        if pending is None:
            return

        last_assistant = state.last_assistant_message()
        if last_assistant is None:
            return

        # Did the agent actually ask? Recorded on the pending action so it
        # survives the turn and cannot be re-derived from a later message.
        if confirmation.asks_for_confirmation(last_assistant):
            pending.asked = True

        if pending.asked and confirmation.is_affirmative(user_message):
            state.confirmed = True

    def _sampling(self) -> dict[str, Any]:
        """The sampling arguments for a request — usually none at all.

        A configured temperature is passed through; an unconfigured one is not
        expressed. The distinction matters: a model that manages its own
        sampling rejects the parameter outright, and sending a default would
        fail every turn rather than quietly doing nothing.

        Returns:
            `{"temperature": value}` when a deployment set one, else `{}`.
        """
        if self.config.temperature is None:
            return {}
        return {"temperature": self.config.temperature}

    def _model_id(self, decision: ModelDecision) -> str:
        """Resolve a tier to the model id the environment configured for it."""
        return (
            self.config.sonnet_model
            if decision.tier is ModelTier.SONNET
            else self.config.haiku_model
        )

    def _trace_api_failure(
        self, state: SessionState, decision: ModelDecision, model_id: str, exc: Exception
    ) -> None:
        """Record an Anthropic failure as a trace, so the outage is visible too.

        Filed under a pseudo tool name because it is not a tool call — but a
        turn that produced nothing needs to show *why* in the same place
        everything else is shown.
        """
        state.tool_traces.append(
            ToolTrace(
                session_id=state.session_id,
                model=model_id,
                model_tier=decision.tier.value,
                tool_name="anthropic.messages.create",
                tool_args={},
                status=ToolStatus.ERROR,
                latency_ms=0.0,
                result_summary="model call failed",
                # The type and message, never the request — a request body
                # carries the transcript, and an API key lives on the client.
                error=f"{type(exc).__name__}: {exc}",
            )
        )


# --- Response helpers ----------------------------------------------------
#
# Written against the block fields rather than the SDK's classes, so the loop
# works the same on a real response and on a stand-in in a test — and so what
# goes into the transcript is plain, serializable data that a Streamlit session
# can hold across a rerun.


def _block_type(block: Any) -> str:
    """The `type` of a content block."""
    return getattr(block, "type", "")


def _browsing_orders(tool_uses: list[Any]) -> bool:
    """Whether this response is reading out several orders rather than opening one.

    A customer with two live orders gets asked which one they mean, and to ask
    that well the agent has to say something about each — so it looks both up in
    the same turn. Adopting the last one read as "the order under discussion"
    would answer the question the agent is in the middle of asking.

    Args:
        tool_uses: The `tool_use` blocks in one assistant response.

    Returns:
        True when two or more *different* orders were looked up together.
    """
    order_ids = {
        (getattr(block, "input", None) or {}).get("order_id")
        for block in tool_uses
        if getattr(block, "name", "") == "lookup_order"
    }
    return len(order_ids - {None}) > 1


def _assistant_text_message(text: str) -> dict[str, Any]:
    """A plain assistant turn, in the shape the Messages API expects."""
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def _append_assistant_text(state: SessionState, text: str) -> None:
    """Record an assistant turn the model did not produce.

    Used for the fallbacks. The customer was told something, so the transcript
    has to say so — otherwise the next turn is built on a conversation where the
    agent appears to have said nothing.
    """
    state.transcript.append(_assistant_text_message(text))


def _assistant_blocks(response: Any) -> list[dict[str, Any]]:
    """Convert an assistant response into transcript blocks.

    Keeps text and tool_use, which are what the conversation needs to continue.
    Anything else is dropped — including thinking, which is not the agent's to
    store, quote, or show.
    """
    blocks: list[dict[str, Any]] = []
    for block in response.content:
        kind = _block_type(block)
        if kind == "text":
            blocks.append({"type": "text", "text": getattr(block, "text", "")})
        elif kind == "tool_use":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "input": getattr(block, "input", None) or {},
                }
            )
    return blocks


def _text_of(response: Any) -> str:
    """Join the text blocks of a response into the customer-facing reply."""
    return "\n\n".join(
        getattr(block, "text", "")
        for block in response.content
        if _block_type(block) == "text"
    ).strip()
