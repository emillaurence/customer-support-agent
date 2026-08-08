"""Deciding whether the customer just said yes — to a specific thing.

`confirmed=True` is the last gate before Bookly's records change, so it is set
in Python, from the conversation, and never by the model asking for it.

Three conditions, all required:

1. **There is a pending action.** `SessionState.pending_return` is set only by a
   successful, eligible `check_return_eligibility` — a trusted tool result, not a
   claim in the transcript. Without one there is nothing to confirm.
2. **The agent asked.** The immediately preceding assistant message has to be a
   question about that return. A customer who says "yes" after being told their
   book is eligible has agreed with a fact, not authorised a write.
3. **The customer agreed.** The reply has to actually be an affirmative.

Miss any one and `confirmed` stays False. This matters most for the case it is
designed against: a bare "yes" arriving with no pending return — answering some
other question, or arriving from a customer who has lost the thread — must never
open a return.

And this is still only the orchestrator's gate. `initiate_return` takes
`confirmed` as an argument and refuses on its own, so a bug in this module
cannot produce a return the customer never agreed to.
"""

from __future__ import annotations

AFFIRMATIVES: frozenset[str] = frozenset(
    {
        "yes", "yes please", "yep", "yeah", "yup", "sure", "ok", "okay",
        "go ahead", "please do", "please go ahead", "do it", "confirm",
        "confirm it", "confirmed", "proceed", "proceed please", "affirmative",
        "yes do it", "yes go ahead", "yes confirm", "sounds good", "that's right",
        "thats right", "correct", "start it", "start the return", "do that",
    }
)
"""Replies that count as agreement, when everything else lines up.

Matched against the whole reply once punctuation is stripped, or against its
opening — "yes please, that one" counts; "yes, but first, what's the window?"
does not, because it opens with a question the agent has not answered.
"""

CONFIRMATION_CUES: tuple[str, ...] = (
    "shall i", "should i", "would you like me to", "do you want me to",
    "confirm", "go ahead", "is that right", "can i start", "may i start",
    "start the return", "start a return", "open the return", "open a return",
)
"""Phrases that make an assistant message a request for permission.

Paired with a question mark below. The agent is instructed to say "Shall I start
a return for <item> on <order>?" before acting, and this is what recognises that
it did — the difference between the agent stating a fact and the agent asking.
"""

_TRAILING = " .!,\t\n"


def is_affirmative(message: str) -> bool:
    """Whether a reply is agreement, on its own terms.

    Args:
        message: What the customer said.

    Returns:
        True for a plain yes. False for anything carrying a question, a
        negation, or a condition — those need another turn, not a write.
    """
    text = message.strip().lower().strip(_TRAILING)
    if not text:
        return False

    # A reply with a question in it is not a confirmation, whatever it starts
    # with. "Yes, but how long do I have?" is a question.
    if "?" in message:
        return False

    if text in AFFIRMATIVES:
        return True

    # "yes please, the paperback" — agreement plus a detail. Allowed, as long as
    # the opening is unambiguous and nothing negates it.
    if _is_negated(text):
        return False
    return any(
        text.startswith(f"{phrase} ") or text.startswith(f"{phrase},")
        for phrase in AFFIRMATIVES
    )


def asks_for_confirmation(assistant_message: str) -> bool:
    """Whether the agent's last message asked the customer to confirm an action.

    Args:
        assistant_message: The agent's most recent reply.

    Returns:
        True only for a question that is asking permission. A message that
        reports eligibility without asking returns False, so agreeing with it
        confirms nothing.
    """
    text = assistant_message.lower()
    return "?" in text and any(cue in text for cue in CONFIRMATION_CUES)


def _is_negated(text: str) -> bool:
    """Whether an otherwise-affirmative reply reverses itself.

    "yes but not that one", "ok wait" — the customer has qualified, so the agent
    needs another turn.
    """
    return any(
        marker in text
        for marker in (" not ", " don't ", " dont ", " no ", " wait", " hold on", " actually", " but ")
    )
