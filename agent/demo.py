"""Putting the demo back to its starting state, so it can be rehearsed.

The hero conversation writes a real RMA into `data/returns.json`. That is the
point of it — nothing is faked — but it also means the second run of the demo is
not the first run: the return already exists, `initiate_return` reports the
existing one instead of creating it, and eligibility refuses a duplicate. Both
are correct behaviours; neither is the demo.

So there is one function that restores the starting state, and two callers: the
`scripts/reset_demo.py` command line and the Streamlit button. Neither holds any
reset logic of its own, because two implementations of "put it back" is how a
rehearsal and a live run end up starting from different places.

Three things get reset, and nothing else:

* **The mutable data.** `data/seed/` is copied back over `data/`. Only
  `returns.json` is in there; customers, orders, and items are never written, so
  restoring them would be theatre.
* **The eligibility tokens.** The store is in-memory and process-global, so a
  token minted before a reset would otherwise still be spendable after one.
* **The session**, when one is passed. A fresh `SessionState` is returned rather
  than the old one being edited, so nothing from the previous run — a verified
  customer, a selected order, a pending confirmation, a trace — survives by
  being missed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent.state import SessionState
from tools import eligibility_tokens, fixtures


class DemoReset(BaseModel):
    """What a reset actually did, for the script to print and the UI to show."""

    restored_files: list[str] = Field(
        default_factory=list, description="Mutable data files copied back from the baseline."
    )
    tokens_cleared: bool = Field(
        default=True, description="Whether the in-memory eligibility token store was emptied."
    )

    @property
    def summary(self) -> str:
        """One line describing the reset."""
        files = ", ".join(self.restored_files) if self.restored_files else "nothing"
        return f"Demo reset — restored {files}; eligibility tokens cleared."


def reset_demo() -> DemoReset:
    """Restore the mutable demo data and drop every outstanding eligibility token.

    Deterministic and idempotent: running it twice leaves exactly the state
    running it once does.

    Does not touch conversation state — a caller holding a `SessionState` should
    replace it with `fresh_session()`. Keeping the two separate means the command
    line, which has no session, and the UI, which has one, can share this.

    Returns:
        What was restored.
    """
    restored = fixtures.restore_seeded_data()
    eligibility_tokens.clear()
    return DemoReset(restored_files=restored)


def fresh_session() -> SessionState:
    """A brand-new session: no verified customer, no orders, no traces.

    A new object rather than a cleared one. `SessionState` grows fields as the
    agent does, and a reset that clears them one by one is a reset that silently
    stops being complete the next time one is added.
    """
    return SessionState()
