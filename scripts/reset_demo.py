#!/usr/bin/env python
"""Put the demo back to its starting state.

    python scripts/reset_demo.py

Restores `data/returns.json` from `data/seed/`, and drops any eligibility tokens
this process is holding. Safe to run at any point, and safe to run twice.

All the work is in `agent.demo.reset_demo`, which the Streamlit "Reset demo"
button calls too — this file is only the command line around it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.demo import reset_demo  # noqa: E402 - after the path fix above


def main() -> int:
    """Reset, and say what was reset."""
    result = reset_demo()
    print(result.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
