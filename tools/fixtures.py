"""Read (and, for returns, write) the mock transactional data in `data/`.

A handful of small functions so the tools do not each open their own files. This
is mock data standing in for Bookly's order system: customers, orders, items,
and returns. Policy is not here — that lives in Neo4j and is reached through
`agent.graph`.

Every call re-reads from disk. Slow and deliberate: the store is a few kilobytes,
and a write from `initiate_return` is visible to the next read with no cache to
invalidate.

`DATA_DIR` is a module attribute so tests can point it at a temporary copy and
exercise the write path without touching the repo's fixtures.

**Mutable versus static.** Only `returns.json` is written at runtime, so only
`returns.json` has a baseline copy in `data/seed/`. Customers, orders, and items
are never written and need no restoring. `restore_seeded_data` copies the
baseline back over the live file, which is what makes the demo repeatable — see
`agent.demo`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from agent.models import Customer, Item, Order, ReturnRecord

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

RETURNS_FILE = "returns.json"

SEED_SUBDIR = "seed"
"""Directory inside `DATA_DIR` holding the pristine copy of each mutable file.

A subdirectory of the data directory rather than a path of its own, so pointing
`DATA_DIR` at a temporary copy moves the baseline with it and a test can reset
without touching the repo.
"""


def _read(filename: str) -> list[dict]:
    """Parse one JSON array from the data directory."""
    return json.loads((DATA_DIR / filename).read_text())


def load_customers() -> list[Customer]:
    """Every customer, in file order."""
    return [Customer.model_validate(row) for row in _read("customers.json")]


def load_items() -> dict[str, Item]:
    """The catalogue, keyed by item id — line items are resolved against it."""
    return {row["item_id"]: Item.model_validate(row) for row in _read("items.json")}


def load_orders() -> list[Order]:
    """Every order, in file order."""
    return [Order.model_validate(row) for row in _read("orders.json")]


def load_returns() -> list[ReturnRecord]:
    """Every return on record, including ones opened during this session."""
    return [ReturnRecord.model_validate(row) for row in _read(RETURNS_FILE)]


def append_return(record: ReturnRecord) -> None:
    """Add one return to the store.

    Read-modify-write on a single small file. Not concurrency-safe, and not
    meant to be: one prototype process, one conversation at a time.

    Args:
        record: The return to persist.
    """
    path = DATA_DIR / RETURNS_FILE
    rows = json.loads(path.read_text())
    rows.append(json.loads(record.model_dump_json(exclude_none=True)))
    path.write_text(json.dumps(rows, indent=2) + "\n")


def seed_dir() -> Path:
    """The baseline directory for the data directory currently in use."""
    return DATA_DIR / SEED_SUBDIR


def restore_seeded_data() -> list[str]:
    """Copy every baseline file in `data/seed/` back over its live counterpart.

    The whole of the reset, and deliberately dumb: it copies files, it does not
    reconstruct JSON. Whatever `data/seed/returns.json` holds is what the demo
    starts from, so there is one place to edit the starting state.

    Idempotent — running it twice leaves the same bytes as running it once.

    Returns:
        The filenames restored, in sorted order. Empty if there is no baseline
        directory, which is not an error: nothing mutable means nothing to reset.
    """
    baseline = seed_dir()
    if not baseline.is_dir():
        return []

    restored = []
    for source in sorted(baseline.glob("*.json")):
        shutil.copyfile(source, DATA_DIR / source.name)
        restored.append(source.name)
    return restored


def next_return_id() -> str:
    """Mint the next return id in the fixture's RET-5001 series.

    Sequential rather than random so the id reads like a real RMA number and is
    easy to quote back to a customer.
    """
    numbers = [
        int(record.return_id.removeprefix("RET-"))
        for record in load_returns()
        if record.return_id.removeprefix("RET-").isdigit()
    ]
    return f"RET-{max(numbers, default=5000) + 1}"
