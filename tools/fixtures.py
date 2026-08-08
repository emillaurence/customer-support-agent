"""Read (and, for returns, write) the mock transactional data in `data/`.

Four small functions so the tools do not each open their own files. This is mock
data standing in for Bookly's order system: customers, orders, items, and
returns. Policy is not here — that lives in Neo4j and is reached through
`agent.graph`.

Every call re-reads from disk. Slow and deliberate: the store is a few kilobytes,
and a write from `initiate_return` is visible to the next read with no cache to
invalidate.

`DATA_DIR` is a module attribute so tests can point it at a temporary copy and
exercise the write path without touching the repo's fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.models import Customer, Item, Order, ReturnRecord

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

RETURNS_FILE = "returns.json"


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
