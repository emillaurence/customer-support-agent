"""Neo4j is mandatory for policy: the failure mode must be an error, not a guess.

These tests DO run and never connect to anything. They pin the two properties
that keep the design honest — an unconfigured database raises instead of
degrading, and no fallback switch has crept back into the repo.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent import graph
from agent.graph import PolicyGraphUnavailableError

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def unconfigured(monkeypatch: pytest.MonkeyPatch):
    """No NEO4J_* in the environment, and no .env to pick them up from."""
    monkeypatch.setattr(graph, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(graph, "_driver", None)
    for name in graph.REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)


def test_missing_configuration_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(PolicyGraphUnavailableError) as exc:
        graph.get_driver()
    assert "NEO4J_URI" in str(exc.value)


def test_partial_configuration_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Half a config is not a config — no silent default for the rest."""
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    with pytest.raises(PolicyGraphUnavailableError) as exc:
        graph.get_driver()
    assert "NEO4J_PASSWORD" in str(exc.value)


def test_unreachable_server_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A driver that cannot verify connectivity fails loudly and is closed."""
    closed = []

    class DeadDriver:
        def verify_connectivity(self) -> None:
            raise OSError("connection refused")

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(graph.GraphDatabase, "driver", lambda *a, **k: DeadDriver())
    for name in graph.REQUIRED_ENV:
        monkeypatch.setenv(name, "x")

    with pytest.raises(PolicyGraphUnavailableError, match="cannot reach Neo4j"):
        graph.get_driver()
    assert closed, "the driver must be closed when connectivity fails"
    assert graph._driver is None, "a failed connection must not be cached"


# --- No fallback switch anywhere in the repo ----------------------------

SOURCE_FILES = [
    path
    for pattern in ("*.py", "*.toml", "*.md", "*.json", "*.cypher", ".env.example")
    for path in ROOT.glob(f"**/{pattern}")
    if ".venv" not in path.parts and "__pycache__" not in path.parts and ".git" not in path.parts
]


@pytest.mark.parametrize("token", ["USE_NEO4J", "no_graph", "policies.json"])
def test_no_bypass_token_survives(token: str) -> None:
    """Ban the names the old JSON fallback went by, this test file aside."""
    offenders = [
        path.relative_to(ROOT)
        for path in SOURCE_FILES
        if path != Path(__file__) and token in path.read_text()
    ]
    assert not offenders, f"{token} still referenced in {offenders}"
