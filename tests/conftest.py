"""Shared test fixtures.

Two things make the tool tests deterministic:

* **A fixed clock.** The order fixtures are written relative to 2026-08-08 —
  "day 11", "day 34", "day 67". The tools default to `datetime.now(UTC)`, and
  every test passes `FIXED_NOW` instead, so the scenarios keep meaning what they
  say however long the repo sits.
* **A copy of the data.** `data_dir` redirects the tools at a temporary copy of
  `data/`, so `initiate_return` can be tested for real without a passing test run
  leaving an RMA in the repo.

For the Phase 4 agent tests there is a third: **a stand-in for Anthropic.**
`FakeAnthropic` replays a script of responses instead of calling the API, so the
tool loop, the routing, the confirmation gate, and the tracing are all exercised
end to end with no network, no key, and no model non-determinism. What is being
tested is the orchestrator's behaviour given a model's output — not the model.

`seeded_graph` is the offline stand-in for Neo4j. It answers the one policy query
the tools make, built from `neo4j/policy_graph.json` — the same seed that was
ingested into the database — so unit tests exercise the real decision logic
without a database. `tests/test_neo4j_integration.py` runs the same cases against
the live graph, which is what proves the stub is honest.
"""

from __future__ import annotations

import copy
import importlib
import json
import shutil
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from agent.config import AnthropicConfig
from agent.orchestrator import BooklyAgent
from agent.state import SessionState

ROOT = Path(__file__).resolve().parent.parent
POLICY_SEED = ROOT / "neo4j" / "policy_graph.json"

FIXED_NOW = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
"""The date the order fixtures are written against."""


@pytest.fixture
def now() -> datetime:
    """The fixed clock, for tests that pass it through to a tool."""
    return FIXED_NOW


@pytest.fixture(autouse=True)
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the tools at a throwaway copy of `data/`.

    Autouse: no test should be able to write to the repo's fixtures, and a test
    that mutates returns.json must not change what the next test reads.
    """
    from tools import fixtures

    copy = tmp_path / "data"
    shutil.copytree(ROOT / "data", copy)
    monkeypatch.setattr(fixtures, "DATA_DIR", copy)
    return copy


@pytest.fixture(autouse=True)
def clean_tokens():
    """Empty the token store around every test.

    The store is process-global, so a token minted by one test would otherwise be
    spendable in another — exactly the confusion the store exists to prevent.
    """
    from tools import eligibility_tokens

    eligibility_tokens.clear()
    yield
    eligibility_tokens.clear()


# --- Offline policy graph -----------------------------------------------


def _seed() -> dict[str, Any]:
    return json.loads(POLICY_SEED.read_text())


def policy_rows_for_category(product_type: str) -> list[dict[str, Any]]:
    """Build what `fetch_policies_for_category` returns, from the seed file.

    Same shape as the Cypher result: the policy's properties, the regions with a
    HAS_OVERRIDE edge to it, and the policy ids it OVERRIDES, ordered by
    precedence descending.

    Args:
        product_type: A category name.

    Returns:
        One row per policy governing that category.
    """
    seed = _seed()
    policies = {policy["policy_id"]: policy for policy in seed["policies"]}
    relationships = seed["relationships"]

    governed = [
        rel["to"]
        for rel in relationships
        if rel["type"] == "GOVERNED_BY" and rel["from"] == product_type
    ]

    rows = [
        {
            "category": product_type,
            # Nulls dropped, because Neo4j does not store null properties: a row
            # off the real graph has the key missing, not set to None. Matching
            # that keeps the stub honest about the shape the tools handle.
            "policy": {
                key: value for key, value in policies[policy_id].items() if value is not None
            },
            "granted_to_regions": [
                rel["from"]
                for rel in relationships
                if rel["type"] == "HAS_OVERRIDE" and rel["to"] == policy_id
            ],
            "outranks": [
                rel["to"]
                for rel in relationships
                if rel["type"] == "OVERRIDES" and rel["from"] == policy_id
            ],
        }
        for policy_id in governed
    ]
    rows.sort(key=lambda row: (-row["policy"]["precedence"], row["policy"]["policy_id"]))
    return rows


def _policy_reading_modules() -> list[Any]:
    """The modules holding a reference to `fetch_policies_for_category`.

    One, now: `tools.policy_rules` is the single read of the policy graph, and
    both policy tools go through it. Patching there covers both, and a tool that
    grew its own graph read would no longer be stubbed — which is the point.

    Fetched from `sys.modules` by name. `tools/__init__` re-exports each tool
    *function* under its module's name, so `tools.search_policy` is the function
    and a plain `import ... as` would hand back the wrong object.
    """
    return [importlib.import_module("tools.policy_rules")]


def break_policy_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every policy read raise, as an unreachable Neo4j would.

    A plain function rather than a fixture, so a test can call it after
    `seeded_graph` has been applied and be sure it wins.

    Args:
        monkeypatch: The calling test's monkeypatch.
    """
    from agent.graph import PolicyGraphUnavailableError

    def unavailable(_product_type: str):
        raise PolicyGraphUnavailableError("cannot reach Neo4j at bolt://localhost:7687")

    for module in _policy_reading_modules():
        monkeypatch.setattr(module, "fetch_policies_for_category", unavailable)


# --- Offline Anthropic ---------------------------------------------------


@dataclass
class FakeBlock:
    """One content block, with just the fields the orchestrator reads.

    Deliberately not an SDK type. The orchestrator reads blocks by attribute
    rather than by class, so a plain object is enough — and building responses
    by hand is what lets a test say "the model asked for verify_identity, then
    replied" without a network call.
    """

    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] = dataclass_field(default_factory=dict)


@dataclass
class FakeResponse:
    """One Anthropic response: a list of content blocks."""

    content: list[FakeBlock]


def text(body: str) -> FakeResponse:
    """A response that is just an assistant reply."""
    return FakeResponse(content=[FakeBlock(type="text", text=body)])


def tool_call(name: str, tool_input: dict[str, Any], block_id: str = "toolu_1", say: str = "") -> FakeResponse:
    """A response asking for one tool, optionally with a line of text first."""
    blocks = [FakeBlock(type="text", text=say)] if say else []
    blocks.append(FakeBlock(type="tool_use", id=block_id, name=name, input=tool_input))
    return FakeResponse(content=blocks)


def tool_calls(*calls: tuple[str, dict[str, Any]]) -> FakeResponse:
    """A response asking for several tools at once, as the API allows."""
    return FakeResponse(
        content=[
            FakeBlock(type="tool_use", id=f"toolu_{index}", name=name, input=args)
            for index, (name, args) in enumerate(calls)
        ]
    )


class FakeMessages:
    """The `client.messages` namespace: hands back the next scripted response."""

    def __init__(self, responses: list[FakeResponse], error: Exception | None = None) -> None:
        self._responses = list(responses)
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        """Record the request and return the next scripted response.

        The request is deep-copied before it is recorded. The orchestrator hands
        its live transcript straight to the client, so holding a reference would
        mean every recorded call aliased the same growing list and a test could
        only ever inspect the final state.

        Raises:
            Exception: The configured error, if one was given — this is how the
                API-unavailable path is tested.
            AssertionError: If the loop asked for more responses than the script
                holds, which means the orchestrator looped further than the test
                expected.
        """
        self.calls.append(copy.deepcopy(kwargs))
        if self._error is not None:
            raise self._error
        assert self._responses, "the agent asked for more model responses than the script provides"
        return self._responses.pop(0)


class FakeAnthropic:
    """A stand-in Anthropic client that replays a script.

    Exposes the one method the orchestrator uses, plus the recorded calls so a
    test can assert on which model was used and what was sent.
    """

    def __init__(self, *responses: FakeResponse, error: Exception | None = None) -> None:
        self.messages = FakeMessages(list(responses), error)

    @property
    def calls(self) -> list[dict[str, Any]]:
        """Every request the agent made, in order."""
        return self.messages.calls

    @property
    def models_used(self) -> list[str]:
        """The model id on each request, in order."""
        return [call["model"] for call in self.messages.calls]


@pytest.fixture
def anthropic_config() -> AnthropicConfig:
    """Configuration with recognisable placeholder model names.

    Not real model ids: the point is that the orchestrator uses whatever the
    environment gave it, so the test asserts on these strings rather than on any
    particular Claude release.
    """
    return AnthropicConfig(
        api_key="sk-ant-test-not-a-real-key",
        haiku_model="test-haiku-model",
        sonnet_model="test-sonnet-model",
    )


@pytest.fixture
def make_agent(anthropic_config: AnthropicConfig):
    """Build a `BooklyAgent` wired to a scripted client and the fixed clock."""

    def build(*responses: FakeResponse, error: Exception | None = None) -> tuple[Any, FakeAnthropic]:
        client = FakeAnthropic(*responses, error=error)
        agent = BooklyAgent(config=anthropic_config, client=client, clock=FIXED_NOW)
        return agent, client

    return build


@pytest.fixture
def verified_state() -> SessionState:
    """A session where CUST-001 has already been verified.

    Saves every order test from replaying the identity turn. CUST-001 has one
    active order, so nothing here is ambiguous.
    """
    return SessionState(
        verified_customer_id="CUST-001",
        customer_region="GB",
        active_order_ids=["ORD-1001"],
        active_order_id="ORD-1001",
    )


@pytest.fixture
def seeded_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer policy queries from the seed file instead of Neo4j.

    Patched on the two tools that read the graph, at the name each of them
    imported, so nothing reaches a driver. Requested explicitly rather than
    autouse — the integration tests must not get it.
    """
    for module in _policy_reading_modules():
        monkeypatch.setattr(module, "fetch_policies_for_category", policy_rows_for_category)
