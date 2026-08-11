"""Shared fixtures: a fixed clock, a throwaway copy of the data, an offline
policy graph, and a stand-in for Anthropic.

Four things make the suite deterministic and offline:

* **A fixed clock.** The order fixtures are written relative to 2026-08-08 —
  "day 11", "day 34", "day 67". The tools default to `datetime.now(UTC)`, and
  every test passes `FIXED_NOW` instead, so the scenarios keep meaning what they
  say however long the repo sits.
* **A copy of the data.** `data_dir` redirects the tools at a temporary copy of
  `data/`, so `initiate_return` can be tested for real without a passing test run
  leaving an RMA in the repo.
* **`seeded_graph`.** The offline stand-in for Neo4j, built from
  `neo4j/policy_graph.json` — the same seed that was ingested — so unit tests
  exercise the real decision logic without a database.
  `tests/test_neo4j_integration.py` runs the same cases against the live graph,
  which is what proves the stub is honest.
* **`FakeAnthropic`.** Replays a script of responses instead of calling the API,
  so the tool loop, the routing, the confirmation gate, and the tracing are all
  exercised with no network, no key, and no model non-determinism. What is under
  test is the agent's behaviour given a model's output — not the model.
"""

from __future__ import annotations

import copy
import json
import shutil
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from agent.agent import AnthropicConfig, BooklyAgent
from agent.state import SessionState

ROOT = Path(__file__).resolve().parent.parent
POLICY_SEED = ROOT / "neo4j" / "policy_graph.json"

FIXED_NOW = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
"""The date the order fixtures are written against."""

# --- The demo customer ---------------------------------------------------
#
# Ada (CUST-001) has two live orders: ORD-1001 holds a physical book delivered 11
# days before the fixed clock, ORD-1002 one delivered 67 days before it. The
# first is the hero return; the second is the outside-the-window case. Both come
# out of the fixtures — no test tells a tool what to decide.

HERO_EMAIL = "ada@example.com"
HERO_CUSTOMER = "CUST-001"
IN_WINDOW_ORDER, IN_WINDOW_ITEM = "ORD-1001", "ITEM-100"
EXPIRED_ORDER, EXPIRED_ITEM = "ORD-1002", "ITEM-101"

CONFIRM_QUESTION = (
    "That one can be returned. Shall I start a return for The Pragmatic Programmer "
    "on order ORD-1001?"
)
"""A cue plus a question mark, which is what `asks_for_confirmation` looks for."""


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
    from agent import tools

    copied = tmp_path / "data"
    shutil.copytree(ROOT / "data", copied)
    monkeypatch.setattr(tools, "DATA_DIR", copied)
    return copied


@pytest.fixture(autouse=True)
def clean_tokens():
    """Empty the token store around every test.

    The store is process-global, so a token minted by one test would otherwise be
    spendable in another — exactly the confusion the store exists to prevent.
    """
    from agent.tools import _clear_eligibility_tokens

    _clear_eligibility_tokens()
    yield
    _clear_eligibility_tokens()


@pytest.fixture(autouse=True)
def log_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the operational log at a throwaway file, and take any handler away
    again afterwards.

    Autouse for the same reason as `data_dir`: the shell writes to
    `logs/bookly.log` beside the code, and a test run must neither append to the
    file a demo is being read from nor leave a handler behind for the next test.
    Set through the environment, not a patched attribute — `AppTest` re-execs
    `app.py` in its own namespace, which only the environment reaches.
    """
    from agent.agent import LOG

    monkeypatch.setenv("BOOKLY_LOG_FILE", str(tmp_path / "logs" / "bookly.log"))
    before = list(LOG.handlers)
    yield
    for handler in LOG.handlers:
        if handler not in before:
            handler.close()
    LOG.handlers = before

    import app

    app.setup_logging.clear()


def returns_in(data_dir: Path) -> list[dict]:
    """Every RMA currently on disk, read from the test's temporary data copy."""
    return json.loads((data_dir / "returns.json").read_text())


# --- Offline policy graph -----------------------------------------------


def policy_rows_for_category(product_type: str) -> list[dict[str, Any]]:
    """Build what `fetch_policies_for_category` returns, from the seed file.

    Same shape as the Cypher result: the policy's properties, the regions with a
    HAS_OVERRIDE edge to it, and the policy ids it OVERRIDES, ordered by
    precedence descending.
    """
    seed = json.loads(POLICY_SEED.read_text())
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
            "policy": {k: v for k, v in policies[policy_id].items() if v is not None},
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


@pytest.fixture
def seeded_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer policy queries from the seed file instead of Neo4j.

    Patched on `policy.policy`, which is the single read of the graph, so both
    policy tools are covered and a tool that grew its own read would not be
    stubbed. Requested explicitly rather than autouse — the integration tests must
    not get it.
    """
    monkeypatch.setattr("policy.policy.fetch_policies_for_category", policy_rows_for_category)


def break_policy_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every policy read raise, as an unreachable Neo4j would.

    A plain function rather than a fixture, so a test can call it after
    `seeded_graph` has been applied and be sure it wins.
    """
    from policy.graph import PolicyGraphUnavailableError

    def unavailable(_product_type: str):
        raise PolicyGraphUnavailableError("cannot reach Neo4j at bolt://localhost:7687")

    monkeypatch.setattr("policy.policy.fetch_policies_for_category", unavailable)


# --- Offline Anthropic ---------------------------------------------------


@dataclass
class FakeBlock:
    """One content block, with just the fields the loop reads.

    Deliberately not an SDK type. The loop reads blocks by attribute rather than
    by class, so a plain object is enough — and building responses by hand is what
    lets a test say "the model asked for verify_identity, then replied".
    """

    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] = dataclass_field(default_factory=dict)


@dataclass
class FakeUsage:
    """The prompt-cache counters off a real response's `usage`.

    Only the two fields the loop reads. A real `usage` carries more, and reports
    None rather than zero when a model or account has no cache activity — which
    is why the loop reads it defensively.
    """

    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


@dataclass
class FakeResponse:
    """One Anthropic response: a list of content blocks, and optionally usage.

    `usage` defaults to None because most tests are about the loop's behaviour,
    not its accounting — and its absence also exercises the path where a response
    reports no usage at all.
    """

    content: list[FakeBlock]
    usage: FakeUsage | None = None


def text(body: str) -> FakeResponse:
    """A response that is just an assistant reply."""
    return FakeResponse(content=[FakeBlock(type="text", text=body)])


def tool_call(
    name: str, tool_input: dict[str, Any], block_id: str = "toolu_1", say: str = ""
) -> FakeResponse:
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

        The request is deep-copied because the loop hands its live transcript
        straight to the client: holding a reference would mean every recorded call
        aliased the same growing list.

        Raises:
            Exception: The configured error, if one was given — this is how the
                API-unavailable path is tested.
            AssertionError: If the loop asked for more responses than the script
                holds, which means it looped further than the test expected.
        """
        self.calls.append(copy.deepcopy(kwargs))
        if self._error is not None:
            raise self._error
        assert self._responses, "the agent asked for more model responses than the script provides"
        return self._responses.pop(0)

    def stream(self, **kwargs: Any) -> FakeMessageStream:
        """The streaming entry point, alongside `create`.

        The agent calls this one exclusively now, so it has to record the
        request and pick the next scripted response exactly as `create` does —
        it just hands the response back wrapped for incremental reading instead
        of all at once. An error is raised immediately, on `__enter__`, the same
        place the real SDK raises one: before any text could have streamed.
        """
        response = self.create(**kwargs)
        return FakeMessageStream(response)


class FakeMessageStream:
    """A stand-in for `anthropic.MessageStreamManager`.

    Offline tests have no real deltas to replay, so a scripted response's text
    is split into a few word-sized chunks — enough to prove the agent relays
    chunks as they arrive rather than buffering the whole reply — and the
    unmodified response is still what `get_final_message` returns, so tool
    execution and transcript handling see exactly what they would from a real
    stream.
    """

    def __init__(self, response: FakeResponse) -> None:
        self._response = response

    def __enter__(self) -> FakeMessageStream:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    @property
    def text_stream(self):
        for block in self._response.content:
            if getattr(block, "type", "") == "text" and block.text:
                words = block.text.split(" ")
                for index, word in enumerate(words):
                    yield word if index == len(words) - 1 else f"{word} "

    def get_final_message(self) -> FakeResponse:
        return self._response


class FakeAnthropic:
    """A stand-in Anthropic client that replays a script."""

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

    Not real model ids: the point is that the agent uses whatever the environment
    gave it, so tests assert on these strings rather than on any Claude release.
    """
    return AnthropicConfig(
        api_key="sk-ant-test-not-a-real-key",
        haiku_model="test-haiku-model",
        sonnet_model="test-sonnet-model",
    )


@pytest.fixture
def make_agent(anthropic_config: AnthropicConfig):
    """Build a `BooklyAgent` wired to a scripted client and the fixed clock."""

    def build(*responses: FakeResponse, error: Exception | None = None):
        client = FakeAnthropic(*responses, error=error)
        agent = BooklyAgent(config=anthropic_config, client=client, clock=FIXED_NOW)
        return agent, client

    return build


# --- Sessions ------------------------------------------------------------


@pytest.fixture
def verified_state() -> SessionState:
    """A session where CUST-001 has already been verified, with one order chosen.

    Saves every order test from replaying the identity turn.
    """
    return SessionState(
        verified_customer_id=HERO_CUSTOMER,
        customer_region="GB",
        active_order_ids=[IN_WINDOW_ORDER],
        active_order_id=IN_WINDOW_ORDER,
    )


@pytest.fixture
def hero_verified() -> SessionState:
    """The hero customer, verified, with neither of her two orders chosen yet."""
    return SessionState(
        verified_customer_id=HERO_CUSTOMER,
        customer_region="GB",
        active_order_ids=[IN_WINDOW_ORDER, EXPIRED_ORDER],
    )


# --- The hero conversation ----------------------------------------------


@pytest.fixture
def hero_script():
    """Anthropic's side of the whole hero conversation, in order.

    Five turns' worth of responses. The tool calls are the ones a model that read
    the schemas would make; the text is what it would say around them.
    """
    return (
        # Turn 1 — unverified, so the only thing to do is ask.
        text("Happy to check. What's the email address on your Bookly account?"),
        # Turn 2 — verify, then list the orders. Ada has two and has named
        # neither, so the list is what the clarifying question is built from.
        tool_call("verify_identity", {"email": HERO_EMAIL}),
        tool_call("lookup_order", {}, block_id="toolu_list"),
        text(
            "Thanks Ada. You've got two orders with us — one with The Pragmatic "
            "Programmer and one with Designing Data-Intensive Applications. Which "
            "one did you mean?"
        ),
        # Turn 3 — the customer picks one by title; the agent opens that order.
        tool_call("lookup_order", {"order_id": IN_WINDOW_ORDER}, block_id="toolu_pick"),
        text("The Pragmatic Programmer was delivered on 28 July and is with you."),
        # Turn 4 — intent changes; eligibility decides, and the agent asks.
        tool_call(
            "check_return_eligibility",
            {
                "order_id": IN_WINDOW_ORDER,
                "item_id": IN_WINDOW_ITEM,
                "reason": "Not what I expected.",
            },
            block_id="toolu_elig",
        ),
        text(CONFIRM_QUESTION),
        # Turn 5 — confirmed. The write happens deterministically, before this
        # response is even asked for — see `_auto_initiate_confirmed_returns` —
        # so this is only ever asked to compose the reply.
        text("Your return is open. You'll get an email with the next steps."),
    )


def run_hero_flow(agent, state: SessionState) -> None:
    """Drive the five customer turns of the hero conversation."""
    agent.respond(state, "Where's my book?")
    agent.respond(state, HERO_EMAIL)
    agent.respond(state, "The Pragmatic Programmer one")
    agent.respond(state, "Actually, I want to return it.")
    agent.respond(state, "Yes please")


def tool_names(state: SessionState) -> list[str]:
    """The tools this session has run, oldest first."""
    return [trace.tool_name for trace in state.tool_traces]


def tiers(state: SessionState) -> list[str]:
    """The model tier chosen for each turn, oldest first."""
    return [turn.model_tier for turn in state.model_turns]
