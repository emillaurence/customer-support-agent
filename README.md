# customer-support-agent

Conversational AI customer support agent with multi-turn workflows, tool use, and intent-aware responses.

This repo is the **Bookly** support agent — an online bookstore selling physical books and ebooks.

Current state: **the agent talks.** The six deterministic tools are wired to the
Anthropic Messages API through a hand-written tool loop, with deterministic
Haiku/Sonnet routing, an enforced confirmation gate, and tool tracing. The
Streamlit shell is still deliberately plain — the trace UI is the next phase.

**Neo4j is a required dependency for policy and return eligibility evaluation.
JSON is used only for mock transactional data.**

## Architecture

```
Customer
  └── Streamlit UI
        └── Bookly agent orchestrator
              ├── model routing (deterministic)  → Haiku | Sonnet
              ├── Anthropic Messages API         → tool selection, language
              └── tools (deterministic Python)
                    ├── JSON (mock data)  → customers, orders, items, returns
                    └── Neo4j (required)  → policy rules and eligibility relationships
```

One agent, a flat set of tools, no framework and no second agent. The division
of labour is the point:

| Claude owns | Python owns |
| --- | --- |
| Natural language, tone | Identity and ownership |
| Intent understanding | Policy truth |
| Clarifying questions | Eligibility decisions |
| Which tool to call | Eligibility tokens |
| Writing the reply | Confirmation enforcement |
| | The return write, and idempotency |

Claude can *ask* for a tool. It cannot decide an answer, and it cannot write a
trusted field. Every value a guard depends on is set by the orchestrator from a
tool result that actually succeeded.

Customer and order data live in flat JSON because it is simple record lookup. Policy lives in a
graph because eligibility is about *relationships* — which policy governs which
item category, in which region, during which promotion, and which override wins.

Policy lookup, eligibility rules, policy overrides, regional overrides, and
explainable rule paths are all decided by Cypher against Neo4j. There is no
JSON policy engine and no bypass switch: if the graph is unreachable,
[agent/graph.py](agent/graph.py) raises `PolicyGraphUnavailableError` and the
tool fails. It never degrades to a fixture or returns a mocked decision.

## Layout

| Path | What it holds |
| --- | --- |
| [app.py](app.py) | Minimal Streamlit chat shell |
| [agent/orchestrator.py](agent/orchestrator.py) | The Anthropic tool loop — one turn in, one reply out |
| [agent/routing.py](agent/routing.py) | Deterministic Haiku/Sonnet selection |
| [agent/tool_registry.py](agent/tool_registry.py) | Tool schemas, dispatch, and trusted-state updates |
| [agent/confirmation.py](agent/confirmation.py) | Whether a "yes" authorises a specific return |
| [agent/tracing.py](agent/tracing.py) | `ToolTrace` and `ModelTurn` — observable execution |
| [agent/config.py](agent/config.py) | Anthropic configuration, validated at startup |
| [agent/prompts.py](agent/prompts.py) | System prompt and guardrail list |
| [agent/state.py](agent/state.py) | `SessionState` — the only thing carried between turns |
| [agent/models.py](agent/models.py) | Typed domain models mirroring the fixtures |
| [tools/](tools/) | The six tools, plus `fixtures.py` (mock data access) and `eligibility_tokens.py` (token store) |
| [agent/graph.py](agent/graph.py) | The required Neo4j connection and the one policy query; raises when it is missing |
| [data/](data/) | Mock transactional JSON only: customers, orders, items, returns |
| [neo4j/](neo4j/) | Policy graph seed, ingestion script, reference Cypher |
| [tests/](tests/) | Unit tests for all six tools, plus a live-Neo4j integration group |

## Tools

| Tool | Purpose | Reads |
| --- | --- | --- |
| [verify_identity](tools/verify_identity.py) | Verify by email; returns `customer_id`, region, and active order ids | JSON |
| [lookup_order](tools/lookup_order.py) | Fetch one owned order, its items, and its shipment | JSON |
| [search_policy](tools/search_policy.py) | Informational policy retrieval — what the rules *are* | Neo4j |
| [check_return_eligibility](tools/check_return_eligibility.py) | The deterministic decision for one item on one order | Neo4j + JSON |
| [initiate_return](tools/initiate_return.py) | The only write. Needs a bound eligibility token *and* `confirmed=True` | JSON |
| [escalate_to_human](tools/escalate_to_human.py) | Mint a case id and hand off | — |

### The schemas are narrower than the functions

Each tool's Python signature takes everything it needs to be safe alone —
`customer_id`, `eligibility_token`, `confirmed`. The JSON schema shown to Claude
exposes only what the *customer* decides: which order, which item, why. The rest
is injected by [agent/tool_registry.py](agent/tool_registry.py) from trusted
session state.

That is the difference between a guard and a suggestion. If `confirmed` were a
schema field, a model that hallucinated `confirmed=true` would satisfy
`initiate_return`'s signature and the customer would get a return they never
asked for. It is not expressible, so it cannot be hallucinated. The same holds
for `customer_id` — an unverified session cannot produce one — and for the
eligibility token, which is also stripped from the tool result before Claude
sees it: the model learns the decision, never the credential behind it.

`search_policy` and `check_return_eligibility` are deliberately not the same tool.
The first answers "can ebooks be returned" without an order. The second answers
"can I return *this*" and is the only thing that decides, and the only thing that
issues a token.

### How eligibility picks a policy

Filter, then rank — never rank, then filter:

1. Read every policy governing the item's category from Neo4j.
2. Drop the ones whose conditions this order and customer do not satisfy. A policy
   reached through `(:Region)-[:HAS_OVERRIDE]->(:Policy)` is offered to that region
   only; a policy with a `promotion_code` is offered only to orders placed under
   that promotion, inside its active dates.
3. Let the highest-precedence survivor decide.

`AU_BOOKLY_EXTENDED_RETURN` outranks `STANDARD_30_DAY`, but a UK customer must
never be handed it. Precedence ranks what already applies; it cannot make
something apply.

### Eligibility tokens

A `uuid4`, opaque, with no meaning of its own. What it permits — customer, order,
item, and the policy that allowed it — is held server-side in
[tools/eligibility_tokens.py](tools/eligibility_tokens.py). A token is issued only
on the eligible path, and `initiate_return` refuses any request that does not
match the grant exactly. The model never generates or modifies one. No JWT: an
unguessable key into a server-side record is the simpler equivalent.

### Identity is mocked — intentionally

**Verification is email-only, and that is a deliberate take-home shortcut, not
production authentication.** A matching customer email is treated as proof of
identity. An email address is a public identifier, not a secret, so anyone who
knows a customer's address passes this check.

In production, identity would not be established by the agent at all. It would
arrive already established: an authenticated session token or user id handed to
the tools by the application, from whatever the storefront already uses. Failing
that, at minimum a one-time code sent to the address on file.

The shape is the part worth keeping, and it is production-shaped: identity is
established once, by a tool; the resulting `customer_id` is trusted session
state the model cannot write or forge; and every tool that reads customer data
demands it and re-checks ownership itself. Swapping the mock for real
authentication changes one function.

See [tools/verify_identity.py](tools/verify_identity.py).

## Model routing

Two models, chosen by a small function in
[agent/routing.py](agent/routing.py) — no classifier, no routing agent. Asking
an LLM which LLM to use costs a round trip to answer a question a boolean can
answer, and makes the decision unreproducible.

The rule: **Haiku answers, Sonnet acts.**

| Haiku | Sonnet |
| --- | --- |
| General policy questions | Return or refund intent |
| Straightforward order status | Eligibility evaluation |
| Simple factual retrieval | Ambiguity needing resolution |
| | An open return workflow |
| | Explicit confirmation |
| | State-changing actions |
| | Escalation |
| | Long multi-turn context |

Session state is checked before message text, so an open return workflow
outranks whatever the customer just typed — a mid-workflow "ok" stays on Sonnet.
Promotion is one-way within a workflow: cheap to promote, expensive to drop the
strong model halfway through a return.

Neither model id appears in the code. `ANTHROPIC_MODEL_HAIKU` and
`ANTHROPIC_MODEL_SONNET` are both required, and a test scans `agent/`, `tools/`,
and `app.py` to keep it that way.

## The agent loop

One hand-written loop in [agent/orchestrator.py](agent/orchestrator.py):

1. Read the customer's message, and decide *first* whether it confirms a pending
   return.
2. Route to Haiku or Sonnet, and record which and why.
3. Call Anthropic with the transcript, the system prompt, and the tool schemas.
4. If Claude asked for tools, run them all, trace them, update trusted state
   from the results that succeeded, feed every result back in one message, and
   loop — up to `MAX_TOOL_ITERATIONS`.
5. When Claude replies in plain text, return it.

Every failure path ends with an honest message, never a fabricated success:
an unknown tool, malformed arguments, a blocked guard, an unreachable Neo4j, an
Anthropic outage, or a loop that hits the iteration cap.

## Confirmation

`confirmed=True` requires three things, all decided in Python:

1. **A pending return exists** — set only by an eligible `check_return_eligibility`.
2. **The agent asked** — its previous message was a question requesting permission.
3. **The customer agreed** — the reply is genuinely affirmative.

A bare "yes" with nothing pending changes nothing. A "yes" agreeing with a
*statement* about eligibility changes nothing. A "yes" given for one item is
dropped the moment the customer switches to another. And `initiate_return` still
re-checks on its own, so a bug in any of this cannot produce a write.

There is no `request_clarification` tool. Clarification is just the model asking
a question in natural language — which it has to do, because when a customer has
two active orders the tools leave it nothing to guess with.

## Observability

Every tool call is recorded on the session as a `ToolTrace`: trace id, timestamp,
session id, model and tier, tool name, sanitized arguments, status, latency, a
one-line result summary, and any error. Every turn is recorded as a `ModelTurn`
with the tier, the model id, and the routing reason. Phase 6 renders them; the
loop does not know a UI exists.

Traces record **observable execution, not reasoning**. Nothing carries
chain-of-thought, the Anthropic key, the Neo4j password, or a spendable
eligibility token. Email addresses are masked to `a***@example.com`.

## Session state

Minimal by design — just what the planned flow needs:

```
verify → find the order → pick the item → check eligibility → confirm → act
```

`messages` and `transcript` (the visible conversation, and the Anthropic one
with its tool blocks), `verified_customer_id`, `customer_region`,
`active_order_ids`, `active_order_id`, `active_item_id`, `return_reason`,
`eligibility`, `eligibility_token`, `pending_return`, `confirmed`, `escalated`,
plus `tool_traces` and `model_turns`.

A write needs all three gates: identity, an eligibility token, and an explicit
confirmation. Switching order or item clears the token, so it can never be spent
on a different item.

Session state is not the safety boundary. `confirmed` is kept here for the
conversation flow, but [initiate_return](tools/initiate_return.py) takes
`eligibility_token` and `confirmed` as arguments and refuses on its own, so an
orchestrator bug cannot produce a write the customer never agreed to.

## Policy model

Held in [neo4j/](neo4j/) only — see [neo4j/README.md](neo4j/README.md).

```
(PhysicalBook)-[:GOVERNED_BY]->(STANDARD_30_DAY)
(EBook)-[:GOVERNED_BY]->(DIGITAL_NO_RETURN)
(HOLIDAY_EXTENDED_RETURN)-[:OVERRIDES]->(STANDARD_30_DAY)
(Australia)-[:HAS_OVERRIDE]->(AU_BOOKLY_EXTENDED_RETURN)
```

`AU_BOOKLY_EXTENDED_RETURN` is a Bookly commercial policy, deliberately not
framed as an Australian legal right. Statutory-rights questions are escalations.

[neo4j/policy_graph.json](neo4j/policy_graph.json) is seed data for ingestion
only — it is not read at runtime. Load it with `python neo4j/ingest.py`
(idempotent), and do that before anything asks a policy question.

## Scenarios

The fixtures seed one deliberate case per behaviour worth testing. Dates assume
today is **2026-08-08**.

| Scenario | Customer | Order | Item | Expected |
| --- | --- | --- | --- | --- |
| Two active orders | CUST-003 | ORD-1004, ORD-1005 | — | Ask which order |
| Physical book in window (day 11) | CUST-001 | ORD-1001 | ITEM-100 | Eligible, `STANDARD_30_DAY` |
| Physical book out of window (day 67) | CUST-001 | ORD-1002 | ITEM-101 | Not eligible |
| Ebook (day 7) | CUST-003 | ORD-1004 | ITEM-200 | Not eligible, `DIGITAL_NO_RETURN` |
| Promotional extension (day 41) | CUST-004 | ORD-1006 | ITEM-103 | Eligible, `HOLIDAY_EXTENDED_RETURN` |
| AU override (day 34) | CUST-002 | ORD-1003 | ITEM-102 | Eligible, `AU_BOOKLY_EXTENDED_RETURN` |
| Ebook not rescued by AU override | CUST-002 | ORD-1003 | ITEM-201 | Not eligible |
| Order in transit | CUST-003 | ORD-1005 | ITEM-101 | No window started |
| Existing RMA (RET-5001) | CUST-002 | ORD-1007 | ITEM-100 | Refuse duplicate |
| Someone else's order | CUST-004 | ORD-1008 | ITEM-101 | Nothing returned to CUST-001 |

## Running

### 1. Configure `.env`

```bash
cp .env.example .env
```

Fill in all six. Every one is required.

```
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL_HAIKU=
ANTHROPIC_MODEL_SONNET=

NEO4J_URI=
NEO4J_USERNAME=
NEO4J_PASSWORD=
```

**Both model names**, because the agent routes between them — an agent with only
the cheap one configured would silently run the consequential turns on the wrong
model, so it refuses to start instead. **All three Neo4j values**, because policy
decisions have nowhere else to come from.

No model id is hardcoded in the code. Swapping models is an `.env` edit.

### 2. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Ingest the policy graph

```bash
python neo4j/ingest.py
```

Idempotent. Run it again whenever `neo4j/policy_graph.json` changes.

### 4. Run the app

```bash
streamlit run app.py
```

The chat works end to end. The UI is still plain — the agent trace is Phase 6 —
but the sidebar shows the verified customer, the return workflow, and which
model handled the last turn and why.

The tools still work without Anthropic: they are ordinary Python functions, and
that is what the Phase 3 tests exercise.

### 5. Run the tests

```bash
pytest                    # everything; the integration group skips if Neo4j is down
pytest -m "not integration"   # unit tests only, no database needed
pytest -m integration         # against the real seeded graph
```

The agent tests stub Anthropic as well as Neo4j: `FakeAnthropic` in
[tests/conftest.py](tests/conftest.py) replays a scripted sequence of responses,
so the tool loop, the routing, the confirmation gate, and the tracing are all
exercised with no network, no API key, and no model non-determinism. What is
under test is the orchestrator's behaviour given a model's output — not the
model.

Unit tests stub the policy graph from `neo4j/policy_graph.json` — the same seed
that was ingested — so they run offline. The integration group takes no stub and
re-checks the same decisions against the live database, which is what keeps the
stub honest. Every test measures dates against a fixed 2026-08-08 clock, so the
fixture scenarios keep meaning what they say; the tools themselves default to
`datetime.now(UTC)`.

Tests never write to `data/` — they run against a temporary copy, so exercising
the real `initiate_return` write leaves no RMA behind.

## Not built yet

- The polished Streamlit UI and the agent trace rendering (the data is captured)
- The hero demo conversation
- Streaming replies, and prompt caching of the system prompt and tool schemas
- Voice, long-term memory, and a model-evaluation dashboard
- Production authentication — see below
