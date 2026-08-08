# customer-support-agent

Conversational AI customer support agent with multi-turn workflows, tool use, and intent-aware responses.

This repo is the **Bookly** support agent — an online bookstore selling physical books and ebooks.

Current state: **all six tools implemented and tested as ordinary Python
functions.** Identity, order lookup, policy retrieval, eligibility, the guarded
return write, and escalation all work end to end against Neo4j and the mock data.
No Anthropic calls yet — the orchestrator still returns a canned reply.

**Neo4j is a required dependency for policy and return eligibility evaluation.
JSON is used only for mock transactional data.**

## Architecture

```
Streamlit UI
  └── single Bookly agent (orchestrator + system prompt)
        └── tools
              ├── JSON (mock data)  → customers, orders, items, returns
              └── Neo4j (required)  → policy rules and eligibility relationships
```

One agent, a flat set of tools, no service layer and no framework. Customer and
order data live in flat JSON because it is simple record lookup. Policy lives in a
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
| [agent/orchestrator.py](agent/orchestrator.py) | The agent loop (stub — returns a canned reply) |
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

### Identity is mocked

A matching customer email is treated as verification. That is a prototype
shortcut, not authentication — an email address is a public identifier. The shape
worth keeping is that identity is established once, by a tool, and every tool that
reads customer data demands the resulting `customer_id` and re-checks ownership
itself. See [tools/verify_identity.py](tools/verify_identity.py).

## Session state

Minimal by design — just what the planned flow needs:

```
verify → find the order → pick the item → check eligibility → confirm → act
```

`messages`, `verified_customer_id`, `customer_region`, `active_order_id`,
`active_item_id`, `return_reason`, `eligibility`, `eligibility_token`,
`confirmed`, `escalated`.

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

Fill in `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, and the three `NEO4J_*`
variables. All of the Neo4j ones are required — policy decisions have nowhere
else to come from.

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

The app echoes input — the agent loop is not implemented. The tools work without
it, and without Anthropic.

### 5. Run the tests

```bash
pytest                    # everything; the integration group skips if Neo4j is down
pytest -m "not integration"   # unit tests only, no database needed
pytest -m integration         # against the real seeded graph
```

Unit tests stub the policy graph from `neo4j/policy_graph.json` — the same seed
that was ingested — so they run offline. The integration group takes no stub and
re-checks the same decisions against the live database, which is what keeps the
stub honest. Every test measures dates against a fixed 2026-08-08 clock, so the
fixture scenarios keep meaning what they say; the tools themselves default to
`datetime.now(UTC)`.

Tests never write to `data/` — they run against a temporary copy, so exercising
the real `initiate_return` write leaves no RMA behind.

## Not built yet

- Anthropic API integration (the orchestrator returns a canned reply)
- The tool-calling loop, and Streamlit wired to tool execution
- Guardrail enforcement in the orchestrator (the tools enforce their own)
- The hero conversation and the agent trace UI
- Production authentication, and long-term memory
