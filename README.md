# customer-support-agent

Conversational AI customer support agent with multi-turn workflows, tool use, and intent-aware responses.

This repo is the **Bookly** support agent — an online bookstore selling physical books and ebooks.

Current state: **architecture and scenarios locked, fixtures and policy model in
place, Neo4j ingestion working.** No Anthropic calls, no eligibility queries, no
business logic. Every tool is a documented stub.

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
| [tools/](tools/) | Six tool stubs with signatures and TODOs |
| [agent/graph.py](agent/graph.py) | The required Neo4j connection; raises when it is missing |
| [data/](data/) | Mock transactional JSON only: customers, orders, items, returns |
| [neo4j/](neo4j/) | Policy graph seed, ingestion script, reference Cypher |
| [tests/](tests/) | Fixture-integrity tests that run; behaviour tests skipped |

## Tools

| Tool | Purpose |
| --- | --- |
| [verify_identity](tools/verify_identity.py) | Confirm the caller matches a known customer |
| [lookup_order](tools/lookup_order.py) | Fetch an order and its line items |
| [search_policy](tools/search_policy.py) | Find the policy relevant to a question |
| [check_return_eligibility](tools/check_return_eligibility.py) | Decide if an item can be returned, and why |
| [initiate_return](tools/initiate_return.py) | Create a return record |
| [escalate_to_human](tools/escalate_to_human.py) | Hand off to a human agent |

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

The app echoes input — the agent loop is not implemented. `pytest` runs the
fixture and configuration tests, and needs no database of its own.

## Not built yet

- Anthropic API integration (the orchestrator returns a canned reply)
- Eligibility queries against Neo4j (ingestion works; `search_policy` and `check_return_eligibility` are stubs)
- All six tools — signatures and TODOs only
- Return eligibility logic, token issuing, and rule-path construction
- Guardrail enforcement in the orchestrator
