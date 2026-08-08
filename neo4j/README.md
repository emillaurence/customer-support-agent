# Policy graph

Neo4j is a **required dependency** for policy and return eligibility
evaluation. Return policy lookup, eligibility rules, policy overrides, regional
overrides, and explainable rule paths are all resolved by Cypher at runtime.
There is no JSON fallback: if the database is unconfigured or unreachable the
tools raise, they do not answer from a file.

Neo4j holds **only** policy and eligibility relationships. Customer, order,
item, and return records stay in [data/](../data/) as flat JSON and are never
ingested here.

Order lookup is a key-value read — a graph adds nothing. Eligibility is not:
"can this be returned" depends on the item category, the customer's region, the
delivery date, whether a promotion applied, and which of several overlapping
rules outranks the others. That is a traversal, and the precedence is data
rather than code.

[policy_graph.json](policy_graph.json) is **seed data**, not a runtime policy
store — it is the source ingestion reads, and nothing loads it while the app is
running. [ingest.py](ingest.py) writes it into Neo4j; nothing is hardcoded in
the script.

## Setup

### 1. Configure `.env`

Copy the template and fill in your instance details:

```bash
cp .env.example .env
```

`ingest.py` and the runtime tools read the same three variables:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
```

Missing any of them is a configuration error, reported as one.

### 2. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Run ingestion

```bash
python neo4j/ingest.py
```

It verifies connectivity, creates the uniqueness constraints, then MERGEs the
nodes and relationships. **Idempotent** — run it as often as you like; repeat
runs update properties in place and create nothing new.

The fixture is validated in full *before* the driver opens, so a bad edge fails
the run instead of leaving a half-populated graph. An unrecognised relationship
type is a hard error, never a silent skip.

### 4. Verify the graph

In Neo4j Browser or `cypher-shell`:

```cypher
MATCH (a)-[r]->(b)
RETURN a, r, b;
```

You should see 9 nodes (2 categories, 4 policies, 3 regions) and 8
relationships.

## Model

```
(:Category)-[:GOVERNED_BY]->(:Policy)
(:Policy)-[:OVERRIDES]->(:Policy)
(:Region)-[:HAS_OVERRIDE]->(:Policy)
```

The four edges the demo turns on:

```
(PhysicalBook)-[:GOVERNED_BY]->(STANDARD_30_DAY)
(EBook)-[:GOVERNED_BY]->(DIGITAL_NO_RETURN)
(HOLIDAY_EXTENDED_RETURN)-[:OVERRIDES]->(STANDARD_30_DAY)
(Australia)-[:HAS_OVERRIDE]->(AU_BOOKLY_EXTENDED_RETURN)
```

Regions are keyed on `code`, so Australia is `AU` in the fixture.

## Fixture sections

| Section | Label | Key | Count |
| --- | --- | --- | --- |
| `categories` | `Category` | `name` | 2 |
| `policies` | `Policy` | `policy_id` | 4 |
| `regions` | `Region` | `code` | 3 |
| `relationships` | — | `type`, `from`, `to` | 8 |

Return windows, promotions, and exceptions are properties on `Policy`
(`window_days`, `window_starts_from`, `promotion_code`,
`promotion_active_from` / `promotion_active_to`, `exceptions`) rather than
separate nodes — three node labels is enough to express the rules, and it keeps
the fixture readable.

## Seeded concepts

| Node | Label | Meaning |
| --- | --- | --- |
| `PhysicalBook` | Category | Printed books — returnable |
| `EBook` | Category | Digital downloads — not returnable |
| `STANDARD_30_DAY` | Policy | Default: 30 days from delivery |
| `DIGITAL_NO_RETURN` | Policy | Ebooks are final sale; no window |
| `HOLIDAY_EXTENDED_RETURN` | Policy | 60 days, for `MIDYEAR_HOLIDAY_SALE_2026` orders |
| `AU_BOOKLY_EXTENDED_RETURN` | Policy | Bookly's own 45-day extension for Australia |
| `AU` / `GB` / `US` | Region | Customer regions |

### A note on `AU_BOOKLY_EXTENDED_RETURN`

This is a **Bookly commercial policy**, not a legal rule. It is named for what
it is, so the agent cannot be read as advising a customer about Australian
consumer law. Statutory rights are not modelled here at all; a customer
asserting them is an escalation.

## How precedence resolves

Highest `precedence` wins, and `OVERRIDES` edges record the same ordering
explicitly so the answer can be explained rather than asserted.

| Policy | precedence |
| --- | --- |
| `DIGITAL_NO_RETURN` | 100 |
| `AU_BOOKLY_EXTENDED_RETURN` | 10 |
| `HOLIDAY_EXTENDED_RETURN` | 5 |
| `STANDARD_30_DAY` | 0 |

Two absences do real work: `DIGITAL_NO_RETURN` has no window and nothing points
at it with `OVERRIDES`, so no region or promotion can rescue an ebook. It also
lists no `exceptions`, so a faulty ebook is an escalation rather than a return.

## Worked paths

| Case | Path | Result |
| --- | --- | --- |
| GB physical, day 11 | `PhysicalBook → STANDARD_30_DAY` (30 days) | eligible |
| GB physical, day 67 | same path, past the window | not eligible |
| AU physical, day 34 | `AU → HAS_OVERRIDE → AU_BOOKLY_EXTENDED_RETURN` (45 days), overrides `STANDARD_30_DAY` | eligible |
| Sale physical, day 41 | `PhysicalBook → HOLIDAY_EXTENDED_RETURN` (60 days), overrides `STANDARD_30_DAY` | eligible |
| Any ebook, day 7 | `EBook → GOVERNED_BY → DIGITAL_NO_RETURN` (no window) | not eligible |

## Files

- [policy_graph.json](policy_graph.json) — the source fixture
- [ingest.py](ingest.py) — loads the fixture into Neo4j, idempotently
- [seed.cypher](seed.cypher) — the same model by hand, for reading; `ingest.py`
  is what you should actually run

## Not implemented yet

The eligibility queries and the agent's use of the graph. `search_policy` and
`check_return_eligibility` are stubs; when they land they will query Neo4j
through [agent/graph.py](../agent/graph.py), which raises
`PolicyGraphUnavailableError` rather than degrading. Example read queries are
commented at the end of [seed.cypher](seed.cypher).

The test suite needs no database — it exercises fixture validation and
configuration errors only. Anything that answers a policy question does.
