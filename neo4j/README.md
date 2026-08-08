# Policy graph

Neo4j holds **only** return policy and eligibility relationships. Customers,
orders, items, and returns stay in [data/](../data/) as flat JSON.

Nothing in this directory is connected yet. [policy_graph.json](policy_graph.json)
is the fixture the tools will read; [seed.cypher](seed.cypher) is the same model
expressed for a real instance. Keep the two in step.

## Why split it this way

Order lookup is a key-value read — a graph adds nothing. Eligibility is not:
"can this be returned" depends on the product category, the customer's region,
the delivery date, whether a promotion was running, and which of several
overlapping rules outranks the others. That is a traversal, and the precedence
is data rather than code.

## What lives in the graph

- item categories (`PhysicalBook`, `EBook`)
- return policies
- return windows
- exceptions
- promotions
- regional overrides
- the explainable path behind an eligibility answer

## Model

```
(:ProductType)-[:GOVERNED_BY]->(:Policy)
(:Policy)-[:HAS_WINDOW]->(:ReturnWindow)
(:Policy)-[:OVERRIDES]->(:Policy)
(:Region)-[:HAS_OVERRIDE]->(:Policy)
(:Promotion)-[:GRANTS]->(:Policy)
(:Exception)-[:WAIVES]->(:Policy)
```

The four relationships that carry the demo:

```
(PhysicalBook)-[:GOVERNED_BY]->(STANDARD_30_DAY)
(EBook)-[:GOVERNED_BY]->(DIGITAL_NO_RETURN)
(HOLIDAY_EXTENDED_RETURN)-[:OVERRIDES]->(STANDARD_30_DAY)
(Australia)-[:HAS_OVERRIDE]->(AU_BOOKLY_EXTENDED_RETURN)
```

Labels: `Policy`, plus `RegionalPolicy` or `PromotionalPolicy` where a rule is
scoped to a place or to a sale.

## Seeded concepts

| Node | Label | Meaning |
| --- | --- | --- |
| `PhysicalBook` | ProductType | Printed books — returnable |
| `EBook` | ProductType | Digital downloads — not returnable |
| `STANDARD_30_DAY` | Policy | Default: 30 days from delivery |
| `DIGITAL_NO_RETURN` | Policy | Ebooks are final sale |
| `HOLIDAY_EXTENDED_RETURN` | PromotionalPolicy | 60 days, granted by a holiday sale |
| `AU_BOOKLY_EXTENDED_RETURN` | RegionalPolicy | Bookly's own 45-day extension for Australia |
| `WINDOW_30_DAY` / `WINDOW_45_DAY` / `WINDOW_60_DAY` | ReturnWindow | Days from `delivered_at` |
| `MIDYEAR_HOLIDAY_SALE_2026` | Promotion | Active 2026-06-15 → 2026-07-15 |
| `DAMAGED_ON_ARRIVAL` | Exception | Waives the window; needs human review |

### A note on `AU_BOOKLY_EXTENDED_RETURN`

This is a **Bookly commercial policy**, not a legal rule. It is named for what
it is so the agent cannot be read as advising a customer about Australian
consumer law. Australian statutory rights are not modelled here at all; a
customer asserting them is an escalation.

## How precedence resolves

Highest `precedence` wins, and `OVERRIDES` edges record the same ordering
explicitly so the answer can be explained rather than asserted.

| Policy | precedence |
| --- | --- |
| `DIGITAL_NO_RETURN` | 100 |
| `AU_BOOKLY_EXTENDED_RETURN` | 10 |
| `HOLIDAY_EXTENDED_RETURN` | 5 |
| `STANDARD_30_DAY` | 0 |

Two absences do real work:

- `DIGITAL_NO_RETURN` has **no** `HAS_WINDOW` edge, and **nothing** points at it
  with `OVERRIDES`. There is no window to extend, so no region or promotion can
  rescue an ebook.
- `DAMAGED_ON_ARRIVAL` does not `WAIVE` `DIGITAL_NO_RETURN`. A faulty ebook is
  an escalation, not a return.

## Worked paths

| Case | Path | Result |
| --- | --- | --- |
| GB physical, day 11 | `PhysicalBook → STANDARD_30_DAY → WINDOW_30_DAY` | eligible |
| GB physical, day 67 | same path, past 30 days | not eligible |
| AU physical, day 34 | `AU → HAS_OVERRIDE → AU_BOOKLY_EXTENDED_RETURN → WINDOW_45_DAY`, overrides `STANDARD_30_DAY` | eligible |
| Sale physical, day 41 | `MIDYEAR_HOLIDAY_SALE_2026 → GRANTS → HOLIDAY_EXTENDED_RETURN → WINDOW_60_DAY`, overrides `STANDARD_30_DAY` | eligible |
| Any ebook, day 7 | `EBook → GOVERNED_BY → DIGITAL_NO_RETURN` (no window) | not eligible |

## Running it, later

```bash
# not wired up yet
cypher-shell -u neo4j -p "$NEO4J_PASSWORD" -f neo4j/seed.cypher
```

The repo runs without Neo4j. `search_policy` will read
[policy_graph.json](policy_graph.json) first; swapping in Cypher must not change
its signature.
