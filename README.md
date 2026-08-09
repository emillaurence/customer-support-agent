# customer-support-agent

Conversational AI customer support agent with multi-turn workflows, tool use, and intent-aware responses.

This repo is the **Bookly** support agent — an online bookstore selling physical books and ebooks.

Current state: **the agent handles a whole customer journey, and shows its
work.** The six deterministic tools are wired to the Anthropic Messages API
through a hand-written tool loop, with deterministic Haiku/Sonnet routing, an
enforced confirmation gate, and tool tracing. A customer can ask where their book
is, be verified, choose between two orders, change their mind and ask for a
return, and get a real RMA — in one session, with nothing scripted. See
[Demo](#demo). Under each reply the Streamlit UI shows which model handled the
turn and a collapsed **Agent trace** for that turn: the tools it ran, in order,
with real latencies, and the policy and graph path behind any eligibility
decision. See [The UI](#the-ui).

**Neo4j is a required dependency for policy and return eligibility evaluation.
JSON is used only for mock transactional data.**

## Architecture

```
Customer
  └── Streamlit UI                      app.py + ui.py
        └── Bookly agent                agent/agent.py
              ├── model routing (deterministic)  → Haiku | Sonnet
              ├── Anthropic Messages API         → tool selection, language
              ├── trusted session state          agent/state.py
              └── six tools (deterministic)      agent/tools.py
                    ├── JSON (mock data)  → customers, orders, items, returns
                    └── policy layer      policy/policy.py + policy/graph.py
                          └── Neo4j (required)  → policy rules, regional overrides,
                                                  precedence, rule paths
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
trusted field. Every value a guard depends on is set by the agent loop from a
tool result that actually succeeded.

Customer and order data live in flat JSON because it is simple record lookup. Policy lives in a
graph because eligibility is about *relationships* — which policy governs which
item category, in which region, during which promotion, and which override wins.

Policy lookup, eligibility rules, policy overrides, regional overrides, and
explainable rule paths are all decided by Cypher against Neo4j. There is no
JSON policy engine and no bypass switch: if the graph is unreachable,
[policy/graph.py](policy/graph.py) raises `PolicyGraphUnavailableError` and the
tool fails. It never degrades to a fixture or returns a mocked decision.

## Layout

Six files hold the whole system. Read them in this order.

| Path | What it holds |
| --- | --- |
| [agent/agent.py](agent/agent.py) | **How the agent works.** Configuration, the system prompt, deterministic Haiku/Sonnet routing, the confirmation gate, and the Anthropic tool loop |
| [agent/tools.py](agent/tools.py) | **What it can do.** The six tools, their schemas, the dispatch that injects trusted arguments, the trusted-state updates, and the mock-data and token helpers behind them |
| [agent/state.py](agent/state.py) | **What it remembers.** `SessionState`, the domain records, and `ToolTrace` / `ModelTurn` with their sanitizing |
| [policy/policy.py](policy/policy.py) | **Where business policy lives.** Region normalization, applicability, precedence, rule paths, and `search_policy` — the one selection both policy tools read |
| [policy/graph.py](policy/graph.py) | The required Neo4j connection and the one policy query; raises when it is missing |
| [ui.py](ui.py) | **How the demo is rendered.** Theme, trace formatting, trace-to-turn mapping, and the Streamlit calls |
| [app.py](app.py) | The Streamlit entry point — transcript in, one turn out; also where logging is configured |
| [.streamlit/config.toml](.streamlit/config.toml) | The palette, as Streamlit's own theming |
| [scripts/reset_demo.py](scripts/reset_demo.py) | `python scripts/reset_demo.py` — the command line around `agent.tools.reset_demo` |
| [data/](data/) | Mock transactional JSON only: customers, orders, items, returns |
| [data/seed/](data/seed/) | Pristine copies of the mutable files, for the reset to restore |
| [neo4j/](neo4j/) | Policy graph seed, ingestion script, reference Cypher |
| [tests/](tests/) | Five suites by behaviour, plus a live-Neo4j integration group |

## Tools

All six are in [agent/tools.py](agent/tools.py), in this order.

| Tool | Purpose | Reads |
| --- | --- | --- |
| `verify_identity` | Verify by email; returns `customer_id`, first name, and region. Identity only | JSON |
| `lookup_order` | Without `order_id`, list the customer's orders and the items on each; with one, that order, its items, and its shipment | JSON |
| `search_policy` | Informational policy retrieval — what the rules *are* (body in [policy/policy.py](policy/policy.py)) | Neo4j |
| `check_return_eligibility` | The deterministic decision for one item on one order | Neo4j + JSON |
| `initiate_return` | The only write. Needs a bound eligibility token *and* `confirmed=True` | JSON |
| `escalate_to_human` | Mint a case id and hand off | — |

### The schemas are narrower than the functions

Each tool's Python signature takes everything it needs to be safe alone —
`customer_id`, `eligibility_token`, `confirmed`. The JSON schema shown to Claude
exposes only what the *customer* decides: which order, which item, why. The rest
is injected by [agent/tools.py](agent/tools.py) from trusted
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

They are, however, the same *policy selection*. Region applicability, category
applicability, precedence, and overrides all come from
[policy/policy.py](policy/policy.py), which both tools read; the tools
differ only in presentation — one describes, one decides. Two implementations of
"which policy governs Australia" is two answers, and one of them is wrong.

### Which region a policy question is about

Fixed precedence, resolved in Python before the tool runs:

1. a region named in the **current question** — "for Australian customers" → `AU`;
2. the **verified customer's** region, from trusted session state;
3. no region at all, i.e. global policy context.

Session state is a fallback, never an override: a verified UK customer asking
about Australian policy is asking about Australia. Names and codes are normalized
from a table in `policy/policy.py` (`Australia`/`Australian`/`AU` → `AU`,
`United Kingdom`/`UK`/`GB` → `GB`), so the country code is never something the
model invents.

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
[agent/tools.py](agent/tools.py). A token is issued only
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

See `verify_identity` in [agent/tools.py](agent/tools.py).

## Model routing

Two models, chosen by a small function in
[agent/agent.py](agent/agent.py) — no classifier, no routing agent. Asking
an LLM which LLM to use costs a round trip to answer a question a boolean can
answer, and makes the decision unreproducible.

The rule: **Haiku answers, Sonnet acts.**

| Haiku | Sonnet |
| --- | --- |
| Informational policy lookups | A customer's own return or refund |
| Straightforward order status | Eligibility evaluation |
| Simple factual retrieval | Ambiguity needing resolution |
| | An open return workflow |
| | Explicit confirmation |
| | State-changing actions |
| | Requests to depart from policy |
| | Escalation |
| | Long multi-turn context |

The word "return" is not the signal — the *subject* is. "What is the return
policy for Australian customers?" and "Can I return my order?" share a keyword
and nothing else: the first is a policy lookup with no customer in it and goes to
Haiku, the second is an eligibility question about a specific record and goes to
Sonnet. First-person phrasing, "this order", an order id, an instruction to act,
or a request for an exception each veto the cheaper model on their own. The trace
says which rule fired — *informational policy lookup* or *return or refund
workflow*.

Session state is checked before message text, so an open return workflow
outranks whatever the customer just typed — a mid-workflow "ok" stays on Sonnet.
Promotion is one-way within a workflow: cheap to promote, expensive to drop the
strong model halfway through a return.

A return workflow opens the moment the customer *asks* for one, not once an item
has been picked. Answering "which one?" with a book title carries no return
vocabulary at all, and that is usually the turn the eligibility check runs on —
so the intent is recorded on the session (`return_intent_expressed`) and cleared
when the return context is.

Neither model id appears in the code. `ANTHROPIC_MODEL_HAIKU` and
`ANTHROPIC_MODEL_SONNET` are both required, and a test scans `agent/`, `policy/`,
`app.py`, and `ui.py` to keep it that way.

## The agent loop

One hand-written loop in [agent/agent.py](agent/agent.py):

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
with the tier, the model id, and the routing reason. [ui.py](ui.py) renders them; the
loop does not know a UI exists.

Traces record **observable execution, not reasoning**. Nothing carries
chain-of-thought, the Anthropic key, the Neo4j password, or a spendable
eligibility token. Email addresses are masked to `a***@example.com`.

Beside the trace, and separate from it, is an operational log: `logs/bookly.log`,
written by Python's `logging` through a `RotatingFileHandler` (1 MB, three
backups) that [app.py](app.py) installs at startup. Startup, Neo4j warm-up, the
tier and reason each turn routed on, and one line per tool call — its status,
latency, sanitized arguments, and the policy, RMA, or case id it produced. It
reads the *same* sanitized values as the trace, so nothing lands there that the
trace would not show either. The trace is what a reviewer reads about one
conversation; the log is what an operator greps afterwards. The UI never reads
it, and it is git-ignored.

```
2026-08-09 10:25:31 INFO agent session=SESS-7B8DD378 model=sonnet route="return or refund workflow"
2026-08-09 10:25:32 INFO tool name=verify_identity status=success latency_ms=0.2 args={'email': 'b***@example.com'}
2026-08-09 10:25:32 INFO tool name=lookup_order status=success latency_ms=0.2 args={'customer_id': 'CUST-002'}
2026-08-09 10:25:32 INFO tool name=check_return_eligibility status=success latency_ms=34.2 args={'order_id': 'ORD-1003', 'item_id': 'ITEM-201', 'customer_id': 'CUST-002'} policy=DIGITAL_NO_RETURN eligible=false
```

## The UI

The conversation is the page. Under each assistant reply there is one quiet line
saying which model handled the turn, what the turn was, and how much ran — and one
collapsed **Agent trace** describing *that* reply:

```
[Sonnet]  Return workflow · 1 tool

▸ Agent trace
    Model · Sonnet · claude-sonnet-5
    Routing · return or refund workflow

    1 · check_return_eligibility  [✓ Success]  390 ms
    eligible=True, policy_id=STANDARD_30_DAY · order_id=ORD-1001, item_id=ITEM-100
    Policy · STANDARD_30_DAY · Decision · Eligible
    Rule path
      PhysicalBook → STANDARD_30_DAY
```

The line outside the trace is written for reading — *Policy lookup*, *Order
lookup*, *Return workflow*. The router's own deterministic reason is not
paraphrased away: it is printed verbatim inside the trace, next to the model id
that served the turn.

Per turn, not one list at the bottom of the page: the point is to connect an
action to the reply that produced it. Tools appear in execution order, and every
latency shown is the one the loop measured around the call — nothing here is generated for
the demo.

The rule path is the graph traversal the decision came from: the item's category,
the region or promotion that made a conditional policy apply, the policy that
won, and the one it outranked. Showing it is what distinguishes a deterministic
evaluation from a model's opinion. The Cypher behind it is not shown — the
traversal is the explanation, the query is an implementation detail.

**What the UI does not show.** No chain-of-thought — none is captured anywhere in
the repo, so there is none to render. No spendable eligibility token: the
developer view reports one as `held`, never its value. No API key, no Neo4j
password, no environment. Customer emails are masked in the trace. And no stack
traces: an Anthropic outage, an unreachable policy graph, and a failed tool each
reach the customer as one sentence, with the technical detail left in the trace.

Model routing is visible but quiet — a small badge, never louder than the answer.
The tier shown is whatever `select_model` in [agent/agent.py](agent/agent.py) decided; the UI
displays the decision and does not influence it.

Beside the chat, in the sidebar, a collapsed **Developer state** expander shows
the session's current trusted fields — verified customer, region, active order and
item, return reason, eligibility, whether a token is held, whether a confirmation
is outstanding, confirmed, escalated. Separate from the trace, and collapsed,
because it is a debugging view rather than part of the customer's experience.

`ui.py` holds no business logic. It calls no tool, evaluates no policy, chooses no
model, and writes no trusted field; `agent/` and `policy/` do not import it.
`ui.capture_turn` slices the session's flat trace list into the turn that
produced it, which is a presentation mapping rather than a change to the loop.

## Session state

Minimal by design — just what the planned flow needs:

```
verify → find the order → pick the item → check eligibility → confirm → act
```

`messages` and `transcript` (the visible conversation, and the Anthropic one
with its tool blocks), `verified_customer_id`, `customer_region`,
`active_order_ids`, `active_order_id`, `active_item_id`, `return_reason`,
`return_intent_expressed`, `eligibility`, `eligibility_token`, `pending_return`,
`confirmed`, `escalated`, plus `tool_traces` and `model_turns`.

A write needs all three gates: identity, an eligibility token, and an explicit
confirmation. Switching order or item clears the token, so it can never be spent
on a different item.

Session state is not the safety boundary. `confirmed` is kept here for the
conversation flow, but `initiate_return` takes `eligibility_token` and
`confirmed` as arguments and refuses on its own, so a bug in the loop cannot
produce a write the customer never agreed to.

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
| Two active orders | CUST-001, CUST-003 | — | — | Ask which order |
| Physical book in window (day 11) | CUST-001 | ORD-1001 | ITEM-100 | Eligible, `STANDARD_30_DAY` |
| Physical book out of window (day 67) | CUST-001 | ORD-1002 | ITEM-101 | Not eligible |
| Ebook (day 7) | CUST-003 | ORD-1004 | ITEM-200 | Not eligible, `DIGITAL_NO_RETURN` |
| Promotional extension (day 41) | CUST-004 | ORD-1006 | ITEM-103 | Eligible, `HOLIDAY_EXTENDED_RETURN` |
| AU override (day 34) | CUST-002 | ORD-1003 | ITEM-102 | Eligible, `AU_BOOKLY_EXTENDED_RETURN` |
| Ebook not rescued by AU override | CUST-002 | ORD-1003 | ITEM-201 | Not eligible |
| Order in transit | CUST-003 | ORD-1005 | ITEM-101 | No window started |
| Existing RMA (RET-5001) | CUST-002 | ORD-1007 | ITEM-100 | Refuse duplicate |
| Someone else's order | CUST-004 | ORD-1008 | ITEM-101 | Nothing returned to CUST-001 |

## Demo

**Ada (`ada@example.com`, CUST-001) is the demo customer.** She has two live
orders, one book inside its return window and one long outside it — so the same
identity carries both the hero journey and the case where the answer is no.
Nothing in the code knows that. She is a row in `data/customers.json`, and the
two outcomes fall out of the dates on her orders.

### The hero journey

One session, five turns, no script:

| Turn | The customer | What happens | Model |
| --- | --- | --- | --- |
| 1 | "Where's my book?" | Not verified, so the agent asks for the email | Haiku |
| 2 | "ada@example.com" | `verify_identity`, then `lookup_order` with no order id → two live orders → *which one?* | Haiku |
| 3 | "The Pragmatic Programmer one" | Selected by title; `lookup_order` reads that order's status out | Haiku |
| 4 | "Actually, I want to return it." | `check_return_eligibility` → eligible → *shall I start it?* | Sonnet |
| 5 | "Yes please" | `initiate_return` → a real RMA in `data/returns.json` | Sonnet |

    verify_identity → lookup_order → check_return_eligibility → initiate_return

A customer who names their book skips the question: if the listing resolves the
title to exactly one item, the agent goes straight to eligibility rather than
asking them to confirm a choice they already made, and does not re-read the
order for detail nobody asked for. Ambiguity is what gets a question.

The wording is not load-bearing. "Can you check my delivery?", "I haven't
received my book yet", "Can I send this back?", "Can I get a refund?", "Go
ahead" all reach the same places — there is no phrase table anywhere, and
[tests/test_agent.py](tests/test_agent.py) holds the alternatives to
keep it that way.

Three things are worth watching in the traces as it runs:

* **The agent does not guess.** Two live orders means `active_order_id` stays
  empty until Ada chooses one. Reading both out to build the question is
  browsing, not selecting — see `_browsing_orders` in
  [agent/agent.py](agent/agent.py).
* **Nothing is asked twice.** `verify_identity` runs once. Changing intent from
  status to return re-uses the verified customer, the region, and the chosen
  order, and a reason given at the eligibility step is still there at the write.
* **The router moves on its own.** Turns 1–3 are retrieval. Turn 4 says
  "return", and from there the session stays on Sonnet until the RMA exists.

### The other ending

Same customer, same tools, different fixture dates:

> "I want to return a book" → "ada@example.com" → "Designing Data-Intensive
> Applications"

Delivered 67 days ago, so `check_return_eligibility` refuses. No token is
minted, so there is nothing for a "yes" to authorise, and `initiate_return`
would refuse on its own if the model asked for it anyway. The agent explains why
and offers a colleague. `data/returns.json` is untouched.

The UI needs no special case for this: it is the same trace, saying no. The
policy is named, the same rule path is shown, and the decision reads
**Not eligible** — which is what makes it clear the refusal was evaluated rather
than improvised.

### Running it again

```bash
python scripts/reset_demo.py
```

Restores `data/returns.json` from `data/seed/` and drops any eligibility tokens
the process is holding. Idempotent. The **Reset demo** button in the Streamlit
sidebar calls the same `agent.tools.reset_demo` function and additionally
clears the conversation, the session state, and the traces — two copies of "put
it back" is how a rehearsal and a live run end up starting from different
places.

Only `returns.json` is restored, because it is the only file anything writes.
Customers, orders, and items are read-only, and a reset that rewrote them would
be theatre.

## Running

### 1. Configure `.env`

```bash
cp .env.example .env
```

Fill in six of the seven. `ANTHROPIC_TEMPERATURE` is the only optional one, and
the only one to leave blank.

```
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL_HAIKU=
ANTHROPIC_MODEL_SONNET=
ANTHROPIC_TEMPERATURE=

NEO4J_URI=
NEO4J_USERNAME=
NEO4J_PASSWORD=
```

**Both model names**, because the agent routes between them — an agent with only
the cheap one configured would silently run the consequential turns on the wrong
model, so it refuses to start instead. **All three Neo4j values**, because policy
decisions have nowhere else to come from.

**`ANTHROPIC_TEMPERATURE` is unset by default, and should stay that way.** A
support agent should answer the same question the same way, and a demo should
behave the same way twice — but the way to get that on a current model is to
send nothing. Sonnet 5 and the rest of the Claude 5 family manage their own
sampling internally and reject an explicit `temperature` with a 400, so a pinned
value would fail every turn rather than making anything more deterministic. When
the variable is blank the parameter is not sent at all.

The setting exists because it is a deployment choice: a deployment running a
model that does accept a temperature can set one without touching the loop. It
then applies to **both tiers** — one value, never one per model, because there
is a single exit from the loop to Anthropic. Either way it reaches only tone;
business truth and every state-changing action stay with the deterministic tools
and the guards in front of them. Set to something that is not a number, it is a
startup error rather than a silent fallback.

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

The chat works end to end, with a per-turn agent trace under each reply — see
[The UI](#the-ui).

Two resets in the sidebar, and they are not the same thing. **Reset conversation**
forgets the conversation; the RMA it created is still a real record.
**Reset demo** also restores `data/returns.json` from `data/seed/` and clears the
eligibility tokens, which is what makes the [demo](#demo) rehearsable. It calls
the same `agent.tools.reset_demo` as `python scripts/reset_demo.py`, rather than a
second copy of the logic.

The tools still work without Anthropic: they are ordinary Python functions, and
that is what [tests/test_tools.py](tests/test_tools.py) exercises.

### 5. Run the tests

```bash
pytest                    # everything; the integration group skips if Neo4j is down
pytest -m "not integration"   # unit tests only, no database needed
pytest -m integration         # against the real seeded graph
```

Five suites, split by behaviour rather than by module:

| Suite | What it covers |
| --- | --- |
| [tests/test_agent.py](tests/test_agent.py) | Configuration, model routing, the tool loop, multi-turn state, tracing, and the hero journey end to end |
| [tests/test_tools.py](tests/test_tools.py) | The six callable tools, identity and order lookup, the write, and the integrity of the mock data |
| [tests/test_policy.py](tests/test_policy.py) | Regional applicability, AU consistency, precedence, ebooks, promotions, the Neo4j requirement, and the graph seed |
| [tests/test_guardrails.py](tests/test_guardrails.py) | Ownership, eligibility tokens, confirmation, idempotency, escalation, and the outside-the-window ending |
| [tests/test_ui.py](tests/test_ui.py) | Formatting, trace sanitization, trace-to-turn association, reset, and `app.py` itself |

The agent tests stub Anthropic as well as Neo4j: `FakeAnthropic` in
[tests/conftest.py](tests/conftest.py) replays a scripted sequence of responses,
so the tool loop, the routing, the confirmation gate, and the tracing are all
exercised with no network, no API key, and no model non-determinism. What is
under test is the agent's behaviour given a model's output — not the
model.

The UI is tested too, at the level worth testing. The display helpers and the
trace-to-turn mapping in [ui.py](ui.py) are pure functions, so they are checked
directly; the same suite then drives `app.py` itself through Streamlit's own
`AppTest` — in process, no browser, no server — typing the hero flow into the chat
box and asserting on what appears under each reply, on both reset buttons, and on
the absence of an email address or a token in the trace. No browser automation.

Unit tests stub the policy graph from `neo4j/policy_graph.json` — the same seed
that was ingested — so they run offline. The integration group takes no stub and
re-checks the same decisions against the live database, which is what keeps the
stub honest. Every test measures dates against a fixed 2026-08-08 clock, so the
fixture scenarios keep meaning what they say; the tools themselves default to
`datetime.now(UTC)`.

Tests never write to `data/` — they run against a temporary copy, so exercising
the real `initiate_return` write leaves no RMA behind.

## Not built yet

- Streaming replies, and prompt caching of the system prompt and tool schemas
- Voice, long-term memory, and a model-evaluation dashboard
- Production authentication — see below
