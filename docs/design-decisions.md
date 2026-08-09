# Design Decisions and Tradeoffs

[← Back to README](../README.md)

Three decisions shaped this prototype more than any others.

## Decision 1: One agent with explicit tools

### Choice

Bookly runs as a single agent with a hand-rolled tool loop against the Anthropic Messages API, exposing exactly six model-callable tools. No agent framework, no planner/executor split, no second model reviewing the first one's output.

### Why

Support conversations for a bookstore fall into a handful of well-understood journeys: order status, returns, policy questions, escalation. One agent with a flat toolset gives each of those journeys a single, traceable path from customer message to tool call to response — easy to reason about, easy to debug from a trace, and easy to extend by adding a tool rather than a new agent role.

### Tradeoff

Less specialization than a multi-agent architecture would offer. There's no dedicated "policy agent" or "returns agent" that could be tuned, prompted, or evaluated independently of the rest.

### Why it was worth it

Focused customer support journeys of this size do not need multi-agent complexity to do their job well, and the complexity a multi-agent split adds — routing between agents, reconciling their state, debugging across agent boundaries — is a cost paid regardless of whether the domain needs it. One agent kept the surface small enough that the trust boundary below could be enforced everywhere, rather than at N boundaries.

## Decision 2: Natural conversation, deterministic action

### Choice

The model owns language, intent, clarification, tool selection, and explanation. Python owns identity verification, order ownership, eligibility, token issuance, confirmation, mutation, and idempotency. The two never trade places: the model cannot set a trusted field, and Python never phrases a sentence to the customer.

### Tradeoff

More explicit backend logic than a design that trusted the model's own judgment about when a return is eligible or when a customer has agreed. Every gate — is this customer verified, does this token match this exact item, did the customer actually say yes to this exact question — is written out as a Python check rather than left to a well-crafted prompt.

### Why it was worth it

It closes off the failure modes that actually matter in support: the model inventing eligibility that doesn't exist, treating an earlier "ok" as authorization for a different action, or reporting a return as opened when the write never happened. None of those are hypothetical prompt-engineering problems — they're the direct result of asking a language model to also be the system of record. Separating the two removes the question entirely: a return either passed every guard or it didn't get written, independent of what the model said.

## Decision 3: Policy as governed data in Neo4j

### Choice

Return policy — windows, regional overrides, promotional extensions, precedence between overlapping rules — is represented as nodes and relationships in Neo4j and resolved by a Cypher traversal, rather than encoded as conditionals in Python or described in the system prompt.

### Why

A graph naturally expresses categories, regional overrides, and precedence between competing policies, and it gives both the informational tool (`search_policy`) and the transactional one (`check_return_eligibility`) a single resolver to go through. Before that resolver existed as a shared function, the two tools could — and did — disagree about what applied to an Australian customer.

### Tradeoff

More infrastructure than a hardcoded if/elif chain or a paragraph of policy in the prompt: a database dependency, a seed/ingest step, and a query to reason about instead of a line of Python.

### Why it was worth it

It guarantees the same answer for "what's the policy" and "is this eligible" by construction, not by convention, and it gives every eligibility decision an explainable rule path — which graph hops produced it — rather than an assertion the customer has to take on faith.

## Design principle

> Trusted autonomy, not maximum autonomy.
