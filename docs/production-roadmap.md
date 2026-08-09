# Production Roadmap

[← Back to README](../README.md)

> The prototype proves the interaction and control model. These are the first changes I would make for production.

## 1. Integrate

- Production authentication/identity — replace email-matching `verify_identity` with a real auth flow (session tokens, SSO, or an OTP step) in front of the agent.
- Real CRM/order/customer systems — replace the flat JSON fixtures in `data/` with live reads against Bookly's actual order and customer systems.
- Production policy management — replace the hand-seeded Neo4j fixture with a policy graph fed by whatever system Bookly's operations team actually uses to define return rules.
- Real human handoff — replace the mocked `escalate_to_human` case reference with a real ticket in a support platform, including the conversation context the human needs to pick it up.

Outcome: the same interaction model, running against real enterprise workflows instead of fixtures.

## 2. Harden

- Authorization / RBAC around what an agent session, and whoever operates it, can see and do.
- Durable workflow state — pending returns and eligibility tokens currently live in process memory and do not survive a restart; production needs them persisted and recoverable.
- Recovery — a partially completed multi-item return (one `initiate_return` succeeds, the next fails) needs a defined retry and reconciliation path, not just an honest error message.
- Audit — every trusted-state change and mutation should be durably logged with enough context to reconstruct a decision after the fact, beyond the current rotating operational log.
- Operational observability — metrics and alerting on top of the current Agent Trace, so a policy-graph outage or a spike in escalations is visible without reading a transcript.
- SLAs on response time and availability, once this sits in front of real customers.
- Security/privacy controls — encryption, retention limits, and access controls appropriate to real customer PII, which the current mocked data does not need.
- Policy governance — a review/approval process for changes to the policy graph, since it is now the actual source of truth for what Bookly will and won't allow.

Outcome: safe to put in front of real customers and real records.

## 3. Scale

- Additional customer journeys beyond returns, status, and policy — cancellations, exchanges, address changes, and the escalation paths each of those needs.
- More markets and regions, extending the policy graph's regional-override model rather than replacing it.
- Additional channels — email, in-app chat, voice — reusing the same tool contracts against a different front end.
- An evaluation framework — a fixed set of scenarios (like the three in the [demo guide](demo-guide.md)) run automatically against every change, so a prompt or routing edit can't silently regress a guarded path.
- Release governance — staged rollout and review for changes to the system prompt, the routing rules, or the policy graph, given how much of the system's behavior lives in each.
- Latency/cost optimization — tuning the Haiku/Sonnet split and prompt-cache usage against real traffic patterns rather than the fixed keyword lists used here.

Outcome: the same control model, repeatable across more of the business.
