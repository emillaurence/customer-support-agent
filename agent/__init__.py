"""The Bookly support agent.

Three modules, and they are the whole thing:

* `agent.agent` — how the agent works: configuration, the system prompt, model
  routing, the confirmation gate, and the Anthropic tool loop.
* `agent.tools` — what it can do: the six model-callable tools, their schemas,
  the dispatch that injects trusted arguments, and the trusted-state updates.
* `agent.state` — what it remembers: the domain records, the execution traces,
  and `SessionState`.

Deliberately empty of imports. Dependencies run one way — `policy` ← `state` ←
`tools` ← `agent` — and a re-export here would make importing any one module
initialise all of them, reintroducing the cycle this layout removed.
"""
