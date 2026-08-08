"""The presentation layer: how a turn is shown, not how it is decided.

Everything in here reads `SessionState`, `ToolTrace`, `ModelTurn`, and
`EligibilityDecision` and turns them into something a reviewer can read. Nothing
in here decides anything: no tool is called, no policy is evaluated, no model is
chosen, and no trusted field is written. `agent/` and `tools/` do not import this
package, and would work identically if it were deleted.

Three modules, split by how testable they are:

* `ui.format` — pure functions. Latency, statuses, argument display, and the
  graph rule path. Every one of them is a string in, a string out, and they are
  where the display tests live.
* `ui.turns` — the mapping from a session's flat lists of traces to the
  assistant turn that produced them. Also pure.
* `ui.render` — the Streamlit calls. Thin, because the two modules above did the
  thinking.
"""
