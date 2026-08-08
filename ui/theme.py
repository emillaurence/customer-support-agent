"""Bookly's look: a page title, an accent colour, and a little breathing room.

The colours live in `.streamlit/config.toml`, which is Streamlit's own theming
mechanism, so the palette applies to widgets and containers without a stylesheet
fighting them. What is left here is the small amount that config cannot express:
the brand header, and spacing.

Deliberately short. A take-home is not the place for a custom frontend, and every
line of CSS here is one that a Streamlit upgrade could quietly break — so there
is only enough to make the chat look considered.
"""

from __future__ import annotations

PAGE_TITLE = "Bookly Support"
PAGE_ICON = "📚"

BRAND_NAME = "Bookly Support"
BRAND_TAGLINE = "Orders, returns, refunds, and Bookly policies."

WELCOME_MESSAGE = (
    "Hi, I'm Bookly Support. I can help with orders, returns, refunds, and Bookly policies."
)
"""Shown above an empty conversation. Static text, not a turn.

It is not written to `SessionState`, so it is not in the transcript the model
sees, does not count as an assistant message for the confirmation check, and
does not shift the router's turn count.
"""

CHAT_PLACEHOLDER = "Message Bookly Support…"

ACCENT = "#1F5C4F"
"""The one accent colour, matching `primaryColor` in `.streamlit/config.toml`."""

CSS = f"""
<style>
  /* Give the conversation the width of the page and a calmer top margin. */
  .block-container {{ padding-top: 2.2rem; max-width: 62rem; }}

  /* The brand header: a rule in the accent colour, and space under it. */
  .bookly-header {{ margin-bottom: 1.4rem; border-bottom: 1px solid #E7E3DB; padding-bottom: .9rem; }}
  .bookly-header h1 {{ font-size: 1.65rem; font-weight: 650; margin: 0; color: #1C2321; letter-spacing: -.01em; }}
  .bookly-header h1 span.mark {{ color: {ACCENT}; }}
  .bookly-header p {{ margin: .25rem 0 0; color: #6B7280; font-size: .92rem; }}

  /* The rule path, and only the rule path, is monospace. */
  .bookly-path {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: .82rem; line-height: 1.55; color: #1C2321;
      white-space: pre; margin: .1rem 0 .4rem;
  }}

  /* The trace sits under the reply, so it should read as secondary to it. */
  div[data-testid="stExpander"] summary p {{ font-size: .86rem; color: #6B7280; }}
</style>
"""
