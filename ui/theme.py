"""Bookly's look: a page title, an accent colour, and a little breathing room.

The colours live in `.streamlit/config.toml`, which is Streamlit's own theming
mechanism, so the palette applies to widgets and containers without a stylesheet
fighting them. What is left here is the small amount that config cannot express:
the brand header, and spacing.

Deliberately restrained. A take-home is not the place for a custom frontend, and
every line of CSS here is one that a Streamlit upgrade could quietly break — so
it does three things and stops: it composes the page's vertical rhythm, it tells
the customer's message apart from the assistant's, and it makes the trace dense
enough to read. Nothing here is a layout of its own; the widgets are Streamlit's.

Sizes are relative and no width is fixed, so a narrow window and a collapsed
sidebar both still work.
"""

from __future__ import annotations

PAGE_TITLE = "Bookly Support"
PAGE_ICON = "📚"

BRAND_NAME = "Bookly Support"
BRAND_TAGLINE = "Orders, returns, refunds, and Bookly policies."
BRAND_STATUS = "Online"
"""A one-word status beside the title. Static: the agent is up if the page is."""

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

INK = "#1C2321"
"""Body text, matching `textColor` in `.streamlit/config.toml`."""

MUTED = "#6B7280"
"""Secondary text: the tagline, the turn metadata, the trace."""

LINE = "#E7E3DB"
"""The one hairline. Warm, so it belongs to the paper rather than to a browser."""

SURFACE = "#FFFFFF"
"""The assistant's card, against the warm bubble the customer's message sits in."""

CONTENT_WIDTH = "52rem"
"""The conversation column, wide enough to read and narrow enough to stay a column.

Applied to the transcript and to the chat input together, so the box lines up
with the messages above it.
"""

CSS = f"""
<style>
  /* --- The rhythm of the page -------------------------------------------
     Streamlit leaves 6rem of air above the first element and 10rem below the
     last, and its app header — an opaque 3.75rem bar that scrolled content
     hides behind — accounts for most of the first number. The bar is kept,
     because it holds the control that reopens a collapsed sidebar; it is made
     shorter, and the conversation is brought up to just under it. */
  header[data-testid="stHeader"] {{ height: 2.5rem; min-height: 2.5rem; }}
  .stMainBlockContainer, .block-container {{
      padding-top: 2.75rem; padding-bottom: 1.5rem; max-width: {CONTENT_WIDTH};
  }}
  [data-testid="stBottomBlockContainer"] {{
      padding-top: .5rem; padding-bottom: 1.1rem; max-width: {CONTENT_WIDTH};
  }}

  /* --- The brand header ------------------------------------------------- */
  .bookly-header {{ margin: 0 0 1rem; border-bottom: 1px solid {LINE}; padding-bottom: .7rem; }}
  .bookly-header .bookly-title {{ display: flex; align-items: baseline; gap: .55rem; flex-wrap: wrap; }}
  .bookly-header h1 {{ font-size: 1.45rem; font-weight: 650; margin: 0; padding: 0; color: {INK}; letter-spacing: -.01em; }}
  .bookly-header h1 span.mark {{ color: {ACCENT}; }}
  .bookly-header p {{ margin: .2rem 0 0; color: {MUTED}; font-size: .9rem; }}
  .bookly-status {{ font-size: .72rem; font-weight: 500; letter-spacing: .02em; color: {ACCENT}; white-space: nowrap; }}
  .bookly-status::before {{ content: "●"; font-size: .55rem; vertical-align: .1em; margin-right: .28rem; }}

  /* --- The messages -----------------------------------------------------
     The customer's turn keeps the filled bubble Streamlit gives it; the
     assistant's gets a white surface and a hairline, so the two read apart
     without either becoming a box. */
  [data-testid="stChatMessage"] {{ padding: .7rem .85rem; border-radius: .7rem; }}
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{
      background: {SURFACE}; border: 1px solid {LINE};
  }}

  /* Vertical space between two stacked elements is the bottom margin of the
     paragraph above, and nothing else: Streamlit offsets every markdown block
     by -1rem and relies on the block's gap to cancel it, so touching the gap
     makes elements overlap. So the reply, its metadata line, and its trace are
     drawn closer by shortening that margin — and only on the last paragraph, so
     a structured answer keeps the space between its own paragraphs. */
  [data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] > p:last-child {{
      margin-bottom: .35rem;
  }}

  /* --- The trace ---------------------------------------------------------
     Secondary to the reply above it, and dense enough that three tool calls
     do not push the next message off the screen. */
  [data-testid="stExpander"] details {{ border-color: {LINE}; }}
  [data-testid="stExpander"] summary {{ padding: .3rem .7rem; }}
  [data-testid="stExpander"] summary p {{ font-size: .82rem; color: {MUTED}; }}
  [data-testid="stExpander"] details > div[class] {{ padding: .3rem .7rem .45rem; }}
  [data-testid="stExpander"] [data-testid="stMarkdownContainer"] > p {{ margin-bottom: .15rem; }}
  /* The one place a smaller gap is safe: a trace is captions and one-line
     headings, so there is no paragraph margin for it to eat into. */
  [data-testid="stExpander"] [class*="stVerticalBlock"] {{ gap: .7rem; }}

  /* The rule path, and only the rule path, is monospace. Wraps rather than
     scrolls, so a long traversal survives a narrow window. Its bottom margin is
     a paragraph's, because the -1rem above it is what the block expects to
     cancel — without it the last line of a trace hangs out of the card. */
  .bookly-path {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: .78rem; line-height: 1.5; color: {INK};
      overflow-wrap: anywhere; margin: .05rem 0 1rem;
  }}

  /* --- The sidebar ------------------------------------------------------
     Demo controls, not a control panel: narrower, tighter, and the headings
     sized like labels. */
  section[data-testid="stSidebar"] {{ width: 17rem !important; min-width: 17rem !important; }}
  [data-testid="stSidebarHeader"] {{ height: 2.5rem; }}
  [data-testid="stSidebarUserContent"] {{ padding-top: .5rem; }}
  [data-testid="stSidebarContent"] h3 {{
      font-size: .78rem; font-weight: 600; letter-spacing: .05em;
      text-transform: uppercase; color: {MUTED};
  }}
  [data-testid="stSidebarContent"] [class*="stVerticalBlock"] {{ gap: .6rem; }}
  [data-testid="stSidebarContent"] button {{ min-height: 2.1rem; }}
  [data-testid="stSidebarContent"] hr {{ margin: .55rem 0; border-color: {LINE}; }}
</style>
"""
