"""Return eligibility, including how overlapping policies resolve.

All skipped — the rules are not implemented. These cases are the reason the
policy lives in a graph. Dates assume "today" is 2026-08-08, matching the
fixtures; the real tests will inject a clock rather than rely on it.
"""

import pytest

pytestmark = pytest.mark.skip(reason="scaffold — eligibility rules not implemented yet")


def test_physical_book_within_30_days_is_eligible() -> None:
    """The default path: STANDARD_30_DAY allows it."""
    # TODO: ORD-1001 / ITEM-100 (day 11) -> eligible, policy STANDARD_30_DAY
    ...


def test_physical_book_after_window_is_not_eligible() -> None:
    """Past 30 days with no override, the return is refused."""
    # TODO: ORD-1002 / ITEM-101 (day 67) -> not eligible, explanation names
    #       the window; no eligibility_token issued
    ...


def test_ebook_is_never_eligible() -> None:
    """DIGITAL_NO_RETURN has no window at all."""
    # TODO: ORD-1004 / ITEM-200 (day 7) -> not eligible,
    #       policy_id == "DIGITAL_NO_RETURN"
    ...


def test_au_customer_gets_extended_window() -> None:
    """AU_BOOKLY_EXTENDED_RETURN outranks STANDARD_30_DAY for AU customers."""
    # TODO: ORD-1003 / ITEM-102, CUST-002 (day 34) -> eligible,
    #       policy AU_BOOKLY_EXTENDED_RETURN
    ...


def test_ebook_not_rescued_by_regional_override() -> None:
    """AU_BOOKLY_EXTENDED_RETURN attaches to PhysicalBook only."""
    # TODO: ORD-1003 / ITEM-201, CUST-002 (AU) -> not eligible
    ...


def test_promotional_order_gets_extended_window() -> None:
    """HOLIDAY_EXTENDED_RETURN applies to orders placed under the sale."""
    # TODO: ORD-1006 / ITEM-103 (day 41, MIDYEAR_HOLIDAY_SALE_2026) -> eligible,
    #       policy HOLIDAY_EXTENDED_RETURN
    ...


def test_non_promotional_order_does_not_get_extension() -> None:
    """The promotion is not a blanket 60 days for everyone."""
    # TODO: ORD-1002 has no promotion_code and stays refused at day 67
    ...


def test_in_transit_order_has_no_window_yet() -> None:
    """No delivered_at means the clock has not started."""
    # TODO: ORD-1005 -> not eligible, explanation says it has not arrived yet
    ...


def test_existing_return_blocks_a_second_one() -> None:
    """RET-5001 is already open against ORD-1007 / ITEM-100."""
    # TODO: -> not eligible even though day 19 is inside the window
    ...


def test_eligible_decision_issues_a_token() -> None:
    """A token exists only on the eligible path, and never otherwise."""
    # TODO: eligible -> eligibility_token is not None
    # TODO: not eligible -> eligibility_token is None
    ...


def test_decision_includes_an_explainable_rule_path() -> None:
    """The graph traversal is reported, not a hardcoded sentence."""
    # TODO: rule_path starts at the product type and ends at the winning policy
    #       or its window
    ...
