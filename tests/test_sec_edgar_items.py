"""
tests/test_sec_edgar_items.py — 8-K item code → label helper.

Pure-string transformation, no network. Tests the format_8k_items helper
that powers the "highlight reel" labels on Dashboard / Telegram / Lookup.
"""

from __future__ import annotations

import pytest

from core.sec_edgar import format_8k_items


# ---------------------------------------------------------------------------
# Empty / invalid input
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty_string():
    assert format_8k_items("") == ""
    assert format_8k_items(None) == ""
    assert format_8k_items("   ") == ""


# ---------------------------------------------------------------------------
# Single-item mapping
# ---------------------------------------------------------------------------

def test_material_agreement_label():
    """1.01 is the most material item — must map to the Material Agreement label."""
    assert "Material Agreement" in format_8k_items("1.01")


def test_earnings_label():
    """2.02 = quarterly earnings."""
    assert "Earnings Results" in format_8k_items("2.02")


def test_dilution_label():
    """3.02 = unregistered equity sale = dilution signal."""
    out = format_8k_items("3.02")
    assert "Unregistered Equity Sale" in out


def test_officer_change_label():
    assert "Officer/Director Change" in format_8k_items("5.02")


def test_unknown_item_falls_back_to_raw():
    """Unknown items are shown as 'Item X.YZ' so users still see SOMETHING."""
    out = format_8k_items("99.99")
    assert "Item 99.99" in out


# ---------------------------------------------------------------------------
# Filler suppression (9.01)
# ---------------------------------------------------------------------------

def test_9_01_alone_is_shown():
    """If 9.01 is the ONLY item, we still show it — not 'empty'."""
    out = format_8k_items("9.01")
    # Either label or raw fallback is fine; the point is non-empty
    assert out != ""


def test_9_01_filler_suppressed_when_other_items_present():
    """The common '1.01,9.01' pattern should display only Material Agreement."""
    out = format_8k_items("1.01,9.01")
    assert "Material Agreement" in out
    # No reference to "9.01" or its label
    assert "9.01" not in out


def test_multiple_informative_items_joined():
    """'1.01,5.02' → both labels with the dot separator."""
    out = format_8k_items("1.01,5.02")
    assert "Material Agreement" in out
    assert "Officer/Director Change" in out
    assert " · " in out


def test_whitespace_in_input_is_tolerated():
    """SEC sometimes returns ' 1.01, 7.01 ' — must still parse."""
    out = format_8k_items(" 1.01, 7.01 ")
    assert "Material Agreement" in out
    assert "Reg FD Disclosure" in out


# ---------------------------------------------------------------------------
# Display semantics
# ---------------------------------------------------------------------------

def test_separator_is_dot_separator():
    """We use ' · ' between items so the output reads cleanly."""
    out = format_8k_items("1.01,2.02")
    assert " · " in out


def test_dilution_warning_visible():
    """A 3.02 (unregistered sale) is a real dilution signal — must surface."""
    out = format_8k_items("3.02")
    assert "Dilution" in out or "Unregistered" in out


def test_earnings_plus_filler_still_shows_earnings():
    """'2.02,9.01' — earnings buried under exhibits — still shows Earnings."""
    out = format_8k_items("2.02,9.01")
    assert "Earnings" in out
    assert "9.01" not in out
