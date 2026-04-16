"""
tests/test_theme_watchlist.py

Tests for core/theme_watchlist.py — pure lookup functions only.
Session-state functions are not tested here (require Streamlit context).
"""

from __future__ import annotations

import pytest

from core.theme_watchlist import (
    AI_INFRASTRUCTURE,
    THEME_WATCHLISTS,
    all_watchlist_tickers,
    get_watchlist,
    is_on_watchlist,
)


# ---------------------------------------------------------------------------
# 1. Default watchlist contents
# ---------------------------------------------------------------------------

def test_ai_infrastructure_is_non_empty():
    assert len(AI_INFRASTRUCTURE) > 0


def test_ai_infrastructure_contains_nvda():
    assert "NVDA" in AI_INFRASTRUCTURE


def test_ai_infrastructure_contains_no_duplicates():
    assert len(AI_INFRASTRUCTURE) == len(set(AI_INFRASTRUCTURE))


def test_theme_watchlists_has_ai_infrastructure_key():
    assert "AI Infrastructure" in THEME_WATCHLISTS


# ---------------------------------------------------------------------------
# 2. get_watchlist
# ---------------------------------------------------------------------------

def test_get_watchlist_returns_list():
    result = get_watchlist("AI Infrastructure")
    assert isinstance(result, list)
    assert len(result) > 0


def test_get_watchlist_unknown_theme_returns_empty():
    result = get_watchlist("Nonexistent Theme XYZ")
    assert result == []


def test_get_watchlist_returns_copy_not_reference():
    a = get_watchlist("AI Infrastructure")
    b = get_watchlist("AI Infrastructure")
    a.append("FAKE")
    assert "FAKE" not in b


# ---------------------------------------------------------------------------
# 3. is_on_watchlist
# ---------------------------------------------------------------------------

def test_is_on_watchlist_nvda_is_true():
    assert is_on_watchlist("NVDA") is True


def test_is_on_watchlist_case_insensitive():
    assert is_on_watchlist("nvda") is True


def test_is_on_watchlist_unknown_ticker_is_false():
    assert is_on_watchlist("ZZZZNOTREAL") is False


# ---------------------------------------------------------------------------
# 4. all_watchlist_tickers
# ---------------------------------------------------------------------------

def test_all_watchlist_tickers_no_duplicates():
    tickers = all_watchlist_tickers()
    assert len(tickers) == len(set(tickers))


def test_all_watchlist_tickers_includes_nvda():
    assert "NVDA" in all_watchlist_tickers()
