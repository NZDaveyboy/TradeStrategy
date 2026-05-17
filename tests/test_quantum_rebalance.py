"""
tests/test_rebalance.py — Rebalance date generation correctness.

Covers:
  - Quarterly dates are quarter-ends
  - Monthly dates are month-ends
  - Annual dates are year-ends
  - Bounds are respected (no dates outside [start, end])
  - Empty result when start > end
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.quantum.utils import quarterly_rebalance_dates


# ---------------------------------------------------------------------------
# Quarterly
# ---------------------------------------------------------------------------

def test_quarterly_returns_quarter_ends():
    """One full year → 4 quarterly rebalance dates."""
    start = pd.Timestamp("2024-01-01")
    end   = pd.Timestamp("2024-12-31")
    dates = quarterly_rebalance_dates(start, end, frequency="Q")
    assert len(dates) == 4
    # Quarter ends are 3-31, 6-30, 9-30, 12-31
    expected_months = [3, 6, 9, 12]
    for d, m in zip(dates, expected_months):
        assert d.month == m


def test_quarterly_respects_bounds_within_year():
    """Mid-year window → only the quarters that fall in range."""
    start = pd.Timestamp("2024-04-01")
    end   = pd.Timestamp("2024-09-30")
    dates = quarterly_rebalance_dates(start, end, frequency="Q")
    # Q2 end (6-30) and Q3 end (9-30) — Q1 and Q4 are outside
    assert len(dates) == 2
    assert dates[0].month == 6
    assert dates[1].month == 9


def test_quarterly_two_year_window():
    """24 months → 8 quarter-end rebalances."""
    start = pd.Timestamp("2023-01-01")
    end   = pd.Timestamp("2024-12-31")
    dates = quarterly_rebalance_dates(start, end, frequency="Q")
    assert len(dates) == 8


def test_quarterly_short_window_returns_empty():
    """Window inside a single month between quarter ends → no rebalances."""
    start = pd.Timestamp("2024-04-01")
    end   = pd.Timestamp("2024-04-30")
    dates = quarterly_rebalance_dates(start, end, frequency="Q")
    assert dates == []


# ---------------------------------------------------------------------------
# Monthly
# ---------------------------------------------------------------------------

def test_monthly_returns_month_ends():
    start = pd.Timestamp("2024-01-01")
    end   = pd.Timestamp("2024-06-30")
    dates = quarterly_rebalance_dates(start, end, frequency="M")
    assert len(dates) == 6


def test_monthly_respects_bounds():
    start = pd.Timestamp("2024-03-15")
    end   = pd.Timestamp("2024-05-15")
    dates = quarterly_rebalance_dates(start, end, frequency="M")
    # 3-31 and 4-30 fall within
    assert len(dates) == 2


# ---------------------------------------------------------------------------
# Yearly
# ---------------------------------------------------------------------------

def test_yearly_returns_year_ends():
    start = pd.Timestamp("2020-01-01")
    end   = pd.Timestamp("2024-12-31")
    dates = quarterly_rebalance_dates(start, end, frequency="Y")
    assert len(dates) == 5
    for d in dates:
        assert d.month == 12
        assert d.day   == 31


def test_yearly_partial_year_returns_empty():
    """Window that doesn't include a year-end → no yearly rebalance."""
    start = pd.Timestamp("2024-03-01")
    end   = pd.Timestamp("2024-09-30")
    dates = quarterly_rebalance_dates(start, end, frequency="Y")
    assert dates == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_invalid_frequency_raises():
    with pytest.raises(ValueError):
        quarterly_rebalance_dates(pd.Timestamp("2024-01-01"),
                                  pd.Timestamp("2024-12-31"),
                                  frequency="WEIRD")


def test_start_after_end_returns_empty():
    dates = quarterly_rebalance_dates(
        pd.Timestamp("2024-12-31"),
        pd.Timestamp("2024-01-01"),
        frequency="Q",
    )
    assert dates == []


def test_quarterly_dates_are_in_ascending_order():
    """Returned dates must be sorted ascending."""
    start = pd.Timestamp("2022-01-01")
    end   = pd.Timestamp("2024-12-31")
    dates = quarterly_rebalance_dates(start, end, frequency="Q")
    for i in range(1, len(dates)):
        assert dates[i] > dates[i - 1]
