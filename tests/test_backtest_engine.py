"""
tests/test_backtest_engine.py

Tests for core/backtest_engine.py — ScreenerStrategy behaviour and
run_backtest() return structure.

Strategy tests use real Backtest runs on controlled DataFrames so the
actual bar-by-bar logic is exercised (no framework mocking).

run_backtest() integration tests mock the provider to avoid network calls.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from backtesting import Backtest

from core.backtest_engine import ScreenerStrategy, run_backtest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(
    start: str = "2024-01-01",
    periods: int = 30,
    price: float = 100.0,
    trend: float = 0.005,
) -> pd.DataFrame:
    """Gently trending daily OHLCV DataFrame with a tz-naive DatetimeIndex."""
    idx    = pd.date_range(start, periods=periods, freq="B")
    prices = [price * (1 + trend) ** i for i in range(periods)]
    return pd.DataFrame(
        {
            "Open":   [p * 0.999 for p in prices],
            "High":   [p * 1.005 for p in prices],
            "Low":    [p * 0.995 for p in prices],
            "Close":  prices,
            "Volume": [1_000_000.0] * periods,
        },
        index=idx,
    )


def _run(data, signal_dates, max_hold_days=10, size=0.95, commission=0.0):
    """Thin wrapper around Backtest.run() for strategy-logic tests."""
    bt = Backtest(data, ScreenerStrategy, cash=10_000,
                  commission=commission, finalize_trades=True)
    return bt.run(signal_dates=signal_dates, max_hold_days=max_hold_days, size=size)


# ---------------------------------------------------------------------------
# ScreenerStrategy — entry behaviour
# ---------------------------------------------------------------------------

def test_no_signals_produces_no_trades():
    data  = _make_ohlcv(periods=30)
    stats = _run(data, signal_dates={})
    assert stats["# Trades"] == 0


def test_signal_on_valid_date_produces_one_trade():
    data        = _make_ohlcv(periods=30)
    signal_date = str(data.index[5].date())
    stats       = _run(data, {signal_date: {"stop": None, "target": None}})
    assert stats["# Trades"] == 1


def test_multiple_signals_produce_multiple_trades():
    data  = _make_ohlcv(periods=60)
    # Space signals far enough apart so first trade exits before next signal fires
    d1    = str(data.index[2].date())
    d2    = str(data.index[20].date())
    stats = _run(data, {d1: {"stop": None, "target": None},
                        d2: {"stop": None, "target": None}},
                 max_hold_days=10)
    assert stats["# Trades"] == 2


def test_signal_not_in_data_produces_no_trades():
    """Signal entirely outside the OHLCV range — no trade, no error."""
    data  = _make_ohlcv(periods=20)
    stats = _run(data, {"2099-01-01": {"stop": None, "target": None}})
    assert stats["# Trades"] == 0


# ---------------------------------------------------------------------------
# run_backtest — weekend / holiday signal date remapping
# ---------------------------------------------------------------------------

def _make_ohlcv_from_dates(dates: list[str], price: float = 100.0) -> pd.DataFrame:
    """Build an OHLCV DataFrame with an explicit, controlled trading-day list."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    n   = len(idx)
    prices = [price * (1.005 ** i) for i in range(n)]
    return pd.DataFrame(
        {
            "Open":   [p * 0.999 for p in prices],
            "High":   [p * 1.010 for p in prices],
            "Low":    [p * 0.990 for p in prices],
            "Close":  prices,
            "Volume": [1_000_000.0] * n,
        },
        index=idx,
    )


@patch("core.backtest_engine._provider.get_ohlcv_range")
def test_saturday_signal_remaps_to_monday(mock_fetch):
    """Signal dated Saturday advances to the following Monday."""
    # OHLCV starts 2025-01-06 (Mon).  backtesting.py skips bar 0 in next(),
    # so the remapped date must not be bar 0 of the data.  We use Saturday
    # 2025-01-11 → Monday 2025-01-13, which is bar 5 in this OHLCV.
    trading_dates = [str(d.date()) for d in pd.date_range("2025-01-06", periods=16, freq="B")]
    data = _make_ohlcv_from_dates(trading_dates)
    mock_fetch.return_value = data

    saturday = "2025-01-11"
    assert saturday not in trading_dates
    assert "2025-01-13" in trading_dates   # Monday Jan 13 is the expected remap target

    result = run_backtest("TEST", [{"date": saturday, "stop": None, "target": None}])

    assert result["error"] is None
    assert result["n_trades"] == 1


@patch("core.backtest_engine._provider.get_ohlcv_range")
def test_sunday_signal_remaps_to_monday(mock_fetch):
    """Signal dated Sunday also advances to the following Monday."""
    trading_dates = [str(d.date()) for d in pd.date_range("2025-01-06", periods=16, freq="B")]
    data = _make_ohlcv_from_dates(trading_dates)
    mock_fetch.return_value = data

    sunday = "2025-01-12"   # → Monday 2025-01-13 (bar 5, safely past bar 0)
    assert sunday not in trading_dates

    result = run_backtest("TEST", [{"date": sunday, "stop": None, "target": None}])

    assert result["error"] is None
    assert result["n_trades"] == 1


@patch("core.backtest_engine._provider.get_ohlcv_range")
def test_market_holiday_signal_remaps_to_next_open_day(mock_fetch):
    """Signal on a market holiday (Monday missing from OHLCV) advances to Tuesday."""
    # Simulate MLK Day 2025-01-20 being absent from the trading calendar.
    # Manually build: Fri Jan 17, then Tue Jan 21 (skipping Mon Jan 20 holiday).
    trading_dates = [
        "2025-01-13", "2025-01-14", "2025-01-15", "2025-01-16", "2025-01-17",
        # 2025-01-20 missing — MLK Day
        "2025-01-21", "2025-01-22", "2025-01-23", "2025-01-24", "2025-01-27",
        "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31", "2025-02-03",
    ]
    data = _make_ohlcv_from_dates(trading_dates)
    mock_fetch.return_value = data

    holiday = "2025-01-20"
    assert holiday not in trading_dates

    result = run_backtest("TEST", [{"date": holiday, "stop": None, "target": None}])

    assert result["error"] is None
    # Remapped to 2025-01-21 (Tuesday); entry fires on 2025-01-22's open
    assert result["n_trades"] == 1


@patch("core.backtest_engine._provider.get_ohlcv_range")
def test_weekend_signal_outside_ohlcv_range_produces_no_trade(mock_fetch):
    """Weekend signal with no trading day within 7 days forward produces no trade."""
    # OHLCV has only one day — Saturday's +1…+7 window has no matching bar
    trading_dates = ["2025-01-06"]   # just Monday Jan 6; Saturday Jan 4 is 2 days before
    data = _make_ohlcv_from_dates(trading_dates)
    mock_fetch.return_value = data

    # Saturday 2025-12-27 — 7 days forward lands on Sat Jan 3 2026, none in OHLCV
    result = run_backtest("TEST", [{"date": "2025-12-27", "stop": None, "target": None}])

    assert result["n_trades"] == 0


@patch("core.backtest_engine._provider.get_ohlcv_range")
def test_saturday_and_sunday_both_map_to_same_monday(mock_fetch):
    """Sat + Sun signals for same ticker both advance to Monday — one trade results."""
    trading_dates = [str(d.date()) for d in pd.date_range("2025-01-06", periods=20, freq="B")]
    data = _make_ohlcv_from_dates(trading_dates)
    mock_fetch.return_value = data

    saturday = "2025-01-11"   # → 2025-01-13 (bar 5, past bar 0)
    sunday   = "2025-01-12"   # → 2025-01-13 (same Monday)
    signals  = [
        {"date": saturday, "stop": None, "target": None},
        {"date": sunday,   "stop": None, "target": None},
    ]

    result = run_backtest("TEST", signals)

    assert result["error"] is None
    # Both map to 2025-01-06; strategy only enters once (one position at a time)
    assert result["n_trades"] == 1


# ---------------------------------------------------------------------------
# ScreenerStrategy — exit behaviour
# ---------------------------------------------------------------------------

def test_time_exit_fires_at_max_hold_days():
    data        = _make_ohlcv(periods=40)
    signal_date = str(data.index[2].date())
    max_hold    = 7
    stats       = _run(data, {signal_date: {"stop": None, "target": None}},
                       max_hold_days=max_hold)

    trades = stats["_trades"]
    assert not trades.empty

    trade      = trades.iloc[0]
    entry_time = pd.Timestamp(trade["EntryTime"])
    exit_time  = pd.Timestamp(trade["ExitTime"])

    # Count trading bars from entry to exit (exclusive of entry bar)
    bars_held = int((data.index > entry_time).sum()) - int((data.index > exit_time).sum())
    # Allow ±1 bar tolerance for boundary handling
    assert max_hold - 1 <= bars_held <= max_hold + 1


def test_time_exit_closes_before_end_of_data():
    """Trade must not stay open all the way to the last bar when max_hold < remaining bars."""
    data        = _make_ohlcv(periods=40)
    signal_date = str(data.index[2].date())
    stats       = _run(data, {signal_date: {"stop": None, "target": None}},
                       max_hold_days=5)

    trades     = stats["_trades"]
    exit_time  = pd.Timestamp(trades.iloc[0]["ExitTime"])
    # Should exit well before the last bar
    assert exit_time < data.index[-1]


def test_stop_loss_exits_trade_early():
    """When price drops through stop, trade exits before max_hold_days.

    Note: backtesting.py skips bar 0 in next() (used as initial state).
    Signal must be placed on bar 1+ to be seen by the strategy.
    """
    # Price is steady then gaps down hard on bar 4, then recovers
    prices = [100, 101, 102, 103, 85, 87, 90, 92, 94, 96, 98, 100, 101, 102, 103]
    idx    = pd.date_range("2024-01-01", periods=len(prices), freq="B")
    data   = pd.DataFrame(
        {
            "Open":   prices,
            "High":   [p + 1 for p in prices],
            "Low":    [p - 2 for p in prices],
            "Close":  prices,
            "Volume": [1_000_000.0] * len(prices),
        },
        index=idx,
    )

    # Bar 1 is the first bar seen by next(); entry executes at bar 2's open (102)
    signal_date = str(data.index[1].date())
    stop_price  = 95.0   # sits between normal prices (96+) and the bar-4 gap (85)

    stats  = _run(data, {signal_date: {"stop": stop_price, "target": None}},
                  max_hold_days=20)
    trades = stats["_trades"]
    assert not trades.empty

    # Stop at 95 fires on bar 4 (price gaps to 85) — well before max_hold=20
    exit_time = pd.Timestamp(trades.iloc[0]["ExitTime"])
    assert exit_time < data.index[-1]


def test_take_profit_exits_trade_early():
    """When price rises through target, trade exits before max_hold_days."""
    prices = [100, 101, 103, 106, 110, 115, 120, 118, 116, 114, 112, 110, 109, 108, 107]
    idx    = pd.date_range("2024-01-01", periods=len(prices), freq="B")
    data   = pd.DataFrame(
        {
            "Open":   prices,
            "High":   [p + 2 for p in prices],
            "Low":    [p - 1 for p in prices],
            "Close":  prices,
            "Volume": [1_000_000.0] * len(prices),
        },
        index=idx,
    )

    # Bar 1 is the first bar seen by next(); entry executes at bar 2's open (103)
    signal_date   = str(data.index[1].date())
    target_price  = 112.0   # within range of the move

    stats  = _run(data, {signal_date: {"stop": None, "target": target_price}},
                  max_hold_days=20)
    trades = stats["_trades"]
    assert not trades.empty

    exit_time = pd.Timestamp(trades.iloc[0]["ExitTime"])
    assert exit_time < data.index[-1]


def test_strategy_does_not_enter_while_in_position():
    """Two overlapping signals: second should be skipped while first trade is open."""
    data  = _make_ohlcv(periods=30)
    d1    = str(data.index[2].date())
    d2    = str(data.index[4].date())   # 2 bars later — first trade still open
    stats = _run(data, {d1: {"stop": None, "target": None},
                        d2: {"stop": None, "target": None}},
                 max_hold_days=15)
    # Only one trade should have executed
    assert stats["# Trades"] == 1


# ---------------------------------------------------------------------------
# run_backtest — return structure
# ---------------------------------------------------------------------------

def _make_tz_naive_ohlcv(periods=30) -> pd.DataFrame:
    return _make_ohlcv(periods=periods)


@patch("core.backtest_engine._provider.get_ohlcv_range")
def test_run_backtest_returns_expected_keys(mock_fetch):
    mock_fetch.return_value = _make_ohlcv(periods=30)
    signals = [{"date": "2024-01-08", "stop": None, "target": None}]

    result = run_backtest("AAPL", signals)

    expected_keys = {
        "ticker", "n_signals", "n_trades",
        "return_pct", "sharpe", "max_drawdown",
        "win_rate", "avg_trade_pct",
        "trades", "equity_curve", "error",
    }
    assert expected_keys == result.keys()
    assert result["ticker"]    == "AAPL"
    assert result["n_signals"] == 1
    assert result["error"]     is None


@patch("core.backtest_engine._provider.get_ohlcv_range")
def test_run_backtest_produces_trade_and_equity_curve(mock_fetch):
    data = _make_ohlcv(periods=30)
    mock_fetch.return_value = data

    signal_date = str(data.index[5].date())
    result = run_backtest("SPY", [{"date": signal_date, "stop": None, "target": None}])

    assert result["error"] is None
    assert result["n_trades"] >= 1
    assert isinstance(result["trades"], pd.DataFrame)
    assert isinstance(result["equity_curve"], pd.DataFrame)
    assert "Equity" in result["equity_curve"].columns


@patch("core.backtest_engine._provider.get_ohlcv_range")
def test_run_backtest_empty_signals_returns_error(mock_fetch):
    result = run_backtest("AAPL", [])
    assert result["error"] == "no signals"
    assert result["n_trades"] == 0
    mock_fetch.assert_not_called()


@patch("core.backtest_engine._provider.get_ohlcv_range")
def test_run_backtest_empty_data_returns_error(mock_fetch):
    mock_fetch.return_value = pd.DataFrame()
    signals = [{"date": "2024-01-08", "stop": None, "target": None}]

    result = run_backtest("FAIL", signals)
    assert result["error"] is not None
    assert result["n_trades"] == 0


@patch("core.backtest_engine._provider.get_ohlcv_range")
def test_run_backtest_fetch_error_returns_error(mock_fetch):
    mock_fetch.side_effect = Exception("yfinance timeout")
    signals = [{"date": "2024-01-08", "stop": None, "target": None}]

    result = run_backtest("FAIL", signals)
    assert result["error"] is not None
    assert "data fetch failed" in result["error"]


@patch("core.backtest_engine._provider.get_ohlcv_range")
def test_run_backtest_strips_tz_from_index(mock_fetch):
    """Provider returns tz-aware index; engine must strip it for backtesting.py."""
    data = _make_ohlcv(periods=30)
    data.index = data.index.tz_localize("UTC")   # simulate yfinance output
    mock_fetch.return_value = data

    signal_date = str(data.index[5].date())
    result = run_backtest("SPY", [{"date": signal_date, "stop": None, "target": None}])

    assert result["error"] is None   # must not raise on tz-aware data
