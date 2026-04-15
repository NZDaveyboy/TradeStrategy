"""
tests/test_analytics.py

Tests for core/analytics.py.

All tests use hand-built DataFrames — no DB or network access.
load_v2_data() is tested against a real in-memory SQLite DB.
"""

from __future__ import annotations

import math
import sqlite3
import tempfile
import os

import pandas as pd
import pytest

from core.analytics import (
    equity_curve,
    load_v2_data,
    performance_by_score_bucket,
    portfolio_stats,
    win_rate_by_setup,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from a list of dicts."""
    return pd.DataFrame(rows)


def _base_row(**kwargs) -> dict:
    """Minimal backtest_v2-style row with sensible defaults."""
    defaults = {
        "ticker":        "TEST",
        "n_signals":     3,
        "n_trades":      3,
        "return_pct":    10.0,
        "sharpe":        1.2,
        "max_drawdown":  -5.0,
        "win_rate":      66.7,
        "avg_trade_pct": 3.3,
        "error":         None,
    }
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# equity_curve
# ---------------------------------------------------------------------------

def test_equity_curve_starts_at_first_compounded_value():
    df = _df([_base_row(ticker="A", return_pct=10.0),
              _base_row(ticker="B", return_pct=20.0)])
    ec = equity_curve(df)
    assert len(ec) == 2
    assert pytest.approx(ec.iloc[0], rel=1e-6) == 1.10
    assert pytest.approx(ec.iloc[1], rel=1e-6) == 1.10 * 1.20


def test_equity_curve_empty_df():
    assert equity_curve(pd.DataFrame()).empty


def test_equity_curve_single_row():
    df = _df([_base_row(return_pct=5.0)])
    ec = equity_curve(df)
    assert len(ec) == 1
    assert pytest.approx(ec.iloc[0], rel=1e-6) == 1.05


def test_equity_curve_negative_returns_decay():
    df = _df([_base_row(ticker="A", return_pct=-10.0),
              _base_row(ticker="B", return_pct=-10.0)])
    ec = equity_curve(df)
    assert ec.iloc[-1] < 1.0   # overall loss


def test_equity_curve_sorted_by_run_at_when_present():
    df = _df([
        {**_base_row(ticker="A", return_pct=50.0), "run_at": "2024-01-20 00:00 UTC"},
        {**_base_row(ticker="B", return_pct=10.0), "run_at": "2024-01-01 00:00 UTC"},
    ])
    ec = equity_curve(df)
    # B (10%) should come first (earlier run_at), then A (50%)
    assert pytest.approx(ec.iloc[0], rel=1e-6) == 1.10
    assert pytest.approx(ec.iloc[1], rel=1e-6) == 1.10 * 1.50


# ---------------------------------------------------------------------------
# portfolio_stats
# ---------------------------------------------------------------------------

def test_portfolio_stats_empty_df():
    stats = portfolio_stats(pd.DataFrame())
    assert stats["total_trades"] == 0
    assert math.isnan(stats["win_rate"])


def test_portfolio_stats_all_winners():
    df = _df([
        _base_row(ticker="A", return_pct=10.0, n_trades=5),
        _base_row(ticker="B", return_pct=20.0, n_trades=3),
        _base_row(ticker="C", return_pct= 5.0, n_trades=2),
    ])
    stats = portfolio_stats(df)
    assert stats["total_trades"] == 10
    assert stats["n_tickers"]    == 3
    assert stats["win_rate"]     == pytest.approx(100.0, abs=0.1)
    assert stats["avg_return"]   > 0
    assert stats["sharpe"]       > 0       # all positive → Sharpe > 0
    assert math.isnan(stats["sortino"])    # no losses → NaN (no downside)


def test_portfolio_stats_mixed_returns():
    df = _df([
        _base_row(ticker="A", return_pct= 20.0, n_trades=4),
        _base_row(ticker="B", return_pct=-10.0, n_trades=2),
        _base_row(ticker="C", return_pct= 15.0, n_trades=3),
        _base_row(ticker="D", return_pct= -5.0, n_trades=1),
    ])
    stats = portfolio_stats(df)
    assert stats["win_rate"]   == pytest.approx(50.0, abs=0.1)
    assert stats["avg_return"] > 0           # net positive
    assert not math.isnan(stats["sharpe"])
    assert not math.isnan(stats["sortino"])
    assert stats["max_drawdown"] <= 0        # always ≤ 0


def test_portfolio_stats_all_losers():
    df = _df([
        _base_row(ticker="A", return_pct=-10.0, n_trades=3),
        _base_row(ticker="B", return_pct=-20.0, n_trades=2),
    ])
    stats = portfolio_stats(df)
    assert stats["win_rate"]     == pytest.approx(0.0, abs=0.1)
    assert stats["avg_return"]   < 0
    assert stats["max_drawdown"] < 0


def test_portfolio_stats_max_drawdown_is_negative_or_zero():
    df = _df([_base_row(ticker=str(i), return_pct=float(i * 5))
              for i in range(1, 5)])
    stats = portfolio_stats(df)
    assert stats["max_drawdown"] <= 0.0


# ---------------------------------------------------------------------------
# win_rate_by_setup
# ---------------------------------------------------------------------------

def _setup_df() -> pd.DataFrame:
    return _df([
        {**_base_row(ticker="A", return_pct= 20.0), "setup_type": "Early breakout"},
        {**_base_row(ticker="B", return_pct=-10.0), "setup_type": "Early breakout"},
        {**_base_row(ticker="C", return_pct= 15.0), "setup_type": "Early breakout"},
        {**_base_row(ticker="D", return_pct=  5.0), "setup_type": "Momentum continuation"},
        {**_base_row(ticker="E", return_pct= -8.0), "setup_type": "Momentum continuation"},
    ])


def test_win_rate_by_setup_returns_expected_setups():
    result = win_rate_by_setup(_setup_df())
    assert set(result["Setup Type"]) == {"Early breakout", "Momentum continuation"}


def test_win_rate_by_setup_correct_win_rate():
    result = win_rate_by_setup(_setup_df())
    eb = result[result["Setup Type"] == "Early breakout"].iloc[0]
    assert eb["Win Rate %"] == pytest.approx(66.7, abs=0.5)   # 2/3


def test_win_rate_by_setup_sorted_descending():
    result = win_rate_by_setup(_setup_df())
    rates = result["Win Rate %"].tolist()
    assert rates == sorted(rates, reverse=True)


def test_win_rate_by_setup_empty_df():
    assert win_rate_by_setup(pd.DataFrame()).empty


def test_win_rate_by_setup_no_setup_type_column():
    df = _df([_base_row()])
    assert win_rate_by_setup(df).empty


def test_win_rate_by_setup_all_null_setup_type():
    df = _df([{**_base_row(), "setup_type": None},
              {**_base_row(ticker="B"), "setup_type": None}])
    assert win_rate_by_setup(df).empty


# ---------------------------------------------------------------------------
# performance_by_score_bucket
# ---------------------------------------------------------------------------

def _bucket_df() -> pd.DataFrame:
    return _df([
        {**_base_row(ticker="A", return_pct= 30.0, n_trades=5), "avg_tradescore": 75.0},
        {**_base_row(ticker="B", return_pct= 10.0, n_trades=3), "avg_tradescore": 82.0},
        {**_base_row(ticker="C", return_pct=-15.0, n_trades=2), "avg_tradescore": 55.0},
        {**_base_row(ticker="D", return_pct=  8.0, n_trades=4), "avg_tradescore": 45.0},
        {**_base_row(ticker="E", return_pct= -5.0, n_trades=1), "avg_tradescore": 20.0},
    ])


def test_score_bucket_returns_expected_buckets():
    result = performance_by_score_bucket(_bucket_df())
    labels = set(result["Score Bucket"])
    # Scores: 75→60-80, 82→80-100, 55→40-60, 45→40-60, 20→0-40
    assert "60\u201380" in labels
    assert "80\u2013100" in labels
    assert "40\u201360" in labels
    assert "0\u201340" in labels


def test_score_bucket_win_rate_high_bucket():
    result = performance_by_score_bucket(_bucket_df())
    high = result[result["Score Bucket"] == "80\u2013100"].iloc[0]
    assert high["Win Rate %"] == pytest.approx(100.0, abs=0.1)  # only B (positive)


def test_score_bucket_empty_df():
    assert performance_by_score_bucket(pd.DataFrame()).empty


def test_score_bucket_no_tradescore_column():
    df = _df([_base_row()])
    assert performance_by_score_bucket(df).empty


def test_score_bucket_all_null_tradescore():
    df = _df([{**_base_row(), "avg_tradescore": float("nan")}])
    assert performance_by_score_bucket(df).empty


# ---------------------------------------------------------------------------
# load_v2_data — in-memory SQLite integration test
# ---------------------------------------------------------------------------

def _make_test_db() -> str:
    """Write a minimal backtest_v2 + results DB to a temp file, return path."""
    tf = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tf.close()
    conn = sqlite3.connect(tf.name)
    conn.execute("""
        CREATE TABLE backtest_v2 (
            ticker TEXT PRIMARY KEY, n_signals INT, n_trades INT,
            return_pct REAL, sharpe REAL, max_drawdown REAL,
            win_rate REAL, avg_trade_pct REAL, commission REAL,
            max_hold_days INT, error TEXT, run_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE results (
            run_date TEXT, ticker TEXT, strategy TEXT, asset TEXT,
            price REAL, change_pct REAL, rvol REAL, ema9 REAL,
            ema20 REAL, ema200 REAL, rsi REAL, atr REAL,
            stop_loss REAL, macd REAL, macd_signal REAL, vwap REAL,
            volume_trend_up INT, score INT, market_cap REAL,
            float_shares REAL, tradescore REAL, explain TEXT,
            setup_type TEXT, rationale TEXT, change_5d REAL, direction TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO backtest_v2 VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("AAPL", 5, 4, 12.5, 1.1, -8.0, 75.0, 3.2, 0.001, 10, None, "2024-01-10 00:00 UTC"),
            ("TSLA", 3, 2, -5.0, 0.3, -20.0, 50.0, -2.5, 0.001, 10, None, "2024-01-11 00:00 UTC"),
            ("SKIP", 1, 0, None, None, None, None, None, 0.001, 10, None, "2024-01-12 00:00 UTC"),
            ("ERR",  2, 0, None, None, None, None, None, 0.001, 10, "data fetch failed", "2024-01-13 00:00 UTC"),
        ],
    )
    conn.executemany(
        "INSERT INTO results (run_date, ticker, setup_type, tradescore) VALUES (?,?,?,?)",
        [
            ("2024-01-05", "AAPL", "Early breakout",        72.0),
            ("2024-01-06", "AAPL", "Early breakout",        68.0),
            ("2024-01-05", "TSLA", "Momentum continuation", 55.0),
        ],
    )
    conn.commit()
    conn.close()
    return tf.name


def test_load_v2_data_filters_errors_and_zero_trades():
    db = _make_test_db()
    try:
        df = load_v2_data(db_path=db)
        assert "SKIP" not in df["ticker"].values
        assert "ERR"  not in df["ticker"].values
        assert set(df["ticker"]) == {"AAPL", "TSLA"}
    finally:
        os.unlink(db)


def test_load_v2_data_joins_setup_type():
    db = _make_test_db()
    try:
        df = load_v2_data(db_path=db)
        aapl = df[df["ticker"] == "AAPL"].iloc[0]
        assert aapl["setup_type"] == "Early breakout"
        assert aapl["avg_tradescore"] == pytest.approx(70.0, abs=0.1)
    finally:
        os.unlink(db)


def test_load_v2_data_missing_table_returns_empty():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        path = tf.name
    try:
        conn = sqlite3.connect(path)
        conn.close()
        df = load_v2_data(db_path=path)
        assert df.empty
    finally:
        os.unlink(path)


def test_load_v2_data_missing_db_returns_empty():
    df = load_v2_data(db_path="/tmp/does_not_exist_phase8_test.db")
    assert df.empty
