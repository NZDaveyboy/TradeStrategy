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
    DEFAULT_PERIODS_PER_YEAR,
    DEFAULT_RISK_FREE_RATE,
    dated_returns_series,
    drawdown_series,
    equity_curve,
    load_v2_data,
    monthly_returns_table,
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
# Sharpe / Sortino annualisation — the math that was wrong before this fix
# ---------------------------------------------------------------------------

def test_portfolio_stats_sharpe_is_properly_annualised():
    """Sharpe must scale by sqrt(periods_per_year), not just mean/std."""
    # Constructed series with known statistics:
    #   returns = [+10%, -2%, +5%, -3%, +8%, -1%, +4%, -2%]
    #   mean ≈ 0.02375, std ≈ 0.0476, excess ≈ 0.02375 - (0.045/252)
    df = _df([
        _base_row(ticker=t, return_pct=r)
        for t, r in [("A",10.0),("B",-2.0),("C",5.0),("D",-3.0),
                     ("E", 8.0),("F",-1.0),("G",4.0),("H",-2.0)]
    ])
    stats = portfolio_stats(df, periods_per_year=252.0, risk_free_rate=0.045)
    # Hand-compute expected:
    rets = [0.10,-0.02,0.05,-0.03,0.08,-0.01,0.04,-0.02]
    mean = sum(rets)/len(rets)
    var  = sum((r-mean)**2 for r in rets)/(len(rets)-1)
    std  = var**0.5
    rf_per = 0.045/252.0
    expected_sharpe = (mean - rf_per) / std * math.sqrt(252.0)
    assert stats["sharpe"] == pytest.approx(expected_sharpe, rel=1e-3)


def test_portfolio_stats_sortino_uses_downside_std():
    """Sortino's denominator is std of returns below zero, not full std."""
    df = _df([
        _base_row(ticker=t, return_pct=r)
        for t, r in [("A",10.0),("B",-2.0),("C",5.0),("D",-3.0),
                     ("E", 8.0),("F",-1.0),("G",4.0),("H",-2.0)]
    ])
    stats = portfolio_stats(df, periods_per_year=252.0, risk_free_rate=0.045)
    # Downside returns: -0.02, -0.03, -0.01, -0.02
    downside = [-0.02,-0.03,-0.01,-0.02]
    d_mean = sum(downside)/len(downside)
    d_var  = sum((r-d_mean)**2 for r in downside)/(len(downside)-1)
    d_std  = d_var**0.5
    rets = [0.10,-0.02,0.05,-0.03,0.08,-0.01,0.04,-0.02]
    excess_mean = sum(rets)/len(rets) - 0.045/252.0
    expected_sortino = excess_mean / d_std * math.sqrt(252.0)
    assert stats["sortino"] == pytest.approx(expected_sortino, rel=1e-3)


def test_portfolio_stats_periods_per_year_override_changes_sharpe():
    """Doubling periods_per_year should scale Sharpe by sqrt(2)."""
    df = _df([
        _base_row(ticker=t, return_pct=r)
        for t, r in [("A",5.0),("B",-2.0),("C",3.0),("D",-1.0),("E",4.0)]
    ])
    s_252 = portfolio_stats(df, periods_per_year=252.0, risk_free_rate=0.0)
    s_504 = portfolio_stats(df, periods_per_year=504.0, risk_free_rate=0.0)
    # With rf=0, sharpe scales purely as sqrt(N): doubling N → ×sqrt(2)
    assert s_504["sharpe"] == pytest.approx(s_252["sharpe"] * math.sqrt(2), rel=1e-4)


def test_portfolio_stats_risk_free_zero_matches_no_rf_math():
    """When rf=0, sharpe = mean/std × sqrt(N) exactly (no excess adjustment)."""
    df = _df([
        _base_row(ticker=t, return_pct=r)
        for t, r in [("A",6.0),("B",-1.0),("C",4.0),("D",-2.0)]
    ])
    stats = portfolio_stats(df, periods_per_year=252.0, risk_free_rate=0.0)
    rets = [0.06,-0.01,0.04,-0.02]
    mean = sum(rets)/len(rets)
    var  = sum((r-mean)**2 for r in rets)/(len(rets)-1)
    std  = var**0.5
    expected = (mean / std) * math.sqrt(252.0)
    assert stats["sharpe"] == pytest.approx(expected, rel=1e-3)


def test_portfolio_stats_returns_methodology_keys():
    """Output must include periods_per_year and risk_free_rate for UI display."""
    df = _df([_base_row()])
    stats = portfolio_stats(df, periods_per_year=126.0, risk_free_rate=0.03)
    assert stats["periods_per_year"] == 126.0
    assert stats["risk_free_rate"] == 0.03


def test_portfolio_stats_avg_strategy_sharpe_from_saved_column():
    """avg_strategy_sharpe = mean of saved per-ticker sharpe column."""
    df = _df([
        _base_row(ticker="A", return_pct=10.0, sharpe=1.5),
        _base_row(ticker="B", return_pct=-2.0, sharpe=0.3),
        _base_row(ticker="C", return_pct= 4.0, sharpe=0.9),
    ])
    stats = portfolio_stats(df)
    assert stats["avg_strategy_sharpe"] == pytest.approx((1.5 + 0.3 + 0.9) / 3, rel=1e-6)


def test_portfolio_stats_defaults_match_documented_constants():
    """Default annualisation = 252, default rf = 4.5%."""
    df = _df([_base_row()])
    stats = portfolio_stats(df)
    assert stats["periods_per_year"] == DEFAULT_PERIODS_PER_YEAR == 252.0
    assert stats["risk_free_rate"]   == DEFAULT_RISK_FREE_RATE   == 0.045


# ---------------------------------------------------------------------------
# Time-series helpers (dated_returns_series / drawdown_series / monthly_returns_table)
# ---------------------------------------------------------------------------

def test_dated_returns_series_empty_df():
    s = dated_returns_series(pd.DataFrame())
    assert s.empty


def test_dated_returns_series_uses_run_at_when_span_large():
    """run_at spans 60 days → use real dates as index."""
    df = _df([
        {**_base_row(ticker="A", return_pct=10.0), "run_at": "2024-01-01"},
        {**_base_row(ticker="B", return_pct=-5.0), "run_at": "2024-02-15"},
        {**_base_row(ticker="C", return_pct=8.0),  "run_at": "2024-03-10"},
    ])
    s = dated_returns_series(df)
    assert len(s) == 3
    # Should be sorted ascending by date
    assert s.index[0] < s.index[-1]
    # First date matches earliest run_at
    assert s.index[0] == pd.Timestamp("2024-01-01")


def test_dated_returns_series_synthetic_when_run_at_narrow():
    """All run_at on same day → synthesise business-day index."""
    df = _df([
        {**_base_row(ticker="A", return_pct=10.0), "run_at": "2024-01-10"},
        {**_base_row(ticker="B", return_pct=-2.0), "run_at": "2024-01-10"},
    ])
    s = dated_returns_series(df)
    assert len(s) == 2
    # All dates should be business days
    assert all(d.weekday() < 5 for d in s.index)
    # Should end no later than today
    assert s.index[-1] <= pd.Timestamp.today().normalize()


def test_dated_returns_series_no_run_at_uses_synthetic():
    """No run_at column → synthetic business-day index."""
    df = _df([_base_row(ticker="A", return_pct=5.0),
              _base_row(ticker="B", return_pct=-3.0)])
    df = df.drop(columns=["run_at"], errors="ignore")  # ensure absent
    s = dated_returns_series(df)
    assert len(s) == 2
    assert all(d.weekday() < 5 for d in s.index)


def test_drawdown_series_always_non_positive():
    """Drawdown is defined as ≤ 0 everywhere."""
    df = _df([
        _base_row(ticker="A", return_pct=15.0),
        _base_row(ticker="B", return_pct=-20.0),  # drawdown event
        _base_row(ticker="C", return_pct= 5.0),
    ])
    dd = drawdown_series(df)
    assert (dd <= 1e-9).all()  # allow tiny floating noise


def test_drawdown_series_zero_at_new_high():
    """Drawdown is exactly 0 when the curve makes a new all-time high."""
    df = _df([
        _base_row(ticker="A", return_pct=10.0),
        _base_row(ticker="B", return_pct=10.0),
        _base_row(ticker="C", return_pct=10.0),
    ])
    dd = drawdown_series(df)
    # Each step is a new high, so drawdown should be 0 throughout
    assert all(abs(v) < 1e-9 for v in dd)


def test_drawdown_series_reaches_minimum_at_trough():
    """Drawdown reaches its minimum (most negative) at the worst trough."""
    df = _df([
        _base_row(ticker="A", return_pct=20.0),   # peak
        _base_row(ticker="B", return_pct=-30.0),  # trough
        _base_row(ticker="C", return_pct= 5.0),   # partial recovery
    ])
    dd = drawdown_series(df)
    # Worst drawdown should be at index 1 (after -30%)
    assert dd.idxmin() == 1


def test_drawdown_series_empty_df():
    dd = drawdown_series(pd.DataFrame())
    assert dd.empty


def test_monthly_returns_table_returns_expected_columns():
    df = _df([
        {**_base_row(ticker="A", return_pct=5.0),  "run_at": "2024-01-15"},
        {**_base_row(ticker="B", return_pct=3.0),  "run_at": "2024-02-10"},
        {**_base_row(ticker="C", return_pct=-2.0), "run_at": "2024-03-05"},
    ])
    table = monthly_returns_table(df)
    assert set(["Year", "Month", "Month_label", "Return", "N_obs"]).issubset(table.columns)


def test_monthly_returns_table_aggregates_compoundly_in_month():
    """Two returns in same month → compounded, not summed."""
    df = _df([
        {**_base_row(ticker="A", return_pct=10.0), "run_at": "2024-01-05"},
        {**_base_row(ticker="B", return_pct=10.0), "run_at": "2024-01-20"},
    ])
    # Make run_at span > 30 days so we use real dates
    df = pd.concat([df, _df([
        {**_base_row(ticker="C", return_pct=0.0), "run_at": "2024-03-15"},
    ])], ignore_index=True)
    table = monthly_returns_table(df)
    jan_row = table[(table["Year"] == 2024) & (table["Month"] == 1)].iloc[0]
    # Two +10% trades compounded = 1.10 * 1.10 - 1 = 0.21 → 21%
    assert jan_row["Return"] == pytest.approx(21.0, abs=0.01)


def test_monthly_returns_table_empty_df():
    table = monthly_returns_table(pd.DataFrame())
    assert table.empty


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
