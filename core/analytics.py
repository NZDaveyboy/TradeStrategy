"""
core/analytics.py — Portfolio-level analytics for backtest_v2 results.

Reads from the backtest_v2 table (created by backtest_v2.py), joins with
results for setup_type / tradescore, then returns DataFrames and dicts
ready for the Streamlit UI.

quantstats is used for max_drawdown on the equity curve. Sharpe / Sortino
are computed as cross-trade ratios (mean / std) — not annualised, since the
input is one return per ticker, not a daily time series.
"""

from __future__ import annotations

import math
import os
import sqlite3

import pandas as pd
import quantstats as qs

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "screener.db")

SCORE_BUCKETS = [(0, 40), (40, 60), (60, 80), (80, 100)]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_v2_data(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Load backtest_v2 joined with aggregated results for setup_type / tradescore.

    Filters to rows where error IS NULL and n_trades > 0. Returns an empty
    DataFrame if the backtest_v2 table does not exist yet.

    Extra columns added on join:
        setup_type      — modal setup_type for that ticker across all results rows
        avg_tradescore  — mean tradescore for that ticker across all results rows
    """
    if not os.path.exists(db_path):
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='backtest_v2'"
        ).fetchone()
        if not exists:
            return pd.DataFrame()

        v2 = pd.read_sql("SELECT * FROM backtest_v2", conn)

        has_results = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='results'"
        ).fetchone()
        if has_results:
            res = pd.read_sql(
                "SELECT ticker, setup_type, tradescore FROM results",
                conn,
            )
            if not res.empty:
                def _mode(x):
                    m = x.dropna().mode()
                    return m.iloc[0] if not m.empty else None

                res_agg = (
                    res.groupby("ticker")
                    .agg(
                        setup_type    =("setup_type",  _mode),
                        avg_tradescore=("tradescore",  "mean"),
                    )
                    .reset_index()
                )
                v2 = v2.merge(res_agg, on="ticker", how="left")
            else:
                v2["setup_type"]     = None
                v2["avg_tradescore"] = float("nan")
        else:
            v2["setup_type"]     = None
            v2["avg_tradescore"] = float("nan")

    finally:
        conn.close()

    # Keep only usable rows
    v2 = v2[v2["error"].isna() & (v2["n_trades"] > 0)].copy()
    for col in ("return_pct", "win_rate", "sharpe", "max_drawdown", "avg_trade_pct"):
        if col in v2.columns:
            v2[col] = pd.to_numeric(v2[col], errors="coerce")
    v2 = v2.dropna(subset=["return_pct"])
    return v2.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _returns_series(df: pd.DataFrame) -> pd.Series:
    """Per-ticker return_pct as decimal fractions, ordered by run_at if present."""
    if "run_at" in df.columns:
        df = df.sort_values("run_at")
    return (df["return_pct"] / 100.0).reset_index(drop=True)


def _safe_float(v) -> float:
    try:
        f = float(v)
        return float("nan") if math.isnan(f) else f
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def equity_curve(df: pd.DataFrame) -> pd.Series:
    """
    Cumulative equity curve from per-ticker return_pct, starting at 1.0.

    Ordered by run_at when that column is present so the curve follows
    chronological backtest order.
    """
    if df.empty:
        return pd.Series(dtype=float)
    rets  = _returns_series(df)
    curve = (1 + rets).cumprod()
    return curve.reset_index(drop=True)


def portfolio_stats(df: pd.DataFrame) -> dict:
    """
    Aggregate portfolio-level stats from per-ticker backtest_v2 rows.

    Returns a dict with:
        total_trades, n_tickers,
        win_rate (%), avg_return (%), expectancy (%),
        sharpe (cross-trade, not annualised),
        sortino (cross-trade, not annualised),
        max_drawdown (%)
    """
    base = {
        "total_trades": 0,
        "n_tickers":    0,
        "win_rate":     float("nan"),
        "avg_return":   float("nan"),
        "expectancy":   float("nan"),
        "sharpe":       float("nan"),
        "sortino":      float("nan"),
        "max_drawdown": float("nan"),
    }
    if df.empty:
        return base

    returns = _returns_series(df)
    curve   = equity_curve(df)

    std     = returns.std()
    neg     = returns[returns < 0]
    neg_std = neg.std() if len(neg) > 1 else float("nan")

    # Expectancy: avg_win * win_rate + avg_loss * loss_rate
    wins   = returns[returns > 0]
    losses = returns[returns < 0]
    wr     = len(wins) / len(returns) if len(returns) else 0.0
    lr     = 1.0 - wr
    avg_w  = wins.mean()  if len(wins)   else 0.0
    avg_l  = losses.mean() if len(losses) else 0.0
    expect = avg_w * wr + avg_l * lr   # in decimal

    # Max drawdown via quantstats (takes price/equity series)
    try:
        max_dd = _safe_float(qs.stats.max_drawdown(curve)) * 100
    except Exception:
        roll_max = curve.cummax()
        dd       = (curve / roll_max) - 1
        max_dd   = _safe_float(dd.min()) * 100

    return {
        "total_trades": int(df["n_trades"].sum()),
        "n_tickers":    len(df),
        "win_rate":     wr * 100,
        "avg_return":   _safe_float(returns.mean()) * 100,
        "expectancy":   expect * 100,
        "sharpe":       _safe_float(returns.mean() / std) if std > 0 else float("nan"),
        "sortino":      _safe_float(returns.mean() / neg_std) if (not math.isnan(neg_std) and neg_std > 0) else float("nan"),
        "max_drawdown": max_dd,
    }


def win_rate_by_setup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Win rate and avg return grouped by setup_type.

    Returns a DataFrame with columns:
        Setup Type, Tickers, Avg Return %, Win Rate %, Total Trades
    Sorted by Win Rate % descending. Returns empty DataFrame if no
    setup_type data is present.
    """
    if df.empty or "setup_type" not in df.columns:
        return pd.DataFrame()

    sub = df.dropna(subset=["setup_type"])
    if sub.empty:
        return pd.DataFrame()

    result = (
        sub.groupby("setup_type")
        .agg(
            Tickers      =("ticker",     "count"),
            avg_return   =("return_pct", "mean"),
            win_rate     =("return_pct", lambda x: (x > 0).mean() * 100),
            total_trades =("n_trades",   "sum"),
        )
        .round(1)
        .reset_index()
    )
    result = result.rename(columns={
        "setup_type":  "Setup Type",
        "avg_return":  "Avg Return %",
        "win_rate":    "Win Rate %",
        "total_trades":"Total Trades",
    })
    return result.sort_values("Win Rate %", ascending=False).reset_index(drop=True)


def performance_by_score_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Avg return and win rate for tradescore buckets: 0-40, 40-60, 60-80, 80-100.

    Uses avg_tradescore joined from the results table. Returns empty DataFrame
    if tradescore data is not present.
    """
    if df.empty or "avg_tradescore" not in df.columns:
        return pd.DataFrame()

    sub = df.dropna(subset=["avg_tradescore"]).copy()
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for lo, hi in SCORE_BUCKETS:
        mask = (sub["avg_tradescore"] >= lo) & (sub["avg_tradescore"] < hi)
        grp  = sub[mask]
        if grp.empty:
            continue
        avg_trade = (
            round(float(grp["avg_trade_pct"].mean()), 1)
            if "avg_trade_pct" in grp.columns and grp["avg_trade_pct"].notna().any()
            else float("nan")
        )
        rows.append({
            "Score Bucket":  f"{lo}\u2013{hi}",
            "Tickers":       len(grp),
            "Total Trades":  int(grp["n_trades"].sum()),
            "Avg Return %":  round(float(grp["return_pct"].mean()), 1),
            "Win Rate %":    round(float((grp["return_pct"] > 0).mean() * 100), 1),
            "Avg Trade %":   avg_trade,
        })
    return pd.DataFrame(rows)
