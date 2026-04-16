"""
core/research/rescore.py — Load and re-score historical signals from screener DB.

Re-scoring limitations (inputs not stored in screener DB):
  - BOB (breakout-from-base) component of EarlyEntryScore → 0  (max 7 pts)
  - dvol_pts and cons_pts of LiquidityScore → 0                (max 11 pts combined)
  - run5_pen uses stored change_5d column as fallback            (fully recovered)

Re-scored values are therefore 0–18 pts lower than screener-computed scores
on high-liquidity breakout names. All param sets in a sweep share this
limitation equally, so relative rankings across the sweep are valid.

Threshold values calibrated on full screener scores need adjusting:
  screener threshold 55 ≈ re-scored threshold ~37 for breakout/liquid names.
"""

from __future__ import annotations

import os
import sqlite3

import pandas as pd

from core.tradescore import compute_tradescore
from core.research.params import SweepParams

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "screener.db")

# Columns we SELECT from results — must all exist in the results table.
_SIGNAL_COLS = [
    "run_date", "ticker", "direction",
    "price", "change_pct", "rvol", "rsi",
    "ema9", "ema20", "ema200", "atr",
    "macd", "macd_signal", "vwap",
    "market_cap", "float_shares", "tradescore",
    "change_5d", "stop_loss", "setup_type",
]


def load_raw_signals(
    db_path: str = DB_PATH,
    start_date: str | None = None,
    end_date:   str | None = None,
) -> pd.DataFrame:
    """
    Load signal rows from the results table.
    Only rows with tradescore IS NOT NULL are included (pre-tradescore rows excluded).

    Args:
        db_path:    path to screener.db
        start_date: inclusive lower bound on run_date (YYYY-MM-DD)
        end_date:   inclusive upper bound on run_date (YYYY-MM-DD)

    Returns:
        DataFrame with one row per (run_date, ticker) signal.
    """
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=_SIGNAL_COLS)

    clauses = ["tradescore IS NOT NULL"]
    params: list = []
    if start_date:
        clauses.append("run_date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("run_date <= ?")
        params.append(end_date)

    where = " AND ".join(clauses)
    cols  = ", ".join(c for c in _SIGNAL_COLS if c != "stop_loss") + ", stop_loss"
    sql   = f"SELECT {cols} FROM results WHERE {where} ORDER BY run_date, ticker"

    conn = sqlite3.connect(db_path)
    df   = pd.read_sql(sql, conn, params=params)
    conn.close()
    return df


def rescore_row(row: dict, params: SweepParams) -> dict:
    """
    Re-invoke compute_tradescore on a DB row dict using SweepParams weight overrides.

    close=None: BOB=0, run5_pen uses stored change_5d.
    data=None:  dvol_pts=0, cons_pts=0.

    Returns the full compute_tradescore output dict (same shape as screener output).
    """
    return compute_tradescore(
        row,
        close=None,
        data=None,
        weights=params.to_weight_overrides(),
    )


def rescore_signals(
    raw_df: pd.DataFrame,
    params: SweepParams,
) -> pd.DataFrame:
    """
    Re-score all rows in raw_df with params and return an enriched DataFrame.

    Adds columns:
        rescore_tradescore  — re-computed TradeScore
        rescore_direction   — re-computed direction
        rescore_setup_type  — re-computed setup label
        stop_reconstructed  — ema20 ± stop_multiplier × atr (replaces DB stop_loss)
    """
    records = raw_df.to_dict("records")
    rescore_scores:      list[float] = []
    rescore_directions:  list[str]   = []
    rescore_setups:      list[str]   = []
    stops_reconstructed: list[float | None] = []

    for r in records:
        ts     = rescore_row(r, params)
        direct = str(r.get("direction") or ts["direction"])

        rescore_scores.append(ts["score"])
        rescore_directions.append(ts["direction"])
        rescore_setups.append(ts["setup_type"])

        # Reconstruct stop from stored EMA20 + ATR using sweep stop_multiplier.
        # The DB stop_loss used the old formula (min(vwap, ema20) - 0.35*atr).
        ema20 = float(r.get("ema20") or 0)
        atr   = float(r.get("atr")   or 0)
        price = float(r.get("price") or 0)
        if ema20 > 0 and atr > 0 and price > 0:
            if direct == "short":
                stop = round(ema20 + params.stop_multiplier * atr, 4)
            else:
                stop = round(ema20 - params.stop_multiplier * atr, 4)
        else:
            stop = None
        stops_reconstructed.append(stop)

    result = raw_df.copy()
    result["rescore_tradescore"] = rescore_scores
    result["rescore_direction"]  = rescore_directions
    result["rescore_setup_type"] = rescore_setups
    result["stop_reconstructed"] = stops_reconstructed
    return result


def filter_signals(df: pd.DataFrame, params: SweepParams) -> pd.DataFrame:
    """
    Apply threshold filters from SweepParams to a re-scored DataFrame.
    Filters on rescore_tradescore and rescore_direction (from rescore_signals output).
    """
    mask = pd.Series([True] * len(df), index=df.index)

    if params.tradescore_threshold > 0:
        mask &= df["rescore_tradescore"] >= params.tradescore_threshold

    if params.min_rvol > 0:
        mask &= df["rvol"].fillna(0) >= params.min_rvol

    if params.rsi_min > 0:
        mask &= df["rsi"].fillna(0) >= params.rsi_min

    if params.rsi_max < 100:
        mask &= df["rsi"].fillna(100) <= params.rsi_max

    if params.direction_filter != "both":
        mask &= df["rescore_direction"] == params.direction_filter

    return df[mask].copy()


def build_signal_groups(df: pd.DataFrame) -> dict[str, list[dict]]:
    """
    Convert a filtered, re-scored DataFrame to the signal dict format expected
    by core.backtest_engine.run_backtest.

    Returns: {ticker: [{"date": YYYY-MM-DD, "stop": float|None, "target": None}, ...]}
    """
    groups: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        ticker = str(row["ticker"])
        groups.setdefault(ticker, []).append({
            "date":   str(row["run_date"]),
            "stop":   row.get("stop_reconstructed"),
            "target": None,  # time-based exit only in v2
            "tradescore": row.get("rescore_tradescore"),
            "setup_type": row.get("rescore_setup_type"),
            "direction":  row.get("rescore_direction"),
        })
    return groups
