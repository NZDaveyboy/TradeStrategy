"""
src/data.py — Price-data fetcher with graceful degradation.

Uses yfinance to pull daily adjusted-close prices for the universe and
benchmarks. If a ticker fails, log the error and continue — the rest of
the universe still produces an index. If a ticker has missing prices at
the start date, it is **only** included from its first valid trading
date (cold-start handling), so the index doesn't get corrupted by
implicit zero-prices.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from core.quantum.utils import get_logger


log = get_logger("quantum_index.data")


def fetch_prices(
    tickers: Iterable[str],
    start: date | str,
    end: date | str | None = None,
) -> pd.DataFrame:
    """Fetch daily adjusted-close prices for a list of tickers.

    Returns a DataFrame indexed by date with one column per ticker.
    Tickers that fail to fetch are dropped from the output (logged at
    WARNING level). Tickers with no data in the window are also dropped.

    Uses yfinance's `auto_adjust=True` so the Close column is already
    split/dividend-adjusted (matches the spec: "Use adjusted close prices").
    """
    tickers = list(dict.fromkeys(tickers))  # de-dupe, preserve order
    # yfinance wants YYYY-MM-DD only — no time component
    start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_str = pd.Timestamp(end).strftime("%Y-%m-%d") if end else None

    log.info(f"Fetching {len(tickers)} tickers from {start_str} to {end_str or 'today'}")

    try:
        raw = yf.download(
            tickers,
            start=start_str,
            end=end_str,
            progress=False,
            auto_adjust=True,
            threads=False,
            group_by="ticker",
        )
    except Exception as e:
        log.error(f"yfinance bulk fetch failed: {e}")
        return pd.DataFrame()

    if raw is None or raw.empty:
        log.warning("yfinance returned empty dataframe")
        return pd.DataFrame()

    closes: dict[str, pd.Series] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if t not in raw.columns.get_level_values(0):
                log.warning(f"  {t}: no data returned")
                continue
            sub = raw[t]
            if "Close" not in sub.columns:
                log.warning(f"  {t}: missing Close column")
                continue
            s = sub["Close"].dropna()
            if s.empty:
                log.warning(f"  {t}: all-NaN Close series")
                continue
            closes[t] = s
    else:
        # Single-ticker path — yfinance flattens columns
        if "Close" in raw.columns:
            s = raw["Close"].dropna()
            if not s.empty:
                closes[tickers[0]] = s

    if not closes:
        log.error("No usable price data for any ticker")
        return pd.DataFrame()

    df = pd.DataFrame(closes)
    # Sort and ensure a continuous date index for the union of all tickers
    df = df.sort_index()
    # Strip timezone info — keeps date math clean and Altair-compatible
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        pass

    log.info(f"  → {df.shape[1]} tickers × {df.shape[0]} trading days")
    return df


def first_valid_date(prices: pd.DataFrame, ticker: str) -> pd.Timestamp | None:
    """Return the first date `ticker` has a non-NaN price, or None."""
    if ticker not in prices.columns:
        return None
    s = prices[ticker].dropna()
    if s.empty:
        return None
    return s.index[0]


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker daily simple returns. NaN where price is missing on either
    side of the diff (correct — those days are excluded from the
    portfolio-return aggregation later)."""
    return prices.pct_change()
