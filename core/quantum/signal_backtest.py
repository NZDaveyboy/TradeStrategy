"""
core/quantum/signal_backtest.py — Walk-forward backtest of the BUY/WATCH/HOLD/SELL classifier.

Answers the only question that matters about the classifier:
    "When the rules said BUY in the past, did those names actually outperform?
     When they said SELL, did those names actually decline?"

Design:

  At each historical sample date `t` (monthly default), we:
    1. Build the Quantum Ecosystem index using ONLY prices up to `t`
       — this gives us the held-return history needed by the classifier.
    2. Truncate the price panel to `≤ t` so classify_constituents can't peek.
    3. Classify every constituent at `t` — record each (date, ticker, signal).
    4. Look forward `lookforward_days` (default 30 trading days) and record
       the actual price return over that window.
    5. Compute the index's forward return over the same window for
       relative-performance comparison.

  After walking the full window, aggregate:
    - Per-signal hit rate (absolute: BUY → return > 0; SELL → return < 0)
    - Per-signal hit rate (relative-to-index: BUY → beat index; SELL → loss to index)
    - Per-signal mean forward return + std
    - Cumulative "follow BUY signals" portfolio curve

CRITICAL: no look-ahead. The classifier at date `t` only sees prices ≤ `t`.
The forward return uses prices > `t`. Tests guard this invariant.

Public API:
    backtest_signals(universe, prices, ...) -> dict
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.quantum.backtest import classify_constituents
from core.quantum.index import IndexBuilder
from core.quantum.utils import Universe


# Lookforward windows we report on (trading days)
DEFAULT_LOOKFORWARDS = (30, 60, 90)
DEFAULT_SAMPLE_FREQ  = "ME"   # month-end (pandas alias)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def backtest_signals(
    universe: Universe,
    prices: pd.DataFrame,
    *,
    start: pd.Timestamp | None = None,
    end:   pd.Timestamp | None = None,
    sample_freq: str = DEFAULT_SAMPLE_FREQ,
    lookforwards: tuple[int, ...] = DEFAULT_LOOKFORWARDS,
) -> dict:
    """Walk forward through the price panel and validate classifier signals.

    Returns dict with:
      summary       DataFrame: one row per (Signal, lookforward) with N samples,
                    Absolute hit %, vs Index hit %, mean fwd return %, vs Index %
      signal_log    DataFrame: every (Date, Ticker, Signal, fwd_30d, fwd_60d, fwd_90d, ...)
      portfolio     dict of {lookforward: pd.Series} — cumulative return of
                    "equal-weighted BUY signals at each sample" vs the index
      sample_dates  list of pd.Timestamp the backtest sampled at
    """
    if prices.empty or not universe.all_companies():
        return {
            "summary":      pd.DataFrame(),
            "signal_log":   pd.DataFrame(),
            "portfolio":    {},
            "sample_dates": [],
        }

    # Determine the window
    if start is None:
        start = prices.index[0] + pd.Timedelta(days=180)  # leave 6mo warmup for indicators
    if end is None:
        # Leave enough room at the end for the longest lookforward
        max_lf = max(lookforwards)
        end = prices.index[-1] - pd.Timedelta(days=max_lf * 1.5)
    start = pd.Timestamp(start)
    end   = pd.Timestamp(end)

    if end <= start:
        return {
            "summary":      pd.DataFrame(),
            "signal_log":   pd.DataFrame(),
            "portfolio":    {},
            "sample_dates": [],
        }

    # Sample dates: month-ends (or whatever freq) within [start, end]
    sample_dates = list(pd.date_range(start=start, end=end, freq=sample_freq))
    # Snap each to the closest available trading day in `prices`
    valid_idx = prices.index
    snapped: list[pd.Timestamp] = []
    for d in sample_dates:
        # Find the largest trading day ≤ d
        loc = valid_idx.searchsorted(d, side="right") - 1
        if loc < 0:
            continue
        snapped.append(valid_idx[loc])
    sample_dates = sorted(set(snapped))

    if not sample_dates:
        return {
            "summary":      pd.DataFrame(),
            "signal_log":   pd.DataFrame(),
            "portfolio":    {},
            "sample_dates": [],
        }

    # Walk forward
    rows: list[dict] = []
    universe_tickers = {c.ticker for c in universe.all_companies()}

    for t in sample_dates:
        # NO LOOK-AHEAD: truncate prices to ≤ t
        past_prices = prices.loc[:t]
        if past_prices.shape[0] < 60:  # need enough history for 3-month return
            continue

        # Build the index using only past data
        try:
            builder = IndexBuilder(universe, past_prices)
            result = builder.build_ecosystem(past_prices.index[0], t)
        except Exception:
            continue

        if result.levels.empty:
            continue

        # Classify using only past data
        try:
            uni_prices = past_prices[
                [c.ticker for c in universe.all_companies() if c.ticker in past_prices.columns]
            ]
            sigs = classify_constituents(result, uni_prices, universe)
        except Exception:
            continue

        if sigs.empty:
            continue

        # Compute forward returns (uses FUTURE data — only for evaluation, not classification)
        for _, sig_row in sigs.iterrows():
            ticker = sig_row["Ticker"]
            if ticker not in prices.columns or ticker not in universe_tickers:
                continue
            price_at_t = prices.at[t, ticker] if t in prices.index else None
            if price_at_t is None or pd.isna(price_at_t):
                continue

            row = {
                "Date":       t,
                "Ticker":     ticker,
                "Signal":     sig_row["Signal"],
                "Conviction": sig_row["Conviction"],
                "Price@t":    float(price_at_t),
            }
            for lf in lookforwards:
                # Locate the trading day `lf` business days after `t`
                t_loc = valid_idx.get_loc(t)
                fwd_loc = t_loc + lf
                if fwd_loc >= len(valid_idx):
                    row[f"fwd_{lf}d"]     = float("nan")
                    row[f"fwd_idx_{lf}d"] = float("nan")
                    continue
                fwd_date = valid_idx[fwd_loc]
                fwd_price = prices.at[fwd_date, ticker]
                if pd.isna(fwd_price):
                    row[f"fwd_{lf}d"]     = float("nan")
                    row[f"fwd_idx_{lf}d"] = float("nan")
                    continue
                row[f"fwd_{lf}d"] = float((fwd_price / float(price_at_t) - 1.0) * 100.0)

                # Index forward return over same window
                idx_t   = result.levels.iloc[-1]   # index level at t
                # Rebuild index using prices up to fwd_date to get its forward level
                try:
                    fwd_prices = prices.loc[:fwd_date]
                    fwd_builder = IndexBuilder(universe, fwd_prices)
                    fwd_result  = fwd_builder.build_ecosystem(fwd_prices.index[0], fwd_date)
                    if not fwd_result.levels.empty:
                        idx_fwd = fwd_result.levels.iloc[-1]
                        row[f"fwd_idx_{lf}d"] = float((idx_fwd / idx_t - 1.0) * 100.0)
                    else:
                        row[f"fwd_idx_{lf}d"] = float("nan")
                except Exception:
                    row[f"fwd_idx_{lf}d"] = float("nan")

            rows.append(row)

    signal_log = pd.DataFrame(rows)
    if signal_log.empty:
        return {
            "summary":      pd.DataFrame(),
            "signal_log":   signal_log,
            "portfolio":    {},
            "sample_dates": sample_dates,
        }

    # ── Aggregate per-signal stats ─────────────────────────────────────────
    summary_rows = []
    for sig in ("BUY", "WATCH", "HOLD", "SELL"):
        sub = signal_log[signal_log["Signal"] == sig]
        if sub.empty:
            continue
        for lf in lookforwards:
            col_t   = f"fwd_{lf}d"
            col_idx = f"fwd_idx_{lf}d"
            if col_t not in sub.columns:
                continue
            valid = sub.dropna(subset=[col_t])
            if valid.empty:
                continue

            n         = len(valid)
            mean_t    = float(valid[col_t].mean())
            std_t     = float(valid[col_t].std()) if n > 1 else 0.0
            # Absolute hit rate
            if sig == "BUY":
                abs_hit = float((valid[col_t] > 0).sum()) / n * 100.0
            elif sig == "SELL":
                abs_hit = float((valid[col_t] < 0).sum()) / n * 100.0
            else:  # HOLD / WATCH
                abs_hit = float((valid[col_t].abs() <= 5.0).sum()) / n * 100.0

            # Relative hit (vs the index over the same window)
            rel_hit = float("nan")
            mean_rel = float("nan")
            if col_idx in valid.columns:
                rel_valid = valid.dropna(subset=[col_idx])
                if not rel_valid.empty:
                    excess = rel_valid[col_t] - rel_valid[col_idx]
                    mean_rel = float(excess.mean())
                    if sig == "BUY":
                        rel_hit = float((excess > 0).sum()) / len(rel_valid) * 100.0
                    elif sig == "SELL":
                        rel_hit = float((excess < 0).sum()) / len(rel_valid) * 100.0
                    # HOLD/WATCH: skip relative hit

            summary_rows.append({
                "Signal":           sig,
                "Lookforward (d)":  lf,
                "N samples":        n,
                "Hit rate %":       round(abs_hit, 1),
                "vs Index hit %":   round(rel_hit, 1) if pd.notna(rel_hit) else float("nan"),
                "Mean fwd %":       round(mean_t, 2),
                "Mean vs Index %":  round(mean_rel, 2) if pd.notna(mean_rel) else float("nan"),
                "Std fwd %":        round(std_t, 2),
            })

    summary = pd.DataFrame(summary_rows)

    # ── Build "follow BUY signals" portfolio curve ────────────────────────
    # At each sample date, if there's at least one BUY signal, equal-weight
    # those BUYs and hold for `lookforward_days`. Otherwise hold cash (0% return).
    portfolio: dict[int, pd.Series] = {}
    for lf in lookforwards:
        col_t = f"fwd_{lf}d"
        if col_t not in signal_log.columns:
            continue
        # For each sample date, mean return across BUY signals
        buys = signal_log[signal_log["Signal"] == "BUY"].dropna(subset=[col_t])
        if buys.empty:
            continue
        per_sample = (
            buys.groupby("Date")[col_t]
                .mean()
                .sort_index()
        )
        # Convert to cumulative — treat each as a non-overlapping chunk
        # (conservative — overlapping holds get complicated)
        cum = (1.0 + per_sample / 100.0).cumprod() * 100.0
        portfolio[lf] = cum

    return {
        "summary":      summary,
        "signal_log":   signal_log,
        "portfolio":    portfolio,
        "sample_dates": sample_dates,
    }
