#!/usr/bin/env python3
"""
research_diag.py — Diagnostic trace for zero-trade sweep bug.

Run with:
    python3 research_diag.py

Traces the full pipeline for ONE param set (ts>=35, rvol>=1.0, stop=0.5,
max_hold=10) against signals from 2026-04-02 to 2026-04-16.

Prints at each stage so the first silent failure is visible.
"""

from __future__ import annotations

import math
import sys
import os

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from core.research.params import SweepParams
from core.research.rescore import (
    build_signal_groups,
    filter_signals,
    load_raw_signals,
    rescore_signals,
)
from core.backtest_engine import fetch_ticker_data, run_backtest

DB = "screener.db"
START = "2026-04-02"
END   = "2026-04-16"

PARAMS = SweepParams(
    tradescore_threshold=35.0,
    min_rvol=1.0,
    stop_multiplier=0.5,
    max_hold_days=10,
    direction_filter="both",   # accept long + short
)


def sep(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ─────────────────────────────────────────────────────────────
# 1. Raw signals from DB
# ─────────────────────────────────────────────────────────────
sep("1. RAW SIGNALS FROM DB")
raw_df = load_raw_signals(db_path=DB, start_date=START, end_date=END)
print(f"  Rows loaded: {len(raw_df)}")
if raw_df.empty:
    print("  ERROR: No signals loaded — check DB path and date range.")
    sys.exit(1)

# Check run_date dtype and str() representation
sample_date = raw_df.iloc[0]["run_date"]
print(f"  run_date dtype : {raw_df['run_date'].dtype}")
print(f"  sample raw val : {sample_date!r}  (type={type(sample_date).__name__})")
print(f"  sample str()   : {str(sample_date)!r}")
print(f"  Date range     : {raw_df['run_date'].min()} → {raw_df['run_date'].max()}")
print(f"  Unique dates   : {sorted(raw_df['run_date'].unique())}")
print(f"  Direction dist :\n{raw_df['direction'].value_counts(dropna=False).to_string()}")
print(f"  TradeScore range: {raw_df['tradescore'].min()} – {raw_df['tradescore'].max()}")


# ─────────────────────────────────────────────────────────────
# 2. Re-score
# ─────────────────────────────────────────────────────────────
sep("2. RESCORE")
scored_df = rescore_signals(raw_df, PARAMS)
print(f"  Rescored rows        : {len(scored_df)}")
print(f"  rescore_tradescore   : min={scored_df['rescore_tradescore'].min():.1f}  "
      f"max={scored_df['rescore_tradescore'].max():.1f}  "
      f"mean={scored_df['rescore_tradescore'].mean():.1f}")
print(f"  rescore_direction    :\n{scored_df['rescore_direction'].value_counts(dropna=False).to_string()}")
print(f"  stop_reconstructed sample: {scored_df['stop_reconstructed'].head(3).tolist()}")


# ─────────────────────────────────────────────────────────────
# 3. Filter
# ─────────────────────────────────────────────────────────────
sep("3. FILTER (ts>=35, rvol>=1.0, direction=both)")
filtered_df = filter_signals(scored_df, PARAMS)
print(f"  Signals after filter: {len(filtered_df)}")
if filtered_df.empty:
    print("  ERROR: All signals filtered out. Check rescore_direction values and threshold.")
    sys.exit(1)
print(f"  Per-date breakdown:")
print(filtered_df.groupby("run_date").size().to_string())


# ─────────────────────────────────────────────────────────────
# 4. Signal groups — check date key format
# ─────────────────────────────────────────────────────────────
sep("4. BUILD SIGNAL GROUPS — date key format check")
groups = build_signal_groups(filtered_df)
tickers = sorted(groups.keys())
print(f"  Unique tickers in groups: {len(tickers)}")

# Show first 5 signal dicts
first5 = [(t, s) for t in tickers[:3] for s in groups[t][:2]][:5]
print(f"\n  First 5 signal dicts:")
for ticker, sig in first5:
    print(f"    ticker={ticker}  date={sig['date']!r}  "
          f"stop={sig['stop']}  direction={sig['direction']}")

# KEY CHECK: date key format
sample_key = next(iter(groups[tickers[0]])  )["date"]
print(f"\n  DATE KEY FORMAT  : {sample_key!r}")
print(f"  len(sample_key)  : {len(sample_key)}")
if len(sample_key) != 10:
    print("  *** WARNING: date key is NOT 'YYYY-MM-DD' (10 chars) — this will break signal lookup! ***")
else:
    print("  date key looks correct (10 chars, YYYY-MM-DD)")


# ─────────────────────────────────────────────────────────────
# 5. OHLCV data for first ticker — check date index format
# ─────────────────────────────────────────────────────────────
sep("5. OHLCV DATA — date index vs signal key comparison")
probe_ticker = tickers[0]
probe_signals = groups[probe_ticker]
print(f"  Probing ticker: {probe_ticker}")
print(f"  Signal dates  : {[s['date'] for s in probe_signals]}")

try:
    ohlcv = fetch_ticker_data(probe_ticker, probe_signals)
    print(f"  OHLCV rows    : {len(ohlcv)}")
    if ohlcv.empty:
        print("  ERROR: OHLCV is empty — yfinance returned nothing.")
    else:
        print(f"  OHLCV index dtype : {ohlcv.index.dtype}")
        print(f"  OHLCV date range  : {ohlcv.index[0]} → {ohlcv.index[-1]}")
        # Show how the strategy sees each bar's date
        sample_bar_dates = [str(d.date()) for d in ohlcv.index[-5:]]
        print(f"  str(index[-5:].date()) : {sample_bar_dates}")

        # Cross-reference: which signal dates appear in the OHLCV index?
        ohlcv_date_strs = {str(d.date()) for d in ohlcv.index}
        for sig in probe_signals:
            key = sig["date"]
            matched = key in ohlcv_date_strs
            # Also check if it's in OHLCV as-is (before the date() conversion)
            matched_raw = any(str(d) == key for d in ohlcv.index)
            print(f"    signal date {key!r}  in OHLCV(str(date())) → {matched}  "
                  f"in OHLCV(str(ts)) → {matched_raw}")

except Exception as e:
    print(f"  ERROR fetching OHLCV: {e}")


# ─────────────────────────────────────────────────────────────
# 6. Full run_backtest for probe ticker — show result + error
# ─────────────────────────────────────────────────────────────
sep("6. run_backtest RESULT for probe ticker")
print(f"  Ticker  : {probe_ticker}")
print(f"  Signals : {probe_signals}")
try:
    result = run_backtest(
        probe_ticker,
        probe_signals,
        max_hold_days=PARAMS.max_hold_days,
    )
    print(f"  n_signals  : {result['n_signals']}")
    print(f"  n_trades   : {result['n_trades']}")
    print(f"  error      : {result['error']!r}")
    print(f"  return_pct : {result['return_pct']}")
    print(f"  win_rate   : {result['win_rate']}")
    if result.get("trades") is not None and not result["trades"].empty:
        print(f"  trades DF  :\n{result['trades'].to_string()}")
    else:
        print("  trades DF  : None / empty")
except Exception as e:
    print(f"  EXCEPTION in run_backtest: {e}")
    import traceback
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────
# 7. Second ticker (to catch ticker-specific vs systemic)
# ─────────────────────────────────────────────────────────────
if len(tickers) >= 2:
    sep("7. run_backtest for SECOND ticker")
    t2 = tickers[1]
    sigs2 = groups[t2]
    print(f"  Ticker  : {t2}")
    print(f"  Signals : {[s['date'] for s in sigs2]}")
    try:
        r2 = run_backtest(t2, sigs2, max_hold_days=PARAMS.max_hold_days)
        print(f"  n_trades: {r2['n_trades']}  error: {r2['error']!r}  "
              f"return_pct: {r2['return_pct']}")
    except Exception as e:
        print(f"  EXCEPTION: {e}")


# ─────────────────────────────────────────────────────────────
# 8. Summary of all errors across all tickers
# ─────────────────────────────────────────────────────────────
sep("8. BATCH — run all filtered tickers, summarise errors")
errors: dict[str, str] = {}
zero_trade_tickers: list[str] = []
traded_tickers: list[str] = []

for ticker in tickers:
    sigs = groups[ticker]
    try:
        r = run_backtest(ticker, sigs, max_hold_days=PARAMS.max_hold_days)
        if r.get("error"):
            errors[ticker] = r["error"]
        elif r.get("n_trades", 0) == 0:
            zero_trade_tickers.append(ticker)
        else:
            traded_tickers.append(f"{ticker}({r['n_trades']}t)")
    except Exception as e:
        errors[ticker] = f"EXCEPTION: {e}"

print(f"  Tickers run     : {len(tickers)}")
print(f"  With trades     : {len(traded_tickers)}")
print(f"  Zero trades     : {len(zero_trade_tickers)}")
print(f"  Errors          : {len(errors)}")
if errors:
    print(f"\n  Error breakdown:")
    for t, msg in list(errors.items())[:10]:
        print(f"    {t}: {msg}")
if traded_tickers:
    print(f"\n  Traded tickers  : {traded_tickers}")
if zero_trade_tickers:
    print(f"\n  Zero-trade (first 10): {zero_trade_tickers[:10]}")
    # For one zero-trade ticker, show the signal date vs OHLCV overlap
    zt = zero_trade_tickers[0]
    print(f"\n  Zero-trade probe: {zt}")
    zt_sigs = groups[zt]
    try:
        zt_ohlcv = fetch_ticker_data(zt, zt_sigs)
        if zt_ohlcv.empty:
            print(f"    OHLCV empty")
        else:
            ohlcv_dates = {str(d.date()) for d in zt_ohlcv.index}
            for sig in zt_sigs:
                hit = sig["date"] in ohlcv_dates
                print(f"    signal {sig['date']!r} → in OHLCV: {hit}  "
                      f"stop={sig['stop']}")
    except Exception as e:
        print(f"    fetch failed: {e}")

print("\nDone.\n")
