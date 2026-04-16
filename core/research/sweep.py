"""
core/research/sweep.py — Orchestrate a parameter sweep over historical signals.

Design:
  1. Load raw signals from screener.db once.
  2. Pre-fetch OHLCV price data for every unique ticker once (shared across all param sets).
  3. For each SweepParams in the grid:
       a. Re-score signals with the param set's weight overrides.
       b. Filter using threshold params.
       c. Build signal groups.
       d. Run backtest for each ticker (using cached OHLCV).
       e. Aggregate metrics across tickers.
       f. Persist to research_runs + research_results in screener.db.

Runtime target: 125-combo sweep in < 10 minutes.
  - Price fetch: ~2s/ticker × 48 tickers ≈ 96s (one-time)
  - Re-score + backtest: ~5ms/ticker × 48 × 125 combos ≈ 30s
  - Total ≈ ~2 min on a warm machine.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import pandas as pd

from core.backtest_engine import fetch_ticker_data, run_backtest
from core.research.params import SweepParams
from core.research.rescore import (
    build_signal_groups,
    filter_signals,
    load_raw_signals,
    rescore_signals,
)
from core.research.storage import DB_PATH, save_run


# ---------------------------------------------------------------------------
# Price data pre-caching
# ---------------------------------------------------------------------------

def prefetch_price_data(
    signal_groups: dict[str, list[dict]],
    *,
    max_workers: int = 8,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for every ticker in signal_groups concurrently.

    Returns {ticker: DataFrame} (empty DataFrame if fetch failed).
    """
    price_cache: dict[str, pd.DataFrame] = {}

    def _fetch(ticker: str, signals: list[dict]) -> tuple[str, pd.DataFrame]:
        try:
            return ticker, fetch_ticker_data(ticker, signals)
        except Exception:
            return ticker, pd.DataFrame()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch, ticker, sigs): ticker
            for ticker, sigs in signal_groups.items()
        }
        for fut in as_completed(futures):
            ticker, data = fut.result()
            price_cache[ticker] = data
            if progress_cb:
                progress_cb(ticker)

    return price_cache


# ---------------------------------------------------------------------------
# Single-param backtest
# ---------------------------------------------------------------------------

def _aggregate_ticker_results(ticker_results: list[dict]) -> dict:
    """
    Aggregate per-ticker backtest results into sweep-level metrics.

    Metrics:
        n_tickers    — tickers with at least 1 trade
        n_trades     — total trades across all tickers
        win_rate     — trade-weighted win rate (%)
        expectancy   — trade-weighted avg_trade_pct (%)
        sharpe       — avg of per-ticker Sharpe (NaN tickers excluded)
        max_drawdown — worst single-ticker max drawdown (%)
        return_pct   — avg of per-ticker return_pct (tickers with trades only)
    """
    active = [r for r in ticker_results if (r.get("n_trades") or 0) > 0]

    n_tickers = len(active)
    n_trades  = sum(r.get("n_trades", 0) for r in active)

    def _safe(v) -> float | None:
        try:
            f = float(v)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    # Trade-weighted win_rate
    if n_trades > 0:
        win_rate = (
            sum(
                (r.get("n_trades", 0)) * (_safe(r.get("win_rate")) or 0)
                for r in active
            )
            / n_trades
        )
    else:
        win_rate = None

    # Trade-weighted expectancy (avg_trade_pct)
    if n_trades > 0:
        expectancy = (
            sum(
                (r.get("n_trades", 0)) * (_safe(r.get("avg_trade_pct")) or 0)
                for r in active
            )
            / n_trades
        )
    else:
        expectancy = None

    # Mean Sharpe (exclude NaN)
    sharpes = [_safe(r.get("sharpe")) for r in active if _safe(r.get("sharpe")) is not None]
    sharpe  = (sum(sharpes) / len(sharpes)) if sharpes else None

    # Worst drawdown
    drawdowns = [_safe(r.get("max_drawdown")) for r in active if _safe(r.get("max_drawdown")) is not None]
    max_drawdown = min(drawdowns) if drawdowns else None   # drawdowns are negative

    # Mean return
    returns    = [_safe(r.get("return_pct")) for r in active if _safe(r.get("return_pct")) is not None]
    return_pct = (sum(returns) / len(returns)) if returns else None

    return {
        "n_tickers":    n_tickers,
        "n_trades":     n_trades,
        "win_rate":     win_rate,
        "expectancy":   expectancy,
        "sharpe":       sharpe,
        "max_drawdown": max_drawdown,
        "return_pct":   return_pct,
    }


def run_param_set(
    params:       SweepParams,
    raw_df:       pd.DataFrame,
    price_cache:  dict[str, pd.DataFrame],
    *,
    train_start:  str | None = None,
    train_end:    str | None = None,
    test_start:   str | None = None,
    test_end:     str | None = None,
    fold:         int | None = None,
    db_path:      str = DB_PATH,
    persist:      bool = True,
) -> dict:
    """
    Run one complete sweep leg for a single SweepParams.

    Returns aggregate metrics dict (same keys as _aggregate_ticker_results, plus
    run_id, label, n_signals_raw, n_signals_filtered, ticker_results).
    """
    # ── Re-score and filter ───────────────────────────────────────────────
    scored_df  = rescore_signals(raw_df, params)
    filtered_df = filter_signals(scored_df, params)

    n_signals_raw      = len(raw_df)
    n_signals_filtered = len(filtered_df)

    if filtered_df.empty:
        agg = {
            "n_tickers": 0, "n_trades": 0,
            "win_rate": None, "expectancy": None,
            "sharpe": None, "max_drawdown": None, "return_pct": None,
        }
        ticker_results: list[dict] = []
    else:
        signal_groups  = build_signal_groups(filtered_df)
        ticker_results = []

        for ticker, signals in signal_groups.items():
            data = price_cache.get(ticker, pd.DataFrame())
            result = run_backtest(
                ticker,
                signals,
                data=data if not data.empty else None,
                max_hold_days=params.max_hold_days,
            )
            ticker_results.append(result)

        agg = _aggregate_ticker_results(ticker_results)

    # ── Persist ───────────────────────────────────────────────────────────
    run_id: int | None = None
    if persist:
        per_ticker_rows = [
            {
                "ticker":       r["ticker"],
                "n_signals":    r.get("n_signals"),
                "n_trades":     r.get("n_trades"),
                "win_rate":     r.get("win_rate"),
                "expectancy":   r.get("avg_trade_pct"),
                "sharpe":       r.get("sharpe"),
                "max_drawdown": r.get("max_drawdown"),
                "return_pct":   r.get("return_pct"),
                "error":        r.get("error"),
            }
            for r in ticker_results
        ]
        run_id = save_run(
            db_path=db_path,
            run_label=params.label,
            param_json=params.to_json(),
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            fold=fold,
            ticker_rows=per_ticker_rows,
            **{k: v for k, v in agg.items()},
        )

    return {
        "run_id":              run_id,
        "label":               params.label,
        "n_signals_raw":       n_signals_raw,
        "n_signals_filtered":  n_signals_filtered,
        "ticker_results":      ticker_results,
        **agg,
    }


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------

def run_sweep(
    param_sets:   list[SweepParams],
    *,
    db_path:      str        = DB_PATH,
    start_date:   str | None = None,
    end_date:     str | None = None,
    train_start:  str | None = None,
    train_end:    str | None = None,
    test_start:   str | None = None,
    test_end:     str | None = None,
    fold:         int | None = None,
    prefetch_workers: int = 8,
    progress_cb:  Callable[[str, int, int], None] | None = None,
) -> list[dict]:
    """
    Run a full parameter sweep.

    Args:
        param_sets:        list of SweepParams from param_grid()
        db_path:           path to screener.db
        start_date:        filter signals >= this date (YYYY-MM-DD)
        end_date:          filter signals <= this date (YYYY-MM-DD)
        train_start/end:   metadata only — stored in research_runs
        test_start/end:    metadata only — stored in research_runs
        fold:              metadata only — stored in research_runs
        prefetch_workers:  threads for concurrent price data fetching
        progress_cb:       called with (label, current_idx, total) after each param set

    Returns:
        list of result dicts (one per param_set), in the same order.
    """
    # Load raw signals once
    raw_df = load_raw_signals(db_path=db_path, start_date=start_date, end_date=end_date)

    if raw_df.empty:
        return [
            {"run_id": None, "label": p.label, "n_signals_raw": 0,
             "n_signals_filtered": 0, "n_tickers": 0, "n_trades": 0,
             "win_rate": None, "expectancy": None, "sharpe": None,
             "max_drawdown": None, "return_pct": None, "ticker_results": []}
            for p in param_sets
        ]

    # Build full signal groups for pre-caching (use default params — we just need dates)
    from core.research.params import SweepParams as _SP
    _default_scored  = rescore_signals(raw_df, _SP())
    _all_groups      = build_signal_groups(_default_scored)

    price_cache = prefetch_price_data(
        _all_groups,
        max_workers=prefetch_workers,
    )

    # Run each param set sequentially
    results: list[dict] = []
    total = len(param_sets)
    for idx, params in enumerate(param_sets):
        result = run_param_set(
            params,
            raw_df,
            price_cache,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            fold=fold,
            db_path=db_path,
            persist=True,
        )
        results.append(result)
        if progress_cb:
            progress_cb(params.label, idx + 1, total)

    return results
