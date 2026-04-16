"""
core/research/walk_forward.py — Walk-forward validation for parameter sweeps.

Two window modes:
  expanding — train window grows each fold; test window stays fixed
  rolling   — both train and test windows slide; train size stays fixed

Fold generation:
  Given total history of N months, train_months T, test_months K:
    n_folds = (N - T) // K

  Expanding:  fold i → train=[start, start + T + i*K], test=(that date, + K)
  Rolling:    fold i → train=[start + i*K, start + i*K + T], test=(end_of_train, + K)

All dates are month-boundary aligned (first day of month).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from core.research.params import SweepParams
from core.research.sweep import prefetch_price_data, run_param_set
from core.research.rescore import (
    build_signal_groups,
    load_raw_signals,
    rescore_signals,
)
from core.research.storage import DB_PATH


# ---------------------------------------------------------------------------
# Split dataclass
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardSplit:
    fold:        int
    train_start: str
    train_end:   str
    test_start:  str
    test_end:    str


# ---------------------------------------------------------------------------
# make_splits
# ---------------------------------------------------------------------------

def make_splits(
    start_date:   str,
    end_date:     str,
    train_months: int,
    test_months:  int,
    mode:         Literal["expanding", "rolling"] = "expanding",
) -> list[WalkForwardSplit]:
    """
    Generate walk-forward splits between start_date and end_date.

    Args:
        start_date:   first date of available history (YYYY-MM-DD)
        end_date:     last date of available history (YYYY-MM-DD)
        train_months: minimum training window length (months)
        test_months:  test window length (months)
        mode:         "expanding" (growing train) or "rolling" (fixed train)

    Returns:
        List of WalkForwardSplit, one per fold. Empty list if insufficient data.
    """
    start = pd.Timestamp(start_date).replace(day=1)
    end   = pd.Timestamp(end_date).replace(day=1)

    # +1 to include the end month itself (Jan→Dec = 12 months, not 11)
    total_months = (end.year - start.year) * 12 + (end.month - start.month) + 1
    n_folds = (total_months - train_months) // test_months

    splits: list[WalkForwardSplit] = []
    for i in range(n_folds):
        if mode == "expanding":
            train_start = start
            train_end   = start + pd.DateOffset(months=train_months + i * test_months)
        else:  # rolling
            train_start = start + pd.DateOffset(months=i * test_months)
            train_end   = train_start + pd.DateOffset(months=train_months)

        test_start = train_end
        test_end   = test_start + pd.DateOffset(months=test_months)

        # Don't emit a fold whose test window goes past end_date
        if test_end > end + pd.DateOffset(months=1):
            break

        splits.append(WalkForwardSplit(
            fold=i + 1,
            train_start=train_start.strftime("%Y-%m-%d"),
            train_end=(train_end - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            test_start=test_start.strftime("%Y-%m-%d"),
            test_end=(test_end - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        ))

    return splits


# ---------------------------------------------------------------------------
# run_walk_forward
# ---------------------------------------------------------------------------

def run_walk_forward(
    param_sets:   list[SweepParams],
    *,
    db_path:        str                            = DB_PATH,
    start_date:     str,
    end_date:       str,
    train_months:   int                            = 12,
    test_months:    int                            = 3,
    mode:           Literal["expanding", "rolling"] = "expanding",
    prefetch_workers: int                          = 8,
    progress_cb     = None,
) -> list[dict]:
    """
    Run walk-forward validation for each param_set across all folds.

    For each fold:
      - Trains (re-scores + filters) on [train_start, train_end] signals.
      - Tests (backtests) on [test_start, test_end] signals.
      - Price data is pre-fetched once per fold across all tickers.

    Returns list of result dicts (one per param_set × fold), each with all
    keys from run_param_set plus fold/train_start/train_end/test_start/test_end.
    """
    splits = make_splits(start_date, end_date, train_months, test_months, mode)
    if not splits:
        return []

    all_results: list[dict] = []

    for split in splits:
        # Load test-period signals (backtest runs only on test window)
        raw_test_df = load_raw_signals(
            db_path=db_path,
            start_date=split.test_start,
            end_date=split.test_end,
        )

        if raw_test_df.empty:
            continue

        # Pre-fetch price data for all tickers once per fold
        _default_scored = rescore_signals(raw_test_df, SweepParams())
        _all_groups     = build_signal_groups(_default_scored)
        price_cache     = prefetch_price_data(_all_groups, max_workers=prefetch_workers)

        total = len(param_sets)
        for idx, params in enumerate(param_sets):
            result = run_param_set(
                params,
                raw_test_df,
                price_cache,
                train_start=split.train_start,
                train_end=split.train_end,
                test_start=split.test_start,
                test_end=split.test_end,
                fold=split.fold,
                db_path=db_path,
                persist=True,
            )
            result["fold"]        = split.fold
            result["train_start"] = split.train_start
            result["train_end"]   = split.train_end
            result["test_start"]  = split.test_start
            result["test_end"]    = split.test_end
            all_results.append(result)

            if progress_cb:
                progress_cb(params.label, idx + 1, total, split.fold)

    return all_results
