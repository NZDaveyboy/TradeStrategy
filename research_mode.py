#!/usr/bin/env python3
"""
research_mode.py — CLI for parameter sweeps and walk-forward validation.

Subcommands:
    sweep         Run a parameter sweep against all historical signals.
    walk-forward  Run walk-forward validation (expanding or rolling windows).
    compare       Load and display results from previous runs.

Usage examples:
    python research_mode.py sweep --mode thresholds
    python research_mode.py sweep --mode weights
    python research_mode.py sweep --tradescore-threshold 35 45 55 --min-rvol 1.0 2.0
    python research_mode.py walk-forward --mode thresholds --train-months 12 --test-months 3
    python research_mode.py compare --top 20
    python research_mode.py compare --label "tradescore_threshold=35|min_rvol=1.0"
"""

from __future__ import annotations

import argparse
import sys

from core.research.compare import load_runs, print_comparison_table, rank_runs
from core.research.params import (
    THRESHOLD_SWEEP_DEFAULTS,
    WEIGHT_SWEEP_DEFAULTS,
    SweepParams,
    param_grid,
)
from core.research.storage import DB_PATH
from core.research.sweep import run_sweep
from core.research.walk_forward import run_walk_forward


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _sweep_progress(label: str, current: int, total: int, fold: int | None = None) -> None:
    fold_tag = f" [fold {fold}]" if fold is not None else ""
    pct      = int(100 * current / total)
    print(f"  [{pct:3d}%] {current}/{total}{fold_tag}  {label}", flush=True)


def _prefetch_progress(ticker: str) -> None:
    print(f"  fetched {ticker}", flush=True)


# ---------------------------------------------------------------------------
# Subcommand: sweep
# ---------------------------------------------------------------------------

def _cmd_sweep(args: argparse.Namespace) -> None:
    param_sets = _build_param_sets(args)
    print(f"\nSweep: {len(param_sets)} param sets")
    print(f"  DB: {args.db}")
    if args.start:
        print(f"  Signal range: {args.start} → {args.end or 'latest'}")
    print()

    results = run_sweep(
        param_sets,
        db_path=args.db,
        start_date=args.start,
        end_date=args.end,
        prefetch_workers=args.workers,
        progress_cb=_sweep_progress,
    )

    # Quick summary
    print(f"\nDone. {len(results)} results saved.\n")
    _print_quick_summary(results)


# ---------------------------------------------------------------------------
# Subcommand: walk-forward
# ---------------------------------------------------------------------------

def _cmd_walk_forward(args: argparse.Namespace) -> None:
    if not args.history_start or not args.history_end:
        print("Error: --history-start and --history-end are required for walk-forward.", file=sys.stderr)
        sys.exit(1)

    param_sets = _build_param_sets(args)
    from core.research.walk_forward import make_splits
    splits = make_splits(
        args.history_start, args.history_end,
        args.train_months, args.test_months,
        args.wf_mode,
    )

    print(f"\nWalk-forward ({args.wf_mode}): {len(splits)} fold(s) × {len(param_sets)} param sets")
    print(f"  DB: {args.db}")
    print(f"  History: {args.history_start} → {args.history_end}")
    print(f"  Train: {args.train_months} months | Test: {args.test_months} months")
    if not splits:
        print("  WARNING: Insufficient data to produce any folds.")
    print()

    results = run_walk_forward(
        param_sets,
        db_path=args.db,
        start_date=args.history_start,
        end_date=args.history_end,
        train_months=args.train_months,
        test_months=args.test_months,
        mode=args.wf_mode,
        prefetch_workers=args.workers,
        progress_cb=_sweep_progress,
    )

    print(f"\nDone. {len(results)} results saved.\n")
    _print_quick_summary(results)


# ---------------------------------------------------------------------------
# Subcommand: compare
# ---------------------------------------------------------------------------

def _cmd_compare(args: argparse.Namespace) -> None:
    df = load_runs(db_path=args.db, run_label=args.label, fold=args.fold)
    if df.empty:
        print("No results found in database.")
        return

    ranked = rank_runs(df)
    print(f"\nLoaded {len(ranked)} run(s) from {args.db}\n")
    print_comparison_table(ranked, top_n=args.top)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_param_sets(args: argparse.Namespace) -> list[SweepParams]:
    """Construct param sets from args — explicit lists or preset modes."""
    if args.mode == "thresholds":
        kwargs = {k: v for k, v in THRESHOLD_SWEEP_DEFAULTS.items()}
    elif args.mode == "weights":
        kwargs = {k: v for k, v in WEIGHT_SWEEP_DEFAULTS.items()}
    else:
        # Manual: pick up any explicit list arguments
        kwargs: dict = {}
        _maybe_add(kwargs, "tradescore_threshold", args.tradescore_threshold)
        _maybe_add(kwargs, "min_rvol",             args.min_rvol)
        _maybe_add(kwargs, "stop_multiplier",      args.stop_multiplier)
        _maybe_add(kwargs, "max_hold_days",        args.max_hold_days)

    if not kwargs:
        # Single default run
        return [SweepParams()]

    return param_grid(**kwargs)


def _maybe_add(d: dict, key: str, val: list | None) -> None:
    if val:
        d[key] = val


def _print_quick_summary(results: list[dict]) -> None:
    """Print top-5 by n_trades as a quick sanity check."""
    if not results:
        return
    sorted_r = sorted(results, key=lambda r: r.get("n_trades") or 0, reverse=True)
    print(f"{'Label':<45} {'Trades':>6} {'Win%':>6} {'Exp%':>6} {'Sharpe':>7}")
    print("-" * 75)
    for r in sorted_r[:5]:
        label  = (r.get("label") or "")[:44]
        trades = r.get("n_trades") or 0
        win    = f"{r['win_rate']:.1f}"   if r.get("win_rate")   is not None else "-"
        exp    = f"{r['expectancy']:.1f}" if r.get("expectancy") is not None else "-"
        sh     = f"{r['sharpe']:.2f}"     if r.get("sharpe")     is not None else "-"
        print(f"{label:<45} {trades:>6} {win:>6} {exp:>6} {sh:>7}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research_mode.py",
        description="Parameter sweep and walk-forward validation for TradeStrategy.",
    )
    parser.add_argument("--db", default=DB_PATH, help="Path to screener.db (default: auto-detected)")

    sub = parser.add_subparsers(dest="command", required=True)

    # ── sweep ──────────────────────────────────────────────────────────────
    sweep_p = sub.add_parser("sweep", help="Run a parameter sweep.")
    sweep_p.add_argument(
        "--mode", choices=["thresholds", "weights", "manual"], default="thresholds",
        help="Preset sweep mode (default: thresholds). 'manual' reads explicit --* args.",
    )
    sweep_p.add_argument("--start", default=None, metavar="YYYY-MM-DD", help="Signal start date filter.")
    sweep_p.add_argument("--end",   default=None, metavar="YYYY-MM-DD", help="Signal end date filter.")
    sweep_p.add_argument("--workers", type=int, default=8, help="Concurrent threads for price data fetch.")
    _add_param_args(sweep_p)

    # ── walk-forward ───────────────────────────────────────────────────────
    wf_p = sub.add_parser("walk-forward", help="Run walk-forward validation.")
    wf_p.add_argument(
        "--mode", choices=["thresholds", "weights", "manual"], default="thresholds",
        help="Preset sweep mode (default: thresholds).",
    )
    wf_p.add_argument("--history-start", default=None, metavar="YYYY-MM-DD", help="Start of historical data.")
    wf_p.add_argument("--history-end",   default=None, metavar="YYYY-MM-DD", help="End of historical data.")
    wf_p.add_argument("--train-months",  type=int, default=12, help="Training window in months (default 12).")
    wf_p.add_argument("--test-months",   type=int, default=3,  help="Test window in months (default 3).")
    wf_p.add_argument(
        "--wf-mode", choices=["expanding", "rolling"], default="expanding",
        help="Walk-forward mode: expanding (default) or rolling.",
    )
    wf_p.add_argument("--workers", type=int, default=8)
    _add_param_args(wf_p)

    # ── compare ────────────────────────────────────────────────────────────
    cmp_p = sub.add_parser("compare", help="Display and rank saved sweep results.")
    cmp_p.add_argument("--top",   type=int, default=20, help="Number of rows to show (default 20).")
    cmp_p.add_argument("--label", default=None, help="Filter by exact run_label.")
    cmp_p.add_argument("--fold",  type=int, default=None, help="Filter by fold number.")

    return parser


def _add_param_args(p: argparse.ArgumentParser) -> None:
    """Add optional explicit parameter list arguments."""
    p.add_argument("--tradescore-threshold", type=float, nargs="+", metavar="N")
    p.add_argument("--min-rvol",             type=float, nargs="+", metavar="N")
    p.add_argument("--stop-multiplier",      type=float, nargs="+", metavar="N")
    p.add_argument("--max-hold-days",        type=int,   nargs="+", metavar="N")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _parser = _build_parser()
    _args   = _parser.parse_args()

    if _args.command == "sweep":
        _cmd_sweep(_args)
    elif _args.command == "walk-forward":
        _cmd_walk_forward(_args)
    elif _args.command == "compare":
        _cmd_compare(_args)
    else:
        _parser.print_help()
        sys.exit(1)
