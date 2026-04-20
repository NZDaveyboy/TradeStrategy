#!/usr/bin/env python3
"""
backtest_v2.py — Formal backtest engine entry point.

Replaces the ad hoc forward-return lookup in backtest.py with a proper
strategy simulation:
  - Market entry on next bar's open after screener signal date
  - Stop-loss from screener stop_loss column
  - Time-based exit after --max-hold-days trading bars (default 10)
  - Configurable commission

Options backtesting (options_backtest.py) is unchanged — backtesting.py
has no options primitives; B-S re-pricing remains the right approach there.

Usage:
    python3 backtest_v2.py
    python3 backtest_v2.py --max-hold-days 5 --commission 0.001

Results are saved to the backtest_v2 table in screener.db.
backtest.py is kept intact until v2 is confirmed working.
"""

import argparse
import math
import os
import time

from core.db import get_connection
from datetime import datetime, timezone

import pandas as pd

from core.backtest_engine import run_backtest

DB_PATH = os.path.join(os.path.dirname(__file__), "screener.db")


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_table():
    conn = get_connection(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS backtest_v2 (
            ticker          TEXT PRIMARY KEY,
            n_signals       INTEGER,
            n_trades        INTEGER,
            return_pct      REAL,
            sharpe          REAL,
            max_drawdown    REAL,
            win_rate        REAL,
            avg_trade_pct   REAL,
            commission      REAL,
            max_hold_days   INTEGER,
            error           TEXT,
            run_at          TEXT
        )
    """)
    conn.commit()
    conn.close()


def _nan_to_none(v: float) -> float | None:
    return None if math.isnan(v) else v


def save_result(r: dict, commission: float, max_hold_days: int):
    conn = get_connection(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO backtest_v2 VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            r["ticker"],
            r["n_signals"],
            r["n_trades"],
            _nan_to_none(r["return_pct"]),
            _nan_to_none(r["sharpe"]),
            _nan_to_none(r["max_drawdown"]),
            _nan_to_none(r["win_rate"]),
            _nan_to_none(r["avg_trade_pct"]),
            commission,
            max_hold_days,
            r.get("error"),
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        ),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Signal loading
# ---------------------------------------------------------------------------

def load_signals() -> dict[str, list[dict]]:
    """Load screener results grouped by ticker.

    Only includes long and neutral signals — short-side replay is Phase 8+.
    stop_loss from the screener row is used as the strategy stop.
    """
    if not os.path.exists(DB_PATH):
        return {}

    conn = get_connection(DB_PATH)
    df = pd.read_sql(
        """
        SELECT run_date, ticker, price, stop_loss, direction, tradescore, setup_type
        FROM results
        WHERE direction IN ('long', 'neutral')
        ORDER BY run_date
        """,
        conn,
    )
    conn.close()

    by_ticker: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        by_ticker.setdefault(row["ticker"], []).append({
            "date":       row["run_date"],
            "stop":       row["stop_loss"] if pd.notna(row["stop_loss"]) else None,
            "target":     None,               # v2: time exit only
            "tradescore": row.get("tradescore"),
            "setup_type": row.get("setup_type"),
            "direction":  row.get("direction"),
        })
    return by_ticker


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Formal backtest engine (v2)")
    parser.add_argument("--max-hold-days", type=int,   default=10,
                        help="Trading bars before forced exit (default 10)")
    parser.add_argument("--commission",    type=float, default=0.001,
                        help="Round-trip commission per side as decimal (default 0.001 = 0.1%%)")
    parser.add_argument("--cash",          type=float, default=10_000,
                        help="Starting capital per ticker (default $10,000)")
    args = parser.parse_args()

    init_table()
    by_ticker = load_signals()

    if not by_ticker:
        print("No screener results found. Run python3 run.py first.")
        return

    tickers = sorted(by_ticker)
    print(
        f"\nBacktest v2  |  {len(tickers)} tickers  |  "
        f"commission {args.commission * 100:.2f}%  |  "
        f"max hold {args.max_hold_days}d  |  "
        f"cash ${args.cash:,.0f}\n"
    )

    rows = []
    for ticker in tickers:
        signals = by_ticker[ticker]
        result  = run_backtest(
            ticker,
            signals,
            cash=args.cash,
            commission=args.commission,
            max_hold_days=args.max_hold_days,
        )
        save_result(result, args.commission, args.max_hold_days)
        rows.append(result)

        if result["error"]:
            print(f"  SKIP  {ticker:<12}  {result['error']}")
        else:
            print(
                f"  {ticker:<12}  "
                f"signals {result['n_signals']:2d}  "
                f"trades {result['n_trades']:2d}  "
                f"ret {result['return_pct']:+6.1f}%  "
                f"sharpe {result['sharpe']:5.2f}  "
                f"wr {result['win_rate']:4.0f}%  "
                f"dd {result['max_drawdown']:5.1f}%"
            )
        time.sleep(0.1)   # avoid yfinance rate-limiting

    # ── Portfolio summary ────────────────────────────────────────────────────
    valid = [r for r in rows if not r["error"] and r["n_trades"] > 0]
    print(f"\n{'─'*60}")
    if valid:
        print(f"Summary  |  {len(valid)} tickers with trades\n")
        print(f"  Avg return     {sum(r['return_pct']    for r in valid) / len(valid):+.1f}%")
        print(f"  Avg win rate   {sum(r['win_rate']      for r in valid) / len(valid):.0f}%")
        print(f"  Avg Sharpe     {sum(r['sharpe']        for r in valid) / len(valid):.2f}")
        print(f"  Avg drawdown   {sum(r['max_drawdown']  for r in valid) / len(valid):.1f}%")
        print(f"  Avg trade      {sum(r['avg_trade_pct'] for r in valid) / len(valid):+.1f}%")
    else:
        print("No completed trades. Check screener has run (python3 run.py).")

    print(f"\nResults saved → backtest_v2 table in screener.db")


if __name__ == "__main__":
    main()
