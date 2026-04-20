#!/usr/bin/env python3
"""
Backtester for TradeStrategy screener results.

For each screener run in the DB, fetches subsequent price data and
calculates forward returns at 1, 3, 5, and 10 trading days.

Entry assumption: close price on the screener run date (already in DB).
You see the screener after close — you enter next morning. This slightly
understates real returns on gap-ups but keeps the numbers honest.

Usage:
    python3 backtest.py

Results are saved to the backtest table in screener.db and displayed
in the Backtest tab of the Streamlit app.
"""

import os
import time

from core.db import get_connection, sync_if_turso
from datetime import datetime, timedelta

import pandas as pd

from providers.yfinance_provider import YFinanceProvider

_provider = YFinanceProvider()

DB_PATH = os.path.join(os.path.dirname(__file__), "screener.db")

FORWARD_DAYS = [1, 3, 5, 10]


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_backtest_table():
    conn = get_connection(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS backtest (
            run_date    TEXT,
            ticker      TEXT,
            strategy    TEXT,
            asset       TEXT,
            score       INTEGER,
            change_pct  REAL,
            rvol        REAL,
            entry_price REAL,
            price_1d    REAL,
            price_3d    REAL,
            price_5d    REAL,
            price_10d   REAL,
            return_1d   REAL,
            return_3d   REAL,
            return_5d   REAL,
            return_10d  REAL,
            PRIMARY KEY (run_date, ticker)
        )
    """)
    conn.commit()
    sync_if_turso(conn)
    conn.close()


def load_screener_results() -> pd.DataFrame:
    conn = get_connection(DB_PATH)
    df = pd.read_sql(
        "SELECT run_date, ticker, strategy, asset, score, change_pct, rvol, price "
        "FROM results ORDER BY run_date",
        conn,
    )
    conn.close()
    return df


def already_processed(run_date: str, ticker: str) -> bool:
    conn = get_connection(DB_PATH)
    row = conn.execute(
        "SELECT 1 FROM backtest WHERE run_date = ? AND ticker = ?",
        (run_date, ticker),
    ).fetchone()
    conn.close()
    return row is not None


def save_backtest_row(row: dict):
    conn = get_connection(DB_PATH)
    conn.execute(
        """
        INSERT OR REPLACE INTO backtest VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            row["run_date"], row["ticker"], row["strategy"], row["asset"],
            row["score"], row["change_pct"], row["rvol"], row["entry_price"],
            row.get("price_1d"),  row.get("price_3d"),
            row.get("price_5d"),  row.get("price_10d"),
            row.get("return_1d"), row.get("return_3d"),
            row.get("return_5d"), row.get("return_10d"),
        ),
    )
    conn.commit()
    sync_if_turso(conn)
    conn.close()


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------

def fetch_forward_prices(ticker: str, run_date: str) -> dict:
    """
    Returns {1: price, 3: price, 5: price, 10: price} for trading days
    after run_date. Any day not yet available is None.
    """
    start = datetime.strptime(run_date, "%Y-%m-%d")
    end   = start + timedelta(days=20)   # enough buffer for weekends/holidays

    try:
        hist = _provider.get_ohlcv_range(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
        )
    except Exception:
        return {}

    if hist.empty:
        return {}

    closes = hist["Close"].reset_index(drop=True)

    result = {}
    for n in FORWARD_DAYS:
        # Index 0 = run_date close (entry), index n = n trading days later
        if len(closes) > n:
            result[n] = float(closes.iloc[n])
        else:
            result[n] = None

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    init_backtest_table()
    results = load_screener_results()

    if results.empty:
        print("No screener results found. Run python3 run.py first.")
        return

    run_dates = results["run_date"].unique()
    print(f"Backtesting {len(results)} screener entries across {len(run_dates)} run date(s)...\n")

    processed = skipped = 0

    for _, row in results.iterrows():
        run_date = row["run_date"]
        ticker   = row["ticker"]

        if already_processed(run_date, ticker):
            skipped += 1
            continue

        prices = fetch_forward_prices(ticker, run_date)
        entry  = float(row["price"])

        bt_row = {
            "run_date":    run_date,
            "ticker":      ticker,
            "strategy":    row["strategy"],
            "asset":       row["asset"],
            "score":       int(row["score"]),
            "change_pct":  float(row["change_pct"]),
            "rvol":        float(row["rvol"]),
            "entry_price": entry,
        }

        for n in FORWARD_DAYS:
            px = prices.get(n)
            key_p = f"price_{n}d"
            key_r = f"return_{n}d"
            if px and entry:
                bt_row[key_p] = round(px, 4)
                bt_row[key_r] = round((px / entry - 1) * 100, 2)
            else:
                bt_row[key_p] = None
                bt_row[key_r] = None

        save_backtest_row(bt_row)

        available = [f"{n}d={bt_row[f'return_{n}d']:+.1f}%" for n in FORWARD_DAYS if bt_row.get(f"return_{n}d") is not None]
        print(
            f"  {ticker:<12}  {run_date}  score={bt_row['score']}  "
            + ("  ".join(available) if available else "no forward data yet")
        )

        processed += 1
        time.sleep(0.15)

    print(f"\nDone. {processed} new entries processed, {skipped} already up to date.")

    # Quick summary
    conn = get_connection(DB_PATH)
    summary = pd.read_sql(
        """
        SELECT score,
               COUNT(*)                        AS trades,
               ROUND(AVG(return_1d), 2)        AS avg_1d,
               ROUND(AVG(return_3d), 2)        AS avg_3d,
               ROUND(AVG(return_5d), 2)        AS avg_5d,
               ROUND(SUM(CASE WHEN return_1d > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS win_rate_1d
        FROM backtest
        WHERE return_1d IS NOT NULL
        GROUP BY score
        ORDER BY score DESC
        """,
        conn,
    )
    conn.close()

    if not summary.empty:
        print("\nReturn by score (all history):")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
