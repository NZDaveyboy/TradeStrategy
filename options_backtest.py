#!/usr/bin/env python3
"""
Options backtester for TradeStrategy.

For each screener entry, simulates buying:
  - ATM call  (30 DTE, strike = current price)
  - OTM call  (30 DTE, strike = price * 1.05)
  - ATM call  (45 DTE)

Uses Black-Scholes with 30-day realised volatility as the IV input.
No historical IV data needed — this is a simulation, not a replay of
real market prices. IV crush is NOT modelled (conservative assumption:
IV stays constant). Real results will be worse when buying into spikes.

Forward returns at 1, 3, 5, 10 trading days are calculated by
re-pricing the option using B-S with reduced time to expiry.

Usage:
    python3 options_backtest.py

Run after each session alongside backtest.py.
"""

import math
import os
import sqlite3
import time

import numpy as np
import pandas as pd

from providers.yfinance_provider import YFinanceProvider

_provider = YFinanceProvider()

DB_PATH = os.path.join(os.path.dirname(__file__), "screener.db")

RISK_FREE = 0.045   # US 10yr proxy

STRATEGIES = [
    {"name": "atm_call_30d", "moneyness": 1.00, "dte": 30, "type": "call"},
    {"name": "otm_call_30d", "moneyness": 1.05, "dte": 30, "type": "call"},
    {"name": "atm_call_45d", "moneyness": 1.00, "dte": 45, "type": "call"},
]

FORWARD_DAYS = [1, 3, 5, 10]


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------

def _ncdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def bs_price(S, K, T, r, sigma, opt="call") -> float:
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if opt == "call" else max(K - S, 0)
        return float(intrinsic)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt == "call":
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def bs_delta(S, K, T, r, sigma, opt="call") -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _ncdf(d1) if opt == "call" else _ncdf(d1) - 1


# ---------------------------------------------------------------------------
# Realised vol
# ---------------------------------------------------------------------------

def realised_vol(ticker: str, as_of: str, window: int = 30) -> float | None:
    """Annualised 30d realised vol from daily returns ending on as_of."""
    try:
        hist = _provider.get_ohlcv(ticker, "90d", "1d")
        if hist.empty or len(hist) < window + 1:
            return None
        hist.index = hist.index.tz_localize(None)
        hist = hist[hist.index.date <= pd.Timestamp(as_of).date()]
        if len(hist) < window + 1:
            return None
        log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        rv = float(log_ret.tail(window).std() * math.sqrt(252))
        return rv if rv > 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Forward prices
# ---------------------------------------------------------------------------

def forward_closes(ticker: str, run_date: str) -> dict:
    """Returns {1: price, 3: price, 5: price, 10: price} after run_date."""
    from datetime import datetime, timedelta
    start = datetime.strptime(run_date, "%Y-%m-%d")
    end   = start + timedelta(days=25)
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
    return {n: float(closes.iloc[n]) for n in FORWARD_DAYS if len(closes) > n}


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_table():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS backtest_options (
            run_date        TEXT,
            ticker          TEXT,
            screener_score  INTEGER,
            strategy_name   TEXT,
            opt_type        TEXT,
            dte             INTEGER,
            entry_stock_px  REAL,
            strike          REAL,
            entry_iv        REAL,
            entry_opt_px    REAL,
            entry_delta     REAL,
            fwd_stock_1d    REAL,  fwd_stock_3d  REAL,
            fwd_stock_5d    REAL,  fwd_stock_10d REAL,
            opt_px_1d       REAL,  opt_px_3d     REAL,
            opt_px_5d       REAL,  opt_px_10d    REAL,
            return_1d       REAL,  return_3d     REAL,
            return_5d       REAL,  return_10d    REAL,
            PRIMARY KEY (run_date, ticker, strategy_name)
        )
    """)
    conn.commit()
    conn.close()


def already_done(run_date, ticker, strategy_name) -> bool:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT 1 FROM backtest_options WHERE run_date=? AND ticker=? AND strategy_name=?",
        (run_date, ticker, strategy_name),
    ).fetchone()
    conn.close()
    return row is not None


def save_row(r: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO backtest_options VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        r["run_date"], r["ticker"], r["screener_score"],
        r["strategy_name"], r["opt_type"], r["dte"],
        r["entry_stock_px"], r["strike"], r["entry_iv"],
        r["entry_opt_px"], r["entry_delta"],
        r.get("fwd_stock_1d"),  r.get("fwd_stock_3d"),
        r.get("fwd_stock_5d"),  r.get("fwd_stock_10d"),
        r.get("opt_px_1d"),     r.get("opt_px_3d"),
        r.get("opt_px_5d"),     r.get("opt_px_10d"),
        r.get("return_1d"),     r.get("return_3d"),
        r.get("return_5d"),     r.get("return_10d"),
    ))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    init_table()

    conn = sqlite3.connect(DB_PATH)
    screener = pd.read_sql(
        "SELECT run_date, ticker, strategy, asset, score, price FROM results "
        "WHERE asset = 'equity' ORDER BY run_date",
        conn,
    )
    conn.close()

    if screener.empty:
        print("No equity screener results. Run python3 run.py first.")
        return

    print(f"Options backtest: {len(screener)} equity entries × {len(STRATEGIES)} strategies\n")

    processed = skipped = errors = 0

    for _, eq in screener.iterrows():
        run_date = eq["run_date"]
        ticker   = eq["ticker"]
        S        = float(eq["price"])

        # Shared data per ticker/date
        iv = None
        fwd = None

        for strat in STRATEGIES:
            sname = strat["name"]
            if already_done(run_date, ticker, sname):
                skipped += 1
                continue

            # Lazy-load once per ticker/date
            if iv is None:
                iv = realised_vol(ticker, run_date) or 0.30
            if fwd is None:
                fwd = forward_closes(ticker, run_date)

            K       = round(S * strat["moneyness"], 2)
            T_entry = strat["dte"] / 365.0
            opt_t   = strat["type"]

            entry_px = bs_price(S, K, T_entry, RISK_FREE, iv, opt_t)
            delta    = bs_delta(S, K, T_entry, RISK_FREE, iv, opt_t)

            if entry_px <= 0:
                errors += 1
                continue

            row = {
                "run_date":       run_date,
                "ticker":         ticker,
                "screener_score": int(eq["score"]),
                "strategy_name":  sname,
                "opt_type":       opt_t,
                "dte":            strat["dte"],
                "entry_stock_px": round(S, 4),
                "strike":         K,
                "entry_iv":       round(iv, 4),
                "entry_opt_px":   round(entry_px, 4),
                "entry_delta":    round(delta, 3),
            }

            for n in FORWARD_DAYS:
                S_fwd = fwd.get(n)
                if S_fwd:
                    T_fwd = max((strat["dte"] - n) / 365.0, 0)
                    opt_px_fwd = bs_price(S_fwd, K, T_fwd, RISK_FREE, iv, opt_t)
                    ret = (opt_px_fwd / entry_px - 1) * 100
                    row[f"fwd_stock_{n}d"] = round(S_fwd, 4)
                    row[f"opt_px_{n}d"]    = round(opt_px_fwd, 4)
                    row[f"return_{n}d"]    = round(ret, 2)

            save_row(row)
            processed += 1

            avail = [f"{n}d={row[f'return_{n}d']:+.1f}%"
                     for n in FORWARD_DAYS if row.get(f"return_{n}d") is not None]
            print(
                f"  {ticker:<10} {run_date}  [{sname}]  "
                f"entry ${entry_px:.3f}  delta {delta:.2f}  "
                + ("  ".join(avail) if avail else "no fwd data")
            )

        time.sleep(0.1)

    print(f"\nDone. {processed} new, {skipped} skipped, {errors} skipped (zero price).")

    # Summary
    conn = sqlite3.connect(DB_PATH)
    summary = pd.read_sql("""
        SELECT strategy_name, screener_score AS score,
               COUNT(*)                               AS trades,
               ROUND(AVG(return_1d), 1)               AS avg_1d,
               ROUND(AVG(return_3d), 1)               AS avg_3d,
               ROUND(AVG(return_5d), 1)               AS avg_5d,
               ROUND(SUM(CASE WHEN return_1d > 0 THEN 1 ELSE 0 END)
                     * 100.0 / COUNT(*), 0)            AS win_1d_pct
        FROM backtest_options
        WHERE return_1d IS NOT NULL
        GROUP BY strategy_name, screener_score
        ORDER BY strategy_name, screener_score DESC
    """, conn)
    conn.close()

    if not summary.empty:
        print("\nOptions return by strategy + score:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
