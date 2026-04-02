#!/usr/bin/env python3
"""
TradeStrategy screener pipeline.

Usage:
    python run.py

Schedule this to run before US market open (e.g. 9am ET).
Results are stored in screener.db.

Cron example (9am ET, Mon-Fri):
    0 13 * * 1-5 cd /path/to/TradeStrategy && python run.py
"""

import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

DB_PATH = os.path.join(os.path.dirname(__file__), "screener.db")

STRATEGY_FILES = {
    "ai":       "tickers_ai.txt",
    "tech":     "tickers_tech.txt",
    "crypto":   "tickers_crypto.txt",
    "momentum": "tickers_momentum.txt",
}

# Momentum filters applied to Finviz top gainers only.
# Curated strategy tickers are always included regardless.
FILTER_MIN_RVOL   = 3.0
FILTER_MIN_CHANGE = 5.0
FILTER_MIN_PRICE  = 1.0
FILTER_MAX_PRICE  = 50.0
FILTER_MAX_RSI    = 75.0


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            run_date         TEXT,
            ticker           TEXT,
            strategy         TEXT,
            asset            TEXT,
            price            REAL,
            change_pct       REAL,
            rvol             REAL,
            ema9             REAL,
            ema20            REAL,
            ema200           REAL,
            rsi              REAL,
            atr              REAL,
            stop_loss        REAL,
            macd             REAL,
            macd_signal      REAL,
            vwap             REAL,
            volume_trend_up  INTEGER,
            score            INTEGER,
            market_cap       REAL,
            float_shares     REAL,
            PRIMARY KEY (run_date, ticker)
        )
    """)
    # Migrate existing DBs that pre-date the new columns
    for col, col_type in [("market_cap", "REAL"), ("float_shares", "REAL")]:
        try:
            conn.execute(f"ALTER TABLE results ADD COLUMN {col} {col_type}")
        except Exception:
            pass  # column already exists
    conn.commit()
    conn.close()


def save_results(run_date: str, rows: list[dict]):
    conn = sqlite3.connect(DB_PATH)
    for r in rows:
        conn.execute(
            """
            INSERT OR REPLACE INTO results VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_date, r["ticker"], r["strategy"], r["asset"],
                r["price"], r["change_pct"], r["rvol"],
                r["ema9"], r["ema20"], r["ema200"],
                r["rsi"], r["atr"], r["stop_loss"],
                r["macd"], r["macd_signal"], r["vwap"],
                r["volume_trend_up"], r["score"],
                r.get("market_cap"), r.get("float_shares"),
            ),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Ticker loading
# ---------------------------------------------------------------------------

def fetch_finviz_gainers(limit: int = 50) -> list[str]:
    url = "https://finviz.com/screener.ashx?v=111&s=ta_topgainers"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        tickers = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("quote.ashx?t="):
                t = href.split("t=")[1].split("&")[0].strip().upper()
                if t and t not in tickers:
                    tickers.append(t)
        print(f"Finviz: {len(tickers)} top gainers")
        return tickers[:limit]
    except Exception as e:
        print(f"Finviz fetch failed: {e}")
        return []


def load_ticker_file(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.strip().upper() for line in f if line.strip()]


def build_ticker_map() -> dict[str, str]:
    """Returns {ticker: strategy} for all tickers to process."""
    base = os.path.dirname(__file__)
    ticker_map: dict[str, str] = {}

    for ticker in fetch_finviz_gainers():
        ticker_map.setdefault(ticker, "general")

    for strategy, fname in STRATEGY_FILES.items():
        for ticker in load_ticker_file(os.path.join(base, fname)):
            ticker_map.setdefault(ticker, strategy)

    return ticker_map


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return float((100 - 100 / (1 + rs)).iloc[-1])


def atr(data: pd.DataFrame, period: int = 14) -> float:
    h, l, c = data["High"], data["Low"], data["Close"]
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------

def screen_ticker(ticker: str, strategy: str) -> dict | None:
    tk   = yf.Ticker(ticker)
    data = tk.history(period="1y", interval="1d")
    if len(data) < 20:
        return None

    close  = data["Close"]
    price  = float(close.iloc[-1])
    change = (price / float(close.iloc[-2]) - 1) * 100
    rvol   = float(data["Volume"].iloc[-1] / data["Volume"].iloc[:-1].mean())

    ema9   = float(close.ewm(span=9,   adjust=False).mean().iloc[-1])
    ema20  = float(close.ewm(span=20,  adjust=False).mean().iloc[-1])
    ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
    rsi_val = rsi(close)
    atr_val = atr(data)

    market_cap   = None
    float_shares = None

    # Finviz top gainers are filtered strictly; curated lists always pass
    if strategy == "general" and not (
        rvol   >= FILTER_MIN_RVOL
        and FILTER_MIN_PRICE <= price <= FILTER_MAX_PRICE
        and change >= FILTER_MIN_CHANGE
        and ema9 > ema20 > ema200
        and rsi_val < FILTER_MAX_RSI
    ):
        return None

    if strategy == "momentum":
        info         = tk.info
        market_cap   = info.get("marketCap")
        float_shares = info.get("floatShares")
        if not (
            market_cap   and market_cap   < 2_000_000_000
            and float_shares and float_shares < 50_000_000
            and rvol   >= 2.0
            and change >= 5.0
        ):
            return None

    exp1   = close.ewm(span=12, adjust=False).mean()
    exp2   = close.ewm(span=26, adjust=False).mean()
    macd   = float((exp1 - exp2).iloc[-1])
    macd_s = float((exp1 - exp2).ewm(span=9, adjust=False).mean().iloc[-1])
    vwap   = float(((data["High"] + data["Low"] + data["Close"]) / 3).iloc[-1])

    is_crypto = ticker.endswith("-USD")
    if not is_crypto and len(data) >= 7:
        vol3 = data["Volume"].rolling(3).mean()
        volume_trend_up = int(float(vol3.iloc[-1]) > float(vol3.iloc[-4]))
    else:
        volume_trend_up = 0

    score = sum([
        macd > macd_s,
        ema9 > ema20 > ema200,
        price > vwap,
        volume_trend_up == 1,
    ])

    return {
        "ticker":          ticker,
        "strategy":        strategy,
        "asset":           "crypto" if is_crypto else "equity",
        "price":           round(price, 2),
        "change_pct":      round(change, 2),
        "rvol":            round(rvol, 2),
        "ema9":            round(ema9, 2),
        "ema20":           round(ema20, 2),
        "ema200":          round(ema200, 2),
        "rsi":             round(rsi_val, 2),
        "atr":             round(atr_val, 4),
        "stop_loss":       round(price - 1.5 * atr_val, 2),
        "macd":            round(macd, 4),
        "macd_signal":     round(macd_s, 4),
        "vwap":            round(vwap, 2),
        "volume_trend_up": volume_trend_up,
        "score":           score,
        "market_cap":      market_cap,
        "float_shares":    float_shares,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    init_db()
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ticker_map = build_ticker_map()
    print(f"Screening {len(ticker_map)} tickers for {run_date}...")

    results = []
    for ticker, strategy in ticker_map.items():
        try:
            result = screen_ticker(ticker, strategy)
            if result:
                results.append(result)
                print(
                    f"  PASS [{strategy:8}] {ticker:10} "
                    f"{result['change_pct']:+.1f}%  "
                    f"RVOL {result['rvol']:.1f}  "
                    f"Score {result['score']}/4"
                )
        except Exception as e:
            print(f"  ERR  {ticker}: {e}")

    if results:
        save_results(run_date, results)
        print(f"\n{len(results)} candidates saved to screener.db")
    else:
        print("\nNo candidates found today.")


if __name__ == "__main__":
    main()
