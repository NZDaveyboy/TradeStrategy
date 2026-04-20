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

import json
import math
import os
from datetime import datetime, timezone

from core.db import get_connection, sync_if_turso

import pandas as pd

from core.tradescore import (
    compute_tradescore,
    conviction_label,
    _momentum_score,
    _early_entry_score,
    _extension_risk_score,
    _liquidity_score,
    _news_catalyst_score,
    _setup_type,
    _build_rationale,
    _clamp,
    _lerp,
    MS_RVOL_IDEAL, MS_RVOL_MAX_PTS, MS_CHG_HI_PCT, MS_CHG_MAX_PTS, MS_MACD_MAX_PTS,
    EE_RSI_LO, EE_RSI_HI, EE_RSI_MAX_PTS, EE_EMA_NEAR_PCT, EE_EMA_FAR_PCT,
    EE_EMA_MAX_PTS, EE_BOB_MAX_PTS,
    ER_RSI_WARN, ER_RSI_HARD, ER_RSI_MAX_PTS, ER_DAY_WARN, ER_DAY_HARD, ER_DAY_MAX_PTS,
    ER_VWAP_WARN, ER_VWAP_HARD, ER_VWAP_MAX_PTS, ER_5D_WARN, ER_5D_HARD, ER_5D_MAX_PTS,
    LQ_DVOL_MIN, LQ_DVOL_FULL, LQ_DVOL_MAX_PTS, LQ_QUAL_MAX_PTS, LQ_CONS_MAX_PTS,
)
from providers.yfinance_provider import FinvizDiscoveryProvider, YFinanceProvider

_provider  = YFinanceProvider()
_discovery = FinvizDiscoveryProvider()

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
    conn = get_connection(DB_PATH)
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
            tradescore       REAL,
            explain          TEXT,
            setup_type       TEXT,
            rationale        TEXT,
            change_5d        REAL,
            direction        TEXT,
            PRIMARY KEY (run_date, ticker)
        )
    """)
    # Migrate existing DBs that pre-date new columns
    for col, col_type in [
        ("market_cap",    "REAL"),
        ("float_shares",  "REAL"),
        ("tradescore",    "REAL"),
        ("explain",       "TEXT"),
        ("setup_type",    "TEXT"),
        ("rationale",     "TEXT"),
        ("change_5d",     "REAL"),
        ("direction",     "TEXT"),
        # Phase 9 — faithful re-scoring inputs
        ("high_20d",      "REAL"),   # 20-session closing high (for BOB)
        ("dollar_volume", "REAL"),   # price × last-bar volume (for dvol)
        ("vol_cv",        "REAL"),   # volume coefficient of variation prior 10 bars
    ]:
        try:
            conn.execute(f"ALTER TABLE results ADD COLUMN {col} {col_type}")
        except Exception:
            pass  # column already exists
    conn.commit()
    sync_if_turso(conn)
    conn.close()


def save_results(run_date: str, rows: list[dict]):
    conn = get_connection(DB_PATH)
    for r in rows:
        conn.execute(
            """
            INSERT OR REPLACE INTO results (
                run_date, ticker, strategy, asset,
                price, change_pct, rvol,
                ema9, ema20, ema200,
                rsi, atr, stop_loss,
                macd, macd_signal, vwap,
                volume_trend_up, score,
                market_cap, float_shares,
                tradescore, explain,
                setup_type, rationale,
                change_5d, direction,
                high_20d, dollar_volume, vol_cv
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
            """,
            (
                run_date, r["ticker"], r["strategy"], r["asset"],
                r["price"], r["change_pct"], r["rvol"],
                r["ema9"], r["ema20"], r["ema200"],
                r["rsi"], r["atr"], r["stop_loss"],
                r["macd"], r["macd_signal"], r["vwap"],
                r["volume_trend_up"], r["score"],
                r.get("market_cap"), r.get("float_shares"),
                r.get("tradescore"), r.get("explain"),
                r.get("setup_type"), r.get("rationale"),
                r.get("change_5d"), r.get("direction"),
                r.get("high_20d"), r.get("dollar_volume"), r.get("vol_cv"),
            ),
        )
    conn.commit()
    sync_if_turso(conn)
    conn.close()


# ---------------------------------------------------------------------------
# Ticker loading
# ---------------------------------------------------------------------------

def fetch_finviz_gainers(limit: int = 50) -> list[str]:
    return _discovery.get_gainers(limit)


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


# Scoring constants and functions are in core/tradescore.py.
# Imported at the top of this file.


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    val = float((100 - 100 / (1 + rs)).iloc[-1])
    return val if not (val != val) else 50.0  # NaN guard — return neutral 50


def atr(data: pd.DataFrame, period: int = 14) -> float:
    h, l, c = data["High"], data["Low"], data["Close"]
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    val = float(tr.rolling(period).mean().iloc[-1])
    return val if not (val != val) else 0.01  # NaN guard — return near-zero


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------

_STABLECOINS = {"USDC-USD", "DAI-USD", "USDT-USD", "BUSD-USD", "TUSD-USD", "USDP-USD"}


def screen_ticker(ticker: str, strategy: str) -> dict | None:
    if ticker in _STABLECOINS:
        return None  # stablecoins have no trading edge — exclude from ranking

    data = _provider.get_ohlcv(ticker, "1y", "1d")
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

    # Market cap via get_quote (fast, ~50ms) for all tickers.
    # Full fundamentals (slow) only for momentum strategy where float_shares is needed.
    market_cap   = _provider.get_quote(ticker).market_cap
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
        fund         = _provider.get_fundamentals(ticker)
        market_cap   = fund.market_cap
        float_shares = fund.float_shares
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
    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    vwap = float(
        (typical_price * data["Volume"]).cumsum().iloc[-1]
        / data["Volume"].cumsum().iloc[-1]
    ) if data["Volume"].sum() > 0 else float(typical_price.iloc[-1])

    is_crypto = ticker.endswith("-USD")

    if is_crypto:
        # Crypto scoring — equity signals don't apply (no market hours, EMA200
        # is useless in multi-month bear cycles, volume has no daily reset).
        #
        # Score 0-4:
        #   1. MACD above signal line          — momentum direction
        #   2. EMA9 > EMA20                    — short-term trend (drop EMA200)
        #   3. RVOL >= 1.5x                    — above-average participation
        #   4. RSI 40–75                       — momentum zone, not exhausted
        volume_trend_up = 0
        rsi_in_zone     = 40 <= rsi_val <= 75
        score = sum([
            macd > macd_s,
            ema9 > ema20,
            rvol >= 1.5,
            rsi_in_zone,
        ])
        stop_loss = round(price - 2.0 * atr_val, 4)   # wider stop for crypto volatility
    else:
        vol3 = data["Volume"].rolling(3).mean()
        volume_trend_up = int(len(data) >= 7 and float(vol3.iloc[-1]) > float(vol3.iloc[-4]))
        score = sum([
            macd > macd_s,
            ema9 > ema20 > ema200,
            price > vwap,
            volume_trend_up == 1,
        ])
        stop_loss = round(price - 1.5 * atr_val, 2)

    row = {
        "ticker":          ticker,
        "strategy":        strategy,
        "asset":           "crypto" if is_crypto else "equity",
        "price":           round(price, 4 if is_crypto else 2),
        "change_pct":      round(change, 2),
        "rvol":            round(rvol, 2),
        "ema9":            round(ema9, 4 if is_crypto else 2),
        "ema20":           round(ema20, 4 if is_crypto else 2),
        "ema200":          round(ema200, 4 if is_crypto else 2),
        "rsi":             round(rsi_val, 2),
        "atr":             round(atr_val, 4),
        "stop_loss":       stop_loss,
        "macd":            round(macd, 4),
        "macd_signal":     round(macd_s, 4),
        "vwap":            round(vwap, 4 if is_crypto else 2),
        "volume_trend_up": volume_trend_up,
        "score":           score,
        "market_cap":      market_cap,
        "float_shares":    float_shares,
    }

    # Compute faithful re-scoring inputs before calling compute_tradescore.
    # These are stored so research sweeps can reconstruct exact sub-scores
    # without needing the original close series or OHLCV DataFrame.
    row["high_20d"] = round(float(close.iloc[-20:].max()), 6) if len(close) >= 20 else None

    _last_vol   = float(data["Volume"].iloc[-1])
    row["dollar_volume"] = round(price * _last_vol, 2)

    _vol_window = data["Volume"].iloc[-11:-1]
    _vol_mean   = float(_vol_window.mean())
    row["vol_cv"] = round(float(_vol_window.std() / _vol_mean), 6) if len(_vol_window) >= 10 and _vol_mean > 0 else None

    ts = compute_tradescore(row, close=close, data=data)
    row["tradescore"] = ts["score"]
    row["setup_type"] = ts["setup_type"]
    row["rationale"]  = ts["rationale"]
    row["change_5d"]  = ts["change_5d"]
    row["direction"]  = ts["direction"]
    row["explain"]    = json.dumps(ts)
    return row


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
                    f"Score {result['score']}/4  "
                    f"TradeScore {result['tradescore']:.0f}  "
                    f"[{result.get('explain') and json.loads(result['explain'])['conviction']}]"
                )
        except Exception as e:
            print(f"  ERR  {ticker}: {e}")

    if results:
        save_results(run_date, results)
        print(f"\n{len(results)} candidates saved to screener.db")

        # Top 5 by TradeScore
        top5 = sorted(results, key=lambda r: r["tradescore"], reverse=True)[:5]
        print("\nTop 5 by TradeScore:")
        hdr = f"  {'Ticker':<10} {'Score':>6} {'SetupType':<22} {'Mom':>5} {'Entry':>6} {'Ext':>5} {'Liq':>5} {'RVOL':>6} {'Chg':>7} {'RSI':>5}"
        print(hdr)
        print("  " + "-" * len(hdr))
        for r in top5:
            ts_data = json.loads(r["explain"])
            print(
                f"  {r['ticker']:<10} {r['tradescore']:>6.1f} "
                f"{ts_data['setup_type']:<22} "
                f"{ts_data['momentum_score']:>5.1f} "
                f"{ts_data['early_entry']:>6.1f} "
                f"{ts_data['extension_risk']:>5.1f} "
                f"{ts_data['liquidity']:>5.1f} "
                f"{r['rvol']:>6.1f}x "
                f"{r['change_pct']:>+6.1f}% "
                f"{r['rsi']:>5.1f}"
            )
            print(f"    → {ts_data['rationale']}")
    else:
        print("\nNo candidates found today.")

    # Send structured daily brief via Telegram
    try:
        from send_brief import send_daily_brief
        send_daily_brief(run_date)
    except Exception as e:
        print(f"\nDaily brief failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
