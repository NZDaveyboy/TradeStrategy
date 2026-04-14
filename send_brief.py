#!/usr/bin/env python3
"""
send_brief.py — Structured daily Telegram brief for TradeStrategy.

Reads today's screener results from screener.db, derives SPY market regime,
and sends a formatted brief via Telegram.

Usage:
    python3 send_brief.py

Can also be imported and called from run.py:
    from send_brief import send_daily_brief
    send_daily_brief()
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Load .env (same pattern as scan_premarket.py)
# ---------------------------------------------------------------------------

def _load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_env()

import requests

from providers.yfinance_provider import YFinanceProvider

_provider = YFinanceProvider()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screener.db")
TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHATID = os.environ.get("TELEGRAM_CHAT_ID", "")
MAX_MSG_CHARS   = 4000

# setup_type values that belong in the avoid/low-quality section
_AVOID_TYPES = {"Avoid", "Low quality / illiquid"}

# setup_type values that indicate an extended setup (watchlist, not active entry)
_EXTENDED_TYPES = {"Strong but extended", "Strong downside setup", "Overextended", "Extended downside move"}


# ---------------------------------------------------------------------------
# Market regime
# ---------------------------------------------------------------------------

def _spy_regime() -> str:
    """
    Returns "Bullish", "Neutral", or "Bearish" based on SPY price vs EMA20.
    Also appends a brief descriptor.
    """
    try:
        hist = _provider.get_ohlcv("SPY", "60d", "1d")
        if len(hist) < 20:
            return "Neutral — insufficient SPY data"
        close = hist["Close"]
        price = float(close.iloc[-1])
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema9  = float(close.ewm(span=9,  adjust=False).mean().iloc[-1])
        chg1  = (price - float(close.iloc[-2])) / float(close.iloc[-2]) * 100

        if price > ema20 and ema9 > ema20:
            regime = "Bullish"
        elif price < ema20 and ema9 < ema20:
            regime = "Bearish"
        else:
            regime = "Neutral"

        return (
            f"{regime} — SPY ${price:.2f}  EMA20 ${ema20:.2f}  "
            f"({chg1:+.2f}% today)"
        )
    except Exception as e:
        return f"Unknown — SPY data error ({e})"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _load_today(run_date: str) -> list[dict]:
    """Load all screener rows for a given run_date."""
    if not os.path.exists(DB_PATH):
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        # Ensure newer columns exist (safe no-op if already present)
        for col, col_type in [
            ("tradescore",  "REAL"),
            ("explain",     "TEXT"),
            ("setup_type",  "TEXT"),
            ("rationale",   "TEXT"),
            ("direction",   "TEXT"),
        ]:
            try:
                conn.execute(f"ALTER TABLE results ADD COLUMN {col} {col_type}")
            except Exception:
                pass
        conn.commit()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT ticker, price, change_pct, rvol, rsi, atr, vwap,
                   ema9, ema20, tradescore, explain, setup_type, rationale,
                   direction, strategy
            FROM results
            WHERE run_date = ?
            ORDER BY tradescore DESC
            """,
            (run_date,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"DB read error: {e}")
        return []


def _parse_explain(row: dict) -> dict:
    try:
        return json.loads(row.get("explain") or "{}")
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _fmt_ticker_long(row: dict) -> str:
    """One block for a long setup."""
    ex       = _parse_explain(row)
    ticker   = row["ticker"]
    score    = int(row.get("tradescore") or 0)
    stype    = row.get("setup_type") or "—"
    price    = row.get("price") or 0
    rvol     = row.get("rvol") or 0
    rsi      = row.get("rsi") or 0
    atr      = row.get("atr") or 0
    vwap     = row.get("vwap") or 0
    ema20    = row.get("ema20") or 0

    entry_zone   = round(price * 1.001, 2)
    invalidation = round(min(vwap, ema20) - 0.35 * atr, 2) if (vwap and ema20 and atr) else "—"
    target       = round(entry_zone + 2.0 * (entry_zone - invalidation), 2) if invalidation != "—" else "—"
    conviction   = ex.get("conviction") or stype

    lines = [
        f"<b>{ticker}</b>  ${price:.2f}  Score {score}  [{stype}]",
        f"   Entry &gt; ${entry_zone:.2f}  |  Stop ${invalidation}  |  Target ${target}",
        f"   RVOL {rvol:.1f}x  RSI {rsi:.0f}  |  {conviction}",
    ]
    return "\n".join(lines)


def _fmt_ticker_short(row: dict) -> str:
    """One block for a short/bearish setup."""
    ex       = _parse_explain(row)
    ticker   = row["ticker"]
    score    = int(row.get("tradescore") or 0)
    stype    = row.get("setup_type") or "—"
    price    = row.get("price") or 0
    rvol     = row.get("rvol") or 0
    rsi      = row.get("rsi") or 0
    atr      = row.get("atr") or 0
    vwap     = row.get("vwap") or 0
    ema20    = row.get("ema20") or 0

    entry_zone   = round(price * 0.999, 2)
    invalidation = round(max(vwap, ema20) + 0.35 * atr, 2) if (vwap and ema20 and atr) else "—"
    target       = round(max(0.01, entry_zone - 2.0 * (invalidation - entry_zone)), 2) if invalidation != "—" else "—"
    conviction   = ex.get("conviction") or stype

    lines = [
        f"<b>{ticker}</b>  ${price:.2f}  Score {score}  [{stype}]",
        f"   Entry &lt; ${entry_zone:.2f}  |  Stop ${invalidation}  |  Target ${target}",
        f"   RVOL {rvol:.1f}x  RSI {rsi:.0f}  |  {conviction}",
    ]
    return "\n".join(lines)


def _fmt_ticker_extended(row: dict) -> str:
    """One block for an extended / watchlist setup."""
    ticker = row["ticker"]
    score  = int(row.get("tradescore") or 0)
    stype  = row.get("setup_type") or "—"
    price  = row.get("price") or 0
    chg    = row.get("change_pct") or 0
    rvol   = row.get("rvol") or 0
    atr    = row.get("atr") or 0
    vwap   = row.get("vwap") or 0
    ema20  = row.get("ema20") or 0

    pullback_zone = round(max(vwap, ema20), 2) if (vwap and ema20) else "—"

    return (
        f"<b>{ticker}</b>  ${price:.2f}  ({chg:+.1f}%)  Score {score}\n"
        f"   Extended — pullback zone ~${pullback_zone}  RVOL {rvol:.1f}x"
    )


def _fmt_ticker_avoid(row: dict) -> str:
    ex     = _parse_explain(row)
    ticker = row["ticker"]
    stype  = row.get("setup_type") or "—"
    reason = ex.get("conviction") or stype
    return f"<b>{ticker}</b> — {reason}"


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def build_brief(rows: list[dict], regime: str, run_date: str) -> str:
    # Partition rows
    longs     = [r for r in rows if r.get("direction") == "long"
                 and r.get("setup_type") not in _AVOID_TYPES
                 and r.get("setup_type") not in _EXTENDED_TYPES]
    shorts    = [r for r in rows if r.get("direction") == "short"
                 and r.get("setup_type") not in _AVOID_TYPES
                 and r.get("setup_type") not in _EXTENDED_TYPES]
    extended  = [r for r in rows if r.get("setup_type") in _EXTENDED_TYPES]
    avoids    = [r for r in rows if r.get("setup_type") in _AVOID_TYPES]

    # Sort each by tradescore desc (already sorted from DB, but partition disturbs order)
    key = lambda r: float(r.get("tradescore") or 0)
    longs.sort(key=key, reverse=True)
    shorts.sort(key=key, reverse=True)
    extended.sort(key=key, reverse=True)

    sections = []

    # Header
    sections.append(
        f"<b>📈 TradeStrategy Daily Brief — {run_date}</b>\n"
        f"<i>Generated {datetime.now(timezone.utc).strftime('%H:%M UTC')}</i>"
    )

    # Market regime
    sections.append(f"<b>MARKET REGIME</b>\n{regime}")

    # Top long setups
    if longs:
        body = "\n\n".join(_fmt_ticker_long(r) for r in longs[:3])
    else:
        body = "<i>No actionable long setups today.</i>"
    sections.append(f"<b>TOP LONG SETUPS</b>\n{body}")

    # Extended / watchlist
    if extended:
        body = "\n\n".join(_fmt_ticker_extended(r) for r in extended[:3])
    else:
        body = "<i>None extended.</i>"
    sections.append(f"<b>EXTENDED MOMENTUM WATCHLIST</b>\n{body}")

    # Short / bearish
    if shorts:
        body = "\n\n".join(_fmt_ticker_short(r) for r in shorts[:3])
    else:
        body = "<i>No actionable short setups today.</i>"
    sections.append(f"<b>SHORT / BEARISH SETUPS</b>\n{body}")

    # Avoid / low quality
    if avoids:
        body = "\n".join(_fmt_ticker_avoid(r) for r in avoids[:6])
    else:
        body = "<i>None flagged.</i>"
    sections.append(f"<b>AVOID / LOW QUALITY</b>\n{body}")

    # Today's focus — top 3 overall, best R/R, most likely too late
    all_valid = [r for r in rows if r.get("setup_type") not in _AVOID_TYPES]
    top3 = all_valid[:3]
    if top3:
        watchlist = "  |  ".join(r["ticker"] for r in top3)
        best_rr   = top3[0]["ticker"] if top3 else "—"
        # "most likely too late" = highest change_pct among extended
        too_late  = max(extended, key=lambda r: abs(r.get("change_pct") or 0))["ticker"] if extended else "—"
    else:
        watchlist = "—"
        best_rr   = "—"
        too_late  = "—"

    sections.append(
        f"<b>TODAY'S FOCUS</b>\n"
        f"Primary watchlist: {watchlist}\n"
        f"Best quality / R:R: {best_rr}\n"
        f"Most likely too late: {too_late}"
    )

    # Static reminders
    sections.append(
        "<b>REMINDERS</b>\n"
        "• Wait for confirmation before entry — no chasing\n"
        "• Size to your stop, not your conviction\n"
        "• If RVOL drops under 2x at entry, skip it\n"
        "• One position per sector at a time\n"
        "• Extended setups are for watching, not buying"
    )

    return "\n\n─────────────────────\n\n".join(sections)


# ---------------------------------------------------------------------------
# Telegram sender (splits at MAX_MSG_CHARS)
# ---------------------------------------------------------------------------

def _send_telegram(message: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHATID:
        print("Telegram not configured — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    # Split on section separator if needed
    chunks = []
    remaining = message
    while len(remaining) > MAX_MSG_CHARS:
        split_at = remaining.rfind("\n\n─", 0, MAX_MSG_CHARS)
        if split_at == -1:
            split_at = MAX_MSG_CHARS
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip()
    chunks.append(remaining)

    success = True
    for chunk in chunks:
        try:
            resp = requests.post(
                url,
                json={
                    "chat_id":    TELEGRAM_CHATID,
                    "text":       chunk,
                    "parse_mode": "HTML",
                },
                timeout=15,
            )
            if resp.status_code == 200:
                print(f"Telegram chunk sent ({len(chunk)} chars).")
            else:
                print(f"Telegram error {resp.status_code}: {resp.text}")
                success = False
        except Exception as e:
            print(f"Telegram request failed: {e}")
            success = False
    return success


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def send_daily_brief(run_date: str | None = None) -> bool:
    """
    Build and send the daily brief.
    run_date defaults to today (UTC) in YYYY-MM-DD format.
    Returns True if Telegram send succeeded.
    """
    if run_date is None:
        run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"\nBuilding daily brief for {run_date}...")

    rows = _load_today(run_date)
    if not rows:
        print(f"No screener results found for {run_date}. Brief not sent.")
        return False

    print(f"Loaded {len(rows)} screener rows.")
    print("Fetching SPY regime...")
    regime = _spy_regime()
    print(f"Regime: {regime}")

    brief = build_brief(rows, regime, run_date)

    print("\n" + "=" * 60)
    print(brief)
    print("=" * 60 + "\n")

    return _send_telegram(brief)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    ok = send_daily_brief(date_arg)
    sys.exit(0 if ok else 1)
