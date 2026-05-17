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
from datetime import datetime, timezone

import pandas as pd

from core.db import get_connection

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

from core.theme_watchlist import is_on_watchlist
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
    Returns a short-term regime label combined with the long-term EMA200 trend.
    Pulls 1y of SPY so EMA200 is properly warmed.
    """
    try:
        hist = _provider.get_ohlcv("SPY", "1y", "1d")
        if len(hist) < 20:
            return "Neutral — insufficient SPY data"
        close = hist["Close"]
        price = float(close.iloc[-1])
        ema20  = float(close.ewm(span=20,  adjust=False).mean().iloc[-1])
        ema9   = float(close.ewm(span=9,   adjust=False).mean().iloc[-1])
        ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1]) if len(close) >= 200 else None
        chg1   = (price - float(close.iloc[-2])) / float(close.iloc[-2]) * 100

        if price > ema20 and ema9 > ema20:
            short_term = "Bullish"
        elif price < ema20 and ema9 < ema20:
            short_term = "Bearish"
        else:
            short_term = "Neutral"

        if ema200 is None:
            regime = short_term
            long_term_note = ""
        else:
            long_term = "above 200 EMA" if price > ema200 else "below 200 EMA"
            long_term_note = f"  EMA200 ${ema200:.2f} ({long_term})"
            # Compose a combined label that flags trend-counter setups
            if short_term == "Bullish" and price > ema200:
                regime = "Bullish (trend-aligned)"
            elif short_term == "Bearish" and price < ema200:
                regime = "Bearish (trend-aligned)"
            elif short_term == "Bullish" and price < ema200:
                regime = "Bullish (counter-trend bounce)"
            elif short_term == "Bearish" and price > ema200:
                regime = "Bearish (pullback in uptrend)"
            else:
                regime = f"{short_term} ({long_term})"

        return (
            f"{regime} — SPY ${price:.2f}  EMA20 ${ema20:.2f}"
            f"{long_term_note}  ({chg1:+.2f}% today)"
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
        conn = get_connection(DB_PATH)
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
        cursor = conn.execute(
            """
            SELECT ticker, price, change_pct, rvol, rsi, atr, vwap,
                   ema9, ema20, tradescore, explain, setup_type, rationale,
                   direction, strategy
            FROM results
            WHERE run_date = ?
            ORDER BY tradescore DESC
            """,
            (run_date,),
        )
        columns = [d[0] for d in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return rows
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

def _fetch_catalyst_line(ticker: str, price: float) -> str | None:
    """Fetch CatalystScore + top tag for a ticker. Returns None on failure.

    Called only for the top picks in the brief (≤6 picks) so network cost
    is bounded. Each lookup hits ~4 yfinance endpoints; for a daily brief
    this is acceptable latency.
    """
    try:
        from core.catalysts import compute_catalyst_score
        cat = compute_catalyst_score(ticker, price)
    except Exception:
        return None
    score = cat.get("score")
    if score is None:
        return None
    tags = cat.get("tags") or []
    # Prefer ⚠ tag (binary event risk) — surfaces the most actionable warning
    warning_tag = next((t for t in tags if "⚠" in t), None)
    top_tag = warning_tag or (tags[0] if tags else "")
    if score >= 65:
        lean = "🟢 bullish-leaning"
    elif score <= 35:
        lean = "🔴 bearish-leaning"
    else:
        lean = "⚪ mixed"
    if top_tag:
        return f"   CatalystScore {score:.0f}/100 ({lean}) — {top_tag}"
    return f"   CatalystScore {score:.0f}/100 ({lean})"


def _fmt_ticker_long(row: dict) -> str:
    """One block for a long setup. CatalystScore is appended when available."""
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
    cat_line = _fetch_catalyst_line(ticker, float(price))
    if cat_line:
        lines.append(cat_line)
    return "\n".join(lines)


def _fmt_ticker_short(row: dict) -> str:
    """One block for a short/bearish setup. CatalystScore appended when available."""
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
    cat_line = _fetch_catalyst_line(ticker, float(price))
    if cat_line:
        lines.append(cat_line)
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
# Dashboard sections — Quantum signals + Earnings + 8-Ks
# Each builder returns a section string (HTML for Telegram) or None on
# failure. Each is wrapped in try/except so the brief never fails because
# of these — they're additive, not load-bearing.
# ---------------------------------------------------------------------------

def _section_quantum_signals() -> str | None:
    """Top BUY + SELL signals from the Quantum Ecosystem classifier.

    Pulls 24 tickers + builds the index + runs classify_constituents.
    ~15-25s. Wrapped in try/except — returns None on any failure.
    """
    try:
        from datetime import date as _date, timedelta as _td
        from core.quantum import (
            load_universe, fetch_prices, IndexBuilder, classify_constituents,
        )

        uni = load_universe()
        tickers = [c.ticker for c in uni.all_companies()] + list(uni.benchmarks)
        today = _date.today()
        start = today - _td(days=400)
        prices = fetch_prices(tickers, start, today)
        if prices.empty:
            return None
        uni_tickers = [c.ticker for c in uni.all_companies()]
        uni_prices  = prices[[t for t in uni_tickers if t in prices.columns]]
        result = IndexBuilder(uni, prices).build_ecosystem(
            pd.Timestamp(start), pd.Timestamp(today),
        )
        sigs = classify_constituents(result, uni_prices, uni)
        if sigs.empty:
            return None
    except Exception as e:
        print(f"  quantum signals failed: {e}")
        return None

    buys  = sigs[sigs["Signal"] == "BUY"].head(3)
    sells = sigs[sigs["Signal"] == "SELL"].head(2)

    if buys.empty and sells.empty:
        return None

    lines: list[str] = []
    for _, r in buys.iterrows():
        # Truncate reason for Telegram compactness
        reason = str(r["Reason"])[:70]
        lines.append(
            f"🟢 <b>{r['Ticker']}</b> BUY  Conv {r['Conviction']:.1f}  "
            f"1m {r['1m %']:+.1f}%  — {reason}"
        )
    for _, r in sells.iterrows():
        reason = str(r["Reason"])[:70]
        lines.append(
            f"🔴 <b>{r['Ticker']}</b> SELL  Held {r['Held return %']:+.0f}%  "
            f"1m {r['1m %']:+.1f}%  — {reason}"
        )

    return "<b>⚛️ QUANTUM SIGNALS</b>\n" + "\n".join(lines)


def _section_upcoming_earnings() -> str | None:
    """Earnings in next 7 days from the Quantum universe."""
    try:
        from core.quantum import load_universe
        from core.catalysts import get_next_earnings

        uni = load_universe()
        rows = []
        for c in uni.all_companies():
            try:
                e = get_next_earnings(c.ticker)
            except Exception:
                continue
            if not e:
                continue
            d = e.get("days_to")
            if d is None or d < 0 or d > 7:
                continue
            rows.append({
                "ticker": c.ticker,
                "date":   e["date"],
                "days":   d,
                "eps":    e.get("eps_estimate"),
            })
    except Exception as e:
        print(f"  earnings section failed: {e}")
        return None

    if not rows:
        return None
    rows.sort(key=lambda x: x["days"])

    lines: list[str] = []
    for r in rows[:5]:
        eps_str = f"  est ${r['eps']:.2f}" if r['eps'] else ""
        lines.append(
            f"📊 <b>{r['ticker']}</b>  {r['date']} (in {r['days']}d){eps_str}"
        )
    return "<b>📅 EARNINGS IN NEXT 7d</b>\n" + "\n".join(lines)


def _section_recent_8ks() -> str | None:
    """Material 8-K filings in the last 7 days across the Quantum universe."""
    try:
        from core.quantum import load_universe
        from core.sec_edgar import get_recent_filings

        uni = load_universe()
        rows = []
        now  = pd.Timestamp.today()
        for c in uni.all_companies():
            try:
                fs = get_recent_filings(c.ticker, limit=3)
            except Exception:
                continue
            for f in fs:
                if (f or {}).get("form") != "8-K":
                    continue
                fd = f.get("filed", "")
                if not fd:
                    continue
                try:
                    age_days = (now - pd.Timestamp(fd)).days
                except Exception:
                    continue
                if age_days > 7:
                    continue
                rows.append({
                    "ticker":      c.ticker,
                    "date":        fd,
                    "url":         f.get("url", ""),
                    "items_label": f.get("items_label", ""),
                })
    except Exception as e:
        print(f"  8-K section failed: {e}")
        return None

    if not rows:
        return None
    rows.sort(key=lambda x: x["date"], reverse=True)

    lines: list[str] = []
    for r in rows[:5]:
        # Highlight-reel format: "QBTS  2026-05-13  📝 Material Agreement  [link]"
        label = r.get("items_label") or "8-K"
        if r["url"]:
            lines.append(
                f'📜 <b>{r["ticker"]}</b>  {r["date"]}  {label}  '
                f'<a href="{r["url"]}">→</a>'
            )
        else:
            lines.append(f'📜 <b>{r["ticker"]}</b>  {r["date"]}  {label}')
    return "<b>📜 RECENT 8-Ks (7d)</b>\n" + "\n".join(lines)


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

    # Theme watchlist on radar (TradeScore > 50)
    watchlist_on_radar = [
        r for r in rows
        if is_on_watchlist(r.get("ticker", ""))
        and float(r.get("tradescore") or 0) > 50
    ]
    watchlist_on_radar.sort(key=key, reverse=True)
    if watchlist_on_radar:
        radar_lines = []
        for r in watchlist_on_radar[:6]:
            ts    = int(r.get("tradescore") or 0)
            rvol  = float(r.get("rvol") or 0)
            price = float(r.get("price") or 0)
            chg   = float(r.get("change_pct") or 0)
            radar_lines.append(
                f"⚡ <b>{r['ticker']}</b>  Score {ts}  "
                f"RVOL {rvol:.1f}x  ${price:.2f}  {chg:+.2f}%"
            )
        body = "\n".join(radar_lines)
    else:
        body = "<i>No theme watchlist tickers above score 50 today.</i>"
    sections.append(f"<b>⚡ WATCHLIST ON RADAR</b>\n{body}")

    # Avoid / low quality
    if avoids:
        body = "\n".join(_fmt_ticker_avoid(r) for r in avoids[:6])
    else:
        body = "<i>None flagged.</i>"
    sections.append(f"<b>AVOID / LOW QUALITY</b>\n{body}")

    # ── Dashboard sections (additive — fail-silent if data unavailable) ──
    qs_section = _section_quantum_signals()
    if qs_section:
        sections.append(qs_section)
    earn_section = _section_upcoming_earnings()
    if earn_section:
        sections.append(earn_section)
    f8k_section = _section_recent_8ks()
    if f8k_section:
        sections.append(f8k_section)

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
