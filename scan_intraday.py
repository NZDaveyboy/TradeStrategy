#!/usr/bin/env python3
"""
Intraday scanner — 15-minute bar RVOL and price momentum.

Compares the current 15m bar's volume against the average volume for the
same time slot across the past 20 sessions. Fires before the daily bar
closes, so you see unusual activity early in the move.

Covers all watchlists including crypto (24/7).

Usage:
    python3 scan_intraday.py

Cron — every 15 min during US market hours (9:30am–4pm EST = 14:30–21:00 UTC):
    */15 14-21 * * 1-5 cd ~/TradeStrategy && python3 scan_intraday.py

For crypto (runs any time):
    */30 * * * * cd ~/TradeStrategy && python3 scan_intraday.py --crypto-only

Required env vars for Telegram alerts:
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone, date

# Load .env from project directory (no extra packages needed)
def _load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip())
_load_env()


import pandas as pd
import requests

from providers.yfinance_provider import YFinanceProvider

_provider = YFinanceProvider()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "alerts.csv")

CSV_FIELDS = [
    "triggered_at", "scan_date", "scan_window",
    "ticker", "alert_type", "value",
    "price", "change_pct", "rvol", "gap_pct",
]

TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHATID = os.environ.get("TELEGRAM_CHAT_ID", "")

INTRADAY_RVOL_THRESHOLD = 3.0   # current 15m bar vs avg for same time slot
CHANGE_THRESHOLD        = 2.0   # % price change from previous session close
GAP_THRESHOLD           = 2.0   # % gap: today's open vs previous close
MIN_SLOT_SESSIONS       = 5     # minimum historical sessions needed to score a slot

WATCHLISTS = {
    "equity": ["tickers_tech.txt", "tickers_ai.txt", "tickers_momentum.txt"],
    "crypto": ["tickers_crypto.txt"],
}


# ---------------------------------------------------------------------------
# Ticker loading
# ---------------------------------------------------------------------------

def load_tickers(crypto_only: bool = False) -> list[str]:
    groups = ["crypto"] if crypto_only else ["equity", "crypto"]
    seen, result = set(), []
    for group in groups:
        for fname in WATCHLISTS[group]:
            path = os.path.join(BASE_DIR, fname)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                for line in f:
                    t = line.strip().upper()
                    if t and t not in seen:
                        seen.add(t)
                        result.append(t)
    return result


# ---------------------------------------------------------------------------
# Intraday scan per ticker
# ---------------------------------------------------------------------------

def scan_ticker(ticker: str) -> dict | None:
    try:
        hist = _provider.get_ohlcv(ticker, "60d", "15m")

        # Some tokens return empty frames or lack timezone info
        if hist.empty or not hasattr(hist.index, "tz") or hist.index.tz is None:
            return None

        # Normalise to a consistent timezone for grouping
        is_crypto = ticker.endswith("-USD")
        tz = "UTC" if is_crypto else "America/New_York"
        hist.index = hist.index.tz_convert(tz)

        today_date = hist.index[-1].date()
        today_bars = hist[hist.index.date == today_date]
        hist_bars  = hist[hist.index.date <  today_date]

        if today_bars.empty or hist_bars.empty:
            return None

        # Most recent 15m bar this session
        current_bar    = today_bars.iloc[-1]
        slot_time      = current_bar.name.time()
        current_vol    = float(current_bar["Volume"])
        current_price  = float(current_bar["Close"])

        # Historical bars at the same time slot (same HH:MM across past sessions)
        same_slot = hist_bars[hist_bars.index.time == slot_time]
        if len(same_slot) < MIN_SLOT_SESSIONS:
            return None

        avg_slot_vol = float(same_slot["Volume"].tail(20).mean())
        if avg_slot_vol == 0:
            return None

        intraday_rvol = current_vol / avg_slot_vol

        # Previous session close — last bar of the session before today
        prev_dates = sorted(set(hist_bars.index.date), reverse=True)
        prev_close_bars = hist_bars[hist_bars.index.date == prev_dates[0]]
        prev_close  = float(prev_close_bars["Close"].iloc[-1])

        today_open  = float(today_bars.iloc[0]["Open"])
        if prev_close <= 0:
            return None
        change_pct  = (current_price - prev_close) / prev_close * 100
        gap_pct     = (today_open    - prev_close) / prev_close * 100

        triggers: list[tuple[str, float]] = []

        if intraday_rvol >= INTRADAY_RVOL_THRESHOLD:
            triggers.append(("rvol_15m", round(intraday_rvol, 2)))
        if abs(change_pct) >= CHANGE_THRESHOLD:
            triggers.append(("change", round(change_pct, 2)))
        if gap_pct >= GAP_THRESHOLD:
            triggers.append(("gap_up", round(gap_pct, 2)))
        elif gap_pct <= -GAP_THRESHOLD:
            triggers.append(("gap_down", round(gap_pct, 2)))

        if not triggers:
            return None

        return {
            "ticker":        ticker,
            "price":         round(current_price, 4),
            "change_pct":    round(change_pct, 2),
            "rvol":          round(intraday_rvol, 2),
            "gap_pct":       round(gap_pct, 2),
            "bar_time":      current_bar.name.strftime("%H:%M"),
            "triggers":      triggers,
        }

    except Exception as e:
        print(f"  ERR  {ticker}: {e}")
        return None


# ---------------------------------------------------------------------------
# CSV storage
# ---------------------------------------------------------------------------

def save_alerts(scan_dt: str, scan_date: str, scan_window: str, alerts: list[dict]):
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for a in alerts:
            for atype, aval in a["triggers"]:
                writer.writerow({
                    "triggered_at": scan_dt,
                    "scan_date":    scan_date,
                    "scan_window":  scan_window,
                    "ticker":       a["ticker"],
                    "alert_type":   atype,
                    "value":        aval,
                    "price":        a["price"],
                    "change_pct":   a["change_pct"],
                    "rvol":         a["rvol"],
                    "gap_pct":      a["gap_pct"],
                })


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

TYPE_EMOJI = {
    "rvol_15m": "⚡",
    "change":   "📈",
    "gap_up":   "⬆️",
    "gap_down": "⬇️",
}
TYPE_LABEL = {
    "rvol_15m": "RVOL-15m",
    "change":   "Change",
    "gap_up":   "Gap Up",
    "gap_down": "Gap Down",
}


def format_message(alerts: list[dict], scan_window: str, scan_dt: str) -> str:
    lines = [
        f"<b>TradeStrategy — Intraday  {scan_window}</b>",
        f"<i>{scan_dt}  •  {len(alerts)} alert(s)</i>",
        "",
    ]
    for a in alerts:
        parts = []
        for atype, aval in a["triggers"]:
            emoji = TYPE_EMOJI.get(atype, "•")
            label = TYPE_LABEL.get(atype, atype)
            parts.append(
                f"{emoji} {label} {aval:.1f}x"
                if "rvol" in atype
                else f"{emoji} {label} {aval:+.1f}%"
            )
        lines.append(
            f"<b>{a['ticker']}</b>  ${a['price']:.4f}  "
            f"{a['change_pct']:+.1f}%  RVOL-15m {a['rvol']:.1f}x  [{a['bar_time']}]"
            f"\n   {'  '.join(parts)}"
        )
    return "\n".join(lines)


def send_telegram(message: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHATID:
        print("Telegram not configured — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHATID, "text": message, "parse_mode": "HTML"},
            timeout=15,
        )
        if resp.status_code == 200:
            print("Telegram message sent.")
            return True
        print(f"Telegram error {resp.status_code}: {resp.text}")
        return False
    except Exception as e:
        print(f"Telegram request failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crypto-only", action="store_true",
                        help="Scan crypto watchlist only (useful for 24/7 cron)")
    args = parser.parse_args()

    now       = datetime.now(timezone.utc)
    scan_dt   = now.strftime("%Y-%m-%d %H:%M UTC")
    scan_date = now.strftime("%Y-%m-%d")

    est_hour = (now.hour - 5) % 24
    est_min  = now.minute
    scan_window = f"intraday {'crypto' if args.crypto_only else 'all'}  {est_hour}:{est_min:02d} EST"

    print(f"\n{'='*60}")
    print(f"Intraday scan  |  {scan_dt}  |  {scan_window}")
    print(f"{'='*60}")
    print(f"Thresholds: RVOL-15m >= {INTRADAY_RVOL_THRESHOLD}x  |  "
          f"change >= {CHANGE_THRESHOLD}%  |  gap >= {GAP_THRESHOLD}%\n")

    tickers = load_tickers(crypto_only=args.crypto_only)
    print(f"Scanning {len(tickers)} tickers...\n")

    triggered = []
    for ticker in tickers:
        result = scan_ticker(ticker)
        if result:
            triggered.append(result)
            types_str = "  ".join(
                f"{TYPE_EMOJI.get(t,'')}{t}={v}"
                for t, v in result["triggers"]
            )
            print(
                f"  ALERT  {ticker:<12}  ${result['price']:.4f}  "
                f"{result['change_pct']:+.1f}%  RVOL-15m {result['rvol']:.1f}x  "
                f"[{result['bar_time']}]  [{types_str}]"
            )
        else:
            print(f"  clear  {ticker}")
        time.sleep(0.2)

    print(f"\n{len(triggered)} alert(s) out of {len(tickers)} tickers.")

    if triggered:
        save_alerts(scan_dt, scan_date, scan_window, triggered)
        n_rows = sum(len(a["triggers"]) for a in triggered)
        print(f"Appended {n_rows} row(s) to alerts.csv")
        msg = format_message(triggered, scan_window, scan_dt)
        send_telegram(msg)
    else:
        print("No alerts — nothing sent to Telegram.")


if __name__ == "__main__":
    main()
