#!/usr/bin/env python3
"""
Pre-market / early-session scanner for TradeStrategy.

Scheduled via GitHub Actions at 9:00am and 9:45am EST Mon-Fri.
Loads tickers from tickers_tech.txt and tickers_ai.txt.

Alert triggers (any one is sufficient):
  - RVOL >= 3x  (today's volume vs 20-day avg)
  - |price change| >= 5%  (current price vs previous close)
  - |gap| >= 3%  (today's open vs previous close)

Sends a single grouped Telegram message and appends all alerts to alerts.csv.

Required environment variables (set as GitHub Actions secrets):
  TELEGRAM_BOT_TOKEN  — token from @BotFather
  TELEGRAM_CHAT_ID    — your chat or channel ID
"""

import csv
import os
import time
from datetime import datetime, timezone

import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(BASE_DIR, "alerts.csv")

CSV_FIELDS = [
    "triggered_at", "scan_date", "scan_window",
    "ticker", "alert_type", "value",
    "price", "change_pct", "rvol", "gap_pct",
]

TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHATID = os.environ.get("TELEGRAM_CHAT_ID", "")

RVOL_THRESHOLD   = 3.0
CHANGE_THRESHOLD = 5.0
GAP_THRESHOLD    = 3.0

WATCHLISTS = ["tickers_tech.txt", "tickers_ai.txt"]


# ---------------------------------------------------------------------------
# CSV storage
# ---------------------------------------------------------------------------

def save_alerts_csv(scan_dt: str, scan_date: str, scan_window: str, alerts: list[dict]):
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
# Tickers
# ---------------------------------------------------------------------------

def load_tickers() -> list[str]:
    seen, result = set(), []
    for fname in WATCHLISTS:
        path = os.path.join(BASE_DIR, fname)
        if not os.path.exists(path):
            print(f"Warning: {fname} not found, skipping.")
            continue
        with open(path) as f:
            for line in f:
                t = line.strip().upper()
                if t and t not in seen:
                    seen.add(t)
                    result.append(t)
    return result


# ---------------------------------------------------------------------------
# Per-ticker scan
# ---------------------------------------------------------------------------

def scan_ticker(ticker: str) -> dict | None:
    try:
        tk   = yf.Ticker(ticker)
        hist = tk.history(period="20d", interval="1d")

        if len(hist) < 3:
            return None

        prev_close = float(hist["Close"].iloc[-2])
        avg_vol    = float(hist["Volume"].iloc[:-1].mean())
        today_vol  = float(hist["Volume"].iloc[-1])
        rvol       = today_vol / avg_vol if avg_vol > 0 else 0.0

        # Live price
        fi = tk.fast_info
        try:
            current_price = float(fi.last_price)
        except Exception:
            current_price = float(hist["Close"].iloc[-1])

        # Today's open for gap calc
        try:
            today_open = float(fi.open)
        except Exception:
            today_open = float(hist["Open"].iloc[-1])

        change_pct = (current_price - prev_close) / prev_close * 100
        gap_pct    = (today_open - prev_close) / prev_close * 100

        triggers: list[tuple[str, float]] = []

        if rvol >= RVOL_THRESHOLD:
            triggers.append(("rvol", round(rvol, 2)))
        if abs(change_pct) >= CHANGE_THRESHOLD:
            triggers.append(("change", round(change_pct, 2)))
        if gap_pct >= GAP_THRESHOLD:
            triggers.append(("gap_up", round(gap_pct, 2)))
        elif gap_pct <= -GAP_THRESHOLD:
            triggers.append(("gap_down", round(gap_pct, 2)))

        if not triggers:
            return None

        return {
            "ticker":     ticker,
            "price":      round(current_price, 4),
            "change_pct": round(change_pct, 2),
            "rvol":       round(rvol, 2),
            "gap_pct":    round(gap_pct, 2),
            "triggers":   triggers,
        }

    except Exception as e:
        print(f"  ERR  {ticker}: {e}")
        return None


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

TYPE_EMOJI = {
    "rvol":     "🔥",
    "change":   "📈",
    "gap_up":   "⬆️",
    "gap_down": "⬇️",
}
TYPE_LABEL = {
    "rvol":     "RVOL",
    "change":   "Change",
    "gap_up":   "Gap Up",
    "gap_down": "Gap Down",
}


def format_message(alerts: list[dict], scan_window: str, scan_dt: str) -> str:
    lines = [
        f"<b>TradeStrategy — {scan_window}</b>",
        f"<i>{scan_dt} UTC  •  {len(alerts)} alert(s)</i>",
        "",
    ]
    for a in alerts:
        parts = []
        for atype, aval in a["triggers"]:
            emoji = TYPE_EMOJI.get(atype, "•")
            label = TYPE_LABEL.get(atype, atype)
            parts.append(
                f"{emoji} {label} {aval:.1f}x"
                if atype == "rvol"
                else f"{emoji} {label} {aval:+.1f}%"
            )
        lines.append(
            f"<b>{a['ticker']}</b>  ${a['price']:.2f}  "
            f"{a['change_pct']:+.1f}%  RVOL {a['rvol']:.1f}x"
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
    now  = datetime.now(timezone.utc)
    scan_dt   = now.strftime("%Y-%m-%d %H:%M UTC")
    scan_date = now.strftime("%Y-%m-%d")

    # Label the scan window in EST (UTC-5)
    est_hour = (now.hour - 5) % 24
    est_min  = now.minute
    scan_window = f"{est_hour}:{est_min:02d} EST"

    print(f"\n{'='*60}")
    print(f"Pre-market scan  |  {scan_dt}  |  {scan_window}")
    print(f"{'='*60}")

    tickers = load_tickers()
    print(f"Watchlist: {len(tickers)} tickers across {len(WATCHLISTS)} files")
    print(f"Thresholds: RVOL >= {RVOL_THRESHOLD}x  |  change >= {CHANGE_THRESHOLD}%  |  gap >= {GAP_THRESHOLD}%\n")

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
                f"  ALERT  {ticker:<10}  ${result['price']:.2f}  "
                f"{result['change_pct']:+.1f}%  RVOL {result['rvol']:.1f}x  [{types_str}]"
            )
        else:
            print(f"  clear  {ticker}")
        time.sleep(0.25)

    print(f"\n{len(triggered)} alert(s) triggered out of {len(tickers)} tickers.")

    if triggered:
        save_alerts_csv(scan_dt, scan_date, scan_window, triggered)
        n_rows = sum(len(a["triggers"]) for a in triggered)
        print(f"Appended {n_rows} row(s) to alerts.csv")
        msg = format_message(triggered, scan_window, scan_dt)
        send_telegram(msg)
    else:
        print("No alerts — nothing sent to Telegram.")


if __name__ == "__main__":
    main()
