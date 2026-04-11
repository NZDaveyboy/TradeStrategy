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
        ("market_cap",  "REAL"),
        ("float_shares","REAL"),
        ("tradescore",  "REAL"),
        ("explain",     "TEXT"),
        ("setup_type",  "TEXT"),
        ("rationale",   "TEXT"),
        ("change_5d",   "REAL"),
        ("direction",   "TEXT"),
    ]:
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
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
# Scoring constants — adjust thresholds here without touching logic
# ---------------------------------------------------------------------------

# MomentumScore (max 25)
MS_RVOL_IDEAL   = 3.0   # RVOL at which full RVOL points are earned
MS_RVOL_MAX_PTS = 10    # sub-cap for RVOL contribution
MS_CHG_HI_PCT   = 8.0   # % change earning full change points (sweet spot ceiling)
MS_CHG_MAX_PTS  = 8     # sub-cap for change% contribution
MS_MACD_MAX_PTS = 7     # sub-cap for MACD contribution

# EarlyEntryScore (max 25)
EE_RSI_LO       = 52    # ideal RSI band lower bound
EE_RSI_HI       = 68    # ideal RSI band upper bound (above here = heating up)
EE_RSI_MAX_PTS  = 10    # sub-cap for RSI contribution
EE_EMA_NEAR_PCT = 5.0   # within this % of EMA20 → full proximity points
EE_EMA_FAR_PCT  = 18.0  # beyond this % from EMA20 → 0 proximity points
EE_EMA_MAX_PTS  = 8     # sub-cap for EMA proximity
EE_BOB_MAX_PTS  = 7     # sub-cap for breakout-from-base contribution

# ExtensionRiskScore (max 20, subtracted from total)
ER_RSI_WARN     = 70    # RSI above this starts penalty
ER_RSI_HARD     = 82    # RSI at this = max RSI penalty
ER_RSI_MAX_PTS  = 6     # max RSI penalty points
ER_DAY_WARN     = 10.0  # single-day % move starts penalty
ER_DAY_HARD     = 22.0  # single-day % move = max penalty
ER_DAY_MAX_PTS  = 6     # max daily-overextension penalty points
ER_VWAP_WARN    = 1.5   # ATR multiples above VWAP starts penalty
ER_VWAP_HARD    = 4.0   # ATR multiples = max VWAP penalty
ER_VWAP_MAX_PTS = 5     # max VWAP extension penalty points
ER_5D_WARN      = 15.0  # 5-session % run starts multi-day penalty
ER_5D_HARD      = 45.0  # 5-session % run = max multi-day penalty
ER_5D_MAX_PTS   = 3     # max multi-day-run penalty points

# LiquidityQualityScore (max 15)
LQ_DVOL_MIN     = 500_000    # dollar volume below this → 0 liquidity points
LQ_DVOL_FULL    = 15_000_000 # dollar volume above this → full points
LQ_DVOL_MAX_PTS = 8          # max dollar-volume points
LQ_QUAL_MAX_PTS = 4          # max float/mcap quality points
LQ_CONS_MAX_PTS = 3          # max volume-consistency points


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _lerp(val: float, lo: float, hi: float) -> float:
    """Linear 0→1 as val goes from lo to hi, clamped."""
    if hi <= lo:
        return 1.0 if val >= hi else 0.0
    return _clamp((val - lo) / (hi - lo))


# ---------------------------------------------------------------------------
# Sub-score functions
# ---------------------------------------------------------------------------

def _momentum_score(row: dict) -> tuple[float, dict]:
    """
    MomentumScore 0–25.
    RVOL (10) + Change% (8) + MACD (7).
    RVOL sub-linear above ideal — extreme RVOL can signal panic/chase, not edge.
    Change% decays above sweet spot so a 15% gap day isn't rewarded more than 8%.
    """
    rvol       = float(row.get("rvol", 0))
    change_pct = float(row.get("change_pct", 0))
    macd       = float(row.get("macd", 0))
    macd_sig   = float(row.get("macd_signal", 0))
    atr_val    = float(row.get("atr", 0.01)) or 0.01

    # RVOL: ramps up to ideal, then slowly diminishes for extremes
    if rvol <= 0:
        rvol_pts = 0.0
    elif rvol <= MS_RVOL_IDEAL:
        rvol_pts = MS_RVOL_MAX_PTS * (rvol / MS_RVOL_IDEAL) ** 0.6
    else:
        excess   = _clamp((rvol - MS_RVOL_IDEAL) / MS_RVOL_IDEAL)
        rvol_pts = MS_RVOL_MAX_PTS * (1.0 - 0.2 * excess)
    rvol_pts = _clamp(rvol_pts, 0, MS_RVOL_MAX_PTS)

    # Change%: full at sweet spot ceiling, decays above it
    if change_pct <= 0:
        chg_pts = 0.0
    elif change_pct <= MS_CHG_HI_PCT:
        chg_pts = MS_CHG_MAX_PTS * (change_pct / MS_CHG_HI_PCT)
    else:
        chg_pts = MS_CHG_MAX_PTS * max(0.4, 1.0 - (change_pct - MS_CHG_HI_PCT) / 25.0)
    chg_pts = _clamp(chg_pts, 0, MS_CHG_MAX_PTS)

    # MACD: normalized to ATR. ±0.10 ATR diff maps to 0–7 pts.
    macd_norm = (macd - macd_sig) / atr_val
    macd_pts  = _clamp((macd_norm + 0.1) / 0.2) * MS_MACD_MAX_PTS

    total = rvol_pts + chg_pts + macd_pts
    return round(total, 2), {
        "rvol_pts":   round(rvol_pts, 2),
        "change_pts": round(chg_pts,  2),
        "macd_pts":   round(macd_pts, 2),
    }


def _early_entry_score(row: dict, close: pd.Series | None) -> tuple[float, dict]:
    """
    EarlyEntryScore 0–25.
    RSI zone (10) + EMA20 proximity (8) + breakout-from-base (7).
    RSI 52–68 is the sweet spot: confirmed momentum, not yet overbought.
    EMA proximity rewards names that haven't yet run far from their trend.
    BOB (breakout-from-base) rewards price at or near 20-session highs.
    """
    rsi_val = float(row.get("rsi", 50))
    ema20   = float(row.get("ema20", 0)) or float(row.get("price", 1))
    price   = float(row.get("price", 1)) or 1.0

    # RSI zone: peak in [EE_RSI_LO, EE_RSI_HI], decays linearly outside
    if EE_RSI_LO <= rsi_val <= EE_RSI_HI:
        rsi_pts = float(EE_RSI_MAX_PTS)
    elif 42 <= rsi_val < EE_RSI_LO:
        rsi_pts = EE_RSI_MAX_PTS * _lerp(rsi_val, 42, EE_RSI_LO)
    elif EE_RSI_HI < rsi_val <= 76:
        rsi_pts = EE_RSI_MAX_PTS * (1.0 - _lerp(rsi_val, EE_RSI_HI, 76))
    else:
        rsi_pts = 0.0

    # EMA20 proximity: the closer to the moving average, the cleaner the entry
    dist_pct = abs(price - ema20) / ema20 * 100 if ema20 > 0 else EE_EMA_FAR_PCT
    if dist_pct <= EE_EMA_NEAR_PCT:
        ema_pts = float(EE_EMA_MAX_PTS)
    elif dist_pct <= EE_EMA_FAR_PCT:
        ema_pts = EE_EMA_MAX_PTS * (1.0 - _lerp(dist_pct, EE_EMA_NEAR_PCT, EE_EMA_FAR_PCT))
    else:
        ema_pts = 0.0

    # Breakout-from-base: is price at or near its 20-session high?
    bob_pts = 0.0
    if close is not None and len(close) >= 20:
        hi_20   = float(close.iloc[-20:].max())
        pct_hi  = price / hi_20 if hi_20 > 0 else 0.0
        if pct_hi >= 0.97:      # at or above 20-day high — fresh breakout
            bob_pts = float(EE_BOB_MAX_PTS)
        elif pct_hi >= 0.85:
            bob_pts = EE_BOB_MAX_PTS * _lerp(pct_hi, 0.85, 0.97)

    total = rsi_pts + ema_pts + bob_pts
    return round(total, 2), {
        "rsi_pts": round(rsi_pts, 2),
        "ema_pts": round(ema_pts, 2),
        "bob_pts": round(bob_pts, 2),
    }


def _extension_risk_score(row: dict, close: pd.Series | None) -> tuple[float, dict]:
    """
    ExtensionRiskScore 0–20 (subtracted). Higher = more dangerous entry.
    RSI overbought (6) + single-day overextension (6) + VWAP distance (5) + 5-day run (3).
    """
    rsi_val    = float(row.get("rsi", 50))
    change_pct = float(row.get("change_pct", 0))
    price      = float(row.get("price", 1))
    vwap       = float(row.get("vwap", price)) or price
    atr_val    = float(row.get("atr", 0.01)) or 0.01

    # RSI overbought penalty
    rsi_pen = ER_RSI_MAX_PTS * _lerp(rsi_val, ER_RSI_WARN, ER_RSI_HARD)

    # Single-day overextension penalty (absolute move)
    day_pen = ER_DAY_MAX_PTS * _lerp(abs(change_pct), ER_DAY_WARN, ER_DAY_HARD)

    # VWAP distance penalty in ATR multiples
    vwap_dist = abs(price - vwap) / atr_val
    vwap_pen  = ER_VWAP_MAX_PTS * _lerp(vwap_dist, ER_VWAP_WARN, ER_VWAP_HARD)

    # 5-session cumulative run penalty
    run5_pen  = 0.0
    if close is not None and len(close) >= 6:
        change_5d = (float(close.iloc[-1]) / float(close.iloc[-6]) - 1) * 100
        run5_pen  = ER_5D_MAX_PTS * _lerp(abs(change_5d), ER_5D_WARN, ER_5D_HARD)

    total = rsi_pen + day_pen + vwap_pen + run5_pen
    return round(_clamp(total, 0, 20), 2), {
        "rsi_pen":  round(rsi_pen,  2),
        "day_pen":  round(day_pen,  2),
        "vwap_pen": round(vwap_pen, 2),
        "run5_pen": round(run5_pen, 2),
    }


def _liquidity_score(row: dict, data: pd.DataFrame | None) -> tuple[float, dict]:
    """
    LiquidityQualityScore 0–15.
    Dollar volume (8) + float/mcap quality tier (4) + volume consistency (3).
    Mid-cap scores highest on quality — cleaner trends, less manipulation risk.
    """
    price      = float(row.get("price", 0))
    market_cap = row.get("market_cap")

    # Dollar volume
    dvol = 0.0
    if data is not None and len(data) > 0:
        dvol = price * float(data["Volume"].iloc[-1])
    if dvol >= LQ_DVOL_FULL:
        dvol_pts = float(LQ_DVOL_MAX_PTS)
    elif dvol >= LQ_DVOL_MIN:
        dvol_pts = LQ_DVOL_MAX_PTS * _lerp(dvol, LQ_DVOL_MIN, LQ_DVOL_FULL)
    else:
        dvol_pts = 0.0

    # Float/mcap quality tier
    float_shares = row.get("float_shares")
    if market_cap:
        if market_cap >= 10_000_000_000:    # large cap — valid but not the focus
            qual_pts = 2.0
        elif market_cap >= 2_000_000_000:   # mid cap — cleanest for trend trades
            qual_pts = float(LQ_QUAL_MAX_PTS)
        elif market_cap >= 500_000_000:     # small cap — higher risk, bigger moves
            qual_pts = 3.0
        else:                               # micro cap — added risk, thin market
            qual_pts = 1.0
        if float_shares and float_shares < 10_000_000:   # very low float = spike risk
            qual_pts = max(0.0, qual_pts - 2.0)
    else:
        qual_pts = 2.0  # unknown = neutral

    # Volume consistency: low coefficient of variation = consistent participation
    cons_pts = 0.0
    if data is not None and len(data) >= 11:
        vols = data["Volume"].iloc[-11:-1]
        mean = vols.mean()
        if mean > 0:
            cv       = vols.std() / mean
            cons_pts = LQ_CONS_MAX_PTS * max(0.0, 1.0 - float(cv))

    total = dvol_pts + qual_pts + cons_pts
    return round(_clamp(total, 0, 15), 2), {
        "dvol_pts": round(dvol_pts, 2),
        "qual_pts": round(qual_pts, 2),
        "cons_pts": round(cons_pts, 2),
    }


def _news_catalyst_score(_row: dict) -> tuple[float, dict]:
    """NewsCatalystScore 0–15. Stubbed pending news integration."""
    return 0.0, {"note": "stub — no news source connected"}


_BEARISH_LABELS = {
    "Overextended":        "Extended downside move",
    "Strong but extended": "Strong downside setup",
    "Early breakout":      "Bearish breakdown",
    "Emerging momentum":   "Emerging weakness",
    "Momentum watchlist":  "Bearish watchlist",
    "Avoid":               "Avoid",
    "Low quality / illiquid": "Low quality / illiquid",
}


def _setup_type(ms: float, ee: float, er: float, lq: float,
                rsi: float, change_5d: float,
                change_pct: float = 0.0, direction: str = "long") -> str:
    """
    Derive a human label from sub-score geometry.
    Order matters — stronger disqualifiers checked first.

    change_pct is today's single-day move. A move >= 15% is treated as
    extended regardless of what ER scores, because the 1-year cumulative
    VWAP used in ER can understate intraday extension for recently beaten-
    down stocks.

    direction: "long" | "short" | "neutral" — bearish setups get bearish labels.
    """
    if lq <= 3:
        label = "Low quality / illiquid"
    elif er >= 15:
        label = "Overextended"
    elif abs(change_pct) >= 15.0 and ms >= 10:
        label = "Strong but extended"
    elif ms >= 13 and er >= 8:
        label = "Strong but extended"
    elif ms >= 17 and ee >= 15 and er <= 7:
        label = "Early breakout"
    elif ms >= 13 and ee >= 11 and er <= 9:
        label = "Emerging momentum"
    elif (ms + ee - er) >= 20:
        label = "Emerging momentum"
    elif (ms + ee - er) >= 10:
        label = "Momentum watchlist"
    else:
        label = "Avoid"

    if direction == "short":
        return _BEARISH_LABELS.get(label, label)
    return label


def _build_rationale(row: dict, ms: float, ee: float, er: float,
                     setup_type: str, change_5d: float) -> str:
    """One-line explanation for why this ticker ranked where it did."""
    price   = float(row.get("price", 0))
    rvol    = float(row.get("rvol", 0))
    rsi_val = float(row.get("rsi", 50))
    chg     = float(row.get("change_pct", 0))
    ema20   = float(row.get("ema20", 0)) or price
    dist    = (price - ema20) / ema20 * 100 if ema20 > 0 else 0.0

    parts = []
    if setup_type == "Early breakout":
        parts.append(f"Breaking out on {rvol:.1f}x RVOL")
    elif setup_type == "Emerging momentum":
        parts.append(f"Building momentum, {rvol:.1f}x RVOL")
    elif setup_type == "Strong but extended":
        parts.append(f"Strong move but stretched")
    elif setup_type == "Overextended":
        parts.append(f"Likely too late — {change_5d:+.0f}% 5-day run")
    elif setup_type == "Momentum watchlist":
        parts.append(f"Needs confirmation")
    elif setup_type == "Low quality / illiquid":
        parts.append(f"Thin liquidity, elevated risk")

    parts.append(f"RSI {rsi_val:.0f}")
    if abs(dist) > 8:
        parts.append(f"{dist:+.0f}% from EMA20")
    if er >= 10:
        parts.append("elevated chase risk")

    return ", ".join(parts)


# ---------------------------------------------------------------------------
# TradeScore — composite
# ---------------------------------------------------------------------------

def compute_tradescore(
    row: dict,
    close: pd.Series | None = None,
    data: pd.DataFrame | None = None,
) -> dict:
    """
    FinalTradeScore = MomentumScore + EarlyEntryScore + LiquidityScore
                      + NewsCatalystScore - ExtensionRiskScore

    Practical range 0–65. Negative values clipped to 0.
    Sub-scores are also returned for display and tuning.
    """
    ms_val,  ms_det  = _momentum_score(row)
    ee_val,  ee_det  = _early_entry_score(row, close)
    er_val,  er_det  = _extension_risk_score(row, close)
    lq_val,  lq_det  = _liquidity_score(row, data)
    nc_val,  nc_det  = _news_catalyst_score(row)

    final = round(max(0.0, ms_val + ee_val + lq_val + nc_val - er_val), 1)

    change_5d = 0.0
    if close is not None and len(close) >= 6:
        change_5d = round((float(close.iloc[-1]) / float(close.iloc[-6]) - 1) * 100, 2)

    change_pct = float(row.get("change_pct", 0))

    # Direction: long when price above VWAP and short-term EMA above medium-term.
    price = float(row.get("price", 0))
    vwap  = float(row.get("vwap", price)) or price
    ema9  = float(row.get("ema9",  price)) or price
    ema20 = float(row.get("ema20", price)) or price
    if price > vwap and ema9 >= ema20:
        direction = "long"
    elif price < vwap and ema9 <= ema20:
        direction = "short"
    else:
        direction = "neutral"

    setup     = _setup_type(ms_val, ee_val, er_val, lq_val,
                            float(row.get("rsi", 50)), change_5d,
                            change_pct, direction)
    rationale = _build_rationale(row, ms_val, ee_val, er_val, setup, change_5d)

    return {
        "score":           final,
        "momentum_score":  ms_val,
        "early_entry":     ee_val,
        "extension_risk":  er_val,
        "liquidity":       lq_val,
        "news_catalyst":   nc_val,
        "direction":       direction,
        "setup_type":      setup,
        "rationale":       rationale,
        "change_5d":       change_5d,
        "conviction":      setup,
        "components": {
            "momentum":    ms_det,
            "early_entry": ee_det,
            "extension":   er_det,
            "liquidity":   lq_det,
        },
    }


def conviction_label(score: float, setup_type: str = "") -> str:
    """Kept for backwards compat — returns the setup_type label directly."""
    return setup_type if setup_type else (
        "Early breakout"    if score >= 52 else
        "Emerging momentum" if score >= 35 else
        "Momentum watchlist" if score >= 20 else
        "Avoid"
    )


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

    # Market cap via fast_info (fast, ~50 ms) for all tickers.
    # Full info (slow) only for momentum strategy where float_shares is needed.
    market_cap   = None
    float_shares = None
    try:
        market_cap = getattr(tk.fast_info, "market_cap", None)
    except Exception:
        pass

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
