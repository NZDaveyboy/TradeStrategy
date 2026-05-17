"""
ui/data.py — Streamlit-cached data fetchers shared across tabs.

Wraps the provider layer + heavy yfinance calls with @st.cache_data so that
multiple tabs (and multiple fragment refreshes) don't repeatedly hit the
network. Cache TTLs are tuned to the volatility of each data source:

  - Intraday bars      : 60s     (chart refresh cadence is 30s)
  - Peer fundamentals  : 1h      (P/E, margins move slowly)
  - Company info       : 24h     (sector/industry rarely change)
  - Institutional 13F  : 24h     (filings update quarterly)
  - Pile-in scan       : 24h     (depends on 13F + scan is expensive)
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.peers import fetch_peer_fundamentals_raw
from providers.yfinance_provider import YFinanceProvider


# Singleton provider — stateless wrapper around yfinance; safe at module load.
_provider = YFinanceProvider()


# ---------------------------------------------------------------------------
# FX + spot quotes
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def fetch_nzdusd() -> float:
    """NZD/USD spot rate — used for converting USD prices to NZD anywhere
    in the app. Falls back to 0.57 if the quote fails."""
    try:
        rate = _provider.get_quote("NZDUSD=X").last_price
        return rate if rate else 0.57
    except Exception:
        return 0.57


@st.cache_data(ttl=300)
def fetch_prices(tickers: tuple) -> dict:
    """Returns {ticker: {"price": float, "prev_close": float}} for many tickers."""
    result: dict = {}
    if not tickers:
        return result
    for ticker in tickers:
        try:
            quote = _provider.get_quote(ticker)
            result[ticker] = {
                "price":      quote.last_price,
                "prev_close": quote.prev_close,
            }
        except Exception:
            result[ticker] = {"price": 0.0, "prev_close": 0.0}
    return result


# ---------------------------------------------------------------------------
# Peer / company fundamentals
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_peer_fundamentals(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Streamlit-cached wrapper around the raw peer fundamentals fetcher."""
    return fetch_peer_fundamentals_raw(tickers)


@st.cache_data(ttl=86400)
def fetch_company_info(ticker: str) -> dict:
    """Company name / sector / industry / website / business summary. Cached
    24h since this stuff rarely changes."""
    try:
        f = _provider.get_fundamentals(ticker)
        return {
            "name":     f.name or ticker,
            "sector":   f.sector or "",
            "industry": f.industry or "",
            "website":  f.website or "",
            "summary":  f.summary or "",
        }
    except Exception:
        return {"name": ticker, "sector": "", "industry": "", "website": "", "summary": ""}


# ---------------------------------------------------------------------------
# Intraday bars (live candlestick chart)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def fetch_intraday_bars(ticker: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    """5-day, 5-min OHLCV bars for the live intraday chart. Cached 60s so
    rapid re-runs of the fragment don't hammer yfinance."""
    try:
        df = _provider.get_ohlcv(ticker, period, interval)
        if df is None or df.empty:
            return pd.DataFrame()
        try:
            df.index = df.index.tz_localize(None)
        except Exception:
            pass
        return df
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Institutional 13F data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400)
def fetch_institutional_data(ticker: str) -> dict:
    """13F top institutional holders + mutual fund holders + ownership
    summary. Returns DataFrames as-is. Cached 24h — 13F filings only
    update quarterly so this is generous."""
    import yfinance as yf
    out: dict = {"summary": None, "institutional": None, "mutualfund": None}
    try:
        tk = yf.Ticker(ticker)
        try:
            mh = tk.major_holders
            if mh is not None and not mh.empty:
                out["summary"] = mh["Value"].to_dict()
        except Exception:
            pass
        try:
            ih = tk.institutional_holders
            if ih is not None and not ih.empty:
                out["institutional"] = ih.copy()
        except Exception:
            pass
        try:
            mf = tk.mutualfund_holders
            if mf is not None and not mf.empty:
                out["mutualfund"] = mf.copy()
        except Exception:
            pass
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Smart Money pile-in scan
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def cached_pile_in_scan(tickers_key: tuple[str, ...], scan_id: str) -> pd.DataFrame:
    """Cached wrapper around scan_pile_ins. The scan_id arg lets a user
    force a refresh by changing it (we use the run_date as the cache key).
    24h TTL — 13F data only updates quarterly."""
    from core.smart_money import scan_pile_ins
    return scan_pile_ins(list(tickers_key))


# ---------------------------------------------------------------------------
# Metals + macro context (shared by Metals / Portfolio / Dashboard tabs)
# ---------------------------------------------------------------------------

METAL_FUTURES = {
    "Gold":      "GC=F",
    "Silver":    "SI=F",
    "Platinum":  "PL=F",
    "Palladium": "PA=F",
}
METAL_FUTURES_REV = {v: k for k, v in METAL_FUTURES.items()}

METAL_ETFS = {
    "GLD":  "SPDR Gold ETF",
    "SLV":  "iShares Silver ETF",
    "GDX":  "VanEck Gold Miners",
    "GDXJ": "VanEck Jr Gold Miners",
}

ALL_METAL_TICKERS = list(METAL_FUTURES.values()) + list(METAL_ETFS.keys())

# Driver tags for portfolio themes — used to annotate each holding with
# a one-line "why this is in the portfolio" caption.
ASSET_DRIVERS = {
    "GC=F":   ("Safe haven", "Inflation hedge", "USD inverse"),
    "SI=F":   ("Industrial demand", "Monetary hedge", "EV/solar input"),
    "PL=F":   ("Auto catalyst", "Industrial", "Supply constrained"),
    "PA=F":   ("Auto catalyst", "Industrial", "Supply constrained"),
    "GLD":    ("Safe haven", "Inflation hedge"),
    "SLV":    ("Industrial demand", "Monetary hedge"),
    "GDX":    ("Levered gold play", "Mining equity", "Operational leverage"),
    "GDXJ":   ("Junior miners", "High beta to gold", "Exploration upside"),
    "BTC-USD":("Digital store of value", "Risk-on crypto", "Institutional adoption"),
    "ETH-USD":("Smart contract platform", "DeFi infrastructure", "Risk-on crypto"),
}


@st.cache_data(ttl=300)
def fetch_metal_prices() -> dict:
    """Last price + previous close for every metal future and ETF."""
    result: dict = {}
    for ticker in ALL_METAL_TICKERS:
        try:
            q = _provider.get_quote(ticker)
            result[ticker] = {"price": q.last_price or 0.0, "prev_close": q.prev_close or 0.0}
        except Exception:
            result[ticker] = {"price": 0.0, "prev_close": 0.0}
    return result


@st.cache_data(ttl=3600)
def fetch_metal_chart(ticker: str) -> pd.DataFrame:
    """1y daily OHLCV for one metal ticker (for chart rendering)."""
    try:
        return _provider.get_ohlcv(ticker, "1y", "1d")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_metal_technicals() -> dict:
    """EMA trends, momentum, and signal for each metal future. Includes the
    EMA200 regime modifier so bear-regime moves are flagged."""
    result: dict = {}
    for name, ticker in METAL_FUTURES.items():
        try:
            df = _provider.get_ohlcv(ticker, "1y", "1d")
            if len(df) < 50:
                continue
            close = df["Close"]
            price  = float(close.iloc[-1])
            ema20  = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50  = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
            ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1]) if len(close) >= 200 else None
            mom_5d  = (price / float(close.iloc[-6])  - 1) * 100 if len(close) >= 6  else None
            mom_20d = (price / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else None
            hi_52   = float(df["High"].max()) if "High" in df.columns else price
            lo_52   = float(df["Low"].min())  if "Low"  in df.columns else price
            pct_from_hi = (price / hi_52 - 1) * 100

            above_ema20  = price > ema20
            above_ema50  = price > ema50
            above_ema200 = (price > ema200) if ema200 else None

            if above_ema20 and above_ema50:
                trend  = "Uptrend"
                signal = "Bullish"
            elif above_ema20 and not above_ema50:
                trend  = "Recovering"
                signal = "Neutral"
            elif not above_ema20 and above_ema50:
                trend  = "Pulling back"
                signal = "Watch"
            else:
                trend  = "Downtrend"
                signal = "Bearish"

            # Long-term regime modifier
            if above_ema200 is True:
                trend = f"{trend} (bull regime)"
            elif above_ema200 is False:
                trend = f"{trend} (bear regime)"
                if signal == "Bullish":
                    signal = "Neutral"
                elif signal == "Bearish":
                    signal = "Strong bearish"

            result[name] = {
                "ticker": ticker, "price": price,
                "ema20": ema20, "ema50": ema50, "ema200": ema200,
                "trend": trend, "signal": signal,
                "mom_5d": mom_5d, "mom_20d": mom_20d,
                "hi_52": hi_52, "lo_52": lo_52, "pct_from_hi": pct_from_hi,
            }
        except Exception:
            pass
    return result


@st.cache_data(ttl=300)
def fetch_market_context() -> dict:
    """Derive macro regime signals (SPY / BTC / USD / 10Y) from live price
    data. Used by both Metals and Portfolio tabs."""
    ctx: dict = {}
    for label, ticker, span in [
        ("spy",  "SPY",     "3mo"),
        ("btc",  "BTC-USD", "3mo"),
        ("usd",  "UUP",     "3mo"),   # USD Bullish ETF proxy for DXY
        ("tnx",  "^TNX",    "3mo"),   # 10-year yield
    ]:
        try:
            df    = _provider.get_ohlcv(ticker, span, "1d")
            close = df["Close"]
            price = float(close.iloc[-1])
            ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
            prev  = float(close.iloc[-2]) if len(close) >= 2 else price
            chg   = (price / prev - 1) * 100
            mom_5d = (price / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 else None
            ctx[label] = {
                "price": price, "ema20": ema20, "ema50": ema50,
                "chg": chg, "mom_5d": mom_5d,
                "above_ema20": price > ema20,
                "above_ema50": price > ema50,
            }
        except Exception:
            ctx[label] = {}
    return ctx


# ---------------------------------------------------------------------------
# Options chain helpers (live data + Greeks enrichment + payoff math)
# Moved from app.py so the Options/Learn tabs can import them directly.
# ---------------------------------------------------------------------------

import math
from datetime import datetime, timezone

import numpy as np

from core.options_math import bs_greeks


@st.cache_data(ttl=300)
def get_chain(ticker: str, expiry: str):
    """Return (calls_df, puts_df, spot) for ticker/expiry. Cached 5 min."""
    calls, puts = _provider.get_option_chain(ticker, expiry)
    spot        = _provider.get_quote(ticker).last_price
    return calls, puts, spot


@st.cache_data(ttl=3600)
def get_rv30(ticker: str) -> float | None:
    """30-day realised volatility (annualised). Cached 1h."""
    try:
        hist = _provider.get_ohlcv(ticker, "90d", "1d")
        if len(hist) < 31:
            return None
        lr = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        return float(lr.tail(30).std() * math.sqrt(252))
    except Exception:
        return None


def enrich_chain(df, spot, expiry_str, opt_type, r: float = 0.045):
    """Add Greeks, mid, break-even to a raw options-chain DataFrame."""
    today  = datetime.now(timezone.utc).date()
    exp_dt = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    T      = max((exp_dt - today).days, 0) / 365.0
    rows = []
    for _, row in df.iterrows():
        K   = float(row["strike"])
        iv  = float(row["impliedVolatility"]) if row.get("impliedVolatility", 0) > 0 else 0.0
        bid = float(row.get("bid") or 0)
        ask = float(row.get("ask") or 0)
        mid = round((bid + ask) / 2, 3) if ask > 0 else 0.0
        prem = mid or float(row.get("lastPrice") or 0)
        g   = bs_greeks(spot, K, T, r, iv, opt_type)
        be  = round(K + prem, 2) if opt_type == "call" else round(K - prem, 2)
        rows.append({
            "Strike": K, "ITM": (spot > K) if opt_type == "call" else (spot < K),
            "Bid": round(bid, 3), "Ask": round(ask, 3), "Mid": mid,
            "IV %": round(iv * 100, 1),
            "Delta": g["delta"], "Gamma": g["gamma"],
            "Theta/day": g["theta"], "Vega/1%": g["vega"],
            "OI": int(row.get("openInterest") or 0),
            "Volume": int(row.get("volume") or 0),
            "Break-even": be,
        })
    out = pd.DataFrame(rows)
    return out[(out["Bid"] > 0) | (out["OI"] > 0)].reset_index(drop=True)


def payoff_df(spot: float, legs: list[dict], price_range_pct: float = 0.30):
    """Build a payoff DataFrame from a list of option legs.

    legs: list of dicts — {type, strike, premium, qty, position}
      type: 'call' or 'put'
      position: 'long' (+1) or 'short' (-1)
      qty: number of contracts
    """
    lo = spot * (1 - price_range_pct)
    hi = spot * (1 + price_range_pct)
    prices = np.linspace(lo, hi, 200)
    total_pnl = np.zeros(len(prices))
    for leg in legs:
        K    = leg["strike"]
        prem = leg["premium"]
        pos  = 1 if leg["position"] == "long" else -1
        qty  = leg.get("qty", 1)
        if leg["type"] == "call":
            intrinsic = np.maximum(prices - K, 0)
        else:
            intrinsic = np.maximum(K - prices, 0)
        total_pnl += pos * qty * (intrinsic - prem)
    return pd.DataFrame({"Stock price": prices, "P&L per share": total_pnl})
