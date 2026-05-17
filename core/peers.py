"""
core/peers.py — Peer-group definitions and fundamentals fetcher.

Single source of truth for which tickers are considered peers, plus a raw
yfinance fetcher that returns a DataFrame of P/E, PEG, margin, ROE, etc.
Both the Lookup tab and the Copilot tab read from here.
"""

from __future__ import annotations

import pandas as pd


PEER_MAP: dict[str, tuple[str, ...]] = {
    # Mega-cap tech
    "AAPL":  ("MSFT", "GOOG", "AMZN", "META"),
    "MSFT":  ("AAPL", "GOOG", "AMZN", "META"),
    "GOOG":  ("MSFT", "AAPL", "META", "AMZN"),
    "GOOGL": ("MSFT", "AAPL", "META", "AMZN"),
    "AMZN":  ("MSFT", "GOOG", "WMT", "META"),
    "META":  ("GOOG", "MSFT", "AAPL", "AMZN"),
    "TSLA":  ("F", "GM", "RIVN", "LCID"),
    # Semis
    "NVDA": ("AMD", "AVGO", "INTC", "TSM"),
    "AMD":  ("NVDA", "INTC", "AVGO", "QCOM"),
    "INTC": ("AMD", "NVDA", "AVGO", "MU"),
    "AVGO": ("NVDA", "AMD", "QCOM", "TXN"),
    "QCOM": ("AVGO", "AMD", "NVDA", "TXN"),
    "TSM":  ("NVDA", "AMD", "AVGO", "INTC"),
    "MU":   ("INTC", "AMD", "TXN", "ON"),
    # Software / cloud
    "CRM":  ("MSFT", "ORCL", "NOW", "WDAY"),
    "ORCL": ("MSFT", "CRM", "SAP", "IBM"),
    "NOW":  ("CRM", "MSFT", "WDAY", "ORCL"),
    "ADBE": ("MSFT", "CRM", "INTU", "ORCL"),
    "PLTR": ("SNOW", "DDOG", "MDB", "CRWD"),
    "SNOW": ("PLTR", "DDOG", "MDB", "NET"),
    "DDOG": ("SNOW", "NET", "CRWD", "ZS"),
    "CRWD": ("ZS", "PANW", "NET", "S"),
    "PANW": ("CRWD", "ZS", "FTNT", "NET"),
    "NET":  ("DDOG", "SNOW", "CRWD", "ZS"),
    # Consumer / streaming
    "NFLX": ("DIS", "PARA", "WBD", "GOOG"),
    "DIS":  ("NFLX", "PARA", "WBD", "CMCSA"),
    # Fintech / payments
    "V":    ("MA", "AXP", "PYPL", "SQ"),
    "MA":   ("V", "AXP", "PYPL", "DFS"),
    "PYPL": ("V", "MA", "SQ", "AFRM"),
    "SOFI": ("LC", "AFRM", "UPST", "PYPL"),
    "AFRM": ("SOFI", "UPST", "PYPL", "SQ"),
    # Banks
    "JPM":  ("BAC", "WFC", "C", "GS"),
    "BAC":  ("JPM", "WFC", "C", "MS"),
    "GS":   ("MS", "JPM", "C", "BAC"),
    # Auto / EV
    "F":    ("GM", "TSLA", "STLA", "RIVN"),
    "GM":   ("F", "TSLA", "STLA", "RIVN"),
    "RIVN": ("LCID", "TSLA", "F", "GM"),
    "LCID": ("RIVN", "TSLA", "F", "GM"),
    # Crypto-adjacent
    "COIN": ("MSTR", "MARA", "RIOT", "HOOD"),
    "HOOD": ("COIN", "SCHW", "IBKR", "SOFI"),
}


def fetch_peer_fundamentals_raw(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Pull fundamentals for each ticker via yfinance. Not cached — callers
    should wrap with their own cache (Streamlit's @st.cache_data, etc.)."""
    import yfinance as yf

    rows = []
    for tkr in tickers:
        try:
            info = yf.Ticker(tkr).info or {}
            rows.append({
                "Ticker":      tkr,
                "Name":        (info.get("shortName") or info.get("longName") or tkr)[:24],
                "Market cap":  info.get("marketCap"),
                "P/E (TTM)":   info.get("trailingPE"),
                "Forward P/E": info.get("forwardPE"),
                "PEG":         info.get("trailingPegRatio") or info.get("pegRatio"),
                "Net margin":  info.get("profitMargins"),
                "ROE":         info.get("returnOnEquity"),
                "Rev growth":  info.get("revenueGrowth"),
            })
        except Exception:
            rows.append({
                "Ticker": tkr, "Name": tkr, "Market cap": None,
                "P/E (TTM)": None, "Forward P/E": None, "PEG": None,
                "Net margin": None, "ROE": None, "Rev growth": None,
            })
    return pd.DataFrame(rows)
