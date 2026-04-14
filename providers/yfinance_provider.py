"""
providers/yfinance_provider.py — Concrete implementations of the provider interfaces.

YFinanceProvider wraps all yf.Ticker() calls in one place.
FinvizDiscoveryProvider wraps the top-gainers HTML scrape.

To swap to a paid data source, implement MarketDataProvider / TickerDiscoveryProvider
and replace the singletons in the modules that import them.
"""

from __future__ import annotations

import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd

from providers.base import (
    Fundamentals,
    MarketDataProvider,
    Quote,
    TickerDiscoveryProvider,
)


class YFinanceProvider(MarketDataProvider):

    def get_ohlcv(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        return yf.Ticker(ticker).history(period=period, interval=interval)

    def get_ohlcv_range(
        self, ticker: str, start: str, end: str, interval: str = "1d"
    ) -> pd.DataFrame:
        return yf.Ticker(ticker).history(start=start, end=end, interval=interval)

    def get_quote(self, ticker: str) -> Quote:
        try:
            fi = yf.Ticker(ticker).fast_info
            return Quote(
                last_price=float(getattr(fi, "last_price",   None) or 0),
                open=      float(getattr(fi, "open",         None) or 0),
                prev_close=float(getattr(fi, "previous_close", None) or 0),
                market_cap=      getattr(fi, "market_cap",   None),
            )
        except Exception:
            return Quote(last_price=0.0, open=0.0, prev_close=0.0, market_cap=None)

    def get_fundamentals(self, ticker: str) -> Fundamentals:
        try:
            info = yf.Ticker(ticker).info
            return Fundamentals(
                name=        info.get("longName") or info.get("shortName") or ticker,
                market_cap=  info.get("marketCap"),
                float_shares=info.get("floatShares"),
                sector=      info.get("sector")               or "",
                industry=    info.get("industry")             or "",
                summary=     info.get("longBusinessSummary")  or "",
                website=     info.get("website")              or "",
            )
        except Exception:
            return Fundamentals(name=ticker, market_cap=None, float_shares=None)

    def get_expiries(self, ticker: str) -> tuple[str, ...]:
        try:
            return yf.Ticker(ticker).options
        except Exception:
            return ()

    def get_option_chain(
        self, ticker: str, expiry: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        chain = yf.Ticker(ticker).option_chain(expiry)
        return chain.calls, chain.puts


class FinvizDiscoveryProvider(TickerDiscoveryProvider):

    _URL = "https://finviz.com/screener.ashx?v=111&s=ta_topgainers"
    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    def get_gainers(self, limit: int = 50) -> list[str]:
        try:
            resp = requests.get(self._URL, headers=self._HEADERS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            tickers: list[str] = []
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
