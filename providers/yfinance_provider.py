"""
providers/yfinance_provider.py — yfinance implementation of MarketDataProvider.

Wraps every `yf.Ticker(...)` call in one place. To swap to a paid data source,
implement `MarketDataProvider` and replace the singletons in callers.

Phase 5 refactor: `FinvizDiscoveryProvider` was moved to
`providers/scraped_provider.py` to separate HTML scraping from the
structured-API yfinance layer. Existing call sites must update their
imports — there is no backward-compat alias here.
"""

from __future__ import annotations

import yfinance as yf
import pandas as pd

from providers.base import MarketDataProvider
from data.models import Fundamentals, Quote


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
