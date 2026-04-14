"""
providers/base.py — Abstract provider interfaces for market data and ticker discovery.

Swap implementations without touching business logic. All call sites depend on
these abstractions, not on yfinance or any specific data source.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Quote:
    """Live quote from fast_info — low latency (~50ms)."""
    last_price: float
    open: float
    prev_close: float
    market_cap: float | None


@dataclass
class Fundamentals:
    """Company fundamentals from the full info dict — slower (~500ms).
    Only call when name, sector, float, or summary are actually needed."""
    name: str
    market_cap: float | None
    float_shares: float | None
    sector: str = ""
    industry: str = ""
    summary: str = ""
    website: str = ""


class MarketDataProvider(ABC):

    @abstractmethod
    def get_ohlcv(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """OHLCV bars for a rolling period. Returns DataFrame with Open/High/Low/Close/Volume."""
        ...

    @abstractmethod
    def get_ohlcv_range(self, ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        """OHLCV bars for a fixed date range. Used by backtests."""
        ...

    @abstractmethod
    def get_quote(self, ticker: str) -> Quote:
        """Live price, open, previous close, market cap.
        Never raises — returns Quote with 0.0 / None on failure."""
        ...

    @abstractmethod
    def get_fundamentals(self, ticker: str) -> Fundamentals:
        """Full company info. Slow — only call when name, sector, or float are needed."""
        ...

    @abstractmethod
    def get_expiries(self, ticker: str) -> tuple[str, ...]:
        """Available option expiry dates, ascending. Returns empty tuple on failure."""
        ...

    @abstractmethod
    def get_option_chain(self, ticker: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returns (calls_df, puts_df) for the given expiry."""
        ...


class TickerDiscoveryProvider(ABC):

    @abstractmethod
    def get_gainers(self, limit: int = 50) -> list[str]:
        """Returns ticker symbols for today's top gainers."""
        ...
