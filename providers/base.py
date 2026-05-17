"""
providers/base.py — Abstract provider interfaces for market data and ticker discovery.

Swap implementations without touching business logic. All call sites depend on
these abstractions, not on yfinance or any specific data source.

The typed data schemas (Quote, Fundamentals, OHLCVBar, OptionContract, etc.)
live in `data/models.py` — this module re-exports them for backward compat
with existing imports. New code should import from `data.models` directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

# Re-export typed schemas from the data layer (Phase 5 refactor).
# Existing call sites that do `from providers.base import Quote, Fundamentals`
# continue to work; new code should import from `data.models` directly.
from data.models import Fundamentals, Quote  # noqa: F401  (re-exported)


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
