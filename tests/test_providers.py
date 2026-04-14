"""
tests/test_providers.py

Verifies that YFinanceProvider and FinvizDiscoveryProvider:
  - call the correct yfinance / requests methods
  - return the correct Quote / Fundamentals shapes
  - degrade gracefully on network errors (no exception raised to caller)

All network I/O is mocked — no real HTTP calls.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from providers.base import Fundamentals, Quote
from providers.yfinance_provider import FinvizDiscoveryProvider, YFinanceProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv() -> pd.DataFrame:
    return pd.DataFrame({
        "Open":   [100.0],
        "High":   [105.0],
        "Low":    [99.0],
        "Close":  [103.0],
        "Volume": [1_000_000],
    })


# ---------------------------------------------------------------------------
# YFinanceProvider.get_ohlcv
# ---------------------------------------------------------------------------

@patch("providers.yfinance_provider.yf.Ticker")
def test_get_ohlcv_calls_history_with_period_and_interval(mock_ticker):
    mock_tk = MagicMock()
    mock_tk.history.return_value = _make_ohlcv()
    mock_ticker.return_value = mock_tk

    df = YFinanceProvider().get_ohlcv("SPY", "20d", "1d")

    mock_ticker.assert_called_once_with("SPY")
    mock_tk.history.assert_called_once_with(period="20d", interval="1d")
    assert set(df.columns) == {"Open", "High", "Low", "Close", "Volume"}


# ---------------------------------------------------------------------------
# YFinanceProvider.get_ohlcv_range
# ---------------------------------------------------------------------------

@patch("providers.yfinance_provider.yf.Ticker")
def test_get_ohlcv_range_calls_history_with_dates(mock_ticker):
    mock_tk = MagicMock()
    mock_tk.history.return_value = _make_ohlcv()
    mock_ticker.return_value = mock_tk

    YFinanceProvider().get_ohlcv_range("AAPL", "2024-01-01", "2024-01-31")

    mock_ticker.assert_called_once_with("AAPL")
    mock_tk.history.assert_called_once_with(
        start="2024-01-01", end="2024-01-31", interval="1d"
    )


@patch("providers.yfinance_provider.yf.Ticker")
def test_get_ohlcv_range_respects_interval_override(mock_ticker):
    mock_tk = MagicMock()
    mock_tk.history.return_value = _make_ohlcv()
    mock_ticker.return_value = mock_tk

    YFinanceProvider().get_ohlcv_range("SPY", "2024-01-01", "2024-01-31", interval="15m")

    mock_tk.history.assert_called_once_with(
        start="2024-01-01", end="2024-01-31", interval="15m"
    )


# ---------------------------------------------------------------------------
# YFinanceProvider.get_quote
# ---------------------------------------------------------------------------

@patch("providers.yfinance_provider.yf.Ticker")
def test_get_quote_returns_correct_values(mock_ticker):
    mock_fi = MagicMock()
    mock_fi.last_price    = 150.25
    mock_fi.open          = 148.00
    mock_fi.previous_close = 147.50
    mock_fi.market_cap    = 2_500_000_000
    mock_ticker.return_value.fast_info = mock_fi

    quote = YFinanceProvider().get_quote("AAPL")

    assert isinstance(quote, Quote)
    assert quote.last_price == 150.25
    assert quote.open       == 148.00
    assert quote.prev_close == 147.50
    assert quote.market_cap == 2_500_000_000


@patch("providers.yfinance_provider.yf.Ticker")
def test_get_quote_returns_safe_defaults_on_error(mock_ticker):
    mock_ticker.side_effect = Exception("network timeout")

    quote = YFinanceProvider().get_quote("FAIL")

    assert isinstance(quote, Quote)
    assert quote.last_price == 0.0
    assert quote.open       == 0.0
    assert quote.prev_close == 0.0
    assert quote.market_cap is None


# ---------------------------------------------------------------------------
# YFinanceProvider.get_fundamentals
# ---------------------------------------------------------------------------

@patch("providers.yfinance_provider.yf.Ticker")
def test_get_fundamentals_returns_all_fields(mock_ticker):
    mock_ticker.return_value.info = {
        "longName":             "Apple Inc.",
        "marketCap":            3_000_000_000_000,
        "floatShares":          15_000_000_000,
        "sector":               "Technology",
        "industry":             "Consumer Electronics",
        "longBusinessSummary":  "Apple designs great products.",
        "website":              "https://apple.com",
    }

    fund = YFinanceProvider().get_fundamentals("AAPL")

    assert isinstance(fund, Fundamentals)
    assert fund.name         == "Apple Inc."
    assert fund.market_cap   == 3_000_000_000_000
    assert fund.float_shares == 15_000_000_000
    assert fund.sector       == "Technology"
    assert fund.industry     == "Consumer Electronics"
    assert fund.summary      == "Apple designs great products."
    assert fund.website      == "https://apple.com"


@patch("providers.yfinance_provider.yf.Ticker")
def test_get_fundamentals_falls_back_to_shortname(mock_ticker):
    mock_ticker.return_value.info = {
        "shortName": "Apple",
        "marketCap": None,
    }

    fund = YFinanceProvider().get_fundamentals("AAPL")
    assert fund.name == "Apple"


@patch("providers.yfinance_provider.yf.Ticker")
def test_get_fundamentals_returns_safe_defaults_on_error(mock_ticker):
    mock_ticker.side_effect = Exception("no data")

    fund = YFinanceProvider().get_fundamentals("FAIL")

    assert isinstance(fund, Fundamentals)
    assert fund.name         == "FAIL"
    assert fund.market_cap   is None
    assert fund.float_shares is None
    assert fund.sector       == ""


# ---------------------------------------------------------------------------
# YFinanceProvider.get_expiries
# ---------------------------------------------------------------------------

@patch("providers.yfinance_provider.yf.Ticker")
def test_get_expiries_returns_tuple_of_strings(mock_ticker):
    mock_ticker.return_value.options = ("2024-02-16", "2024-03-15", "2024-06-21")

    expiries = YFinanceProvider().get_expiries("SPY")
    assert expiries == ("2024-02-16", "2024-03-15", "2024-06-21")


@patch("providers.yfinance_provider.yf.Ticker")
def test_get_expiries_returns_empty_tuple_on_error(mock_ticker):
    mock_ticker.side_effect = Exception("no options")

    assert YFinanceProvider().get_expiries("FAIL") == ()


# ---------------------------------------------------------------------------
# YFinanceProvider.get_option_chain
# ---------------------------------------------------------------------------

@patch("providers.yfinance_provider.yf.Ticker")
def test_get_option_chain_returns_calls_and_puts(mock_ticker):
    calls_df = pd.DataFrame({"strike": [100.0, 105.0], "lastPrice": [3.5, 1.2]})
    puts_df  = pd.DataFrame({"strike": [100.0, 95.0],  "lastPrice": [2.5, 0.8]})
    mock_chain       = MagicMock()
    mock_chain.calls = calls_df
    mock_chain.puts  = puts_df
    mock_ticker.return_value.option_chain.return_value = mock_chain

    calls, puts = YFinanceProvider().get_option_chain("SPY", "2024-02-16")

    mock_ticker.return_value.option_chain.assert_called_once_with("2024-02-16")
    assert "strike"    in calls.columns
    assert "lastPrice" in calls.columns
    assert "strike"    in puts.columns
    assert len(calls) == 2
    assert len(puts)  == 2


# ---------------------------------------------------------------------------
# FinvizDiscoveryProvider.get_gainers
# ---------------------------------------------------------------------------

_FINVIZ_HTML = """
<html><body>
  <a href="quote.ashx?t=NVDA&amp;ty=c&amp;p=d&amp;b=1">NVDA</a>
  <a href="quote.ashx?t=AMD&amp;ty=c">AMD</a>
  <a href="quote.ashx?t=NVDA&amp;ty=c">NVDA</a>
  <a href="other.ashx?t=SKIP">should be skipped</a>
  <a href="quote.ashx?t=tsla&amp;ty=c">tsla</a>
</body></html>
"""


@patch("providers.yfinance_provider.requests.get")
def test_finviz_get_gainers_parses_tickers(mock_get):
    mock_resp = MagicMock()
    mock_resp.text = _FINVIZ_HTML
    mock_resp.raise_for_status = lambda: None
    mock_get.return_value = mock_resp

    tickers = FinvizDiscoveryProvider().get_gainers(limit=10)

    assert "NVDA" in tickers
    assert "AMD"  in tickers
    assert "TSLA" in tickers           # uppercased
    assert "SKIP" not in tickers


@patch("providers.yfinance_provider.requests.get")
def test_finviz_get_gainers_deduplicates(mock_get):
    mock_resp = MagicMock()
    mock_resp.text = _FINVIZ_HTML
    mock_resp.raise_for_status = lambda: None
    mock_get.return_value = mock_resp

    tickers = FinvizDiscoveryProvider().get_gainers(limit=10)
    assert tickers.count("NVDA") == 1  # appears twice in HTML, returned once


@patch("providers.yfinance_provider.requests.get")
def test_finviz_get_gainers_respects_limit(mock_get):
    # 3 unique tickers in HTML; limit=2 should return only 2
    mock_resp = MagicMock()
    mock_resp.text = _FINVIZ_HTML
    mock_resp.raise_for_status = lambda: None
    mock_get.return_value = mock_resp

    tickers = FinvizDiscoveryProvider().get_gainers(limit=2)
    assert len(tickers) <= 2


@patch("providers.yfinance_provider.requests.get")
def test_finviz_get_gainers_returns_empty_list_on_error(mock_get):
    mock_get.side_effect = Exception("connection refused")

    tickers = FinvizDiscoveryProvider().get_gainers()
    assert tickers == []
