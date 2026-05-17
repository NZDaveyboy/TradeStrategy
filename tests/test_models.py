"""
tests/test_models.py — Tests for data/models.py typed schemas.

These verify the contract of the dataclasses (immutability, fields,
defaults) and that the re-exports from providers.base still work for
backward compat.
"""

from __future__ import annotations

from datetime import date

import pytest

from data.models import (
    Fundamentals,
    OHLCVBar,
    OptionChain,
    OptionContract,
    Quote,
)


# ---------------------------------------------------------------------------
# Quote
# ---------------------------------------------------------------------------

def test_quote_has_required_fields():
    q = Quote(last_price=100.0, open=99.0, prev_close=98.0, market_cap=1e9)
    assert q.last_price == 100.0
    assert q.market_cap == 1e9


def test_quote_market_cap_can_be_none():
    q = Quote(last_price=10.0, open=10.0, prev_close=10.0, market_cap=None)
    assert q.market_cap is None


def test_quote_is_frozen():
    """Quotes are immutable — providers shouldn't mutate after returning."""
    q = Quote(last_price=10.0, open=10.0, prev_close=10.0, market_cap=None)
    with pytest.raises((AttributeError, Exception)):
        q.last_price = 20.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

def test_fundamentals_default_fields_are_empty_strings():
    f = Fundamentals(name="Acme", market_cap=1e9, float_shares=1e8)
    assert f.sector   == ""
    assert f.industry == ""
    assert f.summary  == ""
    assert f.website  == ""


def test_fundamentals_market_cap_and_float_can_be_none():
    """Crypto, ADRs, and small-cap with no float disclosure → None fields."""
    f = Fundamentals(name="X", market_cap=None, float_shares=None)
    assert f.market_cap   is None
    assert f.float_shares is None


def test_fundamentals_is_frozen():
    f = Fundamentals(name="X", market_cap=1e9, float_shares=1e8)
    with pytest.raises((AttributeError, Exception)):
        f.name = "Y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# OHLCVBar
# ---------------------------------------------------------------------------

def test_ohlcv_bar_required_fields():
    bar = OHLCVBar(date=date(2024, 1, 3), open=100, high=105, low=99, close=103, volume=1_000_000)
    assert bar.close == 103
    assert bar.volume == 1_000_000


def test_ohlcv_bar_adj_close_defaults_to_none():
    bar = OHLCVBar(date=date(2024, 1, 3), open=100, high=105, low=99, close=103, volume=1_000)
    assert bar.adj_close is None


# ---------------------------------------------------------------------------
# OptionContract + OptionChain
# ---------------------------------------------------------------------------

def test_option_contract_required_fields():
    c = OptionContract(
        contract_symbol="AAPL250620C00150000",
        strike=150.0,
        last_price=5.0,
        bid=4.95,
        ask=5.05,
        volume=1500,
        open_interest=10000,
        implied_volatility=0.32,
        in_the_money=True,
        expiry="2025-06-20",
    )
    assert c.in_the_money is True
    assert c.expiry == "2025-06-20"


def test_option_chain_defaults_to_empty_lists():
    chain = OptionChain(expiry="2025-06-20")
    assert chain.calls == []
    assert chain.puts  == []


def test_option_chain_holds_contracts():
    c = OptionContract(
        contract_symbol="X", strike=100, last_price=1, bid=0.9, ask=1.1,
        volume=10, open_interest=100, implied_volatility=0.3,
        in_the_money=False, expiry="2025-06-20",
    )
    chain = OptionChain(expiry="2025-06-20", calls=[c], puts=[])
    assert len(chain.calls) == 1
    assert chain.calls[0].strike == 100


# ---------------------------------------------------------------------------
# Backward-compat re-exports from providers.base
# ---------------------------------------------------------------------------

def test_providers_base_reexports_quote_and_fundamentals():
    """Existing code does `from providers.base import Quote, Fundamentals` —
    after the Phase 5 refactor, this must still work."""
    from providers.base import Quote as Q2, Fundamentals as F2

    assert Q2 is Quote
    assert F2 is Fundamentals


def test_data_package_init_exports():
    """`from data import Quote` etc. should work via data/__init__.py."""
    from data import Quote as Q2, Fundamentals as F2, OHLCVBar as O2

    assert Q2 is Quote
    assert F2 is Fundamentals
    assert O2 is OHLCVBar
