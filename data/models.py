"""
data/models.py — Typed schemas exchanged between providers and the rest of the app.

These dataclasses are the **contract** for what data shapes flow through the
system. Providers (yfinance, Finviz, future paid feeds) implement methods
that return these types so business logic never depends on any specific
data source's quirks.

Two tiers of typing:

  - **Strict dataclasses** (Quote, Fundamentals) — used as actual return
    values. Concrete types, immediate enforcement.
  - **Documentation schemas** (OHLCVBar, OptionContract) — describe the
    shape of fields inside pandas DataFrames returned by providers
    (`get_ohlcv` returns a DataFrame, but it should have OHLCVBar's
    columns). Acts as a single source of truth for what columns/types
    downstream code can expect. Future Phase 5 work may convert DataFrame
    returns to typed records, at which point these become strict.

Add new schemas here — never inline them in provider classes — so the
contract stays inspectable and testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


# ---------------------------------------------------------------------------
# Strict dataclasses (currently enforced)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Quote:
    """Live quote from a fast-info endpoint — low latency (~50ms).

    Returned by MarketDataProvider.get_quote(). Always non-raising — on
    network failure the provider returns Quote with 0.0 / None values
    so the caller can detect "no data" without a try/except.
    """
    last_price: float
    open:       float
    prev_close: float
    market_cap: float | None


@dataclass(frozen=True)
class Fundamentals:
    """Full company info from the slow-info endpoint (~500ms).

    Only fetched when name, sector, float, or summary are actually needed.
    Returned by MarketDataProvider.get_fundamentals().
    """
    name:         str
    market_cap:   float | None
    float_shares: float | None
    sector:       str = ""
    industry:     str = ""
    summary:      str = ""
    website:      str = ""


# ---------------------------------------------------------------------------
# Documentation schemas — describe DataFrame column contracts
# (not enforced at runtime yet; consumers still receive DataFrames)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OHLCVBar:
    """One row in the DataFrame returned by get_ohlcv / get_ohlcv_range.

    The DataFrame is indexed by date (DatetimeIndex). Each row has:

      - open / high / low / close  — float, USD
      - volume                     — float or int, shares traded
      - adj_close (optional)       — split / dividend adjusted close

    A future Phase 5 refinement may have providers return list[OHLCVBar]
    instead of DataFrame for stricter type-safety. For now the schema
    serves as the column contract.
    """
    date:       date
    open:       float
    high:       float
    low:        float
    close:      float
    volume:     float
    adj_close:  float | None = None


@dataclass(frozen=True)
class OptionContract:
    """One option contract from get_option_chain (calls/puts DataFrames).

    Columns in the DataFrame should match these field names (lowercased,
    but yfinance returns mixed case — current providers preserve that).

    Forward-looking: when this becomes the enforced return type, the
    column-mapping logic moves into yfinance_provider.
    """
    contract_symbol:      str
    strike:               float
    last_price:           float
    bid:                  float
    ask:                  float
    volume:               float | None
    open_interest:        float | None
    implied_volatility:   float | None
    in_the_money:         bool
    expiry:               str        # ISO date string YYYY-MM-DD


@dataclass(frozen=True)
class OptionChain:
    """Full option chain for one expiry — pair of calls + puts.

    Currently providers return a tuple[DataFrame, DataFrame] for backward
    compat; this dataclass documents the future strict form and lets test
    code construct chains without touching DataFrame mechanics.
    """
    expiry: str  # ISO date string YYYY-MM-DD
    calls:  list[OptionContract] = field(default_factory=list)
    puts:   list[OptionContract] = field(default_factory=list)
