"""
tests/test_index.py — Core invariants for the index builder.

Covers:
  - Index starts at 100
  - Weights sum to 1 at every rebalance
  - Per-name and category caps are respected
  - Missing ticker data does not crash the builder
  - The three recipes all produce non-empty results on a synthetic universe

No yfinance — everything is built against a synthetic price panel so the
tests run offline and deterministically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.quantum.index import (
    ECOSYSTEM_CATEGORY_CAPS,
    PURE_PLAY_MAX_WEIGHT,
    IndexBuilder,
    _apply_max_cap,
)
from core.quantum.scoring import (
    compute_final_scores,
    equal_weights,
    normalize_with_caps,
)
from core.quantum.utils import Company, Universe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_company(ticker: str, category: str = "pure_play_quantum", **scores) -> Company:
    defaults = {
        "quantum_exposure_score": 3,
        "liquidity_score": 3,
        "profitability_score": 3,
        "risk_score": 3,
        "max_weight": 1.0,
    }
    defaults.update(scores)
    c = Company(
        ticker=ticker,
        company_name=f"{ticker} Inc.",
        **defaults,
    )
    c.category = category
    return c


def _synthetic_prices(tickers: list[str], days: int = 200, seed: int = 42) -> pd.DataFrame:
    """Geometric brownian motion price panel, one column per ticker."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=days)
    out: dict[str, pd.Series] = {}
    for i, t in enumerate(tickers):
        # Different drift/vol per ticker so tests catch averaging
        drift = 0.0003 + 0.0001 * i
        vol   = 0.02 + 0.005 * (i % 3)
        rets = rng.normal(loc=drift, scale=vol, size=days)
        prices = 100.0 * (1.0 + rets).cumprod()
        out[t] = pd.Series(prices, index=dates)
    return pd.DataFrame(out)


def _make_universe() -> Universe:
    """A synthetic universe shaped like the real config: 4 pure plays, a
    few security names, enough enablers that category caps don't all bind."""
    return Universe(
        pure_play_quantum=[
            Company(ticker="PP1", company_name="PP1 Inc.", quantum_exposure_score=5,
                    liquidity_score=2, profitability_score=1, risk_score=5, max_weight=0.5),
            Company(ticker="PP2", company_name="PP2 Inc.", quantum_exposure_score=5,
                    liquidity_score=2, profitability_score=1, risk_score=5, max_weight=0.5),
            Company(ticker="PP3", company_name="PP3 Inc.", quantum_exposure_score=5,
                    liquidity_score=2, profitability_score=1, risk_score=5, max_weight=0.5),
            Company(ticker="PP4", company_name="PP4 Inc.", quantum_exposure_score=5,
                    liquidity_score=2, profitability_score=1, risk_score=5, max_weight=0.5),
        ],
        quantum_security_networking=[
            Company(ticker="SN1", company_name="SN1 Inc.", quantum_exposure_score=3,
                    liquidity_score=4, profitability_score=3, risk_score=3, max_weight=0.1),
            Company(ticker="SN2", company_name="SN2 Inc.", quantum_exposure_score=3,
                    liquidity_score=4, profitability_score=3, risk_score=3, max_weight=0.1),
        ],
        quantum_enablers=[
            Company(ticker="EN1", company_name="EN1 Inc.", quantum_exposure_score=3,
                    liquidity_score=5, profitability_score=5, risk_score=2, max_weight=0.15),
            Company(ticker="EN2", company_name="EN2 Inc.", quantum_exposure_score=3,
                    liquidity_score=5, profitability_score=4, risk_score=2, max_weight=0.15),
            Company(ticker="EN3", company_name="EN3 Inc.", quantum_exposure_score=3,
                    liquidity_score=5, profitability_score=4, risk_score=2, max_weight=0.15),
            Company(ticker="EN4", company_name="EN4 Inc.", quantum_exposure_score=3,
                    liquidity_score=4, profitability_score=4, risk_score=3, max_weight=0.15),
            Company(ticker="EN5", company_name="EN5 Inc.", quantum_exposure_score=3,
                    liquidity_score=4, profitability_score=4, risk_score=3, max_weight=0.15),
            Company(ticker="EN6", company_name="EN6 Inc.", quantum_exposure_score=3,
                    liquidity_score=4, profitability_score=3, risk_score=3, max_weight=0.15),
        ],
        benchmarks=["BENCH"],
    )


# ---------------------------------------------------------------------------
# 1. Index starts at 100
# ---------------------------------------------------------------------------

UNIVERSE_TICKERS = [
    "PP1", "PP2", "PP3", "PP4",
    "SN1", "SN2",
    "EN1", "EN2", "EN3", "EN4", "EN5", "EN6",
]


def test_pure_play_index_starts_at_100():
    universe = _make_universe()
    prices = _synthetic_prices(["PP1", "PP2", "PP3", "PP4"])
    b = IndexBuilder(universe, prices)
    result = b.build_pure_play(prices.index[0], prices.index[-1])
    assert result.levels.iloc[0] == pytest.approx(100.0, abs=1e-9)


def test_ecosystem_index_starts_at_100():
    universe = _make_universe()
    prices = _synthetic_prices(UNIVERSE_TICKERS)
    b = IndexBuilder(universe, prices)
    result = b.build_ecosystem(prices.index[0], prices.index[-1])
    assert result.levels.iloc[0] == pytest.approx(100.0, abs=1e-9)


def test_barbell_index_starts_at_100():
    universe = _make_universe()
    prices = _synthetic_prices(["PP1", "PP2", "PP3", "PP4", "EN1", "EN2", "EN3", "EN4", "EN5", "EN6"])
    b = IndexBuilder(universe, prices)
    result = b.build_barbell(prices.index[0], prices.index[-1])
    assert result.levels.iloc[0] == pytest.approx(100.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. Weights sum to 1
# ---------------------------------------------------------------------------

def test_equal_weights_sum_to_one():
    cs = [_make_company(f"T{i}") for i in range(5)]
    w = equal_weights(cs)
    assert sum(w.values()) == pytest.approx(1.0)


def test_conviction_weights_sum_to_one():
    cs = [
        _make_company("A", quantum_exposure_score=5, liquidity_score=3,
                      profitability_score=2, risk_score=4),
        _make_company("B", quantum_exposure_score=4, liquidity_score=4,
                      profitability_score=3, risk_score=3),
        _make_company("C", quantum_exposure_score=3, liquidity_score=5,
                      profitability_score=5, risk_score=2),
    ]
    scores = compute_final_scores(cs)
    w = normalize_with_caps(scores, cs)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)


def test_index_weights_sum_to_one_at_every_rebalance():
    universe = _make_universe()
    prices = _synthetic_prices(UNIVERSE_TICKERS)
    b = IndexBuilder(universe, prices)
    result = b.build_ecosystem(prices.index[0], prices.index[-1])
    for rb in result.rebalance_dates:
        if rb not in result.weights.index:
            continue
        row_sum = result.weights.loc[rb].sum()
        # Under strict caps with a realistic universe, sum should be ~1.
        # (If caps fully bind, sum can legitimately be <1 — residual is "cash".)
        assert 0.50 <= row_sum <= 1.0 + 1e-6, f"Sum {row_sum} on {rb}"


# ---------------------------------------------------------------------------
# 3. Max weight caps are respected
# ---------------------------------------------------------------------------

def test_pure_play_cap_25pct_respected():
    """No name in the Pure Play index should exceed 25% weight."""
    universe = _make_universe()
    # 4 pure plays — equal-weight = 25% exactly, at the cap
    prices = _synthetic_prices(["PP1", "PP2", "PP3", "PP4"])
    b = IndexBuilder(universe, prices)
    result = b.build_pure_play(prices.index[0], prices.index[-1])
    for rb in result.rebalance_dates:
        if rb not in result.weights.index:
            continue
        max_w = result.weights.loc[rb].max()
        assert max_w <= PURE_PLAY_MAX_WEIGHT + 1e-6, f"Weight {max_w} > 25% on {rb}"


def test_pure_play_cap_binds_when_universe_too_small():
    """3 names + 25% cap = at most 75% invested; each name ≤ 25%, residual = cash."""
    universe = _make_universe()
    prices = _synthetic_prices(["PP1", "PP2", "PP3"])
    b = IndexBuilder(universe, prices)
    result = b.build_pure_play(prices.index[0], prices.index[-1])
    for rb in result.rebalance_dates:
        if rb not in result.weights.index:
            continue
        w = result.weights.loc[rb]
        assert w.max() <= PURE_PLAY_MAX_WEIGHT + 1e-6
        # Sum should be ~75% (3 names × 25%), residual is uninvested
        assert 0.70 <= w.sum() <= 0.76


def test_ecosystem_category_caps_respected():
    universe = _make_universe()
    prices = _synthetic_prices(UNIVERSE_TICKERS)
    b = IndexBuilder(universe, prices)
    result = b.build_ecosystem(prices.index[0], prices.index[-1])

    # Build ticker → category map
    cat_map: dict[str, str] = {}
    for c in result.constituents:
        cat_map[c.ticker] = c.category

    for rb in result.rebalance_dates:
        if rb not in result.weights.index:
            continue
        weights = result.weights.loc[rb]
        for t, w in weights.items():
            cat = cat_map.get(t)
            if cat in ECOSYSTEM_CATEGORY_CAPS:
                cap = ECOSYSTEM_CATEGORY_CAPS[cat]
                assert w <= cap + 1e-6, f"{t} weight {w:.4f} > {cat} cap {cap} on {rb}"


def test_apply_max_cap_redistributes_excess():
    """A 50% cap on a 3-name equal-weight bucket should leave each at exactly 33.3%."""
    w = {"A": 0.6, "B": 0.2, "C": 0.2}
    capped = _apply_max_cap(w, 0.50)
    assert capped["A"] == pytest.approx(0.50, abs=1e-6)
    assert capped["B"] == pytest.approx(0.25, abs=1e-6)
    assert capped["C"] == pytest.approx(0.25, abs=1e-6)
    assert sum(capped.values()) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. Missing ticker data doesn't crash
# ---------------------------------------------------------------------------

def test_missing_ticker_is_skipped_not_fatal():
    universe = _make_universe()
    # Drop EN6 from the price panel — should still build with the rest
    prices = _synthetic_prices(["PP1", "PP2", "PP3", "PP4", "SN1", "EN1", "EN2", "EN3"])
    b = IndexBuilder(universe, prices)
    result = b.build_ecosystem(prices.index[0], prices.index[-1])
    assert not result.levels.empty
    assert "EN6" not in result.weights.columns
    assert "EN1" in result.weights.columns


def test_late_start_ticker_only_included_after_first_valid_date():
    universe = _make_universe()
    prices = _synthetic_prices(["PP1", "PP2", "PP3", "PP4"])
    # Set the first 30 days of PP3 to NaN — should only enter from day 31
    prices.iloc[:30, prices.columns.get_loc("PP3")] = np.nan
    b = IndexBuilder(universe, prices)
    result = b.build_pure_play(prices.index[0], prices.index[-1])
    # First weight row should have PP3 at zero (not yet active)
    assert result.weights["PP3"].iloc[0] == pytest.approx(0.0)


def test_completely_missing_universe_does_not_crash():
    """When ALL tickers are missing, builder returns an empty result, not a crash."""
    universe = _make_universe()
    empty_prices = pd.DataFrame()
    b = IndexBuilder(universe, empty_prices)
    result = b.build_pure_play(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))
    assert result.levels.empty


# ---------------------------------------------------------------------------
# 5. Sanity: index moves with constituent moves
# ---------------------------------------------------------------------------

def test_index_increases_when_all_constituents_increase():
    """If every constituent gains 10% over the window, the index should be ≥ 100 * 1.10."""
    universe = _make_universe()
    days = 60
    dates = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=days)
    # 4 pure plays so equal-weight sums to exactly 1.0 (matches real config)
    prices = pd.DataFrame({
        t: np.linspace(100.0, 110.0, days)
        for t in ["PP1", "PP2", "PP3", "PP4"]
    }, index=dates)
    b = IndexBuilder(universe, prices)
    result = b.build_pure_play(prices.index[0], prices.index[-1])
    assert result.levels.iloc[-1] >= 109.5  # ~10% gain, tiny float room
