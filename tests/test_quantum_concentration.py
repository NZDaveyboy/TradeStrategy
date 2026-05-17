"""
tests/test_concentration.py — Concentration metrics + exclude-ticker logic.

Covers:
  - HHI bounds: 1/N ≤ HHI ≤ 1
  - top-1 ≤ top-3 ≤ top-5 ≤ 100%
  - exclude_tickers actually removes the named ticker from the result
  - full_attribution returns one row per active constituent
  - diversification label thresholds
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.quantum.backtest import (
    concentration_metrics,
    full_attribution,
)
from core.quantum.index import IndexBuilder
from core.quantum.utils import Company, Universe


def _make_universe() -> Universe:
    return Universe(
        pure_play_quantum=[
            Company(ticker="PP1", company_name="PP1", quantum_exposure_score=5,
                    liquidity_score=2, profitability_score=1, risk_score=5, max_weight=0.5),
            Company(ticker="PP2", company_name="PP2", quantum_exposure_score=5,
                    liquidity_score=2, profitability_score=1, risk_score=5, max_weight=0.5),
            Company(ticker="PP3", company_name="PP3", quantum_exposure_score=5,
                    liquidity_score=2, profitability_score=1, risk_score=5, max_weight=0.5),
            Company(ticker="PP4", company_name="PP4", quantum_exposure_score=5,
                    liquidity_score=2, profitability_score=1, risk_score=5, max_weight=0.5),
        ],
        quantum_security_networking=[],
        quantum_enablers=[
            Company(ticker="EN1", company_name="EN1", quantum_exposure_score=3,
                    liquidity_score=5, profitability_score=5, risk_score=2, max_weight=0.15),
            Company(ticker="EN2", company_name="EN2", quantum_exposure_score=3,
                    liquidity_score=5, profitability_score=4, risk_score=2, max_weight=0.15),
        ],
        benchmarks=[],
    )


def _synthetic_prices(tickers: list[str], seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = 100
    dates = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=days)
    cols = {}
    for i, t in enumerate(tickers):
        drift = 0.0005 + 0.0001 * i
        vol = 0.02 + 0.005 * (i % 3)
        cols[t] = 100.0 * (1.0 + rng.normal(drift, vol, days)).cumprod()
    return pd.DataFrame(cols, index=dates)


# ---------------------------------------------------------------------------
# full_attribution
# ---------------------------------------------------------------------------

def test_full_attribution_has_row_per_ticker():
    universe = _make_universe()
    prices = _synthetic_prices(["PP1", "PP2", "PP3", "PP4"])
    b = IndexBuilder(universe, prices)
    result = b.build_pure_play(prices.index[0], prices.index[-1])
    df = full_attribution(result, prices)
    assert set(df["Ticker"]) == {"PP1", "PP2", "PP3", "PP4"}


def test_full_attribution_sorted_descending_by_contribution():
    universe = _make_universe()
    prices = _synthetic_prices(["PP1", "PP2", "PP3", "PP4"])
    b = IndexBuilder(universe, prices)
    result = b.build_pure_play(prices.index[0], prices.index[-1])
    df = full_attribution(result, prices)
    contribs = df["Contribution %"].tolist()
    assert contribs == sorted(contribs, reverse=True)


def test_full_attribution_share_sums_to_100():
    """Contribution share % across all tickers sums to ~100."""
    universe = _make_universe()
    prices = _synthetic_prices(["PP1", "PP2", "PP3", "PP4"])
    b = IndexBuilder(universe, prices)
    result = b.build_pure_play(prices.index[0], prices.index[-1])
    df = full_attribution(result, prices)
    assert df["Contribution share %"].sum() == pytest.approx(100.0, abs=0.5)


def test_full_attribution_includes_held_columns():
    """Held return %, Held days, Total return % must all be present."""
    universe = _make_universe()
    prices = _synthetic_prices(["PP1", "PP2", "PP3", "PP4"])
    b = IndexBuilder(universe, prices)
    result = b.build_pure_play(prices.index[0], prices.index[-1])
    df = full_attribution(result, prices)
    assert "Held return %"  in df.columns
    assert "Held days"      in df.columns
    assert "Total return %" in df.columns


def test_held_return_matches_held_price_range():
    """Held return % equals (last_held_price/first_held_price - 1) × 100."""
    universe = _make_universe()
    days = 60
    dates = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=days)
    cols = {
        "PP1": np.linspace(100.0, 150.0, days),  # +50%
        "PP2": np.linspace(100.0, 110.0, days),
        "PP3": np.linspace(100.0, 110.0, days),
        "PP4": np.linspace(100.0, 110.0, days),
    }
    prices = pd.DataFrame(cols, index=dates)
    b = IndexBuilder(universe, prices)
    result = b.build_pure_play(prices.index[0], prices.index[-1])
    df = full_attribution(result, prices)
    pp1 = df[df["Ticker"] == "PP1"].iloc[0]
    # PP1 was held every day, so Held return = Total return = +50%
    assert pp1["Held return %"]  == pytest.approx(50.0, abs=0.5)
    assert pp1["Total return %"] == pytest.approx(50.0, abs=0.5)
    assert pp1["Held days"] == days


# ---------------------------------------------------------------------------
# concentration_metrics
# ---------------------------------------------------------------------------

def test_concentration_top1_le_top3_le_top5():
    """top1 ≤ top3 ≤ top5 (monotone in N)."""
    universe = _make_universe()
    prices = _synthetic_prices(["PP1", "PP2", "PP3", "PP4"])
    b = IndexBuilder(universe, prices)
    result = b.build_pure_play(prices.index[0], prices.index[-1])
    cm = concentration_metrics(result, prices)
    assert cm["top1_share_pct"] <= cm["top3_share_pct"] + 1e-6
    assert cm["top3_share_pct"] <= cm["top5_share_pct"] + 1e-6
    assert cm["top5_share_pct"] <= 100.0 + 1e-6


def test_concentration_hhi_bounded():
    """HHI is between 1/N and 1."""
    universe = _make_universe()
    prices = _synthetic_prices(["PP1", "PP2", "PP3", "PP4"])
    b = IndexBuilder(universe, prices)
    result = b.build_pure_play(prices.index[0], prices.index[-1])
    cm = concentration_metrics(result, prices)
    n = cm["n_constituents"]
    assert n > 0
    # 1/N is the theoretical minimum for perfectly equal shares.
    # Realistic backtest will be a bit above.
    assert (1.0 / n) - 1e-6 <= cm["hhi"] <= 1.0 + 1e-6


def test_concentration_label_buckets():
    """HHI label maps correctly to thresholds."""
    universe = _make_universe()
    prices = _synthetic_prices(["PP1", "PP2", "PP3", "PP4"])
    b = IndexBuilder(universe, prices)
    result = b.build_pure_play(prices.index[0], prices.index[-1])
    cm = concentration_metrics(result, prices)
    if cm["hhi"] < 0.15:
        assert cm["diversification_label"] == "Diversified"
    elif cm["hhi"] < 0.25:
        assert cm["diversification_label"] == "Moderately concentrated"
    else:
        assert cm["diversification_label"] == "Highly concentrated"


def test_concentration_empty_result_safe():
    """concentration_metrics on an empty result returns sensible zeros."""
    from core.quantum.index import IndexResult
    empty = IndexResult(
        name="X", levels=pd.Series(dtype=float), weights=pd.DataFrame(),
        constituents=[], rebalance_dates=[],
    )
    cm = concentration_metrics(empty, pd.DataFrame())
    assert cm["n_constituents"] == 0
    assert cm["hhi"] == 0.0


# ---------------------------------------------------------------------------
# exclude_tickers
# ---------------------------------------------------------------------------

def test_exclude_tickers_removes_from_pure_play():
    universe = _make_universe()
    prices = _synthetic_prices(["PP1", "PP2", "PP3", "PP4"])
    b = IndexBuilder(universe, prices)
    result = b.build_pure_play(
        prices.index[0], prices.index[-1],
        exclude_tickers={"PP1"},
    )
    # PP1 should not appear in the weights dataframe at all
    if "PP1" in result.weights.columns:
        assert (result.weights["PP1"] == 0).all()


def test_exclude_tickers_removes_from_ecosystem():
    universe = _make_universe()
    prices = _synthetic_prices(["PP1", "PP2", "PP3", "PP4", "EN1", "EN2"])
    b = IndexBuilder(universe, prices)
    result = b.build_ecosystem(
        prices.index[0], prices.index[-1],
        exclude_tickers={"PP1", "EN1"},
    )
    for ticker in ("PP1", "EN1"):
        if ticker in result.weights.columns:
            assert (result.weights[ticker] == 0).all()


def test_exclude_tickers_removes_from_barbell():
    universe = _make_universe()
    prices = _synthetic_prices(["PP1", "PP2", "PP3", "PP4", "EN1", "EN2"])
    b = IndexBuilder(universe, prices)
    result = b.build_barbell(
        prices.index[0], prices.index[-1],
        exclude_tickers={"PP2"},
    )
    if "PP2" in result.weights.columns:
        assert (result.weights["PP2"] == 0).all()


def test_exclude_tickers_changes_total_return():
    """Excluding the historical top performer should reduce the index return."""
    universe = _make_universe()
    # Manufacture a panel where PP1 is dramatically the best performer
    days = 100
    dates = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=days)
    cols = {
        "PP1": np.linspace(100, 300, days),   # +200%
        "PP2": np.linspace(100, 110, days),   # +10%
        "PP3": np.linspace(100, 110, days),
        "PP4": np.linspace(100, 110, days),
    }
    prices = pd.DataFrame(cols, index=dates)

    b = IndexBuilder(universe, prices)
    full = b.build_pure_play(prices.index[0], prices.index[-1])
    ex1  = b.build_pure_play(prices.index[0], prices.index[-1], exclude_tickers={"PP1"})

    assert full.levels.iloc[-1] > ex1.levels.iloc[-1]
    assert full.total_return_pct() > ex1.total_return_pct()
