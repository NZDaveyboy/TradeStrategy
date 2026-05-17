"""
tests/test_quantum_signals.py — BUY / WATCH / HOLD / SELL classifier.

Covers the explicit rules in core.quantum.backtest._classify_one, plus
the end-to-end classify_constituents that wraps it with held-period
return and RSI computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.quantum.backtest import (
    SIGNAL_BUY, SIGNAL_WATCH, SIGNAL_HOLD, SIGNAL_SELL,
    _classify_one,
    _rsi_14,
    classify_constituents,
)
from core.quantum.index import IndexBuilder
from core.quantum.utils import Company, Universe


# ---------------------------------------------------------------------------
# _classify_one — rule-level
# ---------------------------------------------------------------------------

def test_sell_when_capital_impairment_no_recovery():
    """held_return ≤ -50% and 1m < +5% → SELL."""
    s, _ = _classify_one(score=2.0, held_return=-60.0, ret_1m=2.0, ret_3m=-30.0, rsi=45.0)
    assert s == SIGNAL_SELL


def test_sell_when_underwater_and_falling():
    """held_return < -25% AND 1m < -5% → SELL."""
    s, _ = _classify_one(score=1.5, held_return=-30.0, ret_1m=-10.0, ret_3m=-20.0, rsi=40.0)
    assert s == SIGNAL_SELL


def test_no_sell_when_underwater_but_recovering():
    """held_return = -30% but 1m = +15% (recovering) → NOT SELL."""
    s, _ = _classify_one(score=1.5, held_return=-30.0, ret_1m=15.0, ret_3m=20.0, rsi=60.0)
    assert s != SIGNAL_SELL


def test_buy_when_strong_momentum():
    """1m > +15% AND RSI < 80 → BUY."""
    s, _ = _classify_one(score=1.5, held_return=0.0, ret_1m=20.0, ret_3m=30.0, rsi=70.0)
    assert s == SIGNAL_BUY


def test_buy_when_high_conviction_with_momentum():
    """conviction ≥ 2.0 + 1m > +3% + RSI < 75 → BUY."""
    s, _ = _classify_one(score=2.5, held_return=10.0, ret_1m=5.0, ret_3m=12.0, rsi=55.0)
    assert s == SIGNAL_BUY


def test_buy_when_sustained_3m_momentum():
    """3m > +25%, 1m ≥ 0, RSI < 80 → BUY (even with modest 1m)."""
    s, _ = _classify_one(score=1.5, held_return=20.0, ret_1m=1.0, ret_3m=30.0, rsi=68.0)
    assert s == SIGNAL_BUY


def test_watch_when_overbought():
    """RSI > 75 → WATCH (no BUY signal fired earlier)."""
    s, _ = _classify_one(score=1.5, held_return=5.0, ret_1m=2.0, ret_3m=8.0, rsi=82.0)
    assert s == SIGNAL_WATCH


def test_watch_when_top_conviction_no_momentum():
    """conviction ≥ 2.5 AND |1m| < 5% → WATCH."""
    s, _ = _classify_one(score=2.8, held_return=0.0, ret_1m=1.0, ret_3m=2.0, rsi=50.0)
    assert s == SIGNAL_WATCH


def test_hold_when_nothing_else_triggers():
    """Modest score, modest moves → HOLD."""
    s, _ = _classify_one(score=1.5, held_return=5.0, ret_1m=2.0, ret_3m=5.0, rsi=55.0)
    assert s == SIGNAL_HOLD


def test_classify_returns_4_known_signals():
    """Output is always one of the 4 documented signals."""
    test_inputs = [
        (1.0, -80, -10, -50, 30),
        (2.0,   0,   2,   0, 50),
        (3.0,  50,  20,  60, 70),
        (1.5,  10,   1,   5, 85),
    ]
    for params in test_inputs:
        s, _ = _classify_one(
            score=params[0], held_return=params[1], ret_1m=params[2],
            ret_3m=params[3], rsi=params[4],
        )
        assert s in (SIGNAL_BUY, SIGNAL_WATCH, SIGNAL_HOLD, SIGNAL_SELL)


# ---------------------------------------------------------------------------
# _rsi_14 helper
# ---------------------------------------------------------------------------

def test_rsi_falls_when_prices_consistently_decline():
    """A monotonically decreasing price series → RSI well below 50."""
    s = pd.Series(np.linspace(100, 50, 30))
    assert _rsi_14(s) < 30


def test_rsi_rises_when_prices_consistently_increase():
    s = pd.Series(np.linspace(50, 100, 30))
    assert _rsi_14(s) > 70


def test_rsi_neutral_on_flat_series():
    """Constant prices → RSI 50 (default for divide-by-zero gain/loss)."""
    s = pd.Series([100.0] * 30)
    assert _rsi_14(s) == 50.0


def test_rsi_returns_50_on_insufficient_data():
    assert _rsi_14(pd.Series([100.0, 101.0])) == 50.0
    assert _rsi_14(None) == 50.0


# ---------------------------------------------------------------------------
# classify_constituents — integration
# ---------------------------------------------------------------------------

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
        quantum_enablers=[],
        benchmarks=[],
    )


def test_classify_constituents_returns_expected_columns():
    universe = _make_universe()
    days = 80
    dates = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=days)
    prices = pd.DataFrame({
        t: np.linspace(100.0, 110.0, days) for t in ["PP1", "PP2", "PP3", "PP4"]
    }, index=dates)
    b = IndexBuilder(universe, prices)
    result = b.build_pure_play(prices.index[0], prices.index[-1])
    df = classify_constituents(result, prices, universe)
    expected = {
        "Ticker", "Company", "Category", "Signal", "Reason",
        "Conviction", "Weight %", "Held return %", "1m %", "3m %", "RSI(14)",
    }
    assert expected.issubset(df.columns)


def test_classify_constituents_signal_values_are_valid():
    universe = _make_universe()
    days = 80
    dates = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=days)
    prices = pd.DataFrame({
        t: np.linspace(100.0, 110.0, days) for t in ["PP1", "PP2", "PP3", "PP4"]
    }, index=dates)
    b = IndexBuilder(universe, prices)
    result = b.build_pure_play(prices.index[0], prices.index[-1])
    df = classify_constituents(result, prices, universe)
    valid_signals = {SIGNAL_BUY, SIGNAL_WATCH, SIGNAL_HOLD, SIGNAL_SELL}
    assert set(df["Signal"].unique()).issubset(valid_signals)


def test_classify_constituents_handles_empty_result():
    """Empty index result → empty DataFrame with the right columns."""
    from core.quantum.index import IndexResult
    empty = IndexResult(
        name="X", levels=pd.Series(dtype=float), weights=pd.DataFrame(),
        constituents=[], rebalance_dates=[],
    )
    df = classify_constituents(empty, pd.DataFrame(), _make_universe())
    assert df.empty
    assert "Signal" in df.columns


def test_classify_constituents_strong_uptrend_yields_buy():
    """A noisy +40% uptrend over 80 days should trigger BUY (not WATCH —
    a strictly monotone uptrend would push RSI to 100 and trigger
    overbought-WATCH instead; real uptrends have pullbacks)."""
    universe = _make_universe()
    days = 80
    dates = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=days)
    rng = np.random.default_rng(7)
    base = np.linspace(100.0, 140.0, days)
    prices = pd.DataFrame({
        t: base + rng.normal(0, 1.5, days)
        for t in ["PP1", "PP2", "PP3", "PP4"]
    }, index=dates)
    b = IndexBuilder(universe, prices)
    result = b.build_pure_play(prices.index[0], prices.index[-1])
    df = classify_constituents(result, prices, universe)
    # All four pure plays should be BUY (sustained 3m momentum + RSI < 80)
    assert (df["Signal"] == SIGNAL_BUY).all()


def test_classify_constituents_perfect_uptrend_yields_watch_overbought():
    """A perfectly monotone uptrend gives RSI = 100 → WATCH (overbought).

    This is correct behaviour — real-world uptrends have pullbacks; a no-pullback
    rise IS the textbook overbought condition. Documents the rule explicitly.
    """
    universe = _make_universe()
    days = 80
    dates = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=days)
    prices = pd.DataFrame({
        t: np.linspace(100.0, 140.0, days) for t in ["PP1", "PP2", "PP3", "PP4"]
    }, index=dates)
    b = IndexBuilder(universe, prices)
    result = b.build_pure_play(prices.index[0], prices.index[-1])
    df = classify_constituents(result, prices, universe)
    assert (df["Signal"] == SIGNAL_WATCH).all()


def test_classify_constituents_severe_drawdown_yields_sell():
    """A -70% price collapse with no recovery should trigger SELL."""
    universe = _make_universe()
    days = 80
    dates = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=days)
    prices = pd.DataFrame({
        t: np.linspace(100.0, 30.0, days) for t in ["PP1", "PP2", "PP3", "PP4"]
    }, index=dates)
    b = IndexBuilder(universe, prices)
    result = b.build_pure_play(prices.index[0], prices.index[-1])
    df = classify_constituents(result, prices, universe)
    assert (df["Signal"] == SIGNAL_SELL).all()
