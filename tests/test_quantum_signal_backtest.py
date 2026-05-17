"""
tests/test_quantum_signal_backtest.py — Walk-forward signal backtest validation.

The MOST important test in this file is `test_no_lookahead_in_classifier`:
the classifier at sample date `t` MUST only see prices ≤ `t`. Without this
guarantee the backtest is fiction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.quantum.signal_backtest import backtest_signals, DEFAULT_LOOKFORWARDS
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


def _noisy_uptrend(days: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=days)
    cols = {}
    for i, t in enumerate(["PP1", "PP2", "PP3", "PP4", "EN1", "EN2"]):
        drift = 0.0006 + 0.0001 * i
        vol   = 0.022 + 0.005 * (i % 3)
        rets = rng.normal(drift, vol, days)
        cols[t] = 100.0 * (1.0 + rets).cumprod()
    return pd.DataFrame(cols, index=dates)


def test_backtest_returns_expected_keys():
    uni = _make_universe()
    prices = _noisy_uptrend()
    out = backtest_signals(uni, prices)
    assert set(out.keys()) == {"summary", "signal_log", "portfolio", "sample_dates"}


def test_backtest_summary_has_required_columns():
    uni = _make_universe()
    prices = _noisy_uptrend()
    out = backtest_signals(uni, prices)
    required = {
        "Signal", "Lookforward (d)", "N samples",
        "Hit rate %", "vs Index hit %",
        "Mean fwd %", "Mean vs Index %", "Std fwd %",
    }
    assert required.issubset(out["summary"].columns)


def test_backtest_signal_log_has_lookforward_columns():
    uni = _make_universe()
    prices = _noisy_uptrend()
    out = backtest_signals(uni, prices)
    log = out["signal_log"]
    for lf in DEFAULT_LOOKFORWARDS:
        assert f"fwd_{lf}d" in log.columns


def test_backtest_handles_empty_universe():
    uni = Universe(
        pure_play_quantum=[], quantum_security_networking=[],
        quantum_enablers=[], benchmarks=[],
    )
    out = backtest_signals(uni, pd.DataFrame())
    assert out["summary"].empty
    assert out["signal_log"].empty


def test_backtest_handles_too_short_prices():
    uni = _make_universe()
    days = 20
    dates = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=days)
    prices = pd.DataFrame({t: np.linspace(100, 110, days) for t in
                           ["PP1", "PP2", "PP3", "PP4", "EN1", "EN2"]}, index=dates)
    out = backtest_signals(uni, prices)
    assert out["signal_log"].empty


def test_no_lookahead_classifier_only_sees_past_prices(monkeypatch):
    """At each sample date t, classify_constituents MUST only see prices ≤ t."""
    from core.quantum import signal_backtest as sb

    captured: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def wrapped_classify(result, prices, universe):
        if not prices.empty and not result.levels.empty:
            captured.append((prices.index.max(), result.levels.index[-1]))
        return pd.DataFrame(columns=[
            "Ticker", "Company", "Category", "Signal", "Reason",
            "Conviction", "Weight %", "Held return %", "1m %", "3m %", "RSI(14)",
        ])

    monkeypatch.setattr(sb, "classify_constituents", wrapped_classify)

    uni = _make_universe()
    prices = _noisy_uptrend()
    backtest_signals(uni, prices)

    assert captured, "classify_constituents was never invoked"
    for max_seen, sample_t in captured:
        assert max_seen <= sample_t, (
            f"LOOK-AHEAD BIAS: classifier saw prices up to {max_seen} "
            f"but should only have seen up to {sample_t}"
        )


def test_forward_returns_use_future_prices():
    uni = _make_universe()
    days = 500
    dates = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=days)
    base = (1.0 + 0.002) ** np.arange(days)
    cols = {t: 100.0 * base for t in ["PP1", "PP2", "PP3", "PP4", "EN1", "EN2"]}
    prices = pd.DataFrame(cols, index=dates)
    out = backtest_signals(uni, prices)
    log = out["signal_log"]
    if log.empty:
        pytest.skip("No samples produced — universe too small for window")
    r = log.iloc[0]
    t      = r["Date"]
    ticker = r["Ticker"]
    p_t    = prices.at[t, ticker]
    t_loc  = prices.index.get_loc(t)
    if t_loc + 30 < len(prices):
        p_fwd = prices.iloc[t_loc + 30][ticker]
        expected = (p_fwd / p_t - 1.0) * 100.0
        assert r["fwd_30d"] == pytest.approx(expected, abs=0.01)


def test_monotone_uptrend_buys_have_high_hit_rate():
    uni = _make_universe()
    days = 500
    dates = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=days)
    rng = np.random.default_rng(0)
    base = np.linspace(100, 250, days)
    cols = {t: base + rng.normal(0, 1.5, days)
            for t in ["PP1", "PP2", "PP3", "PP4", "EN1", "EN2"]}
    prices = pd.DataFrame(cols, index=dates)
    out = backtest_signals(uni, prices)
    summary = out["summary"]
    if summary.empty:
        pytest.skip("No samples produced")
    buys = summary[summary["Signal"] == "BUY"]
    if buys.empty:
        pytest.skip("No BUY signals fired")
    assert buys["Hit rate %"].mean() >= 80.0


def test_summary_lookforward_values_are_in_default_set():
    uni = _make_universe()
    prices = _noisy_uptrend()
    out = backtest_signals(uni, prices)
    summary = out["summary"]
    if summary.empty:
        pytest.skip("No samples produced")
    assert set(summary["Lookforward (d)"].unique()).issubset(set(DEFAULT_LOOKFORWARDS))


def test_portfolio_curve_rises_in_full_uptrend():
    uni = _make_universe()
    days = 500
    dates = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=days)
    rng = np.random.default_rng(1)
    base = np.linspace(100, 250, days)
    cols = {t: base + rng.normal(0, 1.5, days)
            for t in ["PP1", "PP2", "PP3", "PP4", "EN1", "EN2"]}
    prices = pd.DataFrame(cols, index=dates)
    out = backtest_signals(uni, prices)
    port = out["portfolio"]
    if not port:
        pytest.skip("No portfolio produced")
    if 30 in port and not port[30].empty:
        assert port[30].iloc[-1] > port[30].iloc[0]
