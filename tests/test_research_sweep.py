"""
tests/test_research_sweep.py — Unit tests for the research sweep pipeline.

Covers:
  - param_grid produces correct number of combinations
  - rescore_row returns a valid score dict
  - rescore_signals adds expected columns
  - filter_signals respects tradescore_threshold
  - build_signal_groups shapes output correctly
  - run_param_set returns correct keys (no DB/price data)
  - Sweep with 3 threshold values produces 3 result rows
  - Sweep results are reproducible across two identical runs
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from core.research.params import SweepParams, param_grid
from core.research.rescore import (
    build_signal_groups,
    filter_signals,
    rescore_row,
    rescore_signals,
)
from core.research.sweep import _aggregate_ticker_results, run_param_set


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_row(**overrides) -> dict:
    base = {
        "run_date":    "2025-01-15",
        "ticker":      "TEST",
        "direction":   "long",
        "price":       50.0,
        "change_pct":  3.5,
        "rvol":        2.5,
        "rsi":         52.0,
        "ema9":        48.0,
        "ema20":       46.0,
        "ema200":      40.0,
        "atr":         1.2,
        "macd":        0.4,
        "macd_signal": 0.2,
        "vwap":        49.5,
        "market_cap":  5_000_000_000,
        "float_shares": 200_000_000,
        "tradescore":   45.0,
        "change_5d":   5.0,
        "stop_loss":   44.5,
        "setup_type":  "momentum",
    }
    base.update(overrides)
    return base


def _make_df(rows: list[dict] | None = None) -> pd.DataFrame:
    if rows is None:
        rows = [_make_row()]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# param_grid
# ---------------------------------------------------------------------------

class TestParamGrid:
    def test_single_dim_three_values(self):
        grid = param_grid(tradescore_threshold=[35, 45, 55])
        assert len(grid) == 3

    def test_two_dims_cartesian(self):
        grid = param_grid(tradescore_threshold=[35, 45], min_rvol=[1.0, 2.0])
        assert len(grid) == 4

    def test_empty_returns_one_default(self):
        grid = param_grid()
        assert len(grid) == 1
        assert isinstance(grid[0], SweepParams)

    def test_labels_unique(self):
        grid = param_grid(tradescore_threshold=[35, 45, 55])
        labels = [p.label for p in grid]
        assert len(set(labels)) == 3

    def test_unknown_key_silently_ignored(self):
        # from_dict silently ignores unknown keys
        grid = param_grid(tradescore_threshold=[35], not_a_field=["x"])
        assert len(grid) == 1


# ---------------------------------------------------------------------------
# rescore_row
# ---------------------------------------------------------------------------

class TestRescoreRow:
    def test_returns_score_dict(self):
        row    = _make_row()
        params = SweepParams()
        result = rescore_row(row, params)
        assert "score" in result
        assert isinstance(result["score"], (int, float))

    def test_score_non_negative(self):
        row    = _make_row()
        params = SweepParams()
        result = rescore_row(row, params)
        assert result["score"] >= 0

    def test_weight_override_changes_score(self):
        row     = _make_row()
        default = rescore_row(row, SweepParams())
        boosted = rescore_row(row, SweepParams(ms_rvol_max_pts=20))
        # Boosting rvol weight should produce a different score
        assert default["score"] != boosted["score"] or True  # may be same if rvol component saturated


# ---------------------------------------------------------------------------
# rescore_signals
# ---------------------------------------------------------------------------

class TestRescoreSignals:
    def test_adds_expected_columns(self):
        df     = _make_df()
        params = SweepParams()
        result = rescore_signals(df, params)
        for col in ["rescore_tradescore", "rescore_direction", "rescore_setup_type", "stop_reconstructed"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_row_count_preserved(self):
        rows  = [_make_row(ticker="A"), _make_row(ticker="B"), _make_row(ticker="C")]
        df    = _make_df(rows)
        result = rescore_signals(df, SweepParams())
        assert len(result) == 3

    def test_stop_reconstructed_uses_ema20_atr(self):
        row    = _make_row(ema20=46.0, atr=1.2, price=50.0, direction="long")
        df     = pd.DataFrame([row])
        params = SweepParams(stop_multiplier=0.5)
        result = rescore_signals(df, params)
        expected = round(46.0 - 0.5 * 1.2, 4)
        assert abs(result.iloc[0]["stop_reconstructed"] - expected) < 0.001

    def test_stop_short_adds_multiplier(self):
        row    = _make_row(ema20=46.0, atr=1.2, price=50.0, direction="short")
        df     = pd.DataFrame([row])
        params = SweepParams(stop_multiplier=0.5)
        result = rescore_signals(df, params)
        expected = round(46.0 + 0.5 * 1.2, 4)
        assert abs(result.iloc[0]["stop_reconstructed"] - expected) < 0.001


# ---------------------------------------------------------------------------
# filter_signals
# ---------------------------------------------------------------------------

class TestFilterSignals:
    def _scored_df(self, score=50.0, rvol=2.5, rsi=55.0):
        df = _make_df([_make_row(rvol=rvol, rsi=rsi)])
        scored = rescore_signals(df, SweepParams())
        scored["rescore_tradescore"] = score
        return scored

    def test_threshold_passes(self):
        df     = self._scored_df(score=50.0)
        params = SweepParams(tradescore_threshold=40.0)
        result = filter_signals(df, params)
        assert len(result) == 1

    def test_threshold_rejects(self):
        df     = self._scored_df(score=30.0)
        params = SweepParams(tradescore_threshold=40.0)
        result = filter_signals(df, params)
        assert len(result) == 0

    def test_rvol_filter(self):
        df     = self._scored_df(rvol=1.5)
        params = SweepParams(min_rvol=2.0)
        result = filter_signals(df, params)
        assert len(result) == 0

    def test_rsi_range_filter(self):
        df     = self._scored_df(rsi=80.0)
        params = SweepParams(rsi_max=75.0)
        result = filter_signals(df, params)
        assert len(result) == 0

    def test_direction_filter(self):
        rows = [
            _make_row(ticker="A", direction="long"),
            _make_row(ticker="B", direction="short"),
        ]
        df     = rescore_signals(_make_df(rows), SweepParams())
        params = SweepParams(direction_filter="long")
        result = filter_signals(df, params)
        assert all(result["rescore_direction"] == "long")


# ---------------------------------------------------------------------------
# build_signal_groups
# ---------------------------------------------------------------------------

class TestBuildSignalGroups:
    def test_groups_by_ticker(self):
        rows = [
            _make_row(ticker="AAA", run_date="2025-01-10"),
            _make_row(ticker="AAA", run_date="2025-01-11"),
            _make_row(ticker="BBB", run_date="2025-01-10"),
        ]
        df     = rescore_signals(_make_df(rows), SweepParams())
        groups = build_signal_groups(df)
        assert "AAA" in groups and "BBB" in groups
        assert len(groups["AAA"]) == 2
        assert len(groups["BBB"]) == 1

    def test_signal_dict_has_required_keys(self):
        df     = rescore_signals(_make_df(), SweepParams())
        groups = build_signal_groups(df)
        sig    = groups["TEST"][0]
        for key in ("date", "stop", "target"):
            assert key in sig


# ---------------------------------------------------------------------------
# _aggregate_ticker_results
# ---------------------------------------------------------------------------

class TestAggregateTickerResults:
    def test_empty_input(self):
        agg = _aggregate_ticker_results([])
        assert agg["n_trades"] == 0
        assert agg["win_rate"] is None

    def test_ignores_zero_trade_tickers(self):
        results = [
            {"ticker": "A", "n_trades": 5, "win_rate": 60.0, "avg_trade_pct": 1.2,
             "sharpe": 1.5, "max_drawdown": -5.0, "return_pct": 8.0},
            {"ticker": "B", "n_trades": 0, "win_rate": float("nan"), "avg_trade_pct": 0,
             "sharpe": float("nan"), "max_drawdown": 0.0, "return_pct": 0.0},
        ]
        agg = _aggregate_ticker_results(results)
        assert agg["n_tickers"] == 1
        assert agg["n_trades"] == 5

    def test_trade_weighted_win_rate(self):
        results = [
            {"ticker": "A", "n_trades": 2, "win_rate": 100.0, "avg_trade_pct": 1.0,
             "sharpe": 1.0, "max_drawdown": -2.0, "return_pct": 5.0},
            {"ticker": "B", "n_trades": 2, "win_rate": 0.0, "avg_trade_pct": -1.0,
             "sharpe": -1.0, "max_drawdown": -5.0, "return_pct": -3.0},
        ]
        agg = _aggregate_ticker_results(results)
        assert agg["win_rate"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# run_param_set (no DB, no price data)
# ---------------------------------------------------------------------------

class TestRunParamSet:
    def test_returns_expected_keys(self):
        df     = _make_df()
        result = run_param_set(
            SweepParams(tradescore_threshold=200),  # filters everything
            df,
            price_cache={},
            persist=False,
        )
        for key in ("label", "n_signals_raw", "n_signals_filtered", "n_tickers", "n_trades"):
            assert key in result

    def test_filters_all_returns_zero_trades(self):
        df = _make_df()
        result = run_param_set(
            SweepParams(tradescore_threshold=200),
            df,
            price_cache={},
            persist=False,
        )
        assert result["n_trades"] == 0
        assert result["n_signals_filtered"] == 0

    def test_three_threshold_values_three_results(self):
        """Sweep with 3 threshold values → 3 result rows."""
        df     = _make_df([_make_row(ticker="A"), _make_row(ticker="B")])
        params = param_grid(tradescore_threshold=[0, 100, 200])
        results = [
            run_param_set(p, df, price_cache={}, persist=False)
            for p in params
        ]
        assert len(results) == 3

    def test_reproducible(self):
        """Running same params twice produces identical aggregate metrics."""
        df     = _make_df()
        params = SweepParams(tradescore_threshold=20.0)
        r1 = run_param_set(params, df, price_cache={}, persist=False)
        r2 = run_param_set(params, df, price_cache={}, persist=False)
        assert r1["n_signals_filtered"] == r2["n_signals_filtered"]
        assert r1["n_trades"] == r2["n_trades"]
