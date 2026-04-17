"""
tests/test_rescore_fidelity.py — Verify that stored high_20d / dollar_volume / vol_cv
enable faithful re-scoring without the original close series or OHLCV DataFrame.

Tests are split into three layers:
  1. Sub-score isolation — each stored input recovers the correct sub-component pts
  2. Null/missing graceful degradation — missing stored values silently score 0
  3. End-to-end — full compute_tradescore(row, close=None, data=None) with all three
     stored fields matches compute_tradescore(row, close=..., data=...) within 0.1 pts

No DB access required — all tests use synthetic rows and controlled Series/DataFrames.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from core.tradescore import (
    EE_BOB_MAX_PTS,
    LQ_CONS_MAX_PTS,
    LQ_DVOL_FULL,
    LQ_DVOL_MIN,
    LQ_DVOL_MAX_PTS,
    _early_entry_score,
    _liquidity_score,
    compute_tradescore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_row(**overrides) -> dict:
    """Minimal valid row for compute_tradescore."""
    row = {
        "price":        50.0,
        "rvol":         2.5,
        "change_pct":   3.0,
        "rsi":          58.0,
        "ema9":         49.0,
        "ema20":        47.0,
        "ema200":       40.0,
        "atr":          1.2,
        "macd":         0.3,
        "macd_signal":  0.15,
        "vwap":         49.5,
        "change_5d":    4.0,
        "market_cap":   5_000_000_000,
        "float_shares": None,
        # New stored fields — None by default (old-row behaviour)
        "high_20d":      None,
        "dollar_volume": None,
        "vol_cv":        None,
    }
    row.update(overrides)
    return row


def _make_close(n: int = 25, hi_val: float = 100.0, current: float = 98.0) -> pd.Series:
    """Close series: flat at hi_val for first n-1 bars, then current price."""
    vals = [hi_val] * (n - 1) + [current]
    return pd.Series(vals, dtype=float)


def _make_ohlcv(n_bars: int = 20, last_volume: float = 1_000_000,
                prior_volumes: list[float] | None = None) -> pd.DataFrame:
    """Minimal OHLCV DataFrame for liquidity scoring."""
    if prior_volumes is None:
        prior_volumes = [last_volume] * 10   # consistent volume
    # Need at least 11 bars: 10 prior + 1 last
    vols = list(prior_volumes[-10:]) + [last_volume]
    n    = len(vols)
    idx  = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open":   [50.0] * n,
            "High":   [51.0] * n,
            "Low":    [49.0] * n,
            "Close":  [50.0] * n,
            "Volume": vols,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# 1. BOB — high_20d stored input
# ---------------------------------------------------------------------------

class TestBOBFidelity:
    def test_full_bob_from_stored_high_20d(self):
        """price/high_20d >= 0.97 → bob_pts == EE_BOB_MAX_PTS."""
        row = _base_row(price=98.0, high_20d=100.0)   # pct_hi = 0.98
        _, det = _early_entry_score(row, close=None)
        assert det["bob_pts"] == float(EE_BOB_MAX_PTS)

    def test_full_bob_exact_threshold(self):
        """price == high_20d → pct_hi == 1.0 → full bob."""
        row = _base_row(price=100.0, high_20d=100.0)
        _, det = _early_entry_score(row, close=None)
        assert det["bob_pts"] == float(EE_BOB_MAX_PTS)

    def test_partial_bob_lerp_range(self):
        """0.85 <= pct_hi < 0.97 → bob_pts is partial (lerp-scaled)."""
        # pct_hi = 90 / 100 = 0.90 — midway through the lerp range [0.85, 0.97]
        row = _base_row(price=90.0, high_20d=100.0)
        _, det = _early_entry_score(row, close=None)
        assert 0.0 < det["bob_pts"] < float(EE_BOB_MAX_PTS)

    def test_zero_bob_below_range(self):
        """pct_hi < 0.85 → bob_pts == 0."""
        row = _base_row(price=84.0, high_20d=100.0)
        _, det = _early_entry_score(row, close=None)
        assert det["bob_pts"] == 0.0

    def test_stored_high_20d_matches_close_series(self):
        """Stored high_20d produces same bob_pts as live close series."""
        close = _make_close(n=25, hi_val=100.0, current=98.0)
        row   = _base_row(price=98.0)

        # With close series
        _, det_live = _early_entry_score(row, close=close)

        # With stored high_20d only
        row_stored = _base_row(price=98.0, high_20d=100.0)
        _, det_stored = _early_entry_score(row_stored, close=None)

        assert det_live["bob_pts"] == det_stored["bob_pts"]

    def test_high_20d_none_gives_zero_bob(self):
        """NULL high_20d and no close series → bob_pts = 0 (old-row graceful degradation)."""
        row = _base_row(price=98.0, high_20d=None)
        _, det = _early_entry_score(row, close=None)
        assert det["bob_pts"] == 0.0

    def test_close_takes_precedence_over_stored(self):
        """When close series is provided it wins over stored high_20d."""
        # close has hi = 200 (price/200 = 0.49 → bob = 0)
        # stored high_20d = 100 (price/100 = 0.98 → bob = max)
        close = _make_close(n=25, hi_val=200.0, current=98.0)
        row   = _base_row(price=98.0, high_20d=100.0)
        _, det = _early_entry_score(row, close=close)
        assert det["bob_pts"] == 0.0   # close wins; pct_hi = 0.49


# ---------------------------------------------------------------------------
# 2. Dollar volume — dollar_volume stored input
# ---------------------------------------------------------------------------

class TestDollarVolumeFidelity:
    def test_full_dvol_pts_from_stored(self):
        """dollar_volume >= LQ_DVOL_FULL → dvol_pts == LQ_DVOL_MAX_PTS."""
        row = _base_row(dollar_volume=float(LQ_DVOL_FULL) + 1.0)
        _, det = _liquidity_score(row, data=None)
        assert det["dvol_pts"] == float(LQ_DVOL_MAX_PTS)

    def test_zero_dvol_pts_below_min(self):
        """dollar_volume < LQ_DVOL_MIN → dvol_pts == 0."""
        row = _base_row(dollar_volume=float(LQ_DVOL_MIN) - 1.0)
        _, det = _liquidity_score(row, data=None)
        assert det["dvol_pts"] == 0.0

    def test_partial_dvol_pts_lerp(self):
        """dollar_volume in (LQ_DVOL_MIN, LQ_DVOL_FULL) → dvol_pts is partial."""
        mid = (LQ_DVOL_MIN + LQ_DVOL_FULL) / 2
        row = _base_row(dollar_volume=mid)
        _, det = _liquidity_score(row, data=None)
        assert 0.0 < det["dvol_pts"] < float(LQ_DVOL_MAX_PTS)

    def test_stored_dollar_volume_matches_ohlcv(self):
        """Stored dollar_volume produces same dvol_pts as live OHLCV data."""
        price       = 50.0
        last_vol    = 500_000.0
        dollar_vol  = price * last_vol

        # With OHLCV
        ohlcv = _make_ohlcv(last_volume=last_vol)
        row   = _base_row(price=price)
        _, det_live = _liquidity_score(row, data=ohlcv)

        # With stored dollar_volume
        row_stored = _base_row(price=price, dollar_volume=dollar_vol)
        _, det_stored = _liquidity_score(row_stored, data=None)

        assert abs(det_live["dvol_pts"] - det_stored["dvol_pts"]) < 0.01

    def test_dollar_volume_none_gives_zero_dvol(self):
        """NULL dollar_volume and no data → dvol_pts = 0."""
        row = _base_row(dollar_volume=None)
        _, det = _liquidity_score(row, data=None)
        assert det["dvol_pts"] == 0.0

    def test_data_takes_precedence_over_stored(self):
        """When OHLCV data is provided it wins over stored dollar_volume."""
        # OHLCV gives tiny dvol (below LQ_DVOL_MIN)
        ohlcv = _make_ohlcv(last_volume=1_000.0)
        # Stored value claims max dvol
        row   = _base_row(price=1.0, dollar_volume=float(LQ_DVOL_FULL) * 10)
        _, det = _liquidity_score(row, data=ohlcv)
        assert det["dvol_pts"] == 0.0   # OHLCV wins; price * 1000 = tiny


# ---------------------------------------------------------------------------
# 3. Volume CV — vol_cv stored input
# ---------------------------------------------------------------------------

class TestVolCVFidelity:
    def test_zero_cv_gives_max_cons_pts(self):
        """vol_cv = 0 (perfectly consistent) → cons_pts == LQ_CONS_MAX_PTS."""
        row = _base_row(vol_cv=0.0)
        _, det = _liquidity_score(row, data=None)
        assert det["cons_pts"] == pytest.approx(float(LQ_CONS_MAX_PTS), abs=0.01)

    def test_cv_above_one_gives_zero_cons_pts(self):
        """vol_cv >= 1.0 → cons_pts == 0 (max(0, 1 - cv) = 0)."""
        row = _base_row(vol_cv=1.5)
        _, det = _liquidity_score(row, data=None)
        assert det["cons_pts"] == 0.0

    def test_mid_cv_gives_partial_cons_pts(self):
        """vol_cv = 0.5 → cons_pts == LQ_CONS_MAX_PTS × 0.5."""
        row = _base_row(vol_cv=0.5)
        _, det = _liquidity_score(row, data=None)
        assert det["cons_pts"] == pytest.approx(LQ_CONS_MAX_PTS * 0.5, abs=0.01)

    def test_stored_vol_cv_matches_ohlcv(self):
        """Stored vol_cv produces same cons_pts as live OHLCV computation."""
        prior = [1_000_000.0, 1_200_000.0, 800_000.0, 1_100_000.0, 950_000.0,
                 1_050_000.0, 1_150_000.0, 900_000.0, 1_000_000.0, 1_080_000.0]
        vol_series = pd.Series(prior, dtype=float)
        cv_val     = float(vol_series.std() / vol_series.mean())

        # With OHLCV
        ohlcv = _make_ohlcv(last_volume=1_000_000.0, prior_volumes=prior)
        row   = _base_row()
        _, det_live = _liquidity_score(row, data=ohlcv)

        # With stored vol_cv
        row_stored = _base_row(vol_cv=cv_val)
        _, det_stored = _liquidity_score(row_stored, data=None)

        assert abs(det_live["cons_pts"] - det_stored["cons_pts"]) < 0.01

    def test_vol_cv_none_gives_zero_cons_pts(self):
        """NULL vol_cv and no data → cons_pts = 0."""
        row = _base_row(vol_cv=None)
        _, det = _liquidity_score(row, data=None)
        assert det["cons_pts"] == 0.0


# ---------------------------------------------------------------------------
# 4. End-to-end: full compute_tradescore fidelity
# ---------------------------------------------------------------------------

class TestEndToEndFidelity:
    def _make_full_inputs(self, price: float = 50.0,
                          hi_20: float = 52.0,
                          last_vol: float = 800_000.0):
        """Build matching (row+close+ohlcv) and (row_stored) for comparison."""
        # Close series: 25 bars, max = hi_20, last = price
        closes     = [hi_20] * 24 + [price]
        close      = pd.Series(closes, dtype=float)

        # Volume: 10 consistent prior bars + last bar
        prior_vols = [last_vol] * 10
        ohlcv      = _make_ohlcv(last_volume=last_vol, prior_volumes=prior_vols)

        # Derived stored values using same formulas as run.py
        vol_series  = pd.Series(prior_vols, dtype=float)
        vol_cv      = float(vol_series.std() / vol_series.mean()) if vol_series.mean() > 0 else None
        dollar_vol  = price * last_vol

        row_base = {
            "price":        price,
            "rvol":         2.0,
            "change_pct":   2.5,
            "rsi":          60.0,
            "ema9":         price * 0.99,
            "ema20":        price * 0.96,
            "ema200":       price * 0.85,
            "atr":          1.0,
            "macd":         0.25,
            "macd_signal":  0.10,
            "vwap":         price * 0.99,
            "change_5d":    3.5,
            "market_cap":   3_000_000_000,
            "float_shares": None,
        }
        row_live   = {**row_base, "high_20d": None, "dollar_volume": None, "vol_cv": None}
        row_stored = {**row_base, "high_20d": hi_20, "dollar_volume": dollar_vol, "vol_cv": vol_cv}

        return close, ohlcv, row_live, row_stored

    def test_full_score_matches_within_tolerance(self):
        """End-to-end: stored inputs produce same score as live series."""
        close, ohlcv, row_live, row_stored = self._make_full_inputs()

        live   = compute_tradescore(row_live,   close=close, data=ohlcv)
        stored = compute_tradescore(row_stored, close=None,  data=None)

        assert abs(live["score"] - stored["score"]) < 0.1, (
            f"Score mismatch: live={live['score']} stored={stored['score']}"
        )

    def test_bob_pts_identical(self):
        """BOB sub-component matches exactly between live and stored paths."""
        close, ohlcv, row_live, row_stored = self._make_full_inputs(
            price=50.0, hi_20=51.0   # pct_hi = 0.98 → full BOB
        )
        live   = compute_tradescore(row_live,   close=close, data=ohlcv)
        stored = compute_tradescore(row_stored, close=None,  data=None)

        assert live["components"]["early_entry"]["bob_pts"] == \
               stored["components"]["early_entry"]["bob_pts"]

    def test_dvol_pts_identical(self):
        """dvol sub-component matches exactly between live and stored paths."""
        close, ohlcv, row_live, row_stored = self._make_full_inputs(
            last_vol=2_000_000.0   # above LQ_DVOL_MIN, below LQ_DVOL_FULL
        )
        live   = compute_tradescore(row_live,   close=close, data=ohlcv)
        stored = compute_tradescore(row_stored, close=None,  data=None)

        assert abs(
            live["components"]["liquidity"]["dvol_pts"] -
            stored["components"]["liquidity"]["dvol_pts"]
        ) < 0.01

    def test_cons_pts_matches_within_tolerance(self):
        """cons_pts sub-component matches within 0.01 pts between live and stored."""
        close, ohlcv, row_live, row_stored = self._make_full_inputs()
        live   = compute_tradescore(row_live,   close=close, data=ohlcv)
        stored = compute_tradescore(row_stored, close=None,  data=None)

        assert abs(
            live["components"]["liquidity"]["cons_pts"] -
            stored["components"]["liquidity"]["cons_pts"]
        ) < 0.01

    def test_partial_row_null_fields_score_lower(self):
        """Row with NULL stored fields scores lower than the equivalent full row
        (old-row behaviour: missing inputs contribute 0 pts)."""
        close, ohlcv, row_live, row_stored = self._make_full_inputs(
            last_vol=2_000_000.0,   # ensures dvol_pts > 0
        )
        full    = compute_tradescore(row_stored, close=None, data=None)
        partial = compute_tradescore(row_live,   close=None, data=None)

        # Full should score >= partial (BOB + dvol + cons are all > 0 here)
        assert full["score"] >= partial["score"]
