"""
tests/test_recommendations.py

Tests for core/recommendations.py — build_recommendation() logic.

No DB, no network. All tests use hand-built dicts.
"""

from __future__ import annotations

import pytest

from core.recommendations import (
    MAX_STOP_DISTANCE_PCT,
    EXTENSION_ATR_THRESHOLD,
    EXTENSION_EMA_PCT_THRESHOLD,
    STOP_ATR_MULTIPLIER,
    Recommendation,
    build_recommendation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "ticker", "direction", "setup_type", "recommendation_category",
    "strategy_name", "entry_reference", "invalidation_price", "target_price",
    "risk_reward", "rationale", "iv_assessment", "warnings", "is_actionable",
}


def _row(**kwargs) -> dict:
    """Minimal screener row with a clean, non-extended long-setup.

    price=150, ema20=145, atr=5:
      ATR distance = (150-145)/5 = 1.0 < 1.5 threshold  → not extended
      pct above    = (150-145)/145 = 3.4% < 8% threshold → not extended
      stop         = 145 - 0.5*5  = 142.5 → 5.0% from price → within 12% cap
    """
    base = {
        "ticker":     "AAPL",
        "price":      150.0,
        "ema20":      145.0,
        "ema9":       147.0,
        "ema200":     130.0,
        "atr":        5.0,
        "vwap":       100.0,      # deliberately broken cumulative VWAP — must be IGNORED
        "direction":  "long",
        "setup_type": "Early breakout",
        "tradescore": 72.0,
        "rvol":       2.5,
    }
    base.update(kwargs)
    return base


def _extended_row(**kwargs) -> dict:
    """Row where price is well above EMA20 — triggers extension logic.

    price=170, ema20=145, atr=5:
      ATR distance = (170-145)/5 = 5.0 > 1.5 → extended
      pct above    = (170-145)/145 = 17.2% > 8% → extended
    """
    base = _row(price=170.0, ema20=145.0, atr=5.0)
    base.update(kwargs)
    return base


def _wide_stop_row(**kwargs) -> dict:
    """Row where stop distance exceeds 12% cap but is NOT extended.

    price=100, ema20=100, atr=30:
      ATR distance = (100-100)/30 = 0.0 → not extended
      stop         = 100 - 0.5*30 = 85  → 15% from price → exceeds 12% cap
    """
    base = _row(price=100.0, ema20=100.0, atr=30.0)
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# 1. All required fields present
# ---------------------------------------------------------------------------

def test_recommendation_contains_all_required_fields():
    rec = build_recommendation(_row())
    actual = {f.name for f in rec.__dataclass_fields__.values()}
    assert REQUIRED_FIELDS <= actual, f"Missing: {REQUIRED_FIELDS - actual}"


def test_recommendation_is_dataclass_instance():
    rec = build_recommendation(_row())
    assert isinstance(rec, Recommendation)


def test_all_required_fields_have_non_none_values_for_clean_long():
    rec = build_recommendation(_row())
    # These must never be None on an actionable long
    assert rec.ticker          is not None
    assert rec.strategy_name   is not None
    assert rec.rationale       != ""
    assert rec.iv_assessment   is not None
    assert isinstance(rec.warnings, list)


# ---------------------------------------------------------------------------
# 2. Extended bullish setup must NOT produce long_call
# ---------------------------------------------------------------------------

def test_extended_bullish_does_not_return_long_call():
    rec = build_recommendation(_extended_row())
    assert rec.strategy_name != "long_call", (
        f"Extended setup returned long_call — extension check must run first. "
        f"Got: {rec.strategy_name}"
    )


def test_extended_bullish_does_not_return_bull_call_spread():
    rec = build_recommendation(_extended_row(), atm_iv=0.45, rv30=0.30)
    assert rec.strategy_name != "bull_call_spread", (
        f"Extended + expensive IV still returned bull_call_spread. Got: {rec.strategy_name}"
    )


def test_extended_bullish_setup_type_is_extended():
    rec = build_recommendation(_extended_row())
    assert rec.setup_type == "extended"


def test_extended_bullish_is_not_actionable():
    rec = build_recommendation(_extended_row())
    assert rec.is_actionable is False


def test_extended_bullish_has_warning():
    rec = build_recommendation(_extended_row())
    assert len(rec.warnings) > 0
    assert any("extended" in w.lower() or "above" in w.lower() for w in rec.warnings)


# ---------------------------------------------------------------------------
# 3. Stop distance > 12% → is_actionable == False
# ---------------------------------------------------------------------------

def test_wide_stop_is_not_actionable():
    rec = build_recommendation(_wide_stop_row())
    assert rec.is_actionable is False, (
        f"Expected is_actionable=False when stop > {MAX_STOP_DISTANCE_PCT*100:.0f}%, "
        f"got {rec.is_actionable}. strategy={rec.strategy_name}"
    )


def test_wide_stop_setup_type_is_pullback_candidate():
    rec = build_recommendation(_wide_stop_row())
    assert rec.setup_type == "pullback_candidate"


def test_wide_stop_strategy_is_wait():
    rec = build_recommendation(_wide_stop_row())
    assert rec.strategy_name == "wait"


def test_within_stop_cap_is_actionable():
    # price=150, ema20=145, atr=5 → stop=142.5 → 5.0% — within 12%
    rec = build_recommendation(_row(price=150.0, ema20=145.0, atr=5.0))
    assert rec.is_actionable is True


# ---------------------------------------------------------------------------
# 4. Advice and Options paths produce identical strategy_name / invalidation
# ---------------------------------------------------------------------------

def test_same_row_same_strategy_no_iv():
    """Without IV data both tabs should produce identical output."""
    row = _row()
    rec_advice  = build_recommendation(row)
    rec_options = build_recommendation(row)
    assert rec_advice.strategy_name      == rec_options.strategy_name
    assert rec_advice.invalidation_price == rec_options.invalidation_price


def test_same_row_same_strategy_with_iv():
    """With IV data both tabs should still produce identical output."""
    row = _row()
    rec_advice  = build_recommendation(row, atm_iv=0.45, rv30=0.30)
    rec_options = build_recommendation(row, atm_iv=0.45, rv30=0.30)
    assert rec_advice.strategy_name      == rec_options.strategy_name
    assert rec_advice.invalidation_price == rec_options.invalidation_price


def test_iv_expensive_selects_spread_not_outright():
    # atm_iv 45% vs rv30 30% → expensive → bull_call_spread
    rec = build_recommendation(_row(), atm_iv=0.45, rv30=0.30)
    assert rec.strategy_name == "bull_call_spread"


def test_iv_cheap_selects_outright_call():
    # atm_iv 20% vs rv30 30% → cheap → long_call
    rec = build_recommendation(_row(), atm_iv=0.20, rv30=0.30)
    assert rec.strategy_name == "long_call"


def test_no_iv_data_selects_long_call_by_default():
    rec = build_recommendation(_row())
    assert rec.strategy_name == "long_call"
    assert rec.iv_assessment == "unavailable"


# ---------------------------------------------------------------------------
# 5. Advice tab top picks sorted by tradescore descending (logic test)
# ---------------------------------------------------------------------------

def test_top_picks_sorted_by_tradescore():
    """Simulate what the Advice tab does: build recs for multiple rows,
    confirm results are sortable by tradescore."""
    rows = [
        _row(ticker="A", tradescore=45.0),
        _row(ticker="B", tradescore=82.0),
        _row(ticker="C", tradescore=63.0),
    ]
    recs = [build_recommendation(r) for r in rows]
    # Sort the same way the Advice tab should
    sorted_recs = sorted(
        zip(rows, recs),
        key=lambda x: x[0].get("tradescore", 0),
        reverse=True,
    )
    tickers = [r[0]["ticker"] for r in sorted_recs]
    assert tickers == ["B", "C", "A"], f"Expected ['B','C','A'], got {tickers}"


# ---------------------------------------------------------------------------
# 6. Bearish setups
# ---------------------------------------------------------------------------

def test_bearish_clean_setup_returns_long_put():
    # price=143, ema20=145, atr=3: ATR dist=(145-143)/3=0.67 < 1.5, pct=1.4% < 8% → not extended
    # stop=145+1.5=146.5 → dist=(146.5-143)/143=2.4% < 12% → actionable
    row = _row(direction="short", price=143.0, ema20=145.0, atr=3.0)
    rec = build_recommendation(row)
    assert rec.strategy_name == "long_put"
    assert rec.is_actionable is True


def test_bearish_extended_is_not_actionable():
    # Price 31% below EMA20, 15 ATRs extended → extended downside
    row = _row(direction="short", price=100.0, ema20=145.0, atr=3.0)
    rec = build_recommendation(row)
    assert rec.is_actionable is False
    assert rec.setup_type == "extended"


def test_bearish_expensive_iv_returns_bear_put_spread():
    row = _row(direction="short", price=143.0, ema20=145.0, atr=3.0)
    rec = build_recommendation(row, atm_iv=0.50, rv30=0.30)
    assert rec.strategy_name == "bear_put_spread"


# ---------------------------------------------------------------------------
# 7. No-direction / neutral
# ---------------------------------------------------------------------------

def test_no_direction_returns_wait():
    rec = build_recommendation(_row(direction="", tradescore=30.0))
    assert rec.strategy_name == "wait"
    assert rec.is_actionable is False
    assert rec.setup_type == "no_edge"


def test_neutral_direction_returns_wait():
    rec = build_recommendation(_row(direction="neutral"))
    assert rec.strategy_name == "wait"


# ---------------------------------------------------------------------------
# 8. Crypto: no options
# ---------------------------------------------------------------------------

def test_crypto_ticker_returns_wait():
    rec = build_recommendation(_row(ticker="BTC-USD"))
    assert rec.strategy_name == "wait"
    assert rec.is_actionable is False
    assert rec.setup_type == "crypto_no_options"


def test_crypto_has_no_options_warning():
    rec = build_recommendation(_row(ticker="ETH-USD"))
    assert any("crypto" in w.lower() for w in rec.warnings)


# ---------------------------------------------------------------------------
# 9. Stop logic: EMA20-based, not VWAP-based
# ---------------------------------------------------------------------------

def test_invalidation_uses_ema20_not_vwap():
    """VWAP is intentionally broken (cumulative) — stop must be EMA20 ± ATR.

    price=148, ema20=145, atr=3:
      ATR dist = (148-145)/3 = 1.0 < 1.5 → not extended
      Old stop = min(vwap=100, ema20=145) - 0.35*atr = 98.95  (absurd — 33% from price)
      New stop = ema20 - 0.5*atr = 145 - 1.5 = 143.50
    """
    rec = build_recommendation(_row(price=148.0, ema20=145.0, atr=3.0, vwap=100.0))
    assert rec.invalidation_price == pytest.approx(143.5, abs=0.1), (
        f"Expected stop ~143.5 (EMA20-based), got {rec.invalidation_price}. "
        "VWAP must not be used as stop anchor."
    )


def test_long_stop_is_below_price():
    rec = build_recommendation(_row(direction="long"))
    assert rec.invalidation_price < rec.entry_reference


def test_short_stop_is_above_price():
    # price=143, ema20=145, atr=3 → clean bearish (see test_bearish_clean_setup)
    rec = build_recommendation(_row(direction="short", price=143.0, ema20=145.0, atr=3.0))
    assert rec.invalidation_price > rec.entry_reference


def test_target_is_2r_above_entry_for_long():
    rec = build_recommendation(_row(price=150.0, ema20=145.0, atr=5.0))
    risk   = rec.entry_reference - rec.invalidation_price
    reward = rec.target_price    - rec.entry_reference
    assert reward == pytest.approx(2 * risk, rel=0.01)
