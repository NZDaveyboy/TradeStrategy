"""
tests/test_tradescore.py

Verifies compute_tradescore() with fixed inputs whose expected outputs
were calculated by hand against the scoring constants in run.py.

Setup types tested:
  - Early breakout   (ms≈24, ee=25, er≈1,  lq=15  → score≈63)
  - Overextended     (er=17 from RSI 85 + 25% daily move + VWAP distance)
  - Low quality      (lq=1  from zero dollar volume + micro-cap)
  - Return shape     (all expected keys present for any input)
"""

import math

import numpy as np
import pandas as pd
import pytest

from run import compute_tradescore


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_close(n: int = 20, start: float = 45.0, end: float = 50.0) -> pd.Series:
    """Linearly increasing close series ending at `end`."""
    return pd.Series(np.linspace(start, end, n))


def _make_data(n: int = 20, vol: int = 2_000_000,
               start: float = 45.0, end: float = 50.0) -> pd.DataFrame:
    prices = np.linspace(start, end, n)
    return pd.DataFrame({
        "Open":   prices * 0.99,
        "High":   prices * 1.01,
        "Low":    prices * 0.98,
        "Close":  prices,
        "Volume": [float(vol)] * n,
    })


def _bullish_row() -> dict:
    """
    Designed to hit 'Early breakout':
      RVOL 3.5x  → rvol_pts  ≈ 9.67  (MS sub-cap 10)
      change 7%  → chg_pts   = 7.0   (MS sub-cap 8)
      MACD/ATR   → macd_pts  = 7.0   (MS sub-cap 7)   → MS ≈ 23.7

      RSI 60 in [52,68]              → rsi_pts = 10
      price within 2% of EMA20       → ema_pts = 8
      price at 20-day high (close)   → bob_pts = 7     → EE = 25

      RSI < 70, change < 10%, VWAP within 2 ATR → ER ≈ 1.0

      $100M dollar vol, mid-cap, consistent vol  → LQ = 15
    """
    return {
        "price":          50.0,
        "change_pct":      7.0,
        "rvol":            3.5,
        "ema9":           49.5,
        "ema20":          49.0,
        "ema200":         40.0,
        "rsi":            60.0,
        "atr":             1.0,
        "stop_loss":      47.5,
        "macd":            0.5,
        "macd_signal":     0.1,
        "vwap":           48.0,
        "volume_trend_up": 1,
        "score":           4,
        "market_cap":  5_000_000_000,   # mid-cap
        "float_shares":   50_000_000,
    }


def _overextended_row() -> dict:
    """
    Designed to hit ER ≥ 15 → 'Overextended':
      RSI 85  → rsi_pen  = 6  (clamped; 85 > ER_RSI_HARD 82)
      25% day → day_pen  = 6  (clamped; 25 > ER_DAY_HARD 22)
      price 50, vwap 40, atr 0.5 → vwap_dist 20 ATR multiples → vwap_pen = 5
      Total ER = 17 → 'Overextended'
    """
    return {
        "price":          50.0,
        "change_pct":     25.0,
        "rvol":            5.0,
        "ema9":           49.0,
        "ema20":          45.0,
        "ema200":         30.0,
        "rsi":            85.0,
        "atr":             0.5,
        "stop_loss":      46.0,
        "macd":            0.5,
        "macd_signal":     0.3,
        "vwap":           40.0,
        "volume_trend_up": 1,
        "score":           4,
        "market_cap":  3_000_000_000,
        "float_shares":   50_000_000,
    }


def _illiquid_row() -> dict:
    """
    Designed to hit LQ ≤ 3 → 'Low quality / illiquid':
      vol=0  → dvol $0       → 0 pts
      $30M market cap (micro) → 1 quality pt
      vol=0  → mean=0        → 0 consistency pts
      Total LQ = 1
    """
    return {
        "price":           3.0,
        "change_pct":      0.5,
        "rvol":            0.3,
        "ema9":            3.1,
        "ema20":           3.2,
        "ema200":          3.5,
        "rsi":            45.0,
        "atr":             0.1,
        "stop_loss":       2.85,
        "macd":           -0.01,
        "macd_signal":     0.01,
        "vwap":            3.15,
        "volume_trend_up": 0,
        "score":           0,
        "market_cap":  30_000_000,   # micro-cap → 1 quality pt
        "float_shares":  200_000_000,
    }


# ---------------------------------------------------------------------------
# Return shape
# ---------------------------------------------------------------------------

def test_compute_tradescore_returns_expected_keys():
    result = compute_tradescore(_bullish_row())
    expected_keys = {
        "score", "setup_type", "direction", "rationale",
        "momentum_score", "early_entry", "extension_risk",
        "liquidity", "news_catalyst", "change_5d", "conviction",
        "components",
    }
    assert expected_keys.issubset(result.keys())


def test_compute_tradescore_score_is_non_negative():
    for row in (_bullish_row(), _overextended_row(), _illiquid_row()):
        result = compute_tradescore(row)
        assert result["score"] >= 0.0


# ---------------------------------------------------------------------------
# Early breakout
# ---------------------------------------------------------------------------

def test_bullish_setup_gets_high_tradescore():
    close = _make_close()
    data  = _make_data()
    result = compute_tradescore(_bullish_row(), close=close, data=data)
    assert result["score"] >= 40


def test_bullish_setup_gets_positive_label():
    close = _make_close()
    data  = _make_data()
    result = compute_tradescore(_bullish_row(), close=close, data=data)
    assert result["setup_type"] in (
        "Early breakout", "Emerging momentum", "Strong but extended"
    )


def test_bullish_setup_direction_is_long():
    """price > vwap and ema9 >= ema20 → long."""
    result = compute_tradescore(_bullish_row())
    assert result["direction"] == "long"


def test_bullish_setup_sub_scores_sum_correctly():
    close = _make_close()
    data  = _make_data()
    result = compute_tradescore(_bullish_row(), close=close, data=data)
    # score = MS + EE + LQ + NC - ER  (floats, clipped at 0)
    expected = max(
        0.0,
        result["momentum_score"]
        + result["early_entry"]
        + result["liquidity"]
        + result["news_catalyst"]
        - result["extension_risk"],
    )
    assert abs(result["score"] - round(expected, 1)) < 0.01


# ---------------------------------------------------------------------------
# Overextended
# ---------------------------------------------------------------------------

def test_overextended_setup_gets_overextended_label():
    result = compute_tradescore(_overextended_row())
    assert result["setup_type"] == "Overextended"


def test_overextended_extension_risk_is_gte_15():
    result = compute_tradescore(_overextended_row())
    assert result["extension_risk"] >= 15


# ---------------------------------------------------------------------------
# Low quality / illiquid
# ---------------------------------------------------------------------------

def test_illiquid_setup_gets_illiquid_label():
    data   = _make_data(vol=0)   # zero volume → dvol = $0 → 0 pts
    result = compute_tradescore(_illiquid_row(), data=data)
    assert result["setup_type"] == "Low quality / illiquid"


def test_illiquid_setup_liquidity_lte_3():
    data   = _make_data(vol=0)
    result = compute_tradescore(_illiquid_row(), data=data)
    assert result["liquidity"] <= 3


# ---------------------------------------------------------------------------
# Direction logic
# ---------------------------------------------------------------------------

def test_bearish_direction_when_price_below_vwap_and_ema9_below_ema20():
    row = _bullish_row()
    row["price"]  = 44.0   # below vwap (48) and ema9 (49.5) < ema20 (49)... need to adjust
    row["vwap"]   = 48.0
    row["ema9"]   = 48.0
    row["ema20"]  = 49.0   # ema9 < ema20
    result = compute_tradescore(row)
    assert result["direction"] == "short"
