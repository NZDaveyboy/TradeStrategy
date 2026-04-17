"""
core/tradescore.py — TradeScore composite scoring engine.

Extracted from run.py so research mode and tests can import scoring logic
without pulling in the full screener (yfinance, Finviz, Streamlit deps).

No behaviour changes from run.py — pure lift.

FinalTradeScore = MomentumScore + EarlyEntryScore + LiquidityScore
                + NewsCatalystScore − ExtensionRiskScore
Practical range: 0–65. Negative values clipped to 0.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Scoring constants — adjust thresholds here, nowhere else
# ---------------------------------------------------------------------------

# MomentumScore (max 25)
MS_RVOL_IDEAL:   float = 3.0    # RVOL at which full RVOL points are earned
MS_RVOL_MAX_PTS: int   = 10     # sub-cap for RVOL contribution
MS_CHG_HI_PCT:   float = 8.0    # % change earning full change points (sweet spot ceiling)
MS_CHG_MAX_PTS:  int   = 8      # sub-cap for change% contribution
MS_MACD_MAX_PTS: int   = 7      # sub-cap for MACD contribution

# EarlyEntryScore (max 25)
EE_RSI_LO:       int   = 52     # ideal RSI band lower bound
EE_RSI_HI:       int   = 68     # ideal RSI band upper bound (above here = heating up)
EE_RSI_MAX_PTS:  int   = 10     # sub-cap for RSI contribution
EE_EMA_NEAR_PCT: float = 5.0    # within this % of EMA20 → full proximity points
EE_EMA_FAR_PCT:  float = 18.0   # beyond this % from EMA20 → 0 proximity points
EE_EMA_MAX_PTS:  int   = 8      # sub-cap for EMA proximity
EE_BOB_MAX_PTS:  int   = 7      # sub-cap for breakout-from-base contribution

# ExtensionRiskScore (max 20, subtracted from total)
ER_RSI_WARN:     int   = 70     # RSI above this starts penalty
ER_RSI_HARD:     int   = 82     # RSI at this = max RSI penalty
ER_RSI_MAX_PTS:  int   = 6      # max RSI penalty points
ER_DAY_WARN:     float = 10.0   # single-day % move starts penalty
ER_DAY_HARD:     float = 22.0   # single-day % move = max penalty
ER_DAY_MAX_PTS:  int   = 6      # max daily-overextension penalty points
ER_VWAP_WARN:    float = 1.5    # ATR multiples above VWAP starts penalty
ER_VWAP_HARD:    float = 4.0    # ATR multiples = max VWAP penalty
ER_VWAP_MAX_PTS: int   = 5      # max VWAP extension penalty points
ER_5D_WARN:      float = 15.0   # 5-session % run starts multi-day penalty
ER_5D_HARD:      float = 45.0   # 5-session % run = max multi-day penalty
ER_5D_MAX_PTS:   int   = 3      # max multi-day-run penalty points

# LiquidityQualityScore (max 15)
LQ_DVOL_MIN:     int   = 500_000      # dollar volume below this → 0 liquidity points
LQ_DVOL_FULL:    int   = 15_000_000   # dollar volume above this → full points
LQ_DVOL_MAX_PTS: int   = 8            # max dollar-volume points
LQ_QUAL_MAX_PTS: int   = 4            # max float/mcap quality points
LQ_CONS_MAX_PTS: int   = 3            # max volume-consistency points


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _lerp(val: float, lo: float, hi: float) -> float:
    """Linear 0→1 as val goes from lo to hi, clamped."""
    if hi <= lo:
        return 1.0 if val >= hi else 0.0
    return _clamp((val - lo) / (hi - lo))


# ---------------------------------------------------------------------------
# Sub-score functions
# ---------------------------------------------------------------------------

def _momentum_score(row: dict, *, weights: dict | None = None) -> tuple[float, dict]:
    """
    MomentumScore 0–25.
    RVOL (10) + Change% (8) + MACD (7).
    RVOL sub-linear above ideal — extreme RVOL can signal panic/chase, not edge.
    Change% decays above sweet spot so a 15% gap day isn't rewarded more than 8%.

    weights: optional dict of constant overrides for research sweeps, e.g.
        {"ms_rvol_max_pts": 12, "ms_chg_max_pts": 6}
    """
    w = weights or {}
    ms_rvol_ideal   = w.get("ms_rvol_ideal",   MS_RVOL_IDEAL)
    ms_rvol_max_pts = w.get("ms_rvol_max_pts", MS_RVOL_MAX_PTS)
    ms_chg_hi_pct   = w.get("ms_chg_hi_pct",   MS_CHG_HI_PCT)
    ms_chg_max_pts  = w.get("ms_chg_max_pts",  MS_CHG_MAX_PTS)
    ms_macd_max_pts = w.get("ms_macd_max_pts", MS_MACD_MAX_PTS)

    rvol       = float(row.get("rvol", 0))
    change_pct = float(row.get("change_pct", 0))
    macd       = float(row.get("macd", 0))
    macd_sig   = float(row.get("macd_signal", 0))
    atr_val    = float(row.get("atr", 0.01)) or 0.01

    # RVOL: ramps up to ideal, then slowly diminishes for extremes
    if rvol <= 0:
        rvol_pts = 0.0
    elif rvol <= ms_rvol_ideal:
        rvol_pts = ms_rvol_max_pts * (rvol / ms_rvol_ideal) ** 0.6
    else:
        excess   = _clamp((rvol - ms_rvol_ideal) / ms_rvol_ideal)
        rvol_pts = ms_rvol_max_pts * (1.0 - 0.2 * excess)
    rvol_pts = _clamp(rvol_pts, 0, ms_rvol_max_pts)

    # Change%: full at sweet spot ceiling, decays above it
    if change_pct <= 0:
        chg_pts = 0.0
    elif change_pct <= ms_chg_hi_pct:
        chg_pts = ms_chg_max_pts * (change_pct / ms_chg_hi_pct)
    else:
        chg_pts = ms_chg_max_pts * max(0.4, 1.0 - (change_pct - ms_chg_hi_pct) / 25.0)
    chg_pts = _clamp(chg_pts, 0, ms_chg_max_pts)

    # MACD: normalized to ATR. ±0.10 ATR diff maps to 0–macd_max pts.
    macd_norm = (macd - macd_sig) / atr_val
    macd_pts  = _clamp((macd_norm + 0.1) / 0.2) * ms_macd_max_pts

    total = rvol_pts + chg_pts + macd_pts
    return round(total, 2), {
        "rvol_pts":   round(rvol_pts, 2),
        "change_pts": round(chg_pts,  2),
        "macd_pts":   round(macd_pts, 2),
    }


def _early_entry_score(
    row: dict, close: pd.Series | None, *, weights: dict | None = None
) -> tuple[float, dict]:
    """
    EarlyEntryScore 0–25.
    RSI zone (10) + EMA20 proximity (8) + breakout-from-base (7).
    RSI 52–68 is the sweet spot: confirmed momentum, not yet overbought.
    EMA proximity rewards names that haven't yet run far from their trend.
    BOB (breakout-from-base) rewards price at or near 20-session highs.

    weights: optional dict of constant overrides for research sweeps, e.g.
        {"ee_rsi_lo": 50, "ee_rsi_max_pts": 12, "ee_bob_max_pts": 5}
    Note: bob_pts is 0 when close=None (close series not stored in screener DB).
    """
    w = weights or {}
    ee_rsi_lo      = w.get("ee_rsi_lo",       EE_RSI_LO)
    ee_rsi_hi      = w.get("ee_rsi_hi",       EE_RSI_HI)
    ee_rsi_max_pts = w.get("ee_rsi_max_pts",  EE_RSI_MAX_PTS)
    ee_ema_near    = w.get("ee_ema_near_pct", EE_EMA_NEAR_PCT)
    ee_ema_far     = w.get("ee_ema_far_pct",  EE_EMA_FAR_PCT)
    ee_ema_max_pts = w.get("ee_ema_max_pts",  EE_EMA_MAX_PTS)
    ee_bob_max_pts = w.get("ee_bob_max_pts",  EE_BOB_MAX_PTS)

    rsi_val = float(row.get("rsi", 50))
    ema20   = float(row.get("ema20", 0)) or float(row.get("price", 1))
    price   = float(row.get("price", 1)) or 1.0

    # RSI zone: peak in [ee_rsi_lo, ee_rsi_hi], decays linearly outside
    if ee_rsi_lo <= rsi_val <= ee_rsi_hi:
        rsi_pts = float(ee_rsi_max_pts)
    elif 42 <= rsi_val < ee_rsi_lo:
        rsi_pts = ee_rsi_max_pts * _lerp(rsi_val, 42, ee_rsi_lo)
    elif ee_rsi_hi < rsi_val <= 76:
        rsi_pts = ee_rsi_max_pts * (1.0 - _lerp(rsi_val, ee_rsi_hi, 76))
    else:
        rsi_pts = 0.0

    # EMA20 proximity: the closer to the moving average, the cleaner the entry
    dist_pct = abs(price - ema20) / ema20 * 100 if ema20 > 0 else ee_ema_far
    if dist_pct <= ee_ema_near:
        ema_pts = float(ee_ema_max_pts)
    elif dist_pct <= ee_ema_far:
        ema_pts = ee_ema_max_pts * (1.0 - _lerp(dist_pct, ee_ema_near, ee_ema_far))
    else:
        ema_pts = 0.0

    # Breakout-from-base: is price at or near its 20-session high?
    # Primary: compute from close series.
    # Fallback: use stored high_20d from DB row (populated by screener since Phase 9).
    # If neither is available, bob_pts stays 0 (old rows pre-dating the column).
    bob_pts = 0.0
    hi_20_src = None
    if close is not None and len(close) >= 20:
        hi_20_src = float(close.iloc[-20:].max())
    elif row.get("high_20d") is not None:
        hi_20_src = float(row["high_20d"])

    if hi_20_src is not None and hi_20_src > 0:
        pct_hi = price / hi_20_src
        if pct_hi >= 0.97:      # at or above 20-day high — fresh breakout
            bob_pts = float(ee_bob_max_pts)
        elif pct_hi >= 0.85:
            bob_pts = ee_bob_max_pts * _lerp(pct_hi, 0.85, 0.97)

    total = rsi_pts + ema_pts + bob_pts
    return round(total, 2), {
        "rsi_pts": round(rsi_pts, 2),
        "ema_pts": round(ema_pts, 2),
        "bob_pts": round(bob_pts, 2),
    }


def _extension_risk_score(
    row: dict, close: pd.Series | None, *, weights: dict | None = None
) -> tuple[float, dict]:
    """
    ExtensionRiskScore 0–20 (subtracted). Higher = more dangerous entry.
    RSI overbought (6) + single-day overextension (6) + VWAP distance (5) + 5-day run (3).

    weights: optional dict of constant overrides for research sweeps.
    Fallback: when close=None, uses row["change_5d"] for the 5-day run penalty
    if that key is present (it is stored in screener DB results).
    """
    w = weights or {}
    er_rsi_warn    = w.get("er_rsi_warn",    ER_RSI_WARN)
    er_rsi_hard    = w.get("er_rsi_hard",    ER_RSI_HARD)
    er_rsi_max_pts = w.get("er_rsi_max_pts", ER_RSI_MAX_PTS)
    er_day_warn    = w.get("er_day_warn",    ER_DAY_WARN)
    er_day_hard    = w.get("er_day_hard",    ER_DAY_HARD)
    er_day_max_pts = w.get("er_day_max_pts", ER_DAY_MAX_PTS)
    er_vwap_warn   = w.get("er_vwap_warn",   ER_VWAP_WARN)
    er_vwap_hard   = w.get("er_vwap_hard",   ER_VWAP_HARD)
    er_vwap_max    = w.get("er_vwap_max_pts",ER_VWAP_MAX_PTS)
    er_5d_warn     = w.get("er_5d_warn",     ER_5D_WARN)
    er_5d_hard     = w.get("er_5d_hard",     ER_5D_HARD)
    er_5d_max_pts  = w.get("er_5d_max_pts",  ER_5D_MAX_PTS)

    rsi_val    = float(row.get("rsi", 50))
    change_pct = float(row.get("change_pct", 0))
    price      = float(row.get("price", 1))
    vwap       = float(row.get("vwap", price)) or price
    atr_val    = float(row.get("atr", 0.01)) or 0.01

    # RSI overbought penalty
    rsi_pen = er_rsi_max_pts * _lerp(rsi_val, er_rsi_warn, er_rsi_hard)

    # Single-day overextension penalty (absolute move)
    day_pen = er_day_max_pts * _lerp(abs(change_pct), er_day_warn, er_day_hard)

    # VWAP distance penalty in ATR multiples
    vwap_dist = abs(price - vwap) / atr_val
    vwap_pen  = er_vwap_max * _lerp(vwap_dist, er_vwap_warn, er_vwap_hard)

    # 5-session cumulative run penalty.
    # Primary: compute from close series. Fallback: use stored change_5d from DB row.
    run5_pen  = 0.0
    if close is not None and len(close) >= 6:
        change_5d = (float(close.iloc[-1]) / float(close.iloc[-6]) - 1) * 100
        run5_pen  = er_5d_max_pts * _lerp(abs(change_5d), er_5d_warn, er_5d_hard)
    elif row.get("change_5d") is not None:
        change_5d = float(row["change_5d"])
        run5_pen  = er_5d_max_pts * _lerp(abs(change_5d), er_5d_warn, er_5d_hard)

    total = rsi_pen + day_pen + vwap_pen + run5_pen
    return round(_clamp(total, 0, 20), 2), {
        "rsi_pen":  round(rsi_pen,  2),
        "day_pen":  round(day_pen,  2),
        "vwap_pen": round(vwap_pen, 2),
        "run5_pen": round(run5_pen, 2),
    }


def _liquidity_score(
    row: dict, data: pd.DataFrame | None, *, weights: dict | None = None
) -> tuple[float, dict]:
    """
    LiquidityQualityScore 0–15.
    Dollar volume (8) + float/mcap quality tier (4) + volume consistency (3).
    Mid-cap scores highest on quality — cleaner trends, less manipulation risk.
    """
    w = weights or {}
    lq_dvol_min     = w.get("lq_dvol_min",     LQ_DVOL_MIN)
    lq_dvol_full    = w.get("lq_dvol_full",    LQ_DVOL_FULL)
    lq_dvol_max_pts = w.get("lq_dvol_max_pts", LQ_DVOL_MAX_PTS)
    lq_qual_max_pts = w.get("lq_qual_max_pts", LQ_QUAL_MAX_PTS)
    lq_cons_max_pts = w.get("lq_cons_max_pts", LQ_CONS_MAX_PTS)

    price      = float(row.get("price", 0))
    market_cap = row.get("market_cap")

    # Dollar volume — requires OHLCV DataFrame.
    # Fallback: use stored dollar_volume from DB row (populated since Phase 9).
    # If neither is available, dvol stays 0 (old rows pre-dating the column).
    dvol = 0.0
    if data is not None and len(data) > 0:
        dvol = price * float(data["Volume"].iloc[-1])
    elif row.get("dollar_volume") is not None:
        dvol = float(row["dollar_volume"])
    if dvol >= lq_dvol_full:
        dvol_pts = float(lq_dvol_max_pts)
    elif dvol >= lq_dvol_min:
        dvol_pts = lq_dvol_max_pts * _lerp(dvol, lq_dvol_min, lq_dvol_full)
    else:
        dvol_pts = 0.0

    # Float/mcap quality tier
    float_shares = row.get("float_shares")
    if market_cap:
        if market_cap >= 10_000_000_000:    # large cap — valid but not the focus
            qual_pts = 2.0
        elif market_cap >= 2_000_000_000:   # mid cap — cleanest for trend trades
            qual_pts = float(lq_qual_max_pts)
        elif market_cap >= 500_000_000:     # small cap — higher risk, bigger moves
            qual_pts = 3.0
        else:                               # micro cap — added risk, thin market
            qual_pts = 1.0
        if float_shares and float_shares < 10_000_000:   # very low float = spike risk
            qual_pts = max(0.0, qual_pts - 2.0)
    else:
        qual_pts = 2.0  # unknown = neutral

    # Volume consistency — requires OHLCV DataFrame.
    # Fallback: use stored vol_cv (coefficient of variation) from DB row.
    # If neither is available, cons_pts stays 0 (old rows pre-dating the column).
    cons_pts = 0.0
    if data is not None and len(data) >= 11:
        vols = data["Volume"].iloc[-11:-1]
        mean = vols.mean()
        if mean > 0:
            cv       = vols.std() / mean
            cons_pts = lq_cons_max_pts * max(0.0, 1.0 - float(cv))
    elif row.get("vol_cv") is not None:
        cv       = float(row["vol_cv"])
        cons_pts = lq_cons_max_pts * max(0.0, 1.0 - cv)

    total = dvol_pts + qual_pts + cons_pts
    return round(_clamp(total, 0, 15), 2), {
        "dvol_pts": round(dvol_pts, 2),
        "qual_pts": round(qual_pts, 2),
        "cons_pts": round(cons_pts, 2),
    }


def _news_catalyst_score(_row: dict) -> tuple[float, dict]:
    """NewsCatalystScore 0–15. Stubbed pending news integration."""
    return 0.0, {"note": "stub — no news source connected"}


_BEARISH_LABELS: dict[str, str] = {
    "Overextended":           "Extended downside move",
    "Strong but extended":    "Strong downside setup",
    "Early breakout":         "Bearish breakdown",
    "Emerging momentum":      "Emerging weakness",
    "Momentum watchlist":     "Bearish watchlist",
    "Avoid":                  "Avoid",
    "Low quality / illiquid": "Low quality / illiquid",
}


def _setup_type(ms: float, ee: float, er: float, lq: float,
                rsi: float, change_5d: float,
                change_pct: float = 0.0, direction: str = "long") -> str:
    """
    Derive a human label from sub-score geometry.
    Order matters — stronger disqualifiers checked first.

    change_pct is today's single-day move. A move >= 15% is treated as
    extended regardless of what ER scores, because the 1-year cumulative
    VWAP used in ER can understate intraday extension for recently beaten-
    down stocks.

    direction: "long" | "short" | "neutral" — bearish setups get bearish labels.
    """
    if lq <= 3:
        label = "Low quality / illiquid"
    elif er >= 15:
        label = "Overextended"
    elif abs(change_pct) >= 15.0 and ms >= 10:
        label = "Strong but extended"
    elif ms >= 13 and er >= 8:
        label = "Strong but extended"
    elif ms >= 17 and ee >= 15 and er <= 7:
        label = "Early breakout"
    elif ms >= 13 and ee >= 11 and er <= 9:
        label = "Emerging momentum"
    elif (ms + ee - er) >= 20:
        label = "Emerging momentum"
    elif (ms + ee - er) >= 10:
        label = "Momentum watchlist"
    else:
        label = "Avoid"

    if direction == "short":
        return _BEARISH_LABELS.get(label, label)
    return label


def _build_rationale(row: dict, ms: float, ee: float, er: float,
                     setup_type: str, change_5d: float) -> str:
    """One-line explanation for why this ticker ranked where it did."""
    price   = float(row.get("price", 0))
    rvol    = float(row.get("rvol", 0))
    rsi_val = float(row.get("rsi", 50))
    chg     = float(row.get("change_pct", 0))
    ema20   = float(row.get("ema20", 0)) or price
    dist    = (price - ema20) / ema20 * 100 if ema20 > 0 else 0.0

    parts = []
    if setup_type == "Early breakout":
        parts.append(f"Breaking out on {rvol:.1f}x RVOL")
    elif setup_type == "Emerging momentum":
        parts.append(f"Building momentum, {rvol:.1f}x RVOL")
    elif setup_type == "Strong but extended":
        parts.append(f"Strong move but stretched")
    elif setup_type == "Overextended":
        parts.append(f"Likely too late — {change_5d:+.0f}% 5-day run")
    elif setup_type == "Momentum watchlist":
        parts.append(f"Needs confirmation")
    elif setup_type == "Low quality / illiquid":
        parts.append(f"Thin liquidity, elevated risk")

    parts.append(f"RSI {rsi_val:.0f}")
    if abs(dist) > 8:
        parts.append(f"{dist:+.0f}% from EMA20")
    if er >= 10:
        parts.append("elevated chase risk")

    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_tradescore(
    row: dict,
    close: pd.Series | None = None,
    data: pd.DataFrame | None = None,
    *,
    weights: dict | None = None,
) -> dict:
    """
    FinalTradeScore = MomentumScore + EarlyEntryScore + LiquidityScore
                      + NewsCatalystScore - ExtensionRiskScore

    Practical range 0–65. Negative values clipped to 0.
    Sub-scores are also returned for display and tuning.

    Args:
        row:     screener result dict containing price, rvol, rsi, ema20,
                 ema9, atr, macd, macd_signal, vwap, change_pct, market_cap,
                 float_shares, change_5d (stored in DB), etc.
        close:   pd.Series of recent close prices (needed for BOB and 5d run).
                 When None, BOB=0; run5_pen uses row["change_5d"] if present.
        data:    full OHLCV DataFrame (needed for dollar volume and vol consistency).
                 When None, dvol_pts=0 and cons_pts=0.
        weights: optional dict of constant overrides for research parameter sweeps.
                 Keys are lowercase constant names, e.g. {"ms_rvol_max_pts": 12}.
                 Unspecified keys use the module-level defaults.

    Returns dict with keys:
        score, momentum_score, early_entry, extension_risk, liquidity,
        news_catalyst, direction, setup_type, rationale, change_5d,
        conviction, components
    """
    ms_val,  ms_det  = _momentum_score(row, weights=weights)
    ee_val,  ee_det  = _early_entry_score(row, close, weights=weights)
    er_val,  er_det  = _extension_risk_score(row, close, weights=weights)
    lq_val,  lq_det  = _liquidity_score(row, data, weights=weights)
    nc_val,  nc_det  = _news_catalyst_score(row)

    final = round(max(0.0, ms_val + ee_val + lq_val + nc_val - er_val), 1)

    change_5d = 0.0
    if close is not None and len(close) >= 6:
        change_5d = round((float(close.iloc[-1]) / float(close.iloc[-6]) - 1) * 100, 2)
    elif row.get("change_5d") is not None:
        change_5d = float(row["change_5d"])

    change_pct = float(row.get("change_pct", 0))

    # Direction: long when price above VWAP and short-term EMA above medium-term.
    price = float(row.get("price", 0))
    vwap  = float(row.get("vwap", price)) or price
    ema9  = float(row.get("ema9",  price)) or price
    ema20 = float(row.get("ema20", price)) or price
    if price > vwap and ema9 >= ema20:
        direction = "long"
    elif price < vwap and ema9 <= ema20:
        direction = "short"
    else:
        direction = "neutral"

    setup     = _setup_type(ms_val, ee_val, er_val, lq_val,
                            float(row.get("rsi", 50)), change_5d,
                            change_pct, direction)
    rationale = _build_rationale(row, ms_val, ee_val, er_val, setup, change_5d)

    return {
        "score":           final,
        "momentum_score":  ms_val,
        "early_entry":     ee_val,
        "extension_risk":  er_val,
        "liquidity":       lq_val,
        "news_catalyst":   nc_val,
        "direction":       direction,
        "setup_type":      setup,
        "rationale":       rationale,
        "change_5d":       change_5d,
        "conviction":      setup,
        "components": {
            "momentum":    ms_det,
            "early_entry": ee_det,
            "extension":   er_det,
            "liquidity":   lq_det,
        },
    }


def conviction_label(score: float, setup_type: str = "") -> str:
    """Kept for backwards compat — returns the setup_type label directly."""
    return setup_type if setup_type else (
        "Early breakout"     if score >= 52 else
        "Emerging momentum"  if score >= 35 else
        "Momentum watchlist" if score >= 20 else
        "Avoid"
    )
