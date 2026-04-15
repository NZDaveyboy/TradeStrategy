"""
core/recommendations.py — Unified recommendation engine.

Single source of truth for options strategy recommendations used by both
the Advice tab and the Options tab. Both tabs call build_recommendation()
and receive an identical Recommendation object.

Key design choices:
  - Extension checked BEFORE strategy assigned (fixes old logic error)
  - Stop uses EMA20 ± STOP_ATR_MULTIPLIER * ATR14, not cumulative VWAP
  - Hard cap MAX_STOP_DISTANCE_PCT: if stop is too far, classify as
    pullback_candidate rather than presenting as actionable
  - IV assessment only when live atm_iv and rv30 are passed in;
    otherwise "unavailable" — never faked
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Tunable constants — adjust here, nowhere else
# ---------------------------------------------------------------------------

STOP_ATR_MULTIPLIER:       float = 0.5    # stop = EMA20 ± (multiplier × ATR)
MAX_STOP_DISTANCE_PCT:     float = 0.12   # if stop > 12% from price → not actionable
EXTENSION_ATR_THRESHOLD:   float = 1.5    # price > EMA20 by more than 1.5 ATRs → extended
EXTENSION_EMA_PCT_THRESHOLD: float = 0.08 # price > EMA20 by more than 8% → extended
IV_EXPENSIVE_THRESHOLD:    float = 1.30   # atm_iv > rv30 × 1.30 → expensive
IV_CHEAP_THRESHOLD:        float = 0.85   # atm_iv < rv30 × 0.85 → cheap

# ---------------------------------------------------------------------------
# Strategy display names — import in app.py so both tabs use the same strings
# ---------------------------------------------------------------------------

STRATEGY_DISPLAY: dict[str, str] = {
    "long_call":        "Long Call",
    "bull_call_spread": "Bull Call Spread",
    "long_put":         "Long Put",
    "bear_put_spread":  "Bear Put Spread",
    "cash_secured_put": "Cash-Secured Put",
    "wait":             "No trade — wait",
}


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    ticker:                  str
    direction:               str             # "long" | "short" | "neutral" | ""
    setup_type:              str             # see below
    recommendation_category: str             # "actionable" | "watchlist" | "avoid"
    strategy_name:           str             # snake_case key into STRATEGY_DISPLAY
    entry_reference:         float           # current price used as entry guide
    invalidation_price:      float | None    # stop level
    target_price:            float | None    # 2R target
    risk_reward:             float | None    # ratio (e.g. 2.0)
    rationale:               str             # specific, not generic
    iv_assessment:           str             # "expensive" | "fair" | "cheap" | "unavailable"
    warnings:                list[str] = field(default_factory=list)
    is_actionable:           bool = False

    # setup_type values:
    #   clean_breakout        — strong trend, IV fair/cheap, stop within cap
    #   emerging_momentum     — trend aligning, lower tradescore
    #   extended              — too far from EMA20/ATR anchor
    #   pullback_candidate    — direction clear but stop too wide to trade now
    #   no_edge               — no clear direction
    #   liquidity_concern     — float/volume flags from screener
    #   crypto_no_options     — crypto ticker, options not applicable


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_extended_bullish(price: float, ema20: float, atr: float) -> bool:
    """True when a bullish setup is over-extended above EMA20."""
    if ema20 <= 0 or atr <= 0:
        return False
    pct_above  = (price - ema20) / ema20
    atr_above  = (price - ema20) / atr
    return pct_above > EXTENSION_EMA_PCT_THRESHOLD or atr_above > EXTENSION_ATR_THRESHOLD


def _is_extended_bearish(price: float, ema20: float, atr: float) -> bool:
    """True when a bearish setup is over-extended below EMA20."""
    if ema20 <= 0 or atr <= 0:
        return False
    pct_below  = (ema20 - price) / ema20
    atr_below  = (ema20 - price) / atr
    return pct_below > EXTENSION_EMA_PCT_THRESHOLD or atr_below > EXTENSION_ATR_THRESHOLD


def _iv_assessment(atm_iv: float | None, rv30: float | None) -> str:
    if atm_iv is None or rv30 is None or rv30 <= 0:
        return "unavailable"
    if atm_iv > rv30 * IV_EXPENSIVE_THRESHOLD:
        return "expensive"
    if atm_iv < rv30 * IV_CHEAP_THRESHOLD:
        return "cheap"
    return "fair"


def _build_rationale(
    direction:        str,
    setup_type:       str,
    price:            float,
    ema20:            float,
    atr:              float,
    invalidation:     float | None,
    target:           float | None,
    stop_dist_pct:    float,
    iv_assessment:    str,
    atm_iv:           float | None,
    rv30:             float | None,
    tradescore:       float,
    rvol:             float,
    warnings:         list[str],
) -> str:
    parts = []

    if setup_type == "extended":
        pct = abs(price - ema20) / ema20 * 100
        atr_dist = abs(price - ema20) / atr if atr > 0 else 0
        side = "above" if direction == "long" else "below"
        parts.append(
            f"Price (${price:.2f}) is {pct:.1f}% {side} EMA20 (${ema20:.2f}) "
            f"and {atr_dist:.1f} ATRs extended."
        )
        if direction == "long":
            parts.append(
                f"Wait for a pullback toward EMA20 (${ema20:.2f}) before taking a directional entry."
            )
        else:
            parts.append(
                f"Avoid chasing the short — wait for a bounce back toward EMA20 (${ema20:.2f})."
            )

    elif setup_type == "pullback_candidate":
        parts.append(
            f"Direction is {direction} but stop distance is {stop_dist_pct*100:.1f}%, "
            f"above the {MAX_STOP_DISTANCE_PCT*100:.0f}% cap."
        )
        parts.append(
            f"Invalidation at ${invalidation:.2f} is too far from ${price:.2f}. "
            "Wait for price to tighten before risking capital."
        )

    elif direction == "long":
        parts.append(
            f"Bullish setup — EMA20 ${ema20:.2f}, stop at ${invalidation:.2f} "
            f"(EMA20 − {STOP_ATR_MULTIPLIER} ATR). "
            f"Risk {stop_dist_pct*100:.1f}%, target ${target:.2f} (2R)."
        )
        if rvol >= 2.0:
            parts.append(f"RVOL {rvol:.1f}× — volume confirming.")
        if tradescore >= 60:
            parts.append(f"TradeScore {tradescore:.0f} — high-conviction setup.")
        if iv_assessment == "expensive":
            parts.append(
                f"IV {atm_iv*100:.0f}% is elevated vs 30d RV {rv30*100:.0f}% — "
                "spread reduces vega cost."
            )
        elif iv_assessment == "cheap":
            parts.append(
                f"IV {atm_iv*100:.0f}% is below 30d RV {rv30*100:.0f}% — "
                "outright call has good value."
            )

    elif direction == "short":
        parts.append(
            f"Bearish setup — EMA20 ${ema20:.2f}, stop at ${invalidation:.2f} "
            f"(EMA20 + {STOP_ATR_MULTIPLIER} ATR). "
            f"Risk {stop_dist_pct*100:.1f}%, target ${target:.2f} (2R)."
        )
        if iv_assessment == "expensive":
            parts.append(
                f"IV {atm_iv*100:.0f}% elevated vs RV {rv30*100:.0f}% — "
                "spread reduces cost of outright put."
            )

    else:
        parts.append("No clear directional edge — wait for a better signal.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_IV_FALLBACK_WARNING = (
    "IV data unavailable — conservative structure applied (spread preferred)."
)


def build_recommendation(
    row: dict,
    *,
    atm_iv:  float | None = None,
    rv30:    float | None = None,
    iv_mode: str = "live",
) -> Recommendation:
    """
    Build a Recommendation from a screener results row dict.

    iv_mode controls how IV data affects strategy selection:

        "live"     — pass atm_iv and rv30; strategy branches on IV state.
                     Expensive IV → spread. Cheap/fair IV → outright.
                     Used by the Options tab which fetches live IV per ticker.

        "fallback" — IV data not available (Advice tab, which would need a
                     yfinance round-trip per ticker in a render loop).
                     iv_assessment is set to "unavailable", strategy defaults
                     to spread (conservative — defined risk regardless of IV).
                     A warning is added so the UI can surface the limitation.

    Same ticker + same iv_mode + same inputs → identical strategy_name.
    Advice tab always uses "fallback". Options tab always uses "live".
    """
    ticker     = str(row.get("ticker") or "")
    price      = float(row.get("price")   or 0)
    ema20      = float(row.get("ema20")   or price)
    atr        = float(row.get("atr")     or price * 0.02)
    direction  = str(row.get("direction") or "").lower().strip()
    tradescore = float(row.get("tradescore") or 0)
    rvol       = float(row.get("rvol") or 1.0)

    # ── Crypto: options not applicable ──────────────────────────────────────
    if ticker.endswith("-USD"):
        return Recommendation(
            ticker=ticker,
            direction=direction,
            setup_type="crypto_no_options",
            recommendation_category="watchlist",
            strategy_name="wait",
            entry_reference=price,
            invalidation_price=None,
            target_price=None,
            risk_reward=None,
            rationale="Options are not available for crypto assets.",
            iv_assessment="unavailable",
            warnings=["Crypto — use directional position sizing instead of options."],
            is_actionable=False,
        )

    # ── Price guard ──────────────────────────────────────────────────────────
    if price <= 0:
        return Recommendation(
            ticker=ticker,
            direction=direction,
            setup_type="no_edge",
            recommendation_category="avoid",
            strategy_name="wait",
            entry_reference=0.0,
            invalidation_price=None,
            target_price=None,
            risk_reward=None,
            rationale="No valid price data available.",
            iv_assessment="unavailable",
            warnings=["Price data missing — cannot build recommendation."],
            is_actionable=False,
        )

    # ── IV assessment ────────────────────────────────────────────────────────
    iv = _iv_assessment(atm_iv, rv30)

    # ── Extension check (runs BEFORE strategy assignment) ────────────────────
    is_bull_extended = (direction == "long")  and _is_extended_bullish(price, ema20, atr)
    is_bear_extended = (direction == "short") and _is_extended_bearish(price, ema20, atr)

    # ── Stop / invalidation calculation ──────────────────────────────────────
    # Primary: EMA20 ± STOP_ATR_MULTIPLIER * ATR14
    # No VWAP anchor — the screener's VWAP is a long-run cumulative average,
    # not a session VWAP, so it produces absurd invalidation levels.
    if direction == "long":
        invalidation = round(ema20 - STOP_ATR_MULTIPLIER * atr, 2)
        stop_dist_pct = (price - invalidation) / price if price > 0 else 1.0
        target_price  = round(price + 2.0 * (price - invalidation), 2)
    elif direction == "short":
        invalidation = round(ema20 + STOP_ATR_MULTIPLIER * atr, 2)
        stop_dist_pct = (invalidation - price) / price if price > 0 else 1.0
        target_price  = round(max(0.01, price - 2.0 * (invalidation - price)), 2)
    else:
        invalidation  = None
        stop_dist_pct = 1.0
        target_price  = None

    # ── Classify setup and assign strategy ───────────────────────────────────
    warnings: list[str] = []

    if direction == "long":
        if is_bull_extended:
            setup_type    = "extended"
            strategy_name = "cash_secured_put"   # income play while waiting for pullback
            rec_cat       = "watchlist"
            is_actionable = False
            ext_pct = (price - ema20) / ema20 * 100
            warnings.append(
                f"Price is {ext_pct:.1f}% above EMA20 (${ema20:.2f}) — "
                "extended. Wait for a pullback before a directional entry."
            )
        elif stop_dist_pct > MAX_STOP_DISTANCE_PCT:
            setup_type    = "pullback_candidate"
            strategy_name = "wait"
            rec_cat       = "watchlist"
            is_actionable = False
            warnings.append(
                f"Stop distance {stop_dist_pct*100:.1f}% exceeds "
                f"{MAX_STOP_DISTANCE_PCT*100:.0f}% cap — risk/reward not viable at current price."
            )
        else:
            if iv_mode == "fallback":
                strategy_name = "bull_call_spread"
                warnings.append(_IV_FALLBACK_WARNING)
            else:
                strategy_name = "bull_call_spread" if iv == "expensive" else "long_call"
            setup_type    = "clean_breakout" if tradescore >= 60 else "emerging_momentum"
            rec_cat       = "actionable"
            is_actionable = True

    elif direction == "short":
        if is_bear_extended:
            setup_type    = "extended"
            strategy_name = "wait"
            rec_cat       = "watchlist"
            is_actionable = False
            ext_pct = (ema20 - price) / ema20 * 100
            warnings.append(
                f"Price is {ext_pct:.1f}% below EMA20 (${ema20:.2f}) — "
                "extended downside. Avoid chasing short."
            )
        elif stop_dist_pct > MAX_STOP_DISTANCE_PCT:
            setup_type    = "pullback_candidate"
            strategy_name = "wait"
            rec_cat       = "watchlist"
            is_actionable = False
            warnings.append(
                f"Stop distance {stop_dist_pct*100:.1f}% exceeds "
                f"{MAX_STOP_DISTANCE_PCT*100:.0f}% cap — wait for better entry."
            )
        else:
            if iv_mode == "fallback":
                strategy_name = "bear_put_spread"
                warnings.append(_IV_FALLBACK_WARNING)
            else:
                strategy_name = "bear_put_spread" if iv == "expensive" else "long_put"
            setup_type    = "clean_breakout"
            rec_cat       = "actionable"
            is_actionable = True

    else:
        setup_type    = "no_edge"
        strategy_name = "wait"
        rec_cat       = "avoid"
        is_actionable = False
        invalidation  = None
        target_price  = None

    # ── Risk/reward ──────────────────────────────────────────────────────────
    if invalidation is not None and target_price is not None and price > 0:
        if direction == "long":
            reward = target_price - price
            risk   = price - invalidation
        else:
            reward = price - target_price
            risk   = invalidation - price
        rr = round(reward / risk, 1) if risk > 0 else None
    else:
        rr = None

    # ── Rationale ────────────────────────────────────────────────────────────
    rationale = _build_rationale(
        direction=direction,
        setup_type=setup_type,
        price=price,
        ema20=ema20,
        atr=atr,
        invalidation=invalidation,
        target=target_price,
        stop_dist_pct=stop_dist_pct,
        iv_assessment=iv,
        atm_iv=atm_iv,
        rv30=rv30,
        tradescore=tradescore,
        rvol=rvol,
        warnings=warnings,
    )

    return Recommendation(
        ticker=ticker,
        direction=direction,
        setup_type=setup_type,
        recommendation_category=rec_cat,
        strategy_name=strategy_name,
        entry_reference=round(price, 2),
        invalidation_price=invalidation,
        target_price=target_price,
        risk_reward=rr,
        rationale=rationale,
        iv_assessment=iv,
        warnings=warnings,
        is_actionable=is_actionable,
    )
