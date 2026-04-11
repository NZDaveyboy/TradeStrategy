"""
core/setups.py — Trade setup builder for TradeStrategy.

Given a screener row, returns a directional trade setup with
entry, stop, target, and risk/reward ratio.

No Streamlit imports. No FX conversion. USD only.
"""

from dataclasses import dataclass


@dataclass
class TradeSetup:
    direction: str    # "long" | "short" | "neutral"
    entry:     float
    stop:      float
    target:    float
    rr:        float  # reward/risk ratio (always positive)
    rationale: str


def compute_trade_setup(row: dict) -> TradeSetup:
    """
    Build a directional trade setup from a screener row.

    Required keys:
        price, vwap, ema9, ema20, atr, day_high, day_low,
        tradescore, conviction (label string)

    Returns a TradeSetup dataclass.
    """
    price      = float(row["price"])
    vwap       = float(row.get("vwap") or price)
    ema9       = float(row.get("ema9") or price)
    ema20      = float(row.get("ema20") or price)
    atr        = float(row.get("atr") or price * 0.02)
    day_high   = float(row.get("day_high") or price)
    day_low    = float(row.get("day_low") or price)
    tradescore = float(row.get("tradescore") or 0)
    conviction = str(row.get("conviction") or "")

    bullish = price > vwap and ema9 >= ema20
    bearish = price < vwap and ema9 <= ema20

    if bullish:
        direction = "long"
        entry     = round(day_high * 1.001, 4)
        stop      = round(min(vwap, ema20) - 0.35 * atr, 4)
        if entry <= stop:
            return TradeSetup(
                direction="neutral", entry=price, stop=price,
                target=price, rr=0.0,
                rationale="Entry would be at or below stop — no valid setup.",
            )
        target = round(entry + 2.0 * (entry - stop), 4)
        rr     = round((target - entry) / (entry - stop), 2)
        rationale = (
            f"{conviction} long. Price ${price:.2f} above VWAP ${vwap:.2f} "
            f"and EMA9 {'>=' if ema9 >= ema20 else '<'} EMA20. "
            f"Buy breakout above day high ${day_high:.2f}. "
            f"Stop below VWAP/EMA20 support at ${stop:.2f}. "
            f"Target ${target:.2f} (2R). TradeScore {tradescore:.0f}."
        )

    elif bearish:
        direction = "short"
        entry     = round(day_low * 0.999, 4)
        stop      = round(max(vwap, ema20) + 0.35 * atr, 4)
        if entry >= stop:
            return TradeSetup(
                direction="neutral", entry=price, stop=price,
                target=price, rr=0.0,
                rationale="Entry would be at or above stop — no valid setup.",
            )
        target = round(entry - 2.0 * (stop - entry), 4)
        rr     = round((entry - target) / (stop - entry), 2)
        rationale = (
            f"{conviction} short. Price ${price:.2f} below VWAP ${vwap:.2f} "
            f"and EMA9 {'<=' if ema9 <= ema20 else '>'} EMA20. "
            f"Short breakdown below day low ${day_low:.2f}. "
            f"Stop above VWAP/EMA20 resistance at ${stop:.2f}. "
            f"Target ${target:.2f} (2R). TradeScore {tradescore:.0f}."
        )

    else:
        return TradeSetup(
            direction="neutral", entry=price, stop=price,
            target=price, rr=0.0,
            rationale=(
                f"No clear directional edge. Price ${price:.2f}, "
                f"VWAP ${vwap:.2f}, EMA9 ${ema9:.2f}, EMA20 ${ema20:.2f}."
            ),
        )

    return TradeSetup(
        direction=direction,
        entry=entry,
        stop=stop,
        target=target,
        rr=rr,
        rationale=rationale,
    )
