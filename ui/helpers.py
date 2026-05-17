"""
ui/helpers.py — Pure display formatters used across multiple tabs.

No Streamlit dependency, no DB access, no global state. Just functions that
take a value and return a string (or similar) for rendering. Safe to import
from any tab module or to unit-test in isolation.
"""

from __future__ import annotations

import pandas as pd


def format_holder_value(v) -> str:
    """Pretty-format an institutional holder's USD position value.
    $242B / $1.5B / $250M. Positive values only — used for 13F position size."""
    if v is None or pd.isna(v):
        return "—"
    v = float(v)
    if v >= 1e12: return f"${v/1e12:.2f}T"
    if v >= 1e9:  return f"${v/1e9:.1f}B"
    if v >= 1e6:  return f"${v/1e6:.0f}M"
    return f"${v:,.0f}"


def qoq_change_label(pct: float) -> str:
    """Render the QoQ % change with a directional indicator. pct is a
    decimal (e.g. 0.025 = +2.5%). +1.0 (100%) means a brand-new position."""
    if pct is None or pd.isna(pct):
        return "—"
    pct = float(pct) * 100
    if pct >= 99.9:    return f"🆕 new"
    if pct >= 10:      return f"🟢 +{pct:.1f}%"
    if pct > 0:        return f"🟢 +{pct:.1f}%"
    if pct == 0:       return "—"
    if pct > -10:      return f"🔴 {pct:.1f}%"
    return f"🔴 {pct:.1f}%"


def regime_label(ctx: dict, key: str) -> str:
    """Coarse 4-state regime label from a market_context dict entry.
    Bullish (above EMA20 + EMA50) / Recovering / Pulling back / Bearish."""
    d = ctx.get(key, {})
    if not d:
        return "—"
    if d.get("above_ema20") and d.get("above_ema50"):
        return "Bullish"
    if d.get("above_ema20"):
        return "Recovering"
    if d.get("above_ema50"):
        return "Pulling back"
    return "Bearish"


def driver_tags(ticker: str, screener_row: dict | None, asset_drivers: dict) -> list[str]:
    """Return macro/thematic driver tags for a ticker.

    Args:
      ticker:        ticker symbol
      screener_row:  optional latest screener row dict (used to derive
                     conviction/momentum tags for equities)
      asset_drivers: dict mapping ticker → tuple of tag strings (the
                     ASSET_DRIVERS constant from ui/data.py)
    """
    if ticker in asset_drivers:
        return list(asset_drivers[ticker])
    if ticker.endswith("-USD"):
        return ["Digital asset", "Risk-on crypto"]
    tags = []
    if screener_row:
        ts        = screener_row.get("tradescore", 0) or 0
        direction = screener_row.get("direction", "")
        setup     = screener_row.get("setup_type", "")
        if ts >= 50:
            tags.append("High conviction")
        if "momentum" in str(setup).lower():
            tags.append("Momentum breakout")
        if "weakness" in str(setup).lower():
            tags.append("Breakdown / short setup")
        if direction == "long":
            tags.append("Bullish trend")
        elif direction == "short":
            tags.append("Bearish trend")
    return tags or ["Equity"]


def fmt_usd_compact(v) -> str:
    """Compact USD formatter with sign handling — used for net-flow values
    that can be negative (e.g. Smart Money pile-in scanner).
    Differs from format_holder_value() in that negative inputs render as
    -$1.5B rather than blank."""
    if v is None or pd.isna(v):
        return "—"
    v = float(v)
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1e12: return f"{sign}${a/1e12:.2f}T"
    if a >= 1e9:  return f"{sign}${a/1e9:.1f}B"
    if a >= 1e6:  return f"{sign}${a/1e6:.0f}M"
    return f"{sign}${a:,.0f}"
