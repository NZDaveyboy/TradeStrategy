"""
core/options_math.py — Black-Scholes pricing and Greeks (no scipy).

Pure math, no Streamlit, no I/O. Imported by the Options tab UI and the
Learn tab UI. Implementations match the originals from `app.py` exactly.
"""

from __future__ import annotations

import math


def _ncdf(x: float) -> float:
    """Standard-normal CDF via erf."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def _npdf(x: float) -> float:
    """Standard-normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, opt: str = "call") -> float:
    """Black-Scholes European-option price.

    Args:
      S:     spot price
      K:     strike
      T:     years to expiry (e.g. 30/365)
      r:     risk-free rate (decimal, e.g. 0.045)
      sigma: implied volatility (decimal, e.g. 0.35)
      opt:   "call" or "put"

    Returns the theoretical option price. Returns intrinsic value when
    inputs are degenerate (T<=0, sigma<=0, etc.) — never raises.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0) if opt == "call" else max(K - S, 0)
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if opt == "call":
            return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
        return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)
    except Exception:
        return 0.0


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, opt: str = "call") -> dict:
    """Black-Scholes Greeks. Returns delta, gamma, theta (per day), vega (per 1% IV).
    Returns zero-greeks when inputs are degenerate — never raises."""
    zero = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return zero
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        pdf1  = _npdf(d1)
        gamma = pdf1 / (S * sigma * math.sqrt(T))
        vega  = S * pdf1 * math.sqrt(T) / 100
        if opt == "call":
            delta = _ncdf(d1)
            theta = (-(S * pdf1 * sigma) / (2 * math.sqrt(T))
                     - r * K * math.exp(-r * T) * _ncdf(d2)) / 365
        else:
            delta = _ncdf(d1) - 1
            theta = (-(S * pdf1 * sigma) / (2 * math.sqrt(T))
                     + r * K * math.exp(-r * T) * _ncdf(-d2)) / 365
        return {
            "delta": round(delta, 3),
            "gamma": round(gamma, 5),
            "theta": round(theta, 4),
            "vega":  round(vega, 4),
        }
    except Exception:
        return zero
