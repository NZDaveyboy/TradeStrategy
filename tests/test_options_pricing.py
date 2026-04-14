"""
tests/test_options_pricing.py

Verifies bs_price() and bs_delta() against hand-calculated Black-Scholes values.

Reference values for S=100, K=100, T=1, r=0.05, sigma=0.2:
  d1 = (ln(1) + (0.05 + 0.02) * 1) / (0.2 * 1) = 0.07 / 0.2 = 0.35
  d2 = 0.35 - 0.20 = 0.15
  N(0.35) = 0.63683,  N(0.15) = 0.55962
  call  = 100 * 0.63683 − 100 * exp(−0.05) * 0.55962 ≈ 10.45
  put   = call − S + K * exp(−r*T) ≈ 10.45 − 4.88     ≈ 5.57
  delta_call = N(d1) ≈ 0.637
  delta_put  = N(d1) − 1 ≈ −0.363
"""

import math

import pytest

from options_backtest import bs_delta, bs_price

# Standard test parameters
S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20


# ---------------------------------------------------------------------------
# Call price
# ---------------------------------------------------------------------------

def test_atm_call_price_matches_reference():
    price = bs_price(S, K, T, R, SIGMA, "call")
    assert abs(price - 10.45) < 0.05


def test_otm_call_cheaper_than_atm_call():
    atm = bs_price(100, 100, T, R, SIGMA, "call")
    otm = bs_price(100, 120, T, R, SIGMA, "call")
    assert otm < atm


def test_higher_vol_gives_higher_call_price():
    low  = bs_price(S, K, T, R, sigma=0.10, opt="call")
    high = bs_price(S, K, T, R, sigma=0.30, opt="call")
    assert high > low


def test_longer_expiry_gives_higher_call_price():
    short = bs_price(S, K, T=0.1, r=R, sigma=SIGMA, opt="call")
    long_ = bs_price(S, K, T=1.0, r=R, sigma=SIGMA, opt="call")
    assert long_ > short


# ---------------------------------------------------------------------------
# Put price
# ---------------------------------------------------------------------------

def test_atm_put_price_matches_reference():
    put = bs_price(S, K, T, R, SIGMA, "put")
    assert abs(put - 5.57) < 0.05


def test_put_price_is_positive():
    put = bs_price(S, K, T, R, SIGMA, "put")
    assert put > 0


# ---------------------------------------------------------------------------
# Put-call parity:  call − put = S − K * exp(−r*T)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spot,strike", [
    (100, 100),
    (100, 90),
    (100, 110),
    (50,  50),
])
def test_put_call_parity(spot, strike):
    call = bs_price(spot, strike, T, R, SIGMA, "call")
    put  = bs_price(spot, strike, T, R, SIGMA, "put")
    lhs  = call - put
    rhs  = spot - strike * math.exp(-R * T)
    assert abs(lhs - rhs) < 1e-6


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_call_at_expiry_equals_intrinsic_itm():
    """T=0, ITM call: price = S − K."""
    price = bs_price(110, 100, T=0, r=R, sigma=SIGMA, opt="call")
    assert abs(price - 10.0) < 1e-9


def test_call_at_expiry_equals_zero_otm():
    """T=0, OTM call: price = 0."""
    price = bs_price(90, 100, T=0, r=R, sigma=SIGMA, opt="call")
    assert price == 0.0


def test_put_at_expiry_equals_intrinsic_itm():
    """T=0, ITM put: price = K − S."""
    price = bs_price(90, 100, T=0, r=R, sigma=SIGMA, opt="put")
    assert abs(price - 10.0) < 1e-9


def test_zero_vol_returns_discounted_intrinsic():
    """sigma=0: call = max(S − K*exp(−r*T), 0) ≈ max(S − K, 0) for short T."""
    price = bs_price(110, 100, T=0, r=R, sigma=0, opt="call")
    assert abs(price - 10.0) < 1e-9


# ---------------------------------------------------------------------------
# Delta
# ---------------------------------------------------------------------------

def test_call_delta_matches_reference():
    """N(d1) with d1=0.35 → ≈ 0.637."""
    delta = bs_delta(S, K, T, R, SIGMA, "call")
    assert abs(delta - 0.637) < 0.005


def test_put_delta_is_negative():
    delta = bs_delta(S, K, T, R, SIGMA, "put")
    assert delta < 0


def test_put_delta_matches_reference():
    """N(d1) − 1 with d1=0.35 → ≈ −0.363."""
    delta = bs_delta(S, K, T, R, SIGMA, "put")
    assert abs(delta - (-0.363)) < 0.005


def test_deep_itm_call_delta_approaches_one():
    """Deep ITM: S >> K → delta → 1."""
    delta = bs_delta(200, 100, T, R, SIGMA, "call")
    assert delta > 0.99


def test_deep_otm_call_delta_approaches_zero():
    """Deep OTM: S << K → delta → 0."""
    delta = bs_delta(50, 100, T, R, SIGMA, "call")
    assert delta < 0.01


def test_atm_short_dated_call_delta_near_half():
    """Very short T, ATM → d1 → 0 → N(d1) → 0.5."""
    delta = bs_delta(100, 100, T=0.001, r=R, sigma=SIGMA, opt="call")
    assert 0.45 <= delta <= 0.55


def test_delta_zero_at_expiry():
    """T=0 or sigma=0 → delta = 0.0 (guard clause in bs_delta)."""
    assert bs_delta(S, K, T=0, r=R, sigma=SIGMA) == 0.0
    assert bs_delta(S, K, T=T, r=R, sigma=0)     == 0.0
