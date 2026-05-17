"""
src/scoring.py — Convert per-company scores into capped index weights.

Two public entry points:

  compute_final_scores(companies)
        Returns a {ticker: final_score} dict using the spec formula:
            final = 0.40*quantum + 0.20*liquidity + 0.20*profitability - 0.20*risk
        Scores ≤ 0 are clamped to 0 (effectively excluded from the index).

  normalize_with_caps(scores, companies, category_caps)
        Convert raw scores → weights that:
          (a) sum to 1.0
          (b) respect each company's max_weight cap
          (c) respect category-level caps
        Iterative redistribution: when a cap binds, the excess is
        re-spread proportionally among the still-uncapped names.
"""

from __future__ import annotations

import numpy as np

from core.quantum.utils import Company


# Conviction-weight formula coefficients (per spec)
W_QUANTUM       = 0.40
W_LIQUIDITY     = 0.20
W_PROFITABILITY = 0.20
W_RISK_PENALTY  = 0.20


def compute_final_scores(companies: list[Company]) -> dict[str, float]:
    """Apply the spec conviction-score formula. Negative scores clamped to 0."""
    out: dict[str, float] = {}
    for c in companies:
        raw = (
            W_QUANTUM       * c.quantum_exposure_score
          + W_LIQUIDITY     * c.liquidity_score
          + W_PROFITABILITY * c.profitability_score
          - W_RISK_PENALTY  * c.risk_score
        )
        out[c.ticker] = max(0.0, raw)
    return out


def equal_weights(companies: list[Company]) -> dict[str, float]:
    """1/N weighting — ignores all scores."""
    if not companies:
        return {}
    w = 1.0 / len(companies)
    return {c.ticker: w for c in companies}


def market_cap_weights(
    companies: list[Company],
    market_caps: dict[str, float],
) -> dict[str, float]:
    """Weight by market cap. Tickers without a cap fall back to equal weight.

    `market_caps` is a {ticker: $market_cap} dict — caller fetches these
    (typically via yfinance .fast_info.market_cap or .info["marketCap"]).
    """
    if not companies:
        return {}
    total = sum(market_caps.get(c.ticker, 0) or 0 for c in companies)
    if total <= 0:
        return equal_weights(companies)
    return {c.ticker: (market_caps.get(c.ticker, 0) or 0) / total for c in companies}


def normalize_with_caps(
    scores: dict[str, float],
    companies: list[Company],
    category_caps: dict[str, float] | None = None,
) -> dict[str, float]:
    """Normalize raw scores to weights, then apply caps iteratively.

    Caps applied in order of precedence:
      1. Per-company max_weight (from YAML)
      2. Category cap (from `category_caps` dict, e.g. {"pure_play_quantum": 0.12})

    When a cap binds for company X, X is set to its cap value, removed from
    the redistribution pool, and the residual is re-spread among the
    remaining names proportionally to their scores. Repeat until no caps
    bind. Guaranteed to terminate (at most N iterations).
    """
    category_caps = category_caps or {}

    # Filter to tickers with positive score AND in the company list
    company_by_ticker = {c.ticker: c for c in companies}
    active = {
        t: s for t, s in scores.items()
        if s > 0 and t in company_by_ticker
    }
    if not active:
        return {}

    total = sum(active.values())
    if total <= 0:
        return {}
    weights = {t: s / total for t, s in active.items()}

    # Build the effective per-company cap (min of per-company and category caps)
    def cap_for(ticker: str) -> float:
        c = company_by_ticker[ticker]
        cap = c.max_weight
        if c.category in category_caps:
            cap = min(cap, category_caps[c.category])
        return cap

    # Iterative cap-and-redistribute
    locked: dict[str, float] = {}
    for _ in range(len(active) + 1):
        binding = []
        for t, w in weights.items():
            if t in locked:
                continue
            if w > cap_for(t) + 1e-9:
                binding.append(t)
        if not binding:
            break

        # Lock binding names at their cap
        excess = 0.0
        for t in binding:
            cap = cap_for(t)
            excess += weights[t] - cap
            locked[t] = cap
            weights[t] = cap

        # Redistribute the excess proportionally among unlocked names
        unlocked = [t for t in weights if t not in locked]
        if not unlocked:
            break
        unlocked_total = sum(weights[t] for t in unlocked)
        if unlocked_total <= 0:
            break
        for t in unlocked:
            weights[t] += excess * (weights[t] / unlocked_total)

    # Float-drift renormalisation only — DO NOT renormalize-away-from-1
    # when caps are binding (that would un-do the caps).
    total = sum(weights.values())
    if total > 1.0 + 1e-6:
        # Sum exceeds 1 — gentle proportional shrink (shouldn't happen
        # post-cap, but guard against float drift on small examples)
        weights = {t: w / total for t, w in weights.items()}
    # else: leave sum-may-be-less-than-1; residual is "uninvested" and
    # the index will move at sum(w)×basket speed, which is correct.

    return weights
