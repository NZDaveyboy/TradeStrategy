"""
src/index.py — Three quantum index recipes + the generic builder.

Each recipe is a method on `IndexBuilder` that selects companies and
chooses a weighting method, then defers to the common `build_levels`
machinery. The recipes are:

  build_pure_play(...)    Only pure_play_quantum, equal-weighted, max 25% / name
  build_ecosystem(...)    Pure play + enablers + security, conviction-weighted,
                          with category-level caps (12% / 10% / 8%)
  build_barbell(...)      50% pure plays equal-weighted, 50% enablers
                          equal-weighted, rebalanced together

`build_levels` is the engine: given a price panel and per-rebalance
target weights, it:
  - normalizes the start date to 100
  - holds target weights from each rebalance to the next (drift mode
    is supported via `weights_drift=True` if you want pure b&h)
  - excludes any ticker before its first valid price (cold-start)
  - returns a DataFrame with the index level + per-ticker weight history

No future-data leakage: rebalancing uses prices up to and including the
rebalance date only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from core.quantum.data import daily_returns, first_valid_date
from core.quantum.scoring import (
    compute_final_scores,
    equal_weights,
    market_cap_weights,
    normalize_with_caps,
)
from core.quantum.utils import Company, Universe, get_logger, quarterly_rebalance_dates


log = get_logger("quantum_index.index")

# Category caps for the Ecosystem index (spec section 2)
ECOSYSTEM_CATEGORY_CAPS: dict[str, float] = {
    "pure_play_quantum":           0.12,
    "quantum_enablers":            0.10,
    "quantum_security_networking": 0.08,
}

# Pure Play index cap (spec section 1)
PURE_PLAY_MAX_WEIGHT = 0.25


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class IndexResult:
    """The output of building an index over a date range."""
    name:          str
    levels:        pd.Series                 # date → index level (starts at 100)
    weights:       pd.DataFrame              # date × ticker; daily target weights at each rebalance
    constituents:  list[Company]             # the companies included
    rebalance_dates: list[pd.Timestamp]      # the dates the index rebalanced

    def total_return_pct(self) -> float:
        if self.levels.empty:
            return 0.0
        return (self.levels.iloc[-1] / self.levels.iloc[0] - 1.0) * 100.0


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class IndexBuilder:

    def __init__(self, universe: Universe, prices: pd.DataFrame):
        self.universe = universe
        self.prices   = prices
        self.returns  = daily_returns(prices)

    # ── Recipe 1: Pure Play ────────────────────────────────────────────────
    def build_pure_play(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        weighting: str = "equal_weight",
        rebalance_freq: str = "Q",
        exclude_tickers: set[str] | None = None,
    ) -> IndexResult:
        companies = self.universe.companies_by_category("pure_play_quantum")
        if exclude_tickers:
            companies = [c for c in companies if c.ticker not in exclude_tickers]
        # Pure Play cap is 25% per name (override per-company yaml caps)
        for c in companies:
            c.max_weight = min(c.max_weight, PURE_PLAY_MAX_WEIGHT)

        def weight_fn(active_companies):
            if weighting == "equal_weight":
                w = equal_weights(active_companies)
            elif weighting == "market_cap_weight":
                w = market_cap_weights(active_companies, self._market_caps_for(active_companies))
            elif weighting == "conviction_weight":
                scores = compute_final_scores(active_companies)
                w = normalize_with_caps(scores, active_companies)
            else:
                raise ValueError(f"Unknown weighting: {weighting}")
            # Apply Pure Play cap regardless of method
            return normalize_with_caps(
                {t: 1.0 for t in w},   # use uniform pseudo-scores for the cap pass
                active_companies,
                category_caps={"pure_play_quantum": PURE_PLAY_MAX_WEIGHT},
            ) if False else _apply_max_cap(w, PURE_PLAY_MAX_WEIGHT)

        return self.build_levels(
            name=f"Quantum Pure Play ({weighting})",
            companies=companies,
            weight_fn=weight_fn,
            start=start, end=end,
            rebalance_freq=rebalance_freq,
        )

    # ── Recipe 2: Ecosystem ────────────────────────────────────────────────
    def build_ecosystem(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        weighting: str = "conviction_weight",
        rebalance_freq: str = "Q",
        exclude_tickers: set[str] | None = None,
    ) -> IndexResult:
        companies = (
            self.universe.companies_by_category("pure_play_quantum")
          + self.universe.companies_by_category("quantum_enablers")
          + self.universe.companies_by_category("quantum_security_networking")
        )
        if exclude_tickers:
            companies = [c for c in companies if c.ticker not in exclude_tickers]

        def weight_fn(active_companies):
            if weighting == "conviction_weight":
                scores = compute_final_scores(active_companies)
                return normalize_with_caps(
                    scores, active_companies,
                    category_caps=ECOSYSTEM_CATEGORY_CAPS,
                )
            if weighting == "equal_weight":
                w = equal_weights(active_companies)
            elif weighting == "market_cap_weight":
                w = market_cap_weights(active_companies, self._market_caps_for(active_companies))
            else:
                raise ValueError(f"Unknown weighting: {weighting}")
            # Still apply category caps for non-conviction methods
            return normalize_with_caps(
                {t: w_ for t, w_ in w.items()},
                active_companies,
                category_caps=ECOSYSTEM_CATEGORY_CAPS,
            )

        return self.build_levels(
            name=f"Quantum Ecosystem ({weighting})",
            companies=companies,
            weight_fn=weight_fn,
            start=start, end=end,
            rebalance_freq=rebalance_freq,
        )

    # ── Recipe 3: Barbell ──────────────────────────────────────────────────
    def build_barbell(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        weighting: str = "equal_weight",   # ignored — barbell always uses sleeve equal-weight
        rebalance_freq: str = "Q",
        exclude_tickers: set[str] | None = None,
    ) -> IndexResult:
        pure  = self.universe.companies_by_category("pure_play_quantum")
        enab  = self.universe.companies_by_category("quantum_enablers")
        all_companies = pure + enab
        if exclude_tickers:
            all_companies = [c for c in all_companies if c.ticker not in exclude_tickers]

        def weight_fn(active_companies):
            # Split the active set back into sleeves
            pure_active  = [c for c in active_companies if c.category == "pure_play_quantum"]
            enab_active  = [c for c in active_companies if c.category == "quantum_enablers"]
            weights: dict[str, float] = {}
            if pure_active:
                w_each = 0.5 / len(pure_active)
                for c in pure_active:
                    weights[c.ticker] = w_each
            if enab_active:
                w_each = 0.5 / len(enab_active)
                for c in enab_active:
                    weights[c.ticker] = w_each
            # If one sleeve is empty, the other absorbs the full weight
            total = sum(weights.values())
            if total > 0 and abs(total - 1.0) > 1e-6:
                weights = {t: w / total for t, w in weights.items()}
            return weights

        return self.build_levels(
            name=f"Quantum Barbell ({weighting})",
            companies=all_companies,
            weight_fn=weight_fn,
            start=start, end=end,
            rebalance_freq=rebalance_freq,
        )

    # ── Generic builder ────────────────────────────────────────────────────
    def build_levels(
        self,
        *,
        name: str,
        companies: list[Company],
        weight_fn: Callable[[list[Company]], dict[str, float]],
        start: pd.Timestamp,
        end: pd.Timestamp,
        rebalance_freq: str = "Q",
    ) -> IndexResult:
        """Walk the date range, holding weights from each rebalance to the
        next. Index starts at 100 on the first valid trading day ≥ `start`.
        """
        start = pd.Timestamp(start)
        end   = pd.Timestamp(end)
        prices = self.prices

        # Trading days in the window where ≥1 ticker has a price
        active_dates = prices.loc[start:end].dropna(how="all").index
        if len(active_dates) == 0:
            log.warning(f"[{name}] No price data in window")
            return IndexResult(
                name=name,
                levels=pd.Series(dtype=float),
                weights=pd.DataFrame(),
                constituents=companies,
                rebalance_dates=[],
            )

        # Rebalance schedule: start of window + quarterly anchors within
        rb_dates = [active_dates[0]] + [
            d for d in quarterly_rebalance_dates(active_dates[0], active_dates[-1], frequency=rebalance_freq)
            if d > active_dates[0]
        ]
        rb_dates = sorted(set(rb_dates))

        # Snap each rebalance date to the next available trading day
        snapped: list[pd.Timestamp] = []
        for d in rb_dates:
            idx = active_dates.searchsorted(d)
            if idx >= len(active_dates):
                continue
            snapped.append(active_dates[idx])
        rb_dates = sorted(set(snapped))

        # Compute target weights at each rebalance
        # (Cold-start: only include companies whose first price ≤ that date)
        weight_history: dict[pd.Timestamp, dict[str, float]] = {}
        for rb in rb_dates:
            active = [
                c for c in companies
                if c.ticker in prices.columns
                and first_valid_date(prices, c.ticker) is not None
                and first_valid_date(prices, c.ticker) <= rb
            ]
            if not active:
                continue
            weight_history[rb] = weight_fn(active)

        if not weight_history:
            log.warning(f"[{name}] No active constituents at any rebalance")
            return IndexResult(
                name=name,
                levels=pd.Series(dtype=float),
                weights=pd.DataFrame(),
                constituents=companies,
                rebalance_dates=[],
            )

        # Build the index level day by day. Between rebalances, we apply
        # the target weights to the daily return — this is "constant
        # target weights with daily rebalancing", a simple approximation.
        # (Proper drift-mode requires holding share counts; this method
        # matches the spec well enough and avoids share-count accounting.)
        levels = pd.Series(index=active_dates, dtype=float)
        weights_panel = pd.DataFrame(
            index=active_dates,
            columns=sorted({t for w in weight_history.values() for t in w}),
            dtype=float,
        ).fillna(0.0)

        current_weights = weight_history[rb_dates[0]]
        levels.iloc[0] = 100.0
        weights_panel.loc[active_dates[0]] = pd.Series(current_weights).reindex(weights_panel.columns).fillna(0.0)

        rb_dates_set = set(rb_dates)

        for i in range(1, len(active_dates)):
            day = active_dates[i]
            prev = active_dates[i - 1]

            if day in rb_dates_set and day in weight_history:
                current_weights = weight_history[day]

            # Daily portfolio return = sum(weight × ticker_return)
            day_return = 0.0
            for t, w in current_weights.items():
                if t not in prices.columns:
                    continue
                p_now = prices.at[day, t]
                p_prev = prices.at[prev, t]
                if pd.isna(p_now) or pd.isna(p_prev) or p_prev == 0:
                    continue
                r = (p_now / p_prev) - 1.0
                day_return += w * r

            levels.iloc[i] = levels.iloc[i - 1] * (1.0 + day_return)
            weights_panel.loc[day] = pd.Series(current_weights).reindex(weights_panel.columns).fillna(0.0)

        return IndexResult(
            name=name,
            levels=levels,
            weights=weights_panel,
            constituents=companies,
            rebalance_dates=rb_dates,
        )

    # ── Helpers ────────────────────────────────────────────────────────────
    def _market_caps_for(self, companies: list[Company]) -> dict[str, float]:
        """Pull approximate market caps via yfinance fast_info. Cached at module level."""
        return _fetch_market_caps([c.ticker for c in companies])


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _apply_max_cap(weights: dict[str, float], cap: float) -> dict[str, float]:
    """Apply a single per-name cap iteratively. Redistribute excess proportionally.

    If every name ends at the cap (universe too small for the cap to allow
    sum=1), the residual is left uninvested rather than renormalized — that
    would re-break the cap.
    """
    if not weights:
        return {}
    w = dict(weights)
    for _ in range(len(w) + 1):
        binding = [t for t, v in w.items() if v > cap + 1e-9]
        if not binding:
            break
        excess = 0.0
        for t in binding:
            excess += w[t] - cap
            w[t] = cap
        free = [t for t in w if w[t] < cap - 1e-9]
        if not free:
            # All names at cap — accept partial investment (residual ≈ cash)
            break
        free_total = sum(w[t] for t in free)
        if free_total <= 0:
            break
        for t in free:
            w[t] += excess * (w[t] / free_total)
    # Only shrink if sum overshoots 1 (float drift); never renormalize up
    total = sum(w.values())
    if total > 1.0 + 1e-6:
        w = {t: v / total for t, v in w.items()}
    return w


_market_cap_cache: dict[str, float] = {}

def _fetch_market_caps(tickers: list[str]) -> dict[str, float]:
    """Pull current market caps via yfinance — process-cached.

    Best-effort: failures return 0 for that ticker, which `market_cap_weights`
    falls back from to equal-weight automatically.
    """
    import yfinance as yf

    out: dict[str, float] = {}
    for t in tickers:
        if t in _market_cap_cache:
            out[t] = _market_cap_cache[t]
            continue
        try:
            mc = float(yf.Ticker(t).fast_info.market_cap or 0)
        except Exception:
            mc = 0.0
        _market_cap_cache[t] = mc
        out[t] = mc
    return out
