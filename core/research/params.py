"""
core/research/params.py — SweepParams dataclass and parameter grid factory.

SweepParams captures every dimension that can be varied in a research sweep:
  - Threshold filters   (tradescore_threshold, min_rvol, rsi_min/max)
  - Backtest params     (stop_multiplier, max_hold_days)
  - Sub-score weights   (ms_*, ee_*, er_*, lq_* matching core/tradescore.py constants)

param_grid(**kwarg_lists) generates the cartesian product of provided value lists,
returning one SweepParams per combination.

Example:
    grid = param_grid(
        tradescore_threshold=[35, 45, 55],
        min_rvol=[1.5, 2.0],
    )
    # → 6 SweepParams objects
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field
from typing import Any

# Default weight grids for --weights CLI mode.
# Three values per component: below default, default, above default.
THRESHOLD_SWEEP_DEFAULTS: dict[str, list] = {
    "tradescore_threshold": [35, 45, 55],
    "min_rvol":             [1.0, 2.0, 3.0],
    "stop_multiplier":      [0.3, 0.5, 1.0],
}

WEIGHT_SWEEP_DEFAULTS: dict[str, list] = {
    "ms_rvol_max_pts": [8,  10, 12],
    "ms_chg_max_pts":  [6,  8,  10],
    "ms_macd_max_pts": [5,  7,  9],
    "ee_rsi_max_pts":  [8,  10, 12],
    "ee_ema_max_pts":  [6,  8,  10],
    "ee_bob_max_pts":  [5,  7,  9],
}


@dataclass
class SweepParams:
    """One complete parameter configuration for a research sweep run."""

    label: str = "default"

    # ── Threshold filters (applied after re-scoring) ──────────────────────
    tradescore_threshold: float = 0.0   # include signals with re-scored tradescore >= this
    min_rvol:             float = 0.0   # include signals with rvol >= this
    rsi_min:              float = 0.0   # include signals with rsi >= this
    rsi_max:              float = 100.0 # include signals with rsi <= this
    direction_filter:     str   = "long"  # "long" | "short" | "both"

    # ── Backtest params ────────────────────────────────────────────────────
    stop_multiplier: float = 0.5   # stop = ema20 ± multiplier × atr (overrides DB stop_loss)
    max_hold_days:   int   = 10

    # ── MomentumScore weight overrides (defaults match core/tradescore.py) ─
    ms_rvol_max_pts: int   = 10
    ms_chg_max_pts:  int   = 8
    ms_macd_max_pts: int   = 7

    # ── EarlyEntryScore weight overrides ───────────────────────────────────
    ee_rsi_max_pts:  int   = 10
    ee_ema_max_pts:  int   = 8
    ee_bob_max_pts:  int   = 7

    # ── ExtensionRisk threshold overrides ──────────────────────────────────
    er_rsi_warn:     int   = 70
    er_rsi_hard:     int   = 82
    er_rsi_max_pts:  int   = 6
    er_day_warn:     float = 10.0
    er_day_hard:     float = 22.0
    er_day_max_pts:  int   = 6
    er_vwap_warn:    float = 1.5
    er_vwap_hard:    float = 4.0
    er_vwap_max_pts: int   = 5
    er_5d_warn:      float = 15.0
    er_5d_hard:      float = 45.0
    er_5d_max_pts:   int   = 3

    # ── LiquidityScore weight overrides ────────────────────────────────────
    lq_dvol_max_pts: int   = 8
    lq_qual_max_pts: int   = 4
    lq_cons_max_pts: int   = 3

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def to_weight_overrides(self) -> dict:
        """Return a weights dict suitable for compute_tradescore(weights=...)."""
        return {
            "ms_rvol_max_pts": self.ms_rvol_max_pts,
            "ms_chg_max_pts":  self.ms_chg_max_pts,
            "ms_macd_max_pts": self.ms_macd_max_pts,
            "ee_rsi_max_pts":  self.ee_rsi_max_pts,
            "ee_ema_max_pts":  self.ee_ema_max_pts,
            "ee_bob_max_pts":  self.ee_bob_max_pts,
            "er_rsi_warn":     self.er_rsi_warn,
            "er_rsi_hard":     self.er_rsi_hard,
            "er_rsi_max_pts":  self.er_rsi_max_pts,
            "er_day_warn":     self.er_day_warn,
            "er_day_hard":     self.er_day_hard,
            "er_day_max_pts":  self.er_day_max_pts,
            "er_vwap_warn":    self.er_vwap_warn,
            "er_vwap_hard":    self.er_vwap_hard,
            "er_vwap_max_pts": self.er_vwap_max_pts,
            "er_5d_warn":      self.er_5d_warn,
            "er_5d_hard":      self.er_5d_hard,
            "er_5d_max_pts":   self.er_5d_max_pts,
            "lq_dvol_max_pts": self.lq_dvol_max_pts,
            "lq_qual_max_pts": self.lq_qual_max_pts,
            "lq_cons_max_pts": self.lq_cons_max_pts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SweepParams":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, s: str) -> "SweepParams":
        return cls.from_dict(json.loads(s))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SweepParams):
            return NotImplemented
        return self.to_json() == other.to_json()

    def __hash__(self) -> int:
        return hash(self.to_json())


def param_grid(**kwarg_lists: list[Any]) -> list[SweepParams]:
    """
    Generate the cartesian product of provided value lists as SweepParams.

    Example:
        param_grid(tradescore_threshold=[35, 45, 55], min_rvol=[1.5, 2.0])
        → [SweepParams(tradescore_threshold=35, min_rvol=1.5),
           SweepParams(tradescore_threshold=35, min_rvol=2.0),
           SweepParams(tradescore_threshold=45, min_rvol=1.5), ...]

    Keys must match SweepParams field names. Unknown keys are silently ignored.
    """
    if not kwarg_lists:
        return [SweepParams()]

    keys   = list(kwarg_lists.keys())
    values = list(kwarg_lists.values())
    result = []
    for combo in itertools.product(*values):
        kwargs = dict(zip(keys, combo))
        parts  = [f"{k}={v}" for k, v in sorted(kwargs.items())]
        kwargs["label"] = "|".join(parts)
        result.append(SweepParams.from_dict(kwargs))
    return result
