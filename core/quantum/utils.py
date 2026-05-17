"""
src/utils.py — Universe loading + validation + common helpers.

Pydantic models enforce the YAML contract: every company has a complete
set of fields with valid score ranges. If the YAML is malformed the
loader raises immediately rather than silently returning bad data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ConfigDict


PACKAGE_DIR   = Path(__file__).resolve().parent
TRADESTRATEGY_ROOT = PACKAGE_DIR.parent.parent
UNIVERSE_YAML = PACKAGE_DIR / "universe.yaml"
OUTPUT_DIR    = TRADESTRATEGY_ROOT / "outputs" / "quantum"


# ---------------------------------------------------------------------------
# Pydantic models — the universe contract
# ---------------------------------------------------------------------------

class Company(BaseModel):
    """One investable name in the universe.

    Scores are integers 1–5; max_weight is a fraction 0–1 used as a hard
    per-name cap in addition to any index-level category caps.
    """
    model_config = ConfigDict(extra="forbid")

    ticker:                 str
    company_name:           str
    quantum_exposure_score: int = Field(ge=1, le=5)
    liquidity_score:        int = Field(ge=1, le=5)
    profitability_score:    int = Field(ge=1, le=5)
    risk_score:             int = Field(ge=1, le=5)
    max_weight:             float = Field(default=1.0, ge=0.0, le=1.0)
    notes:                  str = ""

    # category is set after loading (from the YAML key) — not in the
    # per-company block, so we accept it post-hoc rather than at construct time
    category:               str = ""


class Universe(BaseModel):
    """Whole universe: companies grouped by category + benchmark tickers."""
    model_config = ConfigDict(extra="forbid")

    pure_play_quantum:           list[Company]
    quantum_security_networking: list[Company]
    quantum_enablers:            list[Company]
    benchmarks:                  list[str]

    def all_companies(self) -> list[Company]:
        """Flatten all category lists into one list, with category tagged."""
        out: list[Company] = []
        for cat in ("pure_play_quantum", "quantum_security_networking", "quantum_enablers"):
            for c in getattr(self, cat):
                c.category = cat
                out.append(c)
        return out

    def companies_by_category(self, category: str) -> list[Company]:
        if category == "pure_play_quantum":
            return [self._tag(c, category) for c in self.pure_play_quantum]
        if category == "quantum_security_networking":
            return [self._tag(c, category) for c in self.quantum_security_networking]
        if category == "quantum_enablers":
            return [self._tag(c, category) for c in self.quantum_enablers]
        raise ValueError(f"Unknown category: {category}")

    @staticmethod
    def _tag(c: Company, category: str) -> Company:
        c.category = category
        return c


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_universe(path: Path | None = None) -> Universe:
    """Read + validate the universe YAML. Raises pydantic.ValidationError on bad data."""
    path = path or UNIVERSE_YAML
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Universe(**raw)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str = "quantum_index") -> logging.Logger:
    """Single project logger with reasonable defaults. Idempotent."""
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


# ---------------------------------------------------------------------------
# Rebalance date helpers
# ---------------------------------------------------------------------------

def quarterly_rebalance_dates(start, end, *, frequency: str = "Q") -> list:
    """
    Return rebalance dates at the chosen frequency, clamped to [start, end].

    `frequency`:
        "Q"  — last business day of each calendar quarter (default)
        "M"  — last business day of each calendar month
        "Y"  — last business day of each calendar year
        "W"  — last business day of each calendar week

    Implementation note: pandas date_range with the corresponding freq alias.
    We don't roll forward to the next business day — index returns are
    computed on adjacent close pairs, so quarter-end is a valid rebalance
    anchor even when it falls on a weekend (the previous business day's
    close is used).
    """
    import pandas as pd
    freq_map = {"Q": "QE", "M": "ME", "Y": "YE", "W": "W-FRI"}
    if frequency not in freq_map:
        raise ValueError(f"Unsupported rebalance frequency: {frequency}")
    dates = pd.date_range(start=start, end=end, freq=freq_map[frequency])
    return list(dates)
