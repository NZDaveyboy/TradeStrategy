"""
core/research/compare.py — Load and rank sweep results for comparison.

Primary interface:
    load_runs()              — load research_runs into a DataFrame
    rank_runs()              — sort and score by composite metric
    print_comparison_table() — terminal-formatted results table
"""

from __future__ import annotations

import json
import math

import pandas as pd

from core.research.storage import DB_PATH, load_runs


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

_RANK_WEIGHTS: dict[str, float] = {
    "expectancy":   0.40,
    "win_rate":     0.25,
    "sharpe":       0.20,
    "return_pct":   0.10,
    "max_drawdown": 0.05,   # penalised (higher drawdown = worse)
}


def _safe_float(v) -> float | None:
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _normalise(series: pd.Series, invert: bool = False) -> pd.Series:
    """Min-max normalise; invert for metrics where lower is better (drawdown)."""
    lo = series.min()
    hi = series.max()
    if hi == lo:
        return pd.Series([0.5] * len(series), index=series.index)
    norm = (series - lo) / (hi - lo)
    return 1 - norm if invert else norm


def rank_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a composite_score column and sort descending.

    Only rows with at least 1 trade are scored; zero-trade rows receive
    composite_score = 0 and sort to the bottom.
    """
    if df.empty:
        return df

    result = df.copy()

    # Ensure numeric columns
    for col in ["expectancy", "win_rate", "sharpe", "return_pct", "max_drawdown", "n_trades"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    active = result["n_trades"].fillna(0) > 0
    scored = result[active].copy()

    if scored.empty:
        result["composite_score"] = 0.0
        return result.sort_values("composite_score", ascending=False).reset_index(drop=True)

    score = pd.Series(0.0, index=scored.index)
    for col, weight in _RANK_WEIGHTS.items():
        if col not in scored.columns:
            continue
        col_vals = scored[col].fillna(0)
        invert   = col == "max_drawdown"
        score   += weight * _normalise(col_vals, invert=invert)

    result.loc[active, "composite_score"] = score
    result.loc[~active, "composite_score"] = 0.0

    return result.sort_values("composite_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Terminal display
# ---------------------------------------------------------------------------

_TABLE_COLS = [
    ("rank",            "#",          4),
    ("run_label",       "Label",      40),
    ("n_trades",        "Trades",     7),
    ("win_rate",        "Win%",       7),
    ("expectancy",      "Exp%",       7),
    ("sharpe",          "Sharpe",     7),
    ("max_drawdown",    "DD%",        8),
    ("return_pct",      "Ret%",       8),
    ("composite_score", "Score",      6),
]


def print_comparison_table(df: pd.DataFrame, top_n: int = 20) -> None:
    """Print a fixed-width comparison table to stdout."""
    if df.empty:
        print("No results to display.")
        return

    ranked = rank_runs(df)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))

    display = ranked.head(top_n)

    # Header
    header = "  ".join(label.ljust(width) for _, label, width in _TABLE_COLS)
    print(header)
    print("-" * len(header))

    def _fmt(val, col: str) -> str:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "-"
        if col in ("win_rate", "expectancy", "return_pct", "max_drawdown"):
            return f"{float(val):.1f}"
        if col in ("sharpe", "composite_score"):
            return f"{float(val):.2f}"
        if col == "n_trades":
            return str(int(val))
        return str(val)

    for _, row in display.iterrows():
        parts = []
        for col, _, width in _TABLE_COLS:
            val    = row.get(col)
            fmtted = _fmt(val, col)
            parts.append(fmtted.ljust(width))
        print("  ".join(parts))


# ---------------------------------------------------------------------------
# Param diff helper
# ---------------------------------------------------------------------------

def param_diff(df: pd.DataFrame, run_ids: list[int]) -> pd.DataFrame:
    """
    Return only the parameter columns that differ between the selected run_ids.

    Useful for spotting which knobs separate winners from losers.
    """
    if "param_json" not in df.columns:
        return pd.DataFrame()

    subset = df[df["run_id"].isin(run_ids)].copy()
    if subset.empty:
        return pd.DataFrame()

    parsed = subset["param_json"].apply(json.loads)
    param_df = pd.DataFrame(list(parsed), index=subset.index)

    # Only columns where values differ
    varying = [c for c in param_df.columns if param_df[c].nunique() > 1]
    result  = param_df[varying].copy()
    result.insert(0, "run_id", subset["run_id"].values)
    result.insert(1, "run_label", subset["run_label"].values)
    return result.reset_index(drop=True)
