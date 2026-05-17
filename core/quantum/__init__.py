"""
core/quantum — Custom quantum technology index inside TradeStrategy.

Three model indexes (Pure Play / Ecosystem / Barbell), conviction
scoring, quarterly rebalancing, full concentration analysis, and a
counterfactual "ex-top-N" view to test how dependent the index is on
its top contributors.

The full project also lives standalone at /Users/davemason/quantum-index-builder/
with a dedicated Streamlit dashboard + CLI. The modules here are the
same code; the TradeStrategy Quantum tab wraps them in the unified app.

Public API:
    load_universe(path)                       → Universe (pydantic)
    fetch_prices(tickers, start, end)         → DataFrame
    IndexBuilder(universe, prices).build_*()  → IndexResult
    run_full_backtest(result, ...)            → dict of analytics
"""

from core.quantum.utils    import load_universe, Universe, Company
from core.quantum.data     import fetch_prices
from core.quantum.index    import IndexBuilder, IndexResult
from core.quantum.backtest import (
    run_full_backtest,
    compute_stats,
    full_attribution,
    concentration_metrics,
    classify_constituents,
)
from core.quantum.signal_backtest import backtest_signals

__all__ = [
    "load_universe", "Universe", "Company",
    "fetch_prices",
    "IndexBuilder", "IndexResult",
    "run_full_backtest", "compute_stats",
    "full_attribution", "concentration_metrics",
    "classify_constituents",
    "backtest_signals",
]
