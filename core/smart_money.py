"""
core/smart_money.py — Institutional pile-in scanner.

Aggregates QoQ institutional flow across a universe of tickers using 13F
filings (via yfinance). Surfaces tickers where the biggest reported
holders are *adding* — not just where they sit.

Key signal: net_inflow_usd = sum(pctChange × position_value)
across reported institutional holders. Positive = net adding,
negative = net trimming, large = strong conviction.

Caveat: 13F filings are delayed ~45 days, so this is a *trailing*
signal — good for "where smart money has been positioning",
not for catching moves real-time.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence

import pandas as pd


MAX_WORKERS = 10


def _fetch_one(ticker: str) -> dict | None:
    """Pull institutional_holders for a single ticker. Returns a summary dict
    with net flow stats, or None on failure."""
    import yfinance as yf
    try:
        ih = yf.Ticker(ticker).institutional_holders
    except Exception:
        return None
    if ih is None or ih.empty:
        return None

    # Coerce numeric columns defensively — yfinance occasionally returns strings
    for col in ("pctChange", "Value", "Shares", "pctHeld"):
        if col in ih.columns:
            ih[col] = pd.to_numeric(ih[col], errors="coerce")

    moves = ih.dropna(subset=["pctChange", "Value"])
    if moves.empty:
        return None

    # Dollar-weighted net flow. pctChange capped at +1.0 = brand-new position.
    flows = moves["pctChange"] * moves["Value"]
    net_inflow = float(flows.sum())

    adds_mask  = moves["pctChange"] > 0
    trims_mask = moves["pctChange"] < 0
    new_mask   = moves["pctChange"] >= 0.999

    adds  = flows[adds_mask].sum()
    trims = flows[trims_mask].sum()  # negative

    # Identify the largest single adder (most useful "headline" name)
    top_adder = ""
    if adds_mask.any():
        adders = moves[adds_mask].copy()
        adders["flow"] = adders["pctChange"] * adders["Value"]
        top_row = adders.sort_values("flow", ascending=False).iloc[0]
        top_adder = str(top_row["Holder"])

    return {
        "ticker":           ticker.upper(),
        "net_inflow_usd":   net_inflow,
        "adds_usd":         float(adds),
        "trims_usd":        float(trims),
        "new_positions":    int(new_mask.sum()),
        "adds_count":       int(adds_mask.sum()),
        "trims_count":      int(trims_mask.sum()),
        "reporters":        int(len(moves)),
        "top_adder":        top_adder,
    }


def scan_pile_ins(
    tickers: Sequence[str],
    max_workers: int = MAX_WORKERS,
    progress_cb=None,
) -> pd.DataFrame:
    """Scan a universe of tickers and return a DataFrame ranked by net
    institutional inflow ($).

    Args:
      tickers: list of ticker symbols
      max_workers: parallel yfinance calls
      progress_cb: optional callable(done: int, total: int) called as each
                   ticker completes — useful for Streamlit progress bars

    Returns:
      DataFrame with columns:
        ticker, net_inflow_usd, adds_usd, trims_usd, new_positions,
        adds_count, trims_count, reporters, top_adder
      Sorted by net_inflow_usd descending. Tickers with no data are omitted.
    """
    # De-duplicate, drop crypto (no 13F data)
    universe = [t.upper() for t in dict.fromkeys(tickers) if not t.upper().endswith("-USD")]
    if not universe:
        return pd.DataFrame()

    results: list[dict] = []
    total = len(universe)
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_one, t): t for t in universe}
        for fut in as_completed(futures):
            done += 1
            try:
                r = fut.result()
                if r is not None:
                    results.append(r)
            except Exception:
                pass
            if progress_cb is not None:
                try:
                    progress_cb(done, total)
                except Exception:
                    pass

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values("net_inflow_usd", ascending=False).reset_index(drop=True)
    return df
