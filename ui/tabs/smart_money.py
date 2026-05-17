"""
ui/tabs/smart_money.py — Institutional pile-in scanner tab.

Ranks the latest screener universe by net QoQ institutional inflow ($)
using 13F filings via yfinance. Cross-references each row with the
screener's own TradeScore/direction/setup_type so the user can spot
the intersection of smart-money positioning and a clean technical setup.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import streamlit as st

from ui.data import cached_pile_in_scan
from ui.helpers import fmt_usd_compact


def render(get_conn: Callable, screener_ready: bool) -> None:
    """Render the Smart Money tab.

    Args:
      get_conn:       callable returning a DB connection (no args)
      screener_ready: whether the screener DB has any data yet
    """
    st.subheader("💰 Smart Money — institutional pile-ins")
    st.caption(
        "Scans the latest screener universe and ranks tickers by **net "
        "quarter-over-quarter institutional inflow ($)**. Surfaces names "
        "where the biggest 13F filers are *adding*, not just where they sit. "
        "Cross-references with TradeScore so you can spot the intersection "
        "of smart-money positioning and a clean technical setup."
    )

    st.info(
        "⏱️ **13F filings are delayed ~45 days.** Today's scan reflects "
        "Q1 (Jan-Mar) positioning, filed by mid-May. Treat this as **trailing** "
        "confirmation of where institutions have been positioning — not as "
        "real-time flow."
    )

    if not screener_ready:
        st.warning("No screener data yet. Run `python run.py` first.")
        return

    # Universe = top tickers from latest screener run
    with get_conn() as _conn:
        latest_date = pd.read_sql("SELECT MAX(run_date) AS d FROM results", _conn)["d"].iloc[0]
        universe_df = pd.read_sql(
            "SELECT ticker, tradescore, direction, setup_type, price "
            "FROM results WHERE run_date = ? "
            "ORDER BY COALESCE(tradescore, 0) DESC",
            _conn, params=(latest_date,),
        )

    # Limit to top 250 by tradescore to keep scan time reasonable
    if len(universe_df) > 250:
        universe_df = universe_df.head(250)
        scope_note = f"Top 250 by TradeScore from {len(universe_df)} candidates  •  {latest_date}"
    else:
        scope_note = f"All {len(universe_df)} candidates  •  {latest_date}"
    st.caption(scope_note)

    scan_col1, _scan_col2 = st.columns([1, 4])
    if scan_col1.button("🔍 Scan now", type="primary", width='stretch', key="smart_scan"):
        st.session_state["smart_force_refresh"] = str(pd.Timestamp.now())
    if "smart_force_refresh" in st.session_state:
        scan_id = st.session_state["smart_force_refresh"]
    else:
        scan_id = str(latest_date)

    tickers_key = tuple(universe_df["ticker"].tolist())

    # Try the cache first, only show progress UI if we need to do real work
    try:
        scan_df = cached_pile_in_scan(tickers_key, scan_id)
    except Exception as e:
        st.error(f"Scan failed: {e}")
        scan_df = pd.DataFrame()

    if scan_df.empty:
        with st.spinner(f"Scanning {len(tickers_key)} tickers — ~30s on first run, instant after cache…"):
            progress = st.progress(0.0)

            def _cb(done: int, total: int):
                progress.progress(done / total)

            from core.smart_money import scan_pile_ins
            scan_df = scan_pile_ins(list(tickers_key), progress_cb=_cb)
            progress.empty()

    if scan_df.empty:
        st.warning("No institutional data returned. yfinance may be rate-limiting — try again in a minute.")
        return

    # Join with TradeScore + direction + setup_type for cross-reference
    scan_df = scan_df.merge(universe_df, on="ticker", how="left")

    # Highlight cell — top 5 by net inflow that ALSO pass technicals
    actionable = scan_df[
        (scan_df["net_inflow_usd"] > 0)
        & (scan_df["tradescore"].fillna(0) >= 30)
        & (scan_df["direction"] == "long")
    ].head(5)

    if not actionable.empty:
        st.markdown("### 🎯 Top pile-ins with a clean long setup")
        st.caption(
            "_Institutional adders **and** TradeScore ≥ 30 **and** long direction. "
            "Where two independent signals agree._"
        )
        hot = actionable[["ticker", "net_inflow_usd", "new_positions", "tradescore", "setup_type", "top_adder"]].copy()
        hot["net_inflow_usd"] = hot["net_inflow_usd"].apply(fmt_usd_compact)
        hot.columns = ["Ticker", "Net inflow", "New positions", "TradeScore", "Setup", "Top adder"]
        st.dataframe(
            hot, hide_index=True, width='stretch',
            column_config={
                "Ticker":     st.column_config.TextColumn(width="small"),
                "TradeScore": st.column_config.NumberColumn(format="%.1f"),
                "Top adder":  st.column_config.TextColumn(width="large"),
            },
        )
        st.divider()

    st.markdown("### 📊 All pile-ins ranked by net inflow")
    display = scan_df.copy()
    display["net_inflow_usd"] = display["net_inflow_usd"].apply(fmt_usd_compact)
    display["adds_usd"]       = display["adds_usd"].apply(fmt_usd_compact)
    display["trims_usd"]      = display["trims_usd"].apply(fmt_usd_compact)
    display = display[[
        "ticker", "net_inflow_usd", "new_positions", "adds_count",
        "trims_count", "tradescore", "direction", "setup_type", "top_adder",
    ]]
    display.columns = [
        "Ticker", "Net inflow", "New", "Adders",
        "Trimmers", "TradeScore", "Dir", "Setup", "Top adder",
    ]
    st.dataframe(
        display, hide_index=True, width='stretch',
        column_config={
            "Ticker":     st.column_config.TextColumn(width="small"),
            "TradeScore": st.column_config.NumberColumn(format="%.1f"),
            "Top adder":  st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(
        "_Net inflow = Σ(QoQ % change × position value) across reported 13F holders. "
        "Positive = institutions adding net dollars, negative = trimming. "
        "Cached 24h._"
    )
