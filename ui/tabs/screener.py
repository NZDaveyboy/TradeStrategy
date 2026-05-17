"""
ui/tabs/screener.py — Daily screener results table.

Loads the latest run from the screener DB, applies the sidebar filters
(strategy, asset type, score/change/rvol minimums, setup-type allowlist),
renders the candidates as a sortable table with column tooltips.
Also surfaces "Today's Top Picks" cards at the top.
"""

from __future__ import annotations

import json
from typing import Callable

import pandas as pd
import streamlit as st

from core.theme_watchlist import is_on_watchlist
from ui.tabs.dashboard import pick_top_opportunities, show_opportunity_detail


def render(
    *,
    get_conn:           Callable,
    dates:              list,
    selected_date,
    strategy:           str,
    asset_filter:       str,
    min_score:          int,
    min_change:         float,
    min_rvol:           float,
    setup_type_filter:  list,
    setup_types:        list,
    fetch_early_signals: Callable,
) -> None:
    """Render the Screener tab.

    All sidebar state is passed in as keyword args so this module is
    completely decoupled from the global sidebar widgets.
    """
    if not dates or selected_date is None:
        st.info("No screener data yet. Run `python run.py` first.")
    else:
        # ── Early Signals panel ───────────────────────────────────
        with st.expander("⚡ Early Signals — EDGAR filings", expanded=False):
            _all_conn_es = get_conn()
            _es_df = pd.read_sql(
                "SELECT DISTINCT ticker FROM results WHERE run_date = ?",
                _all_conn_es, params=(selected_date,)
            )
            _all_conn_es.close()
            _screener_tickers = tuple(sorted(_es_df["ticker"].tolist())) if not _es_df.empty else ()
            _signals = fetch_early_signals(_screener_tickers)
            if _signals:
                _sig_df = pd.DataFrame(_signals)
                st.dataframe(
                    _sig_df[["filed_at", "ticker", "filing_type", "company", "match_source", "url"]],
                    width='stretch',
                    hide_index=True,
                    column_config={
                        "url": st.column_config.LinkColumn("Filing"),
                        "filed_at": st.column_config.TextColumn("Filed (UTC)"),
                        "match_source": st.column_config.TextColumn("Match"),
                    },
                )
            else:
                st.caption("No new filings from watchlist or screener universe in the last poll.")

        # ── Mode toggle ───────────────────────────────────────────
        mode = st.segmented_control(
            "View mode",
            options=["Advanced", "Simple"],
            default="Advanced",
            label_visibility="collapsed",
        )

        # ── Top Opportunities ─────────────────────────────────────
        _opp_col1, _opp_col2 = st.columns([3, 1])
        with _opp_col1:
            st.markdown("### 🎯 Top Opportunities")
        with _opp_col2:
            dir_filter = st.segmented_control(
                "Direction filter",
                options=["Long", "Short", "Both"],
                default="Long",
                label_visibility="collapsed",
                key="dir_filter",
            )

        _all_conn = get_conn()
        _all_df = pd.read_sql(
            "SELECT * FROM results WHERE run_date = ?",
            _all_conn, params=(selected_date,)
        )
        _all_conn.close()

        _dir_arg = dir_filter.lower() if dir_filter else "long"
        top_df = pick_top_opportunities(_all_df, direction=_dir_arg)

        def _render_opp_cards(cards_df: pd.DataFrame, key_prefix: str):
            if cards_df.empty:
                return False
            card_cols = st.columns(min(4, len(cards_df)))
            score_col = "tradescore" if "tradescore" in cards_df.columns else "score"
            for i, (_, opp) in enumerate(cards_df.iterrows()):
                try:
                    _ex = json.loads(opp.get("explain") or "{}")
                    _conviction = _ex.get("conviction") or opp.get("setup_type") or "—"
                except Exception:
                    _conviction = opp.get("setup_type") or "—"
                with card_cols[i % 4]:
                    _ticker_label = opp['ticker']
                    st.metric(
                        label=f"**{_ticker_label}**",
                        value=f"{opp[score_col]:.0f}",
                        delta=f"{opp['change_pct']:.2f}%",
                    )
                    _caption_parts = [
                        _conviction,
                        f"RVOL {opp.get('rvol', 0):.1f}x",
                        opp.get('strategy', ''),
                    ]
                    if is_on_watchlist(_ticker_label):
                        _caption_parts.append("⚡ Theme watchlist")
                    st.caption("  ·  ".join(p for p in _caption_parts if p))
                    if st.button("Details", key=f"{key_prefix}_{opp['ticker']}_{i}"):
                        show_opportunity_detail(opp.to_dict())
            return True

        if top_df.empty:
            st.info("No setups match the current direction filter. "
                    "Run the screener or switch to Both.")
        elif _dir_arg == "both":
            long_df  = top_df[top_df.get("direction", pd.Series(dtype=str)) == "long"] \
                       if "direction" in top_df.columns else top_df
            short_df = top_df[top_df.get("direction", pd.Series(dtype=str)) == "short"] \
                       if "direction" in top_df.columns else pd.DataFrame()
            if not long_df.empty:
                st.caption("🟢 Long setups")
                _render_opp_cards(long_df, "opp_l")
            if not short_df.empty:
                st.caption("🔴 Short / bearish setups")
                _render_opp_cards(short_df, "opp_s")
            if long_df.empty and short_df.empty:
                _render_opp_cards(top_df, "opp")
        else:
            _render_opp_cards(top_df, "opp")

        st.divider()

        if mode == "Simple":
            st.stop()

        # ── existing filtered table ───────────────────────────────
        conn = get_conn()
        df = pd.read_sql(
            "SELECT * FROM results WHERE run_date = ?", conn, params=(selected_date,)
        )
        conn.close()

        if strategy != "All":
            df = df[df["strategy"] == strategy]
        if asset_filter != "All":
            df = df[df["asset"] == asset_filter]
        # Setup-type filter: only apply when the user has deselected at least one
        # value (empty list = "filter everything out", any-deselected = restrict).
        # Selecting all = no-op.
        if setup_type_filter and len(setup_type_filter) < len(setup_types):
            df = df[df["setup_type"].isin(setup_type_filter)]
        elif not setup_type_filter:
            df = df.iloc[0:0]  # empty selection = show nothing

        df = df[
            (df["score"] >= min_score)
            & (df["change_pct"] >= min_change)
            & (df["rvol"] >= min_rvol)
        ].sort_values(["score", "change_pct"], ascending=False)

        st.caption(f"{len(df)} candidates  •  {selected_date}")

        if df.empty:
            st.info("No stocks match the current filters.")
        else:
            base_cols = [
                "ticker", "score", "strategy", "asset",
                "price", "change_pct", "rvol", "rsi",
                "ema9", "ema20", "ema200",
                "macd", "macd_signal", "vwap",
                "stop_loss", "volume_trend_up",
            ]
            col_config = {
                "score":           st.column_config.NumberColumn("Score", format="%d/4"),
                "change_pct":      st.column_config.NumberColumn("Change %", format="%.2f%%"),
                "rvol":            st.column_config.NumberColumn("RVOL", format="%.2fx"),
                "volume_trend_up": st.column_config.CheckboxColumn("Vol↑"),
            }

            if strategy == "momentum" and "market_cap" in df.columns:
                display_cols = ["ticker", "score", "change_pct", "rvol",
                                "market_cap", "float_shares",
                                "price", "rsi", "stop_loss", "volume_trend_up"]
                col_config["market_cap"]   = st.column_config.NumberColumn("Mkt Cap", format="$%.0f")
                col_config["float_shares"] = st.column_config.NumberColumn("Float", format="%.0f")
            else:
                display_cols = base_cols

            display_cols = [c for c in display_cols if c in df.columns]

            st.dataframe(
                df[display_cols],
                width='stretch',
                hide_index=True,
                column_config=col_config,
            )

            st.subheader("Top movers")
            st.bar_chart(df.set_index("ticker")["change_pct"].head(20))

        with st.expander("Run history"):
            conn = get_conn()
            history = pd.read_sql(
                """
                SELECT run_date,
                       COUNT(*)                  AS candidates,
                       ROUND(AVG(score), 1)       AS avg_score,
                       ROUND(MAX(change_pct), 1)  AS best_change_pct
                FROM results
                GROUP BY run_date
                ORDER BY run_date DESC
                """,
                conn,
            )
            conn.close()
            st.dataframe(history, width='stretch', hide_index=True)
