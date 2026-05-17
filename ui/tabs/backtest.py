"""
ui/tabs/backtest.py — Backtest results tab.

Renders the walk-forward backtest output stored in the local DB. Shows
equity curve, drawdown chart, by-setup statistics, optimisation log, etc.
Pulls everything from `core.backtest_engine` outputs in the screener DB.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import streamlit as st


def render(get_conn: Callable) -> None:
    """Render the Backtest tab.

    Args:
      get_conn: callable returning a DB connection (no args).
    """

    conn = get_conn()
    bt_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='backtest'"
    ).fetchone()
    bt_df = pd.read_sql("SELECT * FROM backtest", conn) if bt_exists else pd.DataFrame()
    conn.close()

    if bt_df.empty:
        st.info("No backtest data yet. Run `python3 backtest_v2.py` to populate.")
    else:
        bt_df = bt_df.dropna(subset=["return_1d"])   # only rows with forward data

        # -------------------------------------------------------------------
        # Top metrics
        # -------------------------------------------------------------------

        total_trades = len(bt_df)
        overall_win  = (bt_df["return_1d"] > 0).sum()
        win_rate     = overall_win / total_trades * 100 if total_trades else 0
        avg_1d       = bt_df["return_1d"].mean()
        avg_5d       = bt_df["return_5d"].mean() if "return_5d" in bt_df else None

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trades analysed",  total_trades)
        m2.metric("Win rate (1d)",    f"{win_rate:.1f}%")
        m3.metric("Avg return (1d)",  f"{avg_1d:+.2f}%")
        if avg_5d is not None:
            m4.metric("Avg return (5d)", f"{avg_5d:+.2f}%")

        st.divider()

        # -------------------------------------------------------------------
        # Return by score
        # -------------------------------------------------------------------

        st.subheader("Does a higher score predict better returns?")

        score_summary = (
            bt_df.groupby("score")
            .agg(
                trades       =("return_1d", "count"),
                avg_1d       =("return_1d", "mean"),
                avg_3d       =("return_3d", "mean"),
                avg_5d       =("return_5d", "mean"),
                win_rate_1d  =("return_1d", lambda x: (x > 0).mean() * 100),
            )
            .round(2)
            .reset_index()
        )
        score_summary.columns = ["Score", "Trades", "Avg 1d %", "Avg 3d %", "Avg 5d %", "Win rate 1d %"]

        st.dataframe(score_summary, width='stretch', hide_index=True)

        chart_data = score_summary.set_index("Score")[["Avg 1d %", "Avg 3d %", "Avg 5d %"]]
        st.bar_chart(chart_data)

        st.divider()

        # -------------------------------------------------------------------
        # Return by strategy
        # -------------------------------------------------------------------

        st.subheader("Return by strategy")

        strat_summary = (
            bt_df.groupby("strategy")
            .agg(
                trades      =("return_1d", "count"),
                avg_1d      =("return_1d", "mean"),
                avg_5d      =("return_5d", "mean"),
                best_1d     =("return_1d", "max"),
                worst_1d    =("return_1d", "min"),
                win_rate    =("return_1d", lambda x: (x > 0).mean() * 100),
            )
            .round(2)
            .reset_index()
        )
        strat_summary.columns = ["Strategy", "Trades", "Avg 1d %", "Avg 5d %", "Best 1d %", "Worst 1d %", "Win rate %"]
        st.dataframe(strat_summary, width='stretch', hide_index=True)

        st.divider()

        # -------------------------------------------------------------------
        # Filters + full trade log
        # -------------------------------------------------------------------

        st.subheader("Trade log")

        bf1, bf2, bf3 = st.columns(3)
        bt_strat  = bf1.selectbox("Strategy", ["All"] + sorted(bt_df["strategy"].unique().tolist()), key="bt_strat")
        bt_asset  = bf2.selectbox("Asset",    ["All", "equity", "crypto"], key="bt_asset")
        bt_score  = bf3.slider("Min score", 0, 4, 0, key="bt_score")

        filtered_bt = bt_df.copy()
        if bt_strat != "All":
            filtered_bt = filtered_bt[filtered_bt["strategy"] == bt_strat]
        if bt_asset != "All":
            filtered_bt = filtered_bt[filtered_bt["asset"] == bt_asset]
        filtered_bt = filtered_bt[filtered_bt["score"] >= bt_score]
        filtered_bt = filtered_bt.sort_values(["run_date", "score"], ascending=[False, False])

        display_cols = ["run_date", "ticker", "strategy", "score",
                        "entry_price", "return_1d", "return_3d", "return_5d", "return_10d"]
        display_cols = [c for c in display_cols if c in filtered_bt.columns]

        st.caption(f"{len(filtered_bt)} trades")
        st.dataframe(
            filtered_bt[display_cols],
            width='stretch',
            hide_index=True,
            column_config={
                "return_1d":  st.column_config.NumberColumn("1d %",  format="%+.2f%%"),
                "return_3d":  st.column_config.NumberColumn("3d %",  format="%+.2f%%"),
                "return_5d":  st.column_config.NumberColumn("5d %",  format="%+.2f%%"),
                "return_10d": st.column_config.NumberColumn("10d %", format="%+.2f%%"),
                "score":      st.column_config.NumberColumn("Score",  format="%d/4"),
            },
        )

    # -----------------------------------------------------------------------
    # Backtest v2 — Strategy Simulation Analytics
    # -----------------------------------------------------------------------

    st.divider()
    st.subheader("Strategy Simulation Analytics (v2)")
    st.caption(
        "Per-ticker results from `python3 backtest_v2.py` — market entry next bar open, "
        "stop-loss from screener, time exit after max hold days."
    )

    from core.analytics import (
        dated_returns_series  as _drs,
        drawdown_series       as _dds,
        equity_curve          as _ec,
        load_v2_data,
        monthly_returns_table as _mrt,
        performance_by_score_bucket,
        portfolio_stats,
        quantstats_tearsheet_html as _qs_html,
        win_rate_by_setup,
    )

    v2_df = load_v2_data()

    if v2_df.empty:
        st.info("No strategy simulation data yet.")
        st.code("python3 backtest_v2.py", language="bash")
    else:
        # ── Key stats row ────────────────────────────────────────────────
        pstats = portfolio_stats(v2_df)

        def _fmt(v, fmt):
            return fmt.format(v) if not (isinstance(v, float) and __import__("math").isnan(v)) else "—"

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total trades",  pstats["total_trades"])
        s2.metric("Win rate",      _fmt(pstats["win_rate"],     "{:.0f}%"))
        s3.metric("Avg return",    _fmt(pstats["avg_return"],   "{:+.1f}%"))
        s4.metric("Expectancy",    _fmt(pstats["expectancy"],   "{:+.2f}%"))

        s5, s6, s7, s8 = st.columns(4)
        s5.metric(
            "Sharpe (ann.)",
            _fmt(pstats["sharpe"], "{:.2f}"),
            help=(
                "Annualised Sharpe of the per-bet return sequence.\n\n"
                f"Assumes {pstats['periods_per_year']:.0f} bets/year and "
                f"{pstats['risk_free_rate']*100:.1f}% risk-free rate. "
                "Formula: (mean_excess / std) × √N."
            ),
        )
        s6.metric(
            "Sortino (ann.)",
            _fmt(pstats["sortino"], "{:.2f}"),
            help=(
                "Annualised Sortino — same as Sharpe but downside-std only "
                "(MAR = 0). Penalises losses, not volatility."
            ),
        )
        s7.metric(
            "Avg per-ticker Sharpe",
            _fmt(pstats["avg_strategy_sharpe"], "{:.2f}"),
            help=(
                "Mean of the per-ticker Sharpe ratios saved by backtest_v2 "
                "(already annualised by backtesting.py at run time). "
                "Useful as a cross-check on the portfolio Sharpe."
            ),
        )
        s8.metric("Max drawdown",  _fmt(pstats["max_drawdown"], "{:.1f}%"))

        st.caption(
            f"_Sharpe / Sortino annualised at {pstats['periods_per_year']:.0f} bets/year "
            f"with {pstats['risk_free_rate']*100:.1f}% risk-free rate. Hover any metric "
            f"for the formula. See **Learn → Lesson 0 → "
            f"9. Backtest analytics** for full methodology._"
        )

        st.divider()

        # ── Time-series tearsheet ─────────────────────────────────────────
        st.subheader("📊 Equity curve + drawdown")
        st.caption(
            "Cumulative product of per-ticker returns ordered by `run_at`. "
            "Drawdown = (equity / cummax) − 1, in percent. "
            "When `run_at` spans more than 30 days, the time axis uses real "
            "timestamps; otherwise it synthesises one bet per trading day "
            "ending today."
        )

        import altair as _alt

        # Build time-series data
        _dated_rets = _drs(v2_df)
        _ec_series  = _ec(v2_df)
        _dd_series  = _dds(v2_df)

        if len(_dated_rets) > 0 and len(_ec_series) > 0:
            # Equity-curve dataframe with same length as dated returns
            _eq_df = pd.DataFrame({
                "Date":   _dated_rets.index,
                "Equity": _ec_series.values,
                "Drawdown_pct": _dd_series.values,
            })
            try:
                _eq_df["Date"] = _eq_df["Date"].dt.tz_localize(None)
            except Exception:
                pass

            _eq_chart = (
                _alt.Chart(_eq_df)
                .mark_line(color="#1f77b4", strokeWidth=2)
                .encode(
                    x=_alt.X("Date:T", axis=_alt.Axis(title=None)),
                    y=_alt.Y("Equity:Q",
                             axis=_alt.Axis(title="Equity (starts at 1.0)"),
                             scale=_alt.Scale(zero=False)),
                    tooltip=[
                        _alt.Tooltip("Date:T"),
                        _alt.Tooltip("Equity:Q", format=".3f"),
                    ],
                )
                .properties(height=240)
            )
            _dd_chart = (
                _alt.Chart(_eq_df)
                .mark_area(color="#d62728", opacity=0.6)
                .encode(
                    x=_alt.X("Date:T", axis=_alt.Axis(title=None)),
                    y=_alt.Y("Drawdown_pct:Q",
                             axis=_alt.Axis(title="Drawdown (%)"),
                             scale=_alt.Scale(domain=[min(_eq_df["Drawdown_pct"].min(), -1), 0])),
                    tooltip=[
                        _alt.Tooltip("Date:T"),
                        _alt.Tooltip("Drawdown_pct:Q", format=".2f", title="Drawdown %"),
                    ],
                )
                .properties(height=180)
            )
            st.altair_chart(_eq_chart, width='stretch')
            st.altair_chart(_dd_chart, width='stretch')
        else:
            st.info("Not enough data to render the equity / drawdown charts.")

        st.divider()

        # ── Monthly returns heatmap ───────────────────────────────────────
        st.subheader("🗓️ Monthly returns heatmap")
        st.caption(
            "Returns compounded within each calendar month. Green = positive, "
            "red = negative. Useful for spotting regime patterns (e.g. seasonal "
            "weakness) that the headline Sharpe number averages away."
        )

        _monthly_df = _mrt(v2_df)
        if not _monthly_df.empty:
            _month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                            "Jul","Aug","Sep","Oct","Nov","Dec"]
            _heatmap = (
                _alt.Chart(_monthly_df)
                .mark_rect(stroke="white", strokeWidth=2)
                .encode(
                    x=_alt.X("Month_label:N",
                             title=None,
                             sort=_month_order),
                    y=_alt.Y("Year:O",
                             title=None,
                             sort="descending"),
                    color=_alt.Color(
                        "Return:Q",
                        title="Return %",
                        scale=_alt.Scale(
                            domain=[-15, 0, 15],
                            range=["#d62728", "#f5f5f5", "#2ca02c"],
                        ),
                    ),
                    tooltip=[
                        _alt.Tooltip("Year:O"),
                        _alt.Tooltip("Month_label:N", title="Month"),
                        _alt.Tooltip("Return:Q", format="+.2f", title="Return %"),
                        _alt.Tooltip("N_obs:Q", title="Observations"),
                    ],
                )
                .properties(height=max(120, 36 * _monthly_df["Year"].nunique() + 60))
            )
            _labels = (
                _alt.Chart(_monthly_df)
                .mark_text(fontSize=11, fontWeight="bold")
                .encode(
                    x=_alt.X("Month_label:N", sort=_month_order),
                    y=_alt.Y("Year:O", sort="descending"),
                    text=_alt.Text("Return:Q", format="+.1f"),
                    color=_alt.condition(
                        "abs(datum.Return) > 10",
                        _alt.value("white"),
                        _alt.value("#222"),
                    ),
                )
            )
            st.altair_chart(_heatmap + _labels, width='stretch')
        else:
            st.info("Not enough dated data for the monthly heatmap.")

        st.divider()

        # ── Returns distribution ──────────────────────────────────────────
        st.subheader("📐 Returns distribution")
        st.caption(
            "Histogram of per-bet returns. Fat right tail = big wins paying for "
            "many small losses (signal-fund profile). Symmetric distribution "
            "centred near zero = noise."
        )
        _ret_df = pd.DataFrame({"Return_pct": (_dated_rets * 100).values})
        if not _ret_df.empty:
            _hist = (
                _alt.Chart(_ret_df)
                .mark_bar(color="#1f77b4")
                .encode(
                    x=_alt.X("Return_pct:Q",
                             bin=_alt.Bin(maxbins=30),
                             axis=_alt.Axis(title="Return per bet (%)")),
                    y=_alt.Y("count():Q", axis=_alt.Axis(title="Number of bets")),
                    tooltip=[
                        _alt.Tooltip("count():Q", title="Bets"),
                    ],
                )
                .properties(height=200)
            )
            _zero = (
                _alt.Chart(pd.DataFrame({"x": [0.0]}))
                .mark_rule(strokeDash=[4, 4], color="#444")
                .encode(x="x:Q")
            )
            st.altair_chart(_hist + _zero, width='stretch')

        # ── Download full QuantStats tearsheet ────────────────────────────
        st.subheader("📥 Full QuantStats tearsheet")
        st.caption(
            "Downloadable HTML report with all the standard QuantStats analytics "
            "(rolling Sharpe, monthly heatmap, drawdown periods, percentile "
            "ranks, etc.) Computed from the same dated returns series shown above."
        )

        if st.button("Generate full tearsheet", type="primary"):
            with st.spinner("Building QuantStats HTML…"):
                _html_bytes = _qs_html(v2_df, title="TradeStrategy Backtest")
            if _html_bytes:
                st.download_button(
                    "⬇️ Download tearsheet (HTML)",
                    data=_html_bytes,
                    file_name=f"tradestrategy_tearsheet_{pd.Timestamp.today():%Y%m%d}.html",
                    mime="text/html",
                )
                st.success("Tearsheet ready — click the button above to download.")
            else:
                st.warning(
                    "Tearsheet generation failed or returned no data. "
                    "Need at least 10 dated observations."
                )

        st.divider()

        # ── Win rate by setup type ────────────────────────────────────────
        by_setup = win_rate_by_setup(v2_df)
        if not by_setup.empty:
            st.subheader("Win rate by setup type")
            st.bar_chart(by_setup.set_index("Setup Type")[["Win Rate %", "Avg Return %"]])
            st.dataframe(by_setup, width='stretch', hide_index=True,
                column_config={
                    "Win Rate %":   st.column_config.NumberColumn(format="%.1f%%"),
                    "Avg Return %": st.column_config.NumberColumn(format="%+.1f%%"),
                })
            st.divider()

        # ── Score bucket performance ──────────────────────────────────────
        by_bucket = performance_by_score_bucket(v2_df)
        if not by_bucket.empty:
            st.subheader("Performance by TradeScore bucket")
            st.dataframe(by_bucket, width='stretch', hide_index=True,
                column_config={
                    "Avg Return %": st.column_config.NumberColumn(format="%+.1f%%"),
                    "Win Rate %":   st.column_config.NumberColumn(format="%.1f%%"),
                    "Avg Trade %":  st.column_config.NumberColumn(format="%+.1f%%"),
                })
            st.divider()

        # ── Per-ticker results table ──────────────────────────────────────
        with st.expander("Per-ticker results table"):
            disp_cols = [c for c in [
                "ticker", "n_signals", "n_trades", "return_pct",
                "win_rate", "sharpe", "max_drawdown", "avg_trade_pct",
                "setup_type", "avg_tradescore", "error",
            ] if c in v2_df.columns]
            st.dataframe(
                v2_df[disp_cols].sort_values("return_pct", ascending=False),
                width='stretch', hide_index=True,
                column_config={
                    "return_pct":    st.column_config.NumberColumn("Return %",    format="%+.1f%%"),
                    "win_rate":      st.column_config.NumberColumn("Win Rate %",  format="%.1f%%"),
                    "avg_trade_pct": st.column_config.NumberColumn("Avg Trade %", format="%+.1f%%"),
                    "max_drawdown":  st.column_config.NumberColumn("Max DD %",    format="%.1f%%"),
                    "sharpe":        st.column_config.NumberColumn("Sharpe",      format="%.2f"),
                    "avg_tradescore":st.column_config.NumberColumn("Avg Score",   format="%.0f"),
                },
            )

    # -----------------------------------------------------------------------
    # Options backtest
    # -----------------------------------------------------------------------

    st.divider()
    st.subheader("Options backtest")
    st.caption("Simulated ATM and OTM call returns on each screener pick using Black-Scholes with 30d realised vol.")

    conn = get_conn()
    bt_opt_exists_bt = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='backtest_options'"
    ).fetchone()
    bt_opt_bt = pd.read_sql("SELECT * FROM backtest_options", conn) if bt_opt_exists_bt else pd.DataFrame()
    conn.close()

    if bt_opt_bt.empty:
        st.info("No options backtest data yet.")
        st.code("python3 options_backtest.py", language="bash")
        st.warning("IV crush is not modelled. Simulated returns assume IV stays constant after entry.")
    else:
        bt_opt_fwd_bt = bt_opt_bt.dropna(subset=["return_1d"])

        # Top metrics
        ob1, ob2, ob3, ob4 = st.columns(4)
        ob1.metric("Simulated trades",  len(bt_opt_fwd_bt))
        ob2.metric("Avg return (1d)",   f"{bt_opt_fwd_bt['return_1d'].mean():+.1f}%")
        ob3.metric("Win rate (1d)",     f"{(bt_opt_fwd_bt['return_1d'] > 0).mean()*100:.0f}%")
        ob4.metric("Avg return (5d)",   f"{bt_opt_fwd_bt['return_5d'].mean():+.1f}%" if "return_5d" in bt_opt_fwd_bt else "—")

        st.warning("IV crush is not modelled — real options bought into high-RVOL moves will underperform these figures.")

        # Return by strategy + score
        st.subheader("Return by strategy and score")
        opt_summary_bt = (
            bt_opt_fwd_bt.groupby(["strategy_name", "screener_score"])
            .agg(
                trades   =("return_1d", "count"),
                avg_1d   =("return_1d", "mean"),
                avg_3d   =("return_3d", "mean"),
                avg_5d   =("return_5d", "mean"),
                win_rate =("return_1d", lambda x: (x > 0).mean() * 100),
            )
            .round(1)
            .reset_index()
        )
        opt_summary_bt.columns = ["Strategy", "Score", "Trades", "Avg 1d %", "Avg 3d %", "Avg 5d %", "Win rate %"]
        st.dataframe(opt_summary_bt, width='stretch', hide_index=True)

        # Equity vs options comparison
        conn = get_conn()
        eq_bt_comp = pd.read_sql(
            "SELECT run_date, ticker, score, return_1d AS eq_1d, return_3d AS eq_3d FROM backtest WHERE return_1d IS NOT NULL",
            conn,
        ) if bt_exists else pd.DataFrame()
        conn.close()

        if not eq_bt_comp.empty and not bt_opt_fwd_bt.empty:
            st.subheader("Equity vs options — same picks")
            atm_bt = bt_opt_fwd_bt[bt_opt_fwd_bt["strategy_name"] == "atm_call_30d"][
                ["run_date", "ticker", "return_1d", "return_3d"]
            ].rename(columns={"return_1d": "opt_1d", "return_3d": "opt_3d"})
            comp_bt = eq_bt_comp.merge(atm_bt, on=["run_date", "ticker"], how="inner")
            if not comp_bt.empty:
                comp_bt_disp = comp_bt[["ticker", "run_date", "score", "eq_1d", "opt_1d", "eq_3d", "opt_3d"]].copy()
                comp_bt_disp.columns = ["Ticker", "Date", "Score", "Equity 1d %", "Option 1d %", "Equity 3d %", "Option 3d %"]
                st.dataframe(comp_bt_disp.sort_values("Option 1d %", ascending=False),
                    width='stretch', hide_index=True,
                    column_config={
                        "Equity 1d %":  st.column_config.NumberColumn(format="%+.1f%%"),
                        "Option 1d %":  st.column_config.NumberColumn(format="%+.1f%%"),
                        "Equity 3d %":  st.column_config.NumberColumn(format="%+.1f%%"),
                        "Option 3d %":  st.column_config.NumberColumn(format="%+.1f%%"),
                    })
                st.caption("Option returns are simulated (Black-Scholes, constant IV).")

        # Full options trade log
        st.subheader("Options trade log")
        obl1, obl2 = st.columns(2)
        filt_strat_bt = obl1.selectbox("Strategy", ["All"] + sorted(bt_opt_fwd_bt["strategy_name"].unique()), key="bt_opt_strat")
        filt_score_bt = obl2.slider("Min score", 0, 4, 0, key="bt_opt_score")
        filtered_opt_bt = bt_opt_fwd_bt.copy()
        if filt_strat_bt != "All":
            filtered_opt_bt = filtered_opt_bt[filtered_opt_bt["strategy_name"] == filt_strat_bt]
        filtered_opt_bt = filtered_opt_bt[filtered_opt_bt["screener_score"] >= filt_score_bt]
        filtered_opt_bt = filtered_opt_bt.sort_values(["run_date", "screener_score"], ascending=[False, False])

        opt_log_cols = ["run_date", "ticker", "screener_score", "strategy_name",
                        "entry_stock_px", "strike", "entry_iv", "entry_opt_px", "entry_delta",
                        "return_1d", "return_3d", "return_5d", "return_10d"]
        opt_log_cols = [c for c in opt_log_cols if c in filtered_opt_bt.columns]
        st.caption(f"{len(filtered_opt_bt)} simulated trades")
        st.dataframe(filtered_opt_bt[opt_log_cols], width='stretch', hide_index=True,
            column_config={
                "return_1d":      st.column_config.NumberColumn("1d %",  format="%+.1f%%"),
                "return_3d":      st.column_config.NumberColumn("3d %",  format="%+.1f%%"),
                "return_5d":      st.column_config.NumberColumn("5d %",  format="%+.1f%%"),
                "return_10d":     st.column_config.NumberColumn("10d %", format="%+.1f%%"),
                "entry_iv":       st.column_config.NumberColumn("IV",    format="%.1%%"),
                "screener_score": st.column_config.NumberColumn("Score", format="%d/4"),
            })
