"""
ui/tabs/quantum.py — Custom Quantum Technology Index tab.

Three model indexes (Pure Play / Ecosystem / Barbell) tracking quantum-
exposed public companies, with full backtest, signal analysis, and
constituent classification. Talks only to core.quantum.* — no shared
app state needed.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render() -> None:
    """Render the Quantum tab."""

    st.subheader("⚛️ Custom Quantum Technology Index")
    st.caption(
        "Three model indexes tracking public companies exposed to quantum "
        "computing, networking, security, and the enabling semis/tooling "
        "supply chain. **Internal research / watchlist tool — not a licensed "
        "benchmark, not financial advice.**"
    )

    from core.quantum import (
        load_universe   as _q_load_universe,
        fetch_prices    as _q_fetch_prices,
        IndexBuilder    as _q_IndexBuilder,
        run_full_backtest as _q_run_full_backtest,
        compute_stats    as _q_compute_stats,
        classify_constituents as _q_classify_constituents,
        backtest_signals as _q_backtest_signals,
    )
    import plotly.express     as _q_px
    import plotly.graph_objects as _q_go

    # ── Controls ──────────────────────────────────────────────────────────
    q_c1, q_c2, q_c3, q_c4 = st.columns(4)
    with q_c1:
        q_index_choice = st.selectbox(
            "Index",
            ["Quantum Pure Play", "Quantum Ecosystem", "Quantum Barbell"],
            index=1,
            key="quantum_index_choice",
            help=(
                "**Pure Play** = 4 hardware co's only, equal-weighted, 25% cap. "
                "**Ecosystem** = 20 names, conviction-weighted, category caps. "
                "**Barbell** = 50% pure plays + 50% enablers."
            ),
        )
    with q_c2:
        q_start_default = pd.Timestamp("2023-01-03").date()
        q_today = pd.Timestamp.today().normalize().date()
        q_start_date = st.date_input("Start date", value=q_start_default, key="quantum_start")
    with q_c3:
        q_end_date = st.date_input("End date", value=q_today, key="quantum_end")
    with q_c4:
        q_default_weighting = {
            "Quantum Pure Play": 0,   # equal_weight
            "Quantum Ecosystem": 2,   # conviction_weight
            "Quantum Barbell":   0,   # equal_weight (sleeve-based)
        }[q_index_choice]
        q_weighting = st.selectbox(
            "Weighting",
            ["equal_weight", "market_cap_weight", "conviction_weight"],
            index=q_default_weighting,
            key="quantum_weighting",
        )

    q_rebalance = st.radio(
        "Rebalance frequency",
        ["Q", "M", "Y"],
        format_func=lambda x: {"Q": "Quarterly", "M": "Monthly", "Y": "Annually"}[x],
        index=0,
        horizontal=True,
        key="quantum_rebalance",
    )

    # ── Load universe + prices ────────────────────────────────────────────
    @st.cache_data(ttl=3600)
    def _q_load_universe_cached():
        return _q_load_universe()

    @st.cache_data(ttl=1800)
    def _q_fetch_prices_cached(tickers: tuple, start: str, end: str) -> pd.DataFrame:
        return _q_fetch_prices(list(tickers), start, end)

    q_universe = _q_load_universe_cached()
    q_all_tickers = [c.ticker for c in q_universe.all_companies()] + list(q_universe.benchmarks)
    q_prices = _q_fetch_prices_cached(tuple(q_all_tickers), str(q_start_date), str(q_end_date))

    if q_prices.empty:
        st.error("No quantum price data available. Try a different date range.")
        st.stop()

    q_universe_tickers = [c.ticker for c in q_universe.all_companies()]
    q_universe_prices  = q_prices[[t for t in q_universe_tickers if t in q_prices.columns]]
    q_bench_tickers    = [t for t in q_universe.benchmarks if t in q_prices.columns]
    q_bench_prices     = q_prices[q_bench_tickers]

    # ── Build the chosen index ────────────────────────────────────────────
    q_builder = _q_IndexBuilder(q_universe, q_prices)
    q_start_ts = pd.Timestamp(q_start_date)
    q_end_ts   = pd.Timestamp(q_end_date)

    if q_index_choice == "Quantum Pure Play":
        q_result = q_builder.build_pure_play(q_start_ts, q_end_ts, weighting=q_weighting, rebalance_freq=q_rebalance)
    elif q_index_choice == "Quantum Ecosystem":
        q_result = q_builder.build_ecosystem(q_start_ts, q_end_ts, weighting=q_weighting, rebalance_freq=q_rebalance)
    else:
        q_result = q_builder.build_barbell(q_start_ts, q_end_ts, weighting=q_weighting, rebalance_freq=q_rebalance)

    if q_result.levels.empty:
        st.error(f"{q_index_choice} produced no data in the chosen window.")
        st.stop()

    q_bt = _q_run_full_backtest(q_result, q_universe_prices, q_bench_prices)
    q_stats = q_bt["stats"]

    # ── Headline KPIs — custom compact tiles ──────────────────────────────
    # Subtle grey background (works in both light + dark Streamlit themes
    # because we use a semi-transparent overlay), with more breathing room
    # so values don't feel cramped. Single-line HTML to avoid Streamlit
    # markdown interpreting blank lines as paragraph breaks (which would
    # leak the trailing </div> as literal text).
    def _q_tile(col, label: str, value: str, value_color: str | None = None):
        color_style = f"color:{value_color};" if value_color else ""
        html = (
            f'<div style="background:rgba(128,128,128,0.10);border:1px solid rgba(128,128,128,0.15);'
            f'border-radius:8px;padding:14px 16px;line-height:1.3;min-height:78px">'
            f'<div style="font-size:10px;text-transform:uppercase;letter-spacing:0.06em;opacity:0.65;font-weight:500">{label}</div>'
            f'<div style="font-size:18px;font-weight:600;{color_style}margin-top:6px">{value}</div>'
            f'</div>'
        )
        col.markdown(html, unsafe_allow_html=True)

    def _q_color_pct(v: float, neutral_zero: bool = False) -> str:
        if neutral_zero and v == 0:
            return "#222"
        return "#2ca02c" if v > 0 else "#d62728"

    q_k1, q_k2, q_k3, q_k4, q_k5, q_k6 = st.columns(6)
    _q_tile(q_k1, "Index",        q_index_choice.replace("Quantum ", ""))
    _q_tile(q_k2, "Constituents", str(len(q_result.weights.columns)))
    _q_tile(q_k3, "Total return", f"{q_stats.total_return_pct:+.1f}%", _q_color_pct(q_stats.total_return_pct))
    _q_tile(q_k4, "CAGR",         f"{q_stats.cagr_pct:+.1f}%",         _q_color_pct(q_stats.cagr_pct))
    _q_tile(q_k5, "Annual vol",   f"{q_stats.annual_vol_pct:.1f}%")
    _q_tile(q_k6, "Max DD",       f"{q_stats.max_drawdown_pct:.1f}%",  "#d62728")

    q_k7, q_k8, q_k9 = st.columns([1, 1, 2])
    _q_tile(q_k7, "Sharpe (rf=0)",        f"{q_stats.sharpe:.2f}")
    _q_tile(q_k8, "Max DD duration",      f"{q_stats.max_drawdown_days} days")
    _q_tile(q_k9, "Rebalances",           f"{len(q_result.rebalance_dates)} ({q_rebalance})")

    # ── 📝 Commentary card — dynamic plain-English summary ────────────────
    def _q_build_commentary() -> str:
        """Build a 4-5 line plain-English read of the current index view."""
        bits: list[str] = []
        _conc   = q_bt["concentration"]
        _attrib = q_bt["full_attribution"]

        # Performance one-liner
        bits.append(
            f"**{q_index_choice}** over {q_result.levels.index[0].date()} → "
            f"{q_result.levels.index[-1].date()}: "
            f"total return **{q_stats.total_return_pct:+.1f}%** "
            f"(CAGR {q_stats.cagr_pct:+.1f}%, annual vol {q_stats.annual_vol_pct:.1f}%, "
            f"Sharpe {q_stats.sharpe:.2f}). Max drawdown {q_stats.max_drawdown_pct:.1f}% over "
            f"{q_stats.max_drawdown_days} trading days."
        )

        # Concentration read
        bits.append(
            f"**Concentration**: HHI {_conc['hhi']:.3f} — _{_conc['diversification_label']}_. "
            f"Top 3 names account for {_conc['top3_share_pct']:.1f}% of the gross attribution."
        )

        # Top contributors
        if not _attrib.empty:
            top3 = _attrib.head(3)
            names = ", ".join(
                f"**{r['Ticker']}** ({r['Contribution %']:+.1f}%)"
                for _, r in top3.iterrows()
            )
            bits.append(f"**Top 3 contributors**: {names}.")

        # QTUM correlation (closest commercial benchmark)
        if q_bt["corr_matrix"] is not None and not q_bt["corr_matrix"].empty:
            cm = q_bt["corr_matrix"]
            if "INDEX" in cm.index and "QTUM" in cm.columns:
                qcorr = float(cm.loc["INDEX", "QTUM"])
                tracking = (
                    "tight tracking" if qcorr > 0.85
                    else "moderate tracking" if qcorr > 0.65
                    else "loose tracking"
                )
                bits.append(
                    f"**Benchmark fit**: Correlation with QTUM (Defiance Quantum ETF) "
                    f"is **{qcorr:.2f}** — {tracking}."
                )

        # Plain-English read combining HHI + counterfactual
        if _conc["hhi"] > 0.25:
            bits.append(
                "**Read**: This index is _heavily concentrated_ in its top names. "
                "Most of the return comes from a few stocks; drawdown risk reflects "
                "their volatility, not broad-market exposure."
            )
        elif _conc["hhi"] < 0.15:
            bits.append(
                "**Read**: This index is _genuinely diversified_. Returns are "
                "broad-based across the universe — even dropping the top 3 names "
                "leaves a sizable portion of the return intact."
            )
        else:
            bits.append(
                "**Read**: This index sits between concentrated and diversified — "
                "a few names contribute disproportionately but the index isn't a "
                "single-stock bet."
            )

        return "\n\n".join(bits)

    with st.container(border=True):
        st.markdown("### 📝 Commentary")
        st.markdown(_q_build_commentary())

    # ── Index level vs benchmarks ─────────────────────────────────────────
    st.markdown("### 📈 Index level vs benchmarks (normalised to 100)")
    q_chart_df = pd.DataFrame({q_result.name: q_result.levels})
    for b in q_bench_tickers:
        bs = q_bench_prices[b].reindex(q_result.levels.index).dropna()
        if bs.empty:
            continue
        q_chart_df[b] = (bs / bs.iloc[0] * 100.0).reindex(q_result.levels.index)
    q_fig = _q_px.line(q_chart_df, labels={"value": "Level (start = 100)", "variable": ""})
    q_fig.update_layout(hovermode="x unified", height=380, legend=dict(orientation="h"))
    st.plotly_chart(q_fig, width='stretch')

    # ── Drawdown ──────────────────────────────────────────────────────────
    st.markdown("### 📉 Drawdown")
    q_dd = q_bt["drawdown"]
    q_dd_fig = _q_go.Figure()
    q_dd_fig.add_trace(_q_go.Scatter(
        x=q_dd.index, y=q_dd.values,
        fill="tozeroy", mode="lines",
        line=dict(color="#d62728", width=1.5),
        name="Drawdown %",
        hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}%<extra></extra>",
    ))
    q_dd_fig.update_layout(
        height=220, yaxis_title="Drawdown (%)",
        yaxis=dict(range=[min(q_dd.min(), -1), 0]),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(q_dd_fig, width='stretch')

    # ── Current weights ───────────────────────────────────────────────────
    q_col_a, q_col_b = st.columns(2)
    with q_col_a:
        st.markdown("### 🎯 Current weights")
        if not q_result.weights.empty:
            q_latest = q_result.weights.iloc[-1]
            q_latest = q_latest[q_latest > 1e-6].sort_values(ascending=False) * 100
            q_w_df = q_latest.reset_index()
            q_w_df.columns = ["Ticker", "Weight %"]
            q_cat_lookup  = {c.ticker: c.category     for c in q_universe.all_companies()}
            q_name_lookup = {c.ticker: c.company_name for c in q_universe.all_companies()}
            q_w_df["Company"] = q_w_df["Ticker"].map(q_name_lookup)
            st.dataframe(
                q_w_df[["Ticker", "Company", "Weight %"]],
                width='stretch', hide_index=True,
                column_config={
                    "Ticker":   st.column_config.TextColumn("Ticker",   width="small"),
                    "Company":  st.column_config.TextColumn("Company"),
                    "Weight %": st.column_config.NumberColumn("Wt %", format="%.2f%%", width="small"),
                },
            )
            st.caption(f"_As of {q_result.levels.index[-1].date()}. Sum: {q_w_df['Weight %'].sum():.2f}%._")

    with q_col_b:
        st.markdown("### 🔗 Correlation vs benchmarks")
        if q_bt["corr_matrix"] is not None and not q_bt["corr_matrix"].empty:
            q_cm = q_bt["corr_matrix"]
            q_idx_row = q_cm.loc[["INDEX"]] if "INDEX" in q_cm.index else q_cm
            st.dataframe(q_idx_row.reset_index().rename(columns={"index": ""}),
                         width='stretch', hide_index=True)
            st.caption("_Pearson correlation of daily returns. QTUM is the closest commercial benchmark._")

    # ── Signals — Watch / Buy / Hold / Sell ───────────────────────────────
    st.markdown("### 🎯 Signals — Watch / Buy / Hold / Sell")
    st.caption(
        "Rules-based classification using conviction score, held return, "
        "1m and 3m price momentum, and RSI(14). Not a model — just an "
        "explicit set of rules so you can see why each name is in each "
        "bucket. Hover the Reason column for the trigger."
    )

    q_signals_df = _q_classify_constituents(q_result, q_universe_prices, q_universe)

    if not q_signals_df.empty:
        # Summary tiles for each bucket
        q_counts = q_signals_df["Signal"].value_counts()
        s_b, s_w, s_h, s_s = st.columns(4)
        _q_tile(s_b, "🟢 Buy",    str(int(q_counts.get("BUY",   0))), "#2ca02c")
        _q_tile(s_w, "⏰ Watch",  str(int(q_counts.get("WATCH", 0))), "#ff9f0a")
        _q_tile(s_h, "⚪ Hold",   str(int(q_counts.get("HOLD",  0))))
        _q_tile(s_s, "🔴 Sell",   str(int(q_counts.get("SELL",  0))), "#d62728")

        # Filter — let user focus on one bucket
        q_signal_filter = st.radio(
            "Show",
            ["All", "BUY only", "WATCH only", "HOLD only", "SELL only"],
            index=0,
            horizontal=True,
            key="quantum_signal_filter",
        )
        q_filt = q_signals_df
        if q_signal_filter != "All":
            q_filt = q_signals_df[q_signals_df["Signal"] == q_signal_filter.split()[0]]

        # Compact view: drop Company + Category (Ticker covers identity);
        # short numeric headers so the table fits without horizontal scroll.
        q_filt_compact = q_filt[[
            "Ticker", "Signal", "Reason", "Conviction",
            "Weight %", "Held return %", "1m %", "3m %", "RSI(14)",
        ]]
        st.dataframe(
            q_filt_compact,
            width='stretch', hide_index=True,
            column_config={
                "Ticker":        st.column_config.TextColumn("Ticker", width="small"),
                "Signal":        st.column_config.TextColumn("Signal", width="small"),
                "Reason":        st.column_config.TextColumn("Reason"),  # gets the leftover space
                "Conviction":    st.column_config.NumberColumn("Conv",  format="%.2f", width="small",
                                  help="Conviction score from the YAML scoring formula (≈1.5–3.0 range)."),
                "Weight %":      st.column_config.NumberColumn("Wt %",  format="%.1f%%", width="small"),
                "Held return %": st.column_config.NumberColumn("Held %", format="%+.1f%%", width="small",
                                  help="Price return during the period the index held the position."),
                "1m %":          st.column_config.NumberColumn("1m",    format="%+.1f%%", width="small"),
                "3m %":          st.column_config.NumberColumn("3m",    format="%+.1f%%", width="small"),
                "RSI(14)":       st.column_config.NumberColumn("RSI",   format="%.0f",   width="small",
                                  help=">75 = overbought · <30 = oversold."),
            },
        )

        # ── Signal backtest expander (heavy — collapsed by default) ─────
        with st.expander("🧪 Signal backtest — did past signals actually work?", expanded=False):
            st.caption(
                "**Walk-forward validation** of the classifier rules. At each "
                "monthly sample date in the past, we run the same rules using "
                "only data available at that date (no look-ahead), then measure "
                "what the named stocks actually did over the next 30 / 60 / 90 "
                "trading days. This is the honest answer to 'do these signals "
                "actually work?'. Heavy computation — first run takes ~3 minutes."
            )

            @st.cache_data(ttl=86400, show_spinner=False)
            def _q_run_signal_backtest(start_iso: str, end_iso: str):
                _prices = _q_fetch_prices_cached(tuple(q_all_tickers), start_iso, end_iso)
                if _prices.empty:
                    return None
                return _q_backtest_signals(q_universe, _prices)

            if st.button("Run signal backtest", key="quantum_sigbt_btn"):
                with st.spinner("Walking forward through history (~3 minutes)…"):
                    _q_bt_out = _q_run_signal_backtest(str(q_start_date), str(q_end_date))
                if _q_bt_out is None or _q_bt_out["summary"].empty:
                    st.warning("No backtest data produced for this window.")
                else:
                    _sig_summary = _q_bt_out["summary"]
                    _sig_log     = _q_bt_out["signal_log"]
                    _sample_n    = len(_q_bt_out["sample_dates"])

                    st.markdown(
                        f"**Walked forward through {_sample_n} sample dates · "
                        f"{len(_sig_log)} signal observations.**"
                    )

                    # Summary table — hit rates + mean forward returns
                    st.dataframe(
                        _sig_summary,
                        width='stretch', hide_index=True,
                        column_config={
                            "Signal":           st.column_config.TextColumn("Signal", width="small"),
                            "Lookforward (d)":  st.column_config.NumberColumn("Fwd (d)", format="%d", width="small"),
                            "N samples":        st.column_config.NumberColumn("N", format="%d", width="small"),
                            "Hit rate %":       st.column_config.NumberColumn(
                                "Hit %", format="%.1f%%",
                                help="BUY hit = fwd return > 0%. SELL hit = fwd return < 0%. HOLD = within ±5%.",
                            ),
                            "vs Index hit %":   st.column_config.NumberColumn(
                                "vs Idx %", format="%.1f%%",
                                help="BUY: % of signals that BEAT the index over the forward window. SELL: % that LOST to the index. (Index here = the Quantum Ecosystem itself, computed at each sample.)",
                            ),
                            "Mean fwd %":       st.column_config.NumberColumn("Mean fwd %", format="%+.1f%%"),
                            "Mean vs Index %":  st.column_config.NumberColumn(
                                "Mean excess %", format="%+.1f%%",
                                help="Mean forward return MINUS the index's forward return over the same window. The honest measure of edge — anything above 0 means the signal added value over passive exposure.",
                            ),
                            "Std fwd %":        st.column_config.NumberColumn("Std %", format="%.0f%%", width="small"),
                        },
                    )

                    # Cumulative "follow BUY signals" portfolio (30-day window)
                    _port = _q_bt_out.get("portfolio", {})
                    if 30 in _port and not _port[30].empty:
                        st.markdown("##### Cumulative return: equal-weight BUY portfolio (30d holds)")
                        _port_chart = _port[30].reset_index()
                        _port_chart.columns = ["Date", "Portfolio level"]
                        _pchart = _q_px.line(
                            _port_chart, x="Date", y="Portfolio level",
                            labels={"Portfolio level": "Level (start = 100)"},
                        )
                        _pchart.update_layout(height=280, hovermode="x unified",
                                              margin=dict(t=10, b=10))
                        st.plotly_chart(_pchart, width='stretch')
                        _final = float(_port[30].iloc[-1])
                        st.caption(
                            f"_If you'd equal-weighted every BUY signal and held each "
                            f"for 30 trading days, the portfolio would have ended at "
                            f"**{_final:.1f}** (start 100). Note: assumes no overlap, no "
                            f"costs, and is sensitive to the BUY rule definitions._"
                        )

                    # Honest interpretation guide
                    st.markdown("**How to read this:**")
                    st.markdown("""
    - **Hit % > 50** alone isn't enough — in a bull market everything goes up. **The real test is "vs Idx %" (relative hit rate) and "Mean excess %"**. A signal that beats the index has actual edge.
    - **WATCH signals' Hit % is intentionally low** because we define HOLD/WATCH-correct as "within ±5%" — most stocks move more than that in 30+ days. The real WATCH metric is "Mean excess %" — if it's negative, the WATCH signal correctly avoided underperformers.
    - **Std % is the noise floor.** When excess returns are smaller than ~½ × Std, the signal isn't statistically robust at this sample size — treat as directional only.
    - **Look at the 90-day column for the stickiest signal.** Short-window hits can be luck; if the signal still holds 3 months later, it's more likely real.
    """)

                    # Full signal log download
                    _csv = _sig_log.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download full signal log (CSV)",
                        data=_csv,
                        file_name="quantum_signal_backtest.csv",
                        mime="text/csv",
                    )
            else:
                st.info("Click **Run signal backtest** above to validate the classifier rules against historical data.")

        with st.expander("ℹ️ Signal rules (in priority order)"):
            st.markdown("""
    1. **SELL** — held return ≤ −50% _and_ 1m momentum < +5% (no recovery)
    2. **SELL** — held return < −25% _and_ 1m momentum < −5% (still falling)
    3. **BUY** — 1m momentum > +15% _and_ RSI < 80 (strong fresh strength)
    4. **BUY** — conviction ≥ 2.0 _and_ 1m momentum > +3% _and_ RSI < 75
    5. **BUY** — 3m momentum > +25% _and_ 1m momentum ≥ 0% _and_ RSI < 80 (sustained run)
    6. **WATCH** — RSI > 75 (overbought — wait for pullback)
    7. **WATCH** — conviction ≥ 2.5 _and_ |1m %| < 5% (high-quality, awaiting setup)
    8. **HOLD** — everything else

    The conviction score uses your YAML: `0.4×quantum + 0.2×liquidity + 0.2×profitability − 0.2×risk`.
    Range for the current universe is roughly **1.5 (lowest pure play) to 3.0 (best enabler).**
            """)
    else:
        st.info("No constituents to classify.")

    st.divider()


    # ── Concentration analysis ────────────────────────────────────────────
    st.markdown("### 🔬 Concentration analysis")
    st.caption(
        "How much of the index return came from the top few names? "
        "The ex-top counterfactuals show what the index would have done "
        "if you'd missed them."
    )

    q_conc = q_bt["concentration"]
    qc1, qc2, qc3, qc4, qc5 = st.columns(5)
    qc1.metric("Top 1 share", f"{q_conc['top1_share_pct']:.1f}%",
               help="Share of total |contribution| from the single biggest name.")
    qc2.metric("Top 3 share", f"{q_conc['top3_share_pct']:.1f}%")
    qc3.metric("Top 5 share", f"{q_conc['top5_share_pct']:.1f}%")
    qc4.metric("HHI",         f"{q_conc['hhi']:.3f}",
               help="Herfindahl index. <0.15 diversified · 0.15-0.25 moderate · >0.25 highly concentrated.")
    qc5.metric("Diversification", q_conc["diversification_label"])

    # Counterfactual rebuilds — drop top 1 and top 3
    q_attrib = q_bt["full_attribution"]
    if not q_attrib.empty:
        q_top1 = q_attrib.iloc[0]["Ticker"]
        q_top3 = set(q_attrib.head(3)["Ticker"].tolist())

        q_recipe_fn = {
            "Quantum Pure Play": q_builder.build_pure_play,
            "Quantum Ecosystem": q_builder.build_ecosystem,
            "Quantum Barbell":   q_builder.build_barbell,
        }[q_index_choice]

        q_ex1 = q_recipe_fn(q_start_ts, q_end_ts, weighting=q_weighting,
                            rebalance_freq=q_rebalance, exclude_tickers={q_top1})
        q_ex3 = q_recipe_fn(q_start_ts, q_end_ts, weighting=q_weighting,
                            rebalance_freq=q_rebalance, exclude_tickers=q_top3)

        q_cf_df = pd.DataFrame({
            q_result.name: q_result.levels,
            f"ex top-1 ({q_top1})": q_ex1.levels.reindex(q_result.levels.index),
            f"ex top-3":            q_ex3.levels.reindex(q_result.levels.index),
        })
        q_cf_fig = _q_px.line(q_cf_df, labels={"value": "Level (start = 100)", "variable": ""})
        q_cf_fig.update_layout(hovermode="x unified", height=340, legend=dict(orientation="h"))
        st.plotly_chart(q_cf_fig, width='stretch')

        q_stats_ex1 = _q_compute_stats(q_ex1.levels)
        q_stats_ex3 = _q_compute_stats(q_ex3.levels)
        q_cf_table = pd.DataFrame([
            {"Scenario": "Full index",                              "Total %": q_stats.total_return_pct,    "CAGR %": q_stats.cagr_pct,    "Max DD %": q_stats.max_drawdown_pct},
            {"Scenario": f"Ex top-1 (drop {q_top1})",               "Total %": q_stats_ex1.total_return_pct, "CAGR %": q_stats_ex1.cagr_pct, "Max DD %": q_stats_ex1.max_drawdown_pct},
            {"Scenario": f"Ex top-3 ({', '.join(sorted(q_top3))})", "Total %": q_stats_ex3.total_return_pct, "CAGR %": q_stats_ex3.cagr_pct, "Max DD %": q_stats_ex3.max_drawdown_pct},
        ])
        st.dataframe(
            q_cf_table,
            width='stretch', hide_index=True,
            column_config={
                "Scenario": st.column_config.TextColumn("Scenario"),
                "Total %":  st.column_config.NumberColumn("Total %",  format="%+.1f%%", width="small"),
                "CAGR %":   st.column_config.NumberColumn("CAGR %",   format="%+.1f%%", width="small"),
                "Max DD %": st.column_config.NumberColumn("Max DD %", format="%.1f%%",  width="small"),
            },
        )

    with st.expander("📋 Full attribution (every constituent)", expanded=False):
        # Compact view: drop the less-honest "Total return %" column
        # (Held return % is the better number); short headers.
        q_attrib_compact = q_attrib[[
            "Ticker", "Held return %", "Held days",
            "Avg weight %", "Contribution %", "Contribution share %",
        ]]
        st.dataframe(
            q_attrib_compact,
            width='stretch', hide_index=True,
            column_config={
                "Ticker":               st.column_config.TextColumn("Ticker", width="small"),
                "Held return %":        st.column_config.NumberColumn(
                    "Held %", format="%+.1f%%", width="small",
                    help="Price return during the period the index held this position. The honest 'did this make money' number.",
                ),
                "Held days":            st.column_config.NumberColumn(
                    "Days", format="%d", width="small",
                    help="Trading days the index held this name with non-zero weight.",
                ),
                "Avg weight %":         st.column_config.NumberColumn(
                    "Avg wt %", format="%.2f%%", width="small",
                ),
                "Contribution %":       st.column_config.NumberColumn(
                    "Contrib %", format="%+.2f%%", width="small",
                    help=(
                        "Linear (arithmetic) attribution: Σ(weight × daily return). "
                        "Industry standard but CAN DIVERGE from intuition for highly "
                        "volatile names — vol-pumping makes the arithmetic sum "
                        "differ from the compound return. Cross-check with Held %."
                    ),
                ),
                "Contribution share %": st.column_config.NumberColumn(
                    "Share %", format="%.1f%%", width="small",
                    help="|Contribution| as % of the gross attribution sum.",
                ),
            },
        )
        st.caption(
            "_**On the math**: Contribution % is the standard arithmetic Brinson "
            "attribution — it's what the index's daily P&L summed to from each name. "
            "For a high-volatility constituent, this can disagree in sign with "
            "Held return %. When that happens, Held return % is the more honest "
            "answer to 'did this position make money?', and the Contribution divergence "
            "comes from arithmetic-vs-geometric drift, not a bug._"
        )

    # ── CSV export ────────────────────────────────────────────────────────
    st.divider()

    def _q_build_csv() -> bytes:
        import io
        buf = io.StringIO()
        buf.write(f"# Quantum Index export — {q_result.name}\n")
        buf.write(f"# Window: {q_result.levels.index[0].date()} → {q_result.levels.index[-1].date()}\n")
        buf.write(f"# Rebalance: {q_rebalance}\n\n# --- Headline stats ---\n")
        for k, v in q_stats.to_dict().items():
            buf.write(f"# {k}: {v}\n")
        buf.write("\n# --- Index levels ---\n")
        q_result.levels.rename("level").to_csv(buf)
        buf.write("\n# --- Daily weights ---\n")
        q_result.weights.to_csv(buf)
        return buf.getvalue().encode("utf-8")

    st.download_button(
        "⬇️ Download quantum index CSV",
        data=_q_build_csv(),
        file_name=f"quantum_{q_index_choice.lower().replace(' ', '_')}_{q_result.levels.index[-1].date()}.csv",
        mime="text/csv",
        type="primary",
    )

    st.caption(
        "⚠️ _Custom research tool — not a licensed benchmark, not financial "
        "advice. The standalone project at `/Users/davemason/quantum-index-builder/` "
        "is the same code with a CLI runner and richer outputs._"
    )
