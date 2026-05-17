"""
ui/tabs/lookup.py — Free-form ticker analysis tab.

Pulls live screener data for any ticker (equity or crypto), shows:
  - Snapshot row (price, market cap, scores)
  - About the company tile
  - Institutional ownership tile (with QoQ piling-in)
  - Direction + setup_type + rationale
  - Recommendation card
  - 6mo daily price chart with EMAs
  - Live intraday candlestick (auto-refreshing fragment)
  - Peer comparison
  - Catalyst signals
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.peers import PEER_MAP
from core.recommendations import STRATEGY_DISPLAY, build_recommendation
from ui.data import (
    _provider as _provider,
    fetch_intraday_bars      as _fetch_intraday_bars,
    fetch_peer_fundamentals,
    fetch_company_info,
    fetch_institutional_data,
)
from ui.helpers import (
    format_holder_value as _format_holder_value,
    qoq_change_label    as _qoq_change_label,
)


@st.fragment(run_every="30s")
def _render_live_intraday(ticker: str, daily_atr: float | None = None) -> None:
    """Self-refreshing intraday candlestick chart for one ticker.
    Re-runs every 30s — only the chart panel updates, not the whole tab."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = _fetch_intraday_bars(ticker)
    if df.empty or "Close" not in df.columns:
        st.caption("_Live chart unavailable — provider returned no intraday data._")
        return

    # Intraday overlays
    close = df["Close"]
    df = df.copy()
    df["EMA9"]  = close.ewm(span=9,  adjust=False).mean()
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    # Session-anchored VWAP — reset cumulative sums at each date boundary
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    tpv = tp * df["Volume"].fillna(0)
    grouper = df.index.normalize() if hasattr(df.index, "normalize") else None
    if grouper is not None:
        df["VWAP"] = tpv.groupby(grouper).cumsum() / df["Volume"].fillna(0).groupby(grouper).cumsum().replace(0, pd.NA)
    else:
        df["VWAP"] = (tpv.cumsum() / df["Volume"].fillna(0).cumsum().replace(0, pd.NA))

    last_price = float(df["Close"].iloc[-1])
    last_time  = df.index[-1]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="Price",
            # Colourblind-safe Wong palette — blue-green up, vermilion down
            increasing_line_color="#009E73", increasing_fillcolor="#009E73",
            decreasing_line_color="#D55E00", decreasing_fillcolor="#D55E00",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA9"], mode="lines", name="EMA9",
        line=dict(color="#E69F00", width=1.4),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA20"], mode="lines", name="EMA20",
        line=dict(color="#56B4E9", width=1.4, dash="dash"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["VWAP"], mode="lines", name="VWAP",
        line=dict(color="#CC79A7", width=1.6),
    ), row=1, col=1)

    # Volume bars
    if "Volume" in df.columns:
        # Colour each bar to match the candle direction
        bar_colors = [
            "#009E73" if c >= o else "#D55E00"
            for o, c in zip(df["Open"], df["Close"])
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=bar_colors, marker_line_width=0,
            showlegend=False, opacity=0.5,
        ), row=2, col=1)

    # Last-price horizontal reference line + annotation
    fig.add_hline(
        y=last_price, line_color="#444", line_width=1, line_dash="dot",
        annotation_text=f"${last_price:.2f}", annotation_position="right",
        row=1, col=1,
    )

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        dragmode="pan",
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    st.plotly_chart(fig, width='stretch', config={"scrollZoom": True})

    # Status line below the chart
    bits = [
        f"**${last_price:.2f}**",
        f"Last bar: {last_time.strftime('%Y-%m-%d %H:%M')}",
        f"VWAP ${float(df['VWAP'].iloc[-1]):.2f}" if pd.notna(df['VWAP'].iloc[-1]) else None,
        f"EMA9 ${float(df['EMA9'].iloc[-1]):.2f}",
        f"Refreshed: {pd.Timestamp.now().strftime('%H:%M:%S')}",
    ]
    st.caption("  •  ".join(b for b in bits if b))


# fetch_institutional_data moved to ui/data.py (imported at top of file).


# _format_holder_value and _qoq_change_label moved to ui/helpers.py (imported above).

# ---------------------------------------------------------------------------
# Sidebar — screener filters only

# ---------------------------------------------------------------------------
# Screener — opportunity detail dialog
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Lookup tab."""

    st.subheader("🔎 Ticker Lookup")
    st.caption(
        "Type any ticker (equity or `XYZ-USD` crypto). Runs the same TradeScore + "
        "Recommendation pipeline used by the Screener and Advice tabs — no filter "
        "gate, so you get a result whether the ticker passes the daily screen or not."
    )

    col_in, col_btn = st.columns([3, 1])
    with col_in:
        lookup_ticker = st.text_input(
            "Ticker",
            value="",
            placeholder="e.g. SOFI, NVDA, BTC-USD",
            label_visibility="collapsed",
            key="lookup_ticker_input",
        ).strip().upper()
    with col_btn:
        do_lookup = st.button("Analyze", type="primary", width='stretch')

    if do_lookup and lookup_ticker:
        with st.spinner(f"Pulling data for {lookup_ticker}…"):
            try:
                from run import screen_ticker
                from core.recommendations import build_recommendation, STRATEGY_DISPLAY
                row = screen_ticker(lookup_ticker, "lookup")
            except Exception as e:
                st.error(f"Lookup failed: {e}")
                row = None

        if row is None:
            st.warning(
                f"No usable data for **{lookup_ticker}**. The ticker may be "
                "delisted, mistyped, or have fewer than 20 trading days of history."
            )
        else:
            # -----------------------------------------------------------------
            # Snapshot row
            # -----------------------------------------------------------------
            # Catalyst score is computed up-front so the top row has both
            # the price/volume score (TradeScore) AND the catalyst score.
            try:
                from core.catalysts import compute_catalyst_score
                catalyst = compute_catalyst_score(lookup_ticker, row.get("price"))
            except Exception as _e:
                catalyst = {"score": None, "components": {}, "tags": [], "data": {}}

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric(
                "Price",
                f"${row['price']:.2f}",
                f"{row['change_pct']:+.2f}% today",
            )
            mc = row.get("market_cap")
            m2.metric(
                "Market cap",
                f"${mc/1e9:.2f}B" if mc and mc >= 1e9 else (f"${mc/1e6:.0f}M" if mc else "—"),
            )
            m3.metric("TradeScore", f"{row['tradescore']:.1f} / 65")
            m4.metric("RSI(14)", f"{row['rsi']:.1f}")
            m5.metric("RVOL", f"{row['rvol']:.2f}×")
            m6.metric(
                "CatalystScore",
                f"{catalyst['score']:.0f} / 100" if catalyst.get("score") is not None else "—",
            )

            st.divider()

            # -----------------------------------------------------------------
            # About the company (name, sector, summary)
            # -----------------------------------------------------------------
            if not lookup_ticker.endswith("-USD"):
                _info = fetch_company_info(lookup_ticker)
                if _info["name"] or _info["summary"]:
                    st.markdown(f"### 🏢 {_info['name']}")
                    _bits = []
                    if _info["sector"]:
                        _bits.append(f"**Sector:** {_info['sector']}")
                    if _info["industry"]:
                        _bits.append(f"**Industry:** {_info['industry']}")
                    if _info["website"]:
                        _bits.append(f"[{_info['website'].replace('https://', '').replace('http://', '').rstrip('/')}]({_info['website']})")
                    if _bits:
                        st.caption("  •  ".join(_bits))
                    if _info["summary"]:
                        _summary = _info["summary"]
                        if len(_summary) > 320:
                            st.markdown(_summary[:320].rsplit(" ", 1)[0] + "…")
                            with st.expander("Show full description"):
                                st.markdown(_summary)
                        else:
                            st.markdown(_summary)
                    st.divider()

            # -----------------------------------------------------------------
            # Institutional ownership — top 13F filers + QoQ "piling in" signal
            # -----------------------------------------------------------------
            if not lookup_ticker.endswith("-USD"):
                _inst = fetch_institutional_data(lookup_ticker)
                _ih = _inst.get("institutional")
                _summary = _inst.get("summary") or {}

                if _ih is not None or _summary:
                    st.markdown("### 🏦 Institutional ownership")

                    # Ownership concentration summary line
                    _sum_bits = []
                    if "institutionsPercentHeld" in _summary:
                        _sum_bits.append(f"**{_summary['institutionsPercentHeld']*100:.1f}%** held by institutions")
                    if "insidersPercentHeld" in _summary:
                        _ins = _summary["insidersPercentHeld"] * 100
                        if _ins >= 0.01:
                            _sum_bits.append(f"**{_ins:.2f}%** insider")
                    if "institutionsCount" in _summary:
                        _sum_bits.append(f"**{int(_summary['institutionsCount']):,}** holders on the register")
                    if _sum_bits:
                        st.caption("  •  ".join(_sum_bits))

                    if _ih is not None:
                        # Flag the biggest moves — adding vs trimming — as a
                        # plain-English summary before the table
                        moves = _ih.dropna(subset=["pctChange"]).copy()
                        if not moves.empty:
                            adding   = moves[moves["pctChange"] >  0.005].sort_values("pctChange", ascending=False).head(3)
                            trimming = moves[moves["pctChange"] < -0.005].sort_values("pctChange").head(3)

                            def _short_holder_name(name: str) -> str:
                                return name.split(",")[0].split(" Inc")[0][:32]

                            def _change_str(pct: float) -> str:
                                if pct >= 0.999:
                                    return "new"
                                return f"{pct*100:+.1f}%"

                            _move_lines = []
                            if not adding.empty:
                                _entries = [
                                    f"{_short_holder_name(r['Holder'])} ({_change_str(r['pctChange'])})"
                                    for _, r in adding.iterrows()
                                ]
                                _move_lines.append(f"🟢 **Adding:** {' · '.join(_entries)}")
                            if not trimming.empty:
                                _entries = [
                                    f"{_short_holder_name(r['Holder'])} ({_change_str(r['pctChange'])})"
                                    for _, r in trimming.iterrows()
                                ]
                                _move_lines.append(f"🔴 **Trimming:** {' · '.join(_entries)}")
                            if _move_lines:
                                for _l in _move_lines:
                                    st.markdown(_l)

                        # Top holders table — sorted by Date Reported (newest first)
                        # so timely Q1 filings sit on top, stale Q4 carryovers below.
                        display = _ih.copy()
                        display["_date_sort"] = pd.to_datetime(display["Date Reported"], errors="coerce")
                        display = display.sort_values("_date_sort", ascending=False).reset_index(drop=True)
                        # Normalise Date Reported to a YYYY-MM-DD string for display + grouping
                        display["Date Reported"] = display["_date_sort"].dt.strftime("%Y-%m-%d")

                        # Freshness summary: how many holders are on the latest filing
                        # vs older ones?
                        _date_counts = display["Date Reported"].value_counts().sort_index(ascending=False)
                        if len(_date_counts) > 1:
                            _latest_date = _date_counts.index[0]
                            _latest_n    = int(_date_counts.iloc[0])
                            _stale_n     = int(display.shape[0] - _latest_n)
                            _other_dates = ", ".join(str(d) for d in _date_counts.index[1:])
                            st.caption(
                                f"📅 **Filing freshness:** {_latest_n} of {len(display)} holders "
                                f"reported through **{_latest_date}** (timely). "
                                f"{_stale_n} still on older filings ({_other_dates}) — those rows' "
                                f"QoQ change reflects the *prior* quarter, not the latest."
                            )

                        display["% held"]   = display["pctHeld"] * 100
                        display["QoQ"]      = display["pctChange"].apply(_qoq_change_label)
                        display["Value"]    = display["Value"].apply(_format_holder_value)
                        display["Shares"]   = display["Shares"].apply(
                            lambda s: f"{s/1e6:.1f}M" if s and s >= 1e6 else f"{s:,.0f}" if s else "—"
                        )
                        display = display[["Holder", "% held", "Shares", "Value", "QoQ", "Date Reported"]]
                        st.dataframe(
                            display,
                            hide_index=True,
                            width='stretch',
                            column_config={
                                "Holder":        st.column_config.TextColumn(width="large"),
                                "% held":        st.column_config.NumberColumn(format="%.2f%%"),
                                "Date Reported": st.column_config.TextColumn(width="small"),
                            },
                        )
                        st.caption(
                            "_13F filings updated quarterly, sorted newest first. QoQ change is "
                            "vs each holder's prior filing — so rows with older dates reflect "
                            "an earlier quarter's move. 🆕 new entry, 🟢 adding, 🔴 trimming._"
                        )
                    st.divider()

            # -----------------------------------------------------------------
            # Direction + setup_type + rationale (from TradeScore)
            # -----------------------------------------------------------------
            dir_label = {"long": "🟢 Long", "short": "🔴 Short", "neutral": "⚪ Neutral"}.get(
                row.get("direction", ""), "⚪ Neutral"
            )
            c1, c2 = st.columns([1, 3])
            c1.markdown(f"### {dir_label}")
            c1.caption(f"Setup: `{row.get('setup_type', '—')}`")
            c2.markdown("**Why this read:**")
            c2.markdown(row.get("rationale", "_No rationale generated._"))

            st.divider()

            # -----------------------------------------------------------------
            # Recommendation card (entry / stop / target / strategy)
            # -----------------------------------------------------------------
            try:
                rec = build_recommendation(row, iv_mode="fallback", catalyst=catalyst)
            except Exception as e:
                st.error(f"Recommendation engine failed: {e}")
                rec = None

            if rec:
                st.markdown("### 💡 Recommendation")

                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Category", rec.recommendation_category.title())
                rc2.metric("Strategy", STRATEGY_DISPLAY.get(rec.strategy_name, rec.strategy_name))
                rc3.metric(
                    "Entry ref",
                    f"${rec.entry_reference:.2f}" if rec.entry_reference else "—",
                )
                rc4.metric(
                    "R:R",
                    f"{rec.risk_reward:.1f}" if rec.risk_reward else "—",
                )

                sc1, sc2 = st.columns(2)
                sc1.metric(
                    "Invalidation (stop)",
                    f"${rec.invalidation_price:.2f}" if rec.invalidation_price else "—",
                )
                sc2.metric(
                    "Target (2R)",
                    f"${rec.target_price:.2f}" if rec.target_price else "—",
                )

                st.markdown("**Rationale**")
                st.markdown(rec.rationale or "_No detailed rationale._")

                if rec.iv_assessment and rec.iv_assessment != "unavailable":
                    st.caption(f"IV assessment: **{rec.iv_assessment}**")

                if rec.warnings:
                    for w in rec.warnings:
                        st.warning(f"⚠️ {w}")

                if rec.is_actionable:
                    st.success(
                        "**Actionable** — setup meets criteria. See Options tab for "
                        "contract pricing."
                    )
                else:
                    st.info(
                        "**Not actionable right now** — either no clear edge, the "
                        "stop is too wide, or the move is extended. Watch for a better setup."
                    )

            st.divider()

            # -----------------------------------------------------------------
            # Price chart with EMA20 / EMA50 / EMA200 and trade levels
            # Pull 2y so EMA200 is fully warmed; display last ~6 months.
            # -----------------------------------------------------------------
            st.markdown("### 📈 Price chart — 6 months daily")

            try:
                import altair as alt
                ohlcv = _provider.get_ohlcv(lookup_ticker, "2y", "1d")
            except Exception as e:
                ohlcv = None
                st.caption(f"_Price chart unavailable: {e}_")

            if ohlcv is not None and not ohlcv.empty:
                close = ohlcv["Close"]
                ema20_s  = close.ewm(span=20,  adjust=False).mean()
                ema50_s  = close.ewm(span=50,  adjust=False).mean()
                ema200_s = close.ewm(span=200, adjust=False).mean()

                # Long-format DataFrame for layered Altair lines
                chart_df = pd.DataFrame({
                    "Date":    close.index,
                    "Close":   close.values,
                    "EMA20":   ema20_s.values,
                    "EMA50":   ema50_s.values,
                    "EMA200":  ema200_s.values,
                })
                # Slice to last ~6 months of trading days for display
                chart_df = chart_df.tail(126).reset_index(drop=True)
                # Strip timezone (altair doesn't render tz-aware datetimes well)
                try:
                    chart_df["Date"] = chart_df["Date"].dt.tz_localize(None)
                except Exception:
                    pass
                long_df = chart_df.melt(
                    id_vars="Date",
                    value_vars=["Close", "EMA20", "EMA50", "EMA200"],
                    var_name="Series",
                    value_name="Price",
                )

                # Colourblind-safe palette (Wong) + line-style redundancy so the
                # series are distinguishable without relying on hue alone.
                line = (
                    alt.Chart(long_df)
                    .mark_line()
                    .encode(
                        x=alt.X("Date:T", axis=alt.Axis(title=None)),
                        y=alt.Y("Price:Q", axis=alt.Axis(title="Price ($)"),
                                scale=alt.Scale(zero=False)),
                        color=alt.Color(
                            "Series:N",
                            scale=alt.Scale(
                                domain=["Close", "EMA20", "EMA50", "EMA200"],
                                range=["#000000", "#E69F00", "#56B4E9", "#CC79A7"],
                            ),
                            legend=alt.Legend(orient="top", title=None),
                        ),
                        strokeDash=alt.StrokeDash(
                            "Series:N",
                            scale=alt.Scale(
                                domain=["Close", "EMA20", "EMA50", "EMA200"],
                                range=[[1, 0], [1, 0], [5, 3], [1, 0]],
                            ),
                            legend=None,
                        ),
                        tooltip=[
                            alt.Tooltip("Date:T", title="Date"),
                            alt.Tooltip("Series:N"),
                            alt.Tooltip("Price:Q", format="$.2f"),
                        ],
                    )
                )

                # Horizontal rules for entry / stop / target when actionable.
                # We always show Entry as a reference; stop / target only when
                # we have meaningful levels.
                rules_rows = []
                if rec and rec.entry_reference:
                    rules_rows.append({"Level": "Entry",  "Price": rec.entry_reference, "Colour": "#444"})
                if rec and rec.invalidation_price:
                    rules_rows.append({"Level": "Stop",   "Price": rec.invalidation_price, "Colour": "#d62728"})
                if rec and rec.target_price:
                    rules_rows.append({"Level": "Target", "Price": rec.target_price, "Colour": "#2ca02c"})

                layered = line
                if rules_rows:
                    rules_df = pd.DataFrame(rules_rows)
                    rules = (
                        alt.Chart(rules_df)
                        .mark_rule(strokeDash=[4, 4], strokeWidth=1.5)
                        .encode(
                            y="Price:Q",
                            color=alt.Color(
                                "Level:N",
                                scale=alt.Scale(
                                    domain=["Entry", "Stop", "Target"],
                                    range=["#444444", "#d62728", "#2ca02c"],
                                ),
                                legend=None,
                            ),
                            tooltip=[
                                alt.Tooltip("Level:N"),
                                alt.Tooltip("Price:Q", format="$.2f"),
                            ],
                        )
                    )
                    # Labels at the right edge of the chart for each rule
                    labels = (
                        alt.Chart(rules_df)
                        .mark_text(align="left", dx=5, fontWeight="bold", fontSize=11)
                        .encode(
                            y="Price:Q",
                            text=alt.Text("Label:N"),
                            color=alt.Color(
                                "Level:N",
                                scale=alt.Scale(
                                    domain=["Entry", "Stop", "Target"],
                                    range=["#444444", "#d62728", "#2ca02c"],
                                ),
                                legend=None,
                            ),
                        )
                        .transform_calculate(Label="datum.Level + ' $' + format(datum.Price, '.2f')")
                    )
                    # Position labels at the right edge by using the last date
                    last_date = chart_df["Date"].max()
                    rules_df_labels = rules_df.copy()
                    rules_df_labels["Date"] = last_date
                    labels = (
                        alt.Chart(rules_df_labels)
                        .mark_text(align="left", dx=5, fontWeight="bold", fontSize=11)
                        .encode(
                            x="Date:T",
                            y="Price:Q",
                            text=alt.Text("Label:N"),
                            color=alt.Color(
                                "Level:N",
                                scale=alt.Scale(
                                    domain=["Entry", "Stop", "Target"],
                                    range=["#444444", "#d62728", "#2ca02c"],
                                ),
                                legend=None,
                            ),
                        )
                        .transform_calculate(Label="datum.Level + ' $' + format(datum.Price, '.2f')")
                    )

                    layered = (line + rules + labels)

                chart = layered.properties(height=400).interactive()
                st.altair_chart(chart, width='stretch')

                # Compact summary under the chart
                last_close = float(close.iloc[-1])
                pct_to_ema20  = ((last_close - float(ema20_s.iloc[-1]))  / float(ema20_s.iloc[-1])  * 100) if ema20_s.iloc[-1]  else 0
                pct_to_ema200 = ((last_close - float(ema200_s.iloc[-1])) / float(ema200_s.iloc[-1]) * 100) if ema200_s.iloc[-1] else 0
                regime = "bull regime" if pct_to_ema200 > 0 else "bear regime"
                if rec and rec.invalidation_price and rec.target_price:
                    risk_pct = abs(rec.entry_reference - rec.invalidation_price) / rec.entry_reference * 100
                    rew_pct  = abs(rec.target_price    - rec.entry_reference) / rec.entry_reference * 100
                    st.caption(
                        f"_Price ${last_close:.2f} ({pct_to_ema20:+.1f}% vs EMA20, "
                        f"{pct_to_ema200:+.1f}% vs EMA200 — {regime}). "
                        f"Trade structure: risk **{risk_pct:.1f}%** to stop, "
                        f"reward **{rew_pct:.1f}%** to target. Dashed lines: "
                        f"<span style='color:#444'>**Entry**</span> · "
                        f"<span style='color:#d62728'>**Stop**</span> · "
                        f"<span style='color:#2ca02c'>**Target**</span>._",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption(
                        f"_Price ${last_close:.2f} ({pct_to_ema20:+.1f}% vs EMA20, "
                        f"{pct_to_ema200:+.1f}% vs EMA200 — {regime}). "
                        f"No actionable levels shown — recommendation is "
                        f"watchlist/avoid for now._"
                    )

            st.divider()

            # -----------------------------------------------------------------
            # Live intraday candlestick — 5d / 5m bars, auto-refresh every 30s
            # -----------------------------------------------------------------
            st.markdown("### 🕯️ Live intraday — 5d / 5m bars")
            st.caption(
                "_Auto-refreshes every 30 seconds. Free yfinance data is "
                "15-minute delayed on US equities — adequate for swing context, "
                "not for scalping._"
            )
            _render_live_intraday(lookup_ticker)
            st.divider()

            # -----------------------------------------------------------------
            # Peer comparison — fundamentals snapshot vs close competitors
            # -----------------------------------------------------------------
            peers = PEER_MAP.get(lookup_ticker, ())
            if peers and not lookup_ticker.endswith("-USD"):
                st.markdown("### 🆚 Peer comparison")
                with st.spinner("Fetching peer fundamentals…"):
                    peer_df = fetch_peer_fundamentals((lookup_ticker,) + peers)

                if peer_df.empty or peer_df.drop(columns=["Ticker", "Name"]).isna().all().all():
                    st.caption("_Peer fundamentals unavailable — yfinance returned no data._")
                else:
                    st.dataframe(
                        peer_df,
                        hide_index=True,
                        width='stretch',
                        column_config={
                            "Ticker":      st.column_config.TextColumn(width="small"),
                            "Name":        st.column_config.TextColumn(width="medium"),
                            "Market cap":  st.column_config.NumberColumn(format="$%.2s"),
                            "P/E (TTM)":   st.column_config.NumberColumn(format="%.1f"),
                            "Forward P/E": st.column_config.NumberColumn(format="%.1f"),
                            "PEG":         st.column_config.NumberColumn(format="%.2f"),
                            "Net margin":  st.column_config.NumberColumn(format="%.1%"),
                            "ROE":         st.column_config.NumberColumn(format="%.1%"),
                            "Rev growth":  st.column_config.NumberColumn(format="%.1%"),
                        },
                    )
                    st.caption(
                        f"_Peers from built-in map: {', '.join(peers)}. "
                        f"Data via yfinance, cached 1h. Edit `PEER_MAP` in `app.py` to customise._"
                    )
                st.divider()

            # -----------------------------------------------------------------
            # Catalyst signals (earnings / news / analyst actions)
            # -----------------------------------------------------------------
            st.markdown("### 🎯 Catalyst signals")

            if catalyst.get("score") is None:
                st.caption(
                    "_No catalyst data — common for crypto, ETFs, ADRs, and "
                    "thinly-covered tickers._"
                )
            else:
                # Tag banner — plain-English summary of what's driving the score
                if catalyst.get("tags"):
                    for tag in catalyst["tags"]:
                        if "⚠" in tag or "miss" in tag.lower() or "cut" in tag.lower() or "downgrade" in tag.lower():
                            st.markdown(f"- 🔴 {tag}")
                        elif "beat" in tag.lower() or "upgrade" in tag.lower() or "raise" in tag.lower() or "upside" in tag.lower() or "strong buy" in tag.lower() or "buy" in tag.lower():
                            st.markdown(f"- 🟢 {tag}")
                        else:
                            st.markdown(f"- ⚪ {tag}")

                # Earnings expander
                earnings_next = catalyst["data"].get("earnings_next")
                earnings_hist = catalyst["data"].get("earnings_history") or []
                with st.expander("📅 Earnings — next date + last 4 quarters", expanded=False):
                    if earnings_next:
                        e1, e2, e3 = st.columns(3)
                        e1.metric(
                            "Next earnings",
                            earnings_next["date"],
                            f"in {earnings_next['days_to']}d" if earnings_next['days_to'] >= 0 else f"{abs(earnings_next['days_to'])}d ago",
                        )
                        if earnings_next.get("eps_estimate") is not None:
                            e2.metric("EPS estimate", f"${earnings_next['eps_estimate']:.3f}")
                        if earnings_next.get("revenue_estimate") is not None:
                            rev = earnings_next["revenue_estimate"]
                            e3.metric("Revenue estimate", f"${rev/1e9:.2f}B" if rev >= 1e9 else f"${rev/1e6:.0f}M")
                    else:
                        st.caption("_No upcoming earnings date available._")

                    if earnings_hist:
                        st.markdown("**Last 4 quarters:**")
                        hist_df = pd.DataFrame(earnings_hist)
                        st.dataframe(
                            hist_df,
                            width='stretch',
                            hide_index=True,
                            column_config={
                                "quarter":      st.column_config.TextColumn("Quarter"),
                                "eps_actual":   st.column_config.NumberColumn("EPS actual",   format="$%.3f"),
                                "eps_estimate": st.column_config.NumberColumn("EPS estimate", format="$%.3f"),
                                "surprise_pct": st.column_config.NumberColumn("Surprise %",   format="%+.1f%%"),
                            },
                        )
                    else:
                        st.caption("_No earnings history available._")

                # News expander
                news = catalyst["data"].get("news") or []
                with st.expander(f"📰 Recent news ({len(news)})", expanded=False):
                    if news:
                        for n in news[:10]:
                            ts = (n.get("published_at") or "")[:10]
                            url = n.get("url") or ""
                            title = n.get("title") or "(untitled)"
                            pub = n.get("publisher") or "—"
                            if url:
                                st.markdown(f"- **{ts}** [{title}]({url}) — _{pub}_")
                            else:
                                st.markdown(f"- **{ts}** {title} — _{pub}_")
                            if n.get("summary"):
                                st.caption(n["summary"][:240] + ("…" if len(n["summary"]) > 240 else ""))
                    else:
                        st.caption("_No recent news on Yahoo Finance for this ticker._")

                # Analyst actions expander
                analyst_data = catalyst["data"].get("analyst") or {}
                with st.expander("📈 Analyst actions (last 90 days)", expanded=False):
                    consensus_label = analyst_data.get("consensus_label", "—")
                    n_analysts = analyst_data.get("num_analysts")
                    tgt_mean = analyst_data.get("target_mean")
                    tgt_low = analyst_data.get("target_low")
                    tgt_high = analyst_data.get("target_high")

                    if n_analysts:
                        a1, a2, a3, a4 = st.columns(4)
                        a1.metric("Consensus", str(consensus_label).replace("_", " ").title())
                        a2.metric("Analysts", str(n_analysts))
                        if tgt_mean:
                            upside = ((tgt_mean - row["price"]) / row["price"] * 100) if row.get("price") else 0
                            a3.metric("Target (mean)", f"${tgt_mean:.2f}", f"{upside:+.1f}% vs price")
                        if tgt_low is not None and tgt_high is not None:
                            a4.metric("Target range", f"${tgt_low:.0f}-${tgt_high:.0f}")
                    else:
                        st.caption("_No analyst coverage available for this ticker._")

                    actions = analyst_data.get("recent_actions") or []
                    if actions:
                        st.markdown(f"**{len(actions)} actions in last 90 days:**")
                        act_df = pd.DataFrame(actions)
                        # Format action descriptions
                        def _fmt_action(r):
                            ta = r.get("target_action") or ""
                            old, new = r.get("target_old"), r.get("target_new")
                            if old and new:
                                arrow = "↑" if new > old else ("↓" if new < old else "→")
                                return f"{ta.title()} ${old:.0f} {arrow} ${new:.0f}"
                            return ta.title() if ta else r.get("action", "").title()
                        act_df["change"] = act_df.apply(_fmt_action, axis=1)
                        act_df["grade"] = act_df.apply(
                            lambda r: f"{r['from_grade']} → {r['to_grade']}" if r.get("from_grade") and r.get("to_grade") else (r.get("to_grade") or ""),
                            axis=1,
                        )
                        st.dataframe(
                            act_df[["date", "firm", "grade", "change"]],
                            width='stretch',
                            hide_index=True,
                            column_config={
                                "date":   st.column_config.TextColumn("Date",   width="small"),
                                "firm":   st.column_config.TextColumn("Firm"),
                                "grade":  st.column_config.TextColumn("Grade"),
                                "change": st.column_config.TextColumn("Target change"),
                            },
                        )
                    else:
                        st.caption("_No analyst actions logged in last 90 days._")

                # Insider activity expander — Form 4 buys/sells aggregated
                insider_data = catalyst["data"].get("insider") or {}
                with st.expander("👤 Insider activity (Form 4, last 90 days)", expanded=False):
                    txs = insider_data.get("transactions") or []
                    if not txs and not insider_data.get("net_6m_summary"):
                        st.caption(
                            "_No recent insider activity available (or ticker has "
                            "no SEC-registered insiders — common for ADRs, ETFs, "
                            "and crypto)._"
                        )
                    else:
                        # Summary row
                        ic1, ic2, ic3, ic4 = st.columns(4)
                        ic1.metric(
                            "Buyers (90d)",
                            insider_data.get("buyer_count", 0),
                            help="Unique insiders who placed open-market purchases. Clusters of 3+ = strong signal.",
                        )
                        ic2.metric(
                            "Sellers (90d)",
                            insider_data.get("seller_count", 0),
                            help="Unique insiders who sold. Mostly noise — selling has many non-bearish reasons.",
                        )
                        ic3.metric(
                            "Total bought",
                            f"${insider_data.get('total_buy_value', 0)/1e6:.2f}M",
                        )
                        ic4.metric(
                            "Total sold",
                            f"${insider_data.get('total_sell_value', 0)/1e6:.2f}M",
                            f"Net ${insider_data.get('net_value', 0)/1e6:+.2f}M",
                        )

                        # Recent transaction table
                        if txs:
                            tx_df = pd.DataFrame(txs)
                            # Colour kind for readability
                            kind_emoji = {"buy": "🟢 Buy", "sell": "🔴 Sell", "neutral": "⚪ Other", "other": "—"}
                            tx_df["kind_disp"] = tx_df["kind"].map(kind_emoji).fillna("—")
                            tx_df["value_fmt"] = tx_df["value"].apply(
                                lambda v: f"${v:,.0f}" if v else "—"
                            )
                            tx_df["shares_fmt"] = tx_df["shares"].apply(
                                lambda s: f"{s:,.0f}" if s else "—"
                            )
                            st.markdown(f"**{len(txs)} transactions in window:**")
                            st.dataframe(
                                tx_df[["date", "insider", "position", "kind_disp", "shares_fmt", "value_fmt"]],
                                width='stretch',
                                hide_index=True,
                                column_config={
                                    "date":       st.column_config.TextColumn("Date",      width="small"),
                                    "insider":    st.column_config.TextColumn("Insider"),
                                    "position":   st.column_config.TextColumn("Position"),
                                    "kind_disp":  st.column_config.TextColumn("Kind",      width="small"),
                                    "shares_fmt": st.column_config.TextColumn("Shares",    width="small"),
                                    "value_fmt":  st.column_config.TextColumn("Value",     width="small"),
                                },
                            )

                        # 6-month yfinance aggregate for cross-reference
                        summary_6m = insider_data.get("net_6m_summary") or {}
                        if summary_6m:
                            st.markdown("**6-month aggregate (yfinance):**")
                            # Render the typical 6 rows: Purchases, Sales, Net, Held, % buy, % sell
                            summ_df = pd.DataFrame(
                                [{"Metric": k, "Value": v} for k, v in summary_6m.items()]
                            )
                            st.dataframe(summ_df, width='stretch', hide_index=True)

                        st.caption(
                            "_**Buying is signal, selling is mostly noise.** Insiders "
                            "sell for many reasons (tax, diversification, 10b5-1 plans); "
                            "they buy for one (they think the stock will rise). The "
                            "CatalystScore reflects this asymmetry — strong reward for "
                            "buyer clusters, small penalty only for lopsided heavy selling._"
                        )

            st.divider()

            # -----------------------------------------------------------------
            # Recent SEC filings (catalyst context)
            # -----------------------------------------------------------------
            with st.expander("📄 Recent SEC filings", expanded=False):
                try:
                    from core.sec_edgar import get_recent_filings
                    filings = get_recent_filings(lookup_ticker, limit=10)
                except Exception as e:
                    filings = None
                    st.caption(f"_SEC filings lookup failed: {e}_")

                if filings is None:
                    pass
                elif not filings:
                    st.caption(
                        "_No recent filings found (or ticker has no SEC CIK — "
                        "common for crypto and ADRs)._"
                    )
                else:
                    filings_df = pd.DataFrame(filings)
                    # Use the items_label as a clearer "What's in it" column for 8-Ks;
                    # fall back to description for other forms.
                    if "items_label" in filings_df.columns:
                        filings_df["What's in it"] = filings_df.apply(
                            lambda r: r.get("items_label") or r.get("description") or "",
                            axis=1,
                        )
                    else:
                        filings_df["What's in it"] = filings_df.get("description", "")
                    st.dataframe(
                        filings_df[["form", "filed", "What's in it", "url"]],
                        width='stretch',
                        hide_index=True,
                        column_config={
                            "form":         st.column_config.TextColumn("Form",  width="small"),
                            "filed":        st.column_config.TextColumn("Filed", width="small"),
                            "What's in it": st.column_config.TextColumn(
                                "What's in it",
                                help="For 8-Ks: SEC item codes mapped to plain English. "
                                     "For other forms: the SEC's filing description.",
                            ),
                            "url":          st.column_config.LinkColumn(
                                "Filing", width="small", display_text="Open ↗",
                            ),
                        },
                    )
                    st.caption(
                        "_Forms tracked: 8-K (material events), 10-Q (quarterly), "
                        "10-K (annual), S-1 (registration), 4 (insider trades)._"
                    )

            # -----------------------------------------------------------------
            # Recent federal contracts (catalyst — only relevant for contractors)
            # -----------------------------------------------------------------
            with st.expander("🏛️ Recent federal contracts (USAspending)", expanded=False):
                try:
                    from core.usaspending import get_recent_contracts
                    contracts = get_recent_contracts(lookup_ticker, days=180, min_amount=250_000)
                except Exception as e:
                    contracts = None
                    st.caption(f"_USAspending lookup failed: {e}_")

                if contracts is None:
                    pass
                elif not contracts:
                    st.caption(
                        "_No federal contracts in the last 180 days for this ticker. "
                        "Most public companies aren't federal contractors — this is "
                        "only a catalyst signal for defense, govtech, IT services, "
                        "and similar names._"
                    )
                else:
                    contracts_df = pd.DataFrame(contracts)
                    contracts_df["amount_fmt"] = contracts_df["amount"].apply(
                        lambda v: f"${v:,.0f}"
                    )
                    st.dataframe(
                        contracts_df[["action_date", "amount_fmt", "kind", "agency", "recipient"]],
                        width='stretch',
                        hide_index=True,
                        column_config={
                            "action_date": st.column_config.TextColumn("Action date", width="small"),
                            "amount_fmt":  st.column_config.TextColumn("Amount", width="small"),
                            "kind":        st.column_config.TextColumn("Kind", width="small"),
                            "agency":      st.column_config.TextColumn("Agency"),
                            "recipient":   st.column_config.TextColumn("Recipient (USAspending name)"),
                        },
                    )
                    st.caption(
                        "_Source: USAspending.gov. Publishes 30-90 days after award action. "
                        "**NEW** = new award; **MOD P000X** = modification / option exercise. "
                        "Non-DoD agency awards historically showed stronger post-event drift "
                        "than DoD (the DoD daily contracts feed is well front-run)._"
                    )

            # -----------------------------------------------------------------
            # Raw indicator row (for power users)
            # -----------------------------------------------------------------
            with st.expander("🔬 Raw indicator row", expanded=False):
                display_cols = [
                    "ticker", "price", "change_pct", "rvol",
                    "ema9", "ema20", "ema200", "rsi", "atr",
                    "macd", "macd_signal", "vwap",
                    "tradescore", "setup_type", "direction",
                    "market_cap", "float_shares",
                ]
                row_display = {k: row.get(k) for k in display_cols if k in row}
                st.json(row_display)

    elif do_lookup and not lookup_ticker:
        st.warning("Enter a ticker first.")
    else:
        st.info("Enter a ticker and hit **Analyze** to see the full recommendation.")
