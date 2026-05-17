"""
ui/tabs/metals.py — Precious metals dashboard.

Macro drivers banner, futures table, ETF table, chart with EMAs, and
ETF spreads. Imports data from ui/data and helpers from app.py for
the regime label / driver-tag helpers (those will move to ui/helpers
in a follow-up step).
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import streamlit as st

from core.db import get_conn
from ui.data import (
    fetch_nzdusd,
    fetch_metal_prices,
    fetch_metal_chart,
    fetch_metal_technicals,
    fetch_market_context,
    METAL_FUTURES     as _METAL_FUTURES,
    METAL_FUTURES_REV as _METAL_FUTURES_REV,
    METAL_ETFS        as _METAL_ETFS,
    ALL_METAL_TICKERS as _ALL_METAL_TICKERS,
    ASSET_DRIVERS     as _ASSET_DRIVERS,
)


def render(regime_label: Callable, driver_tags: Callable) -> None:
    """Render the Metals tab.

    Args:
      regime_label: callable(ctx_dict, key) -> str — formats macro regime label
      driver_tags:  callable(ticker, screener_row) -> list[str] — driver annotations
    """

    nzdusd_m  = fetch_nzdusd()
    metal_px  = fetch_metal_prices()
    metal_tech = fetch_metal_technicals()
    mctx      = fetch_market_context()

    # -----------------------------------------------------------------------
    # Macro drivers banner
    # -----------------------------------------------------------------------

    spy_regime = regime_label(mctx, "spy")
    usd_regime = regime_label(mctx, "usd")
    gold_signal = metal_tech.get("Gold", {}).get("signal", "—")
    gold_mom5   = metal_tech.get("Gold", {}).get("mom_5d")

    # Derive narrative
    usd_d = mctx.get("usd", {})
    usd_falling = usd_d.get("above_ema20") is False  # USD below EMA20 = tailwind for metals

    drivers = []
    if usd_falling:
        drivers.append("USD weakening — metals benefit from inverse dollar relationship")
    elif usd_d.get("above_ema20") and usd_d.get("above_ema50"):
        drivers.append("USD strengthening — headwind for USD-denominated metals")
    if gold_signal == "Bullish":
        drivers.append("Gold in uptrend — safe haven / inflation hedge demand active")
    elif gold_signal == "Bearish":
        drivers.append("Gold in downtrend — risk-on environment reducing safe haven demand")
    if spy_regime in ("Bullish", "Recovering"):
        drivers.append("Equity market bullish — watch for rotation out of safe havens into equities")
    else:
        drivers.append("Equity market under pressure — flight to quality supporting metals")

    tnx_d = mctx.get("tnx", {})
    if tnx_d.get("above_ema20"):
        drivers.append("Rates rising — pressure on non-yielding metals; watch real rate moves")
    else:
        drivers.append("Rates easing — supportive for gold as opportunity cost of holding declines")

    gs_ratio = None
    g_p = metal_px.get("GC=F", {}).get("price", 0)
    s_p = metal_px.get("SI=F", {}).get("price", 0)
    if g_p and s_p:
        gs_ratio = g_p / s_p
        if gs_ratio > 80:
            drivers.append(f"Gold/Silver ratio {gs_ratio:.0f} — historically elevated; silver historically cheap vs gold")
        elif gs_ratio < 60:
            drivers.append(f"Gold/Silver ratio {gs_ratio:.0f} — silver expensive vs gold historically")
        else:
            drivers.append(f"Gold/Silver ratio {gs_ratio:.0f} — within normal historical range (~60–80)")

    # Miners vs gold
    gdx_p  = metal_px.get("GDX",  {}).get("price", 0)
    gdx_pc = metal_px.get("GDX",  {}).get("prev_close", 0)
    gld_p  = metal_px.get("GLD",  {}).get("price", 0)
    gld_pc = metal_px.get("GLD",  {}).get("prev_close", 0)
    if gdx_p and gdx_pc and gld_p and gld_pc:
        gdx_chg = (gdx_p / gdx_pc - 1) * 100
        gld_chg = (gld_p / gld_pc - 1) * 100
        if gdx_chg > gld_chg + 0.5:
            drivers.append("Miners outperforming gold today — risk appetite within metals sector, levered upside in play")
        elif gdx_chg < gld_chg - 0.5:
            drivers.append("Miners underperforming gold today — investors preferring physical over operational leverage")

    with st.container(border=True):
        st.markdown("**Market drivers**")
        for d in drivers:
            st.markdown(f"- {d}")
        col_regime1, col_regime2, col_regime3, col_regime4 = st.columns(4)
        col_regime1.metric("Equity regime",  spy_regime)
        col_regime2.metric("USD trend",      usd_regime)
        col_regime3.metric("Gold signal",    gold_signal)
        if tnx_d.get("price"):
            col_regime4.metric("10yr yield",  f"{tnx_d['price']:.2f}%",
                               delta=f"{tnx_d.get('chg', 0):+.2f}%" if tnx_d.get("chg") is not None else None,
                               delta_color="inverse")

    st.divider()

    # -----------------------------------------------------------------------
    # Spot price cards
    # -----------------------------------------------------------------------

    st.subheader("Spot prices (USD/oz)")

    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, (name, ticker) in zip([mc1, mc2, mc3, mc4], _METAL_FUTURES.items()):
        px   = metal_px.get(ticker, {})
        p    = px.get("price", 0.0)
        prev = px.get("prev_close", 0.0)
        chg  = ((p - prev) / prev * 100) if prev else 0.0
        p_nzd = p / nzdusd_m if nzdusd_m else p
        col.metric(
            f"{name}",
            f"${p:,.2f}" if p else "—",
            delta=f"{chg:+.2f}%  (NZD {p_nzd:,.0f}/oz)" if p else None,
            delta_color="normal",
        )

    # -----------------------------------------------------------------------
    # Gold/Silver ratio
    # -----------------------------------------------------------------------

    st.divider()
    gs_col, etf_col = st.columns([1, 3])

    with gs_col:
        if gs_ratio:
            st.metric(
                "Gold/Silver ratio",
                f"{gs_ratio:.1f}",
                help="oz of silver to buy 1 oz gold. Historical avg ~60. >80 = silver cheap vs gold.",
            )
            if gs_ratio > 80:
                st.caption("Silver historically undervalued vs gold at this ratio.")
            elif gs_ratio < 60:
                st.caption("Silver historically expensive vs gold at this ratio.")
            else:
                st.caption("Ratio within normal historical range.")

    with etf_col:
        ec1, ec2, ec3, ec4 = st.columns(4)
        for col, (etf, label) in zip([ec1, ec2, ec3, ec4], _METAL_ETFS.items()):
            px   = metal_px.get(etf, {})
            p    = px.get("price", 0.0)
            prev = px.get("prev_close", 0.0)
            chg  = ((p - prev) / prev * 100) if prev else 0.0
            col.metric(etf, f"${p:.2f}" if p else "—",
                       delta=f"{chg:+.2f}%" if p else None,
                       delta_color="normal", help=label)

    st.divider()

    # -----------------------------------------------------------------------
    # Technical signals table
    # -----------------------------------------------------------------------

    st.subheader("Technical signals")

    sig_rows = []
    for name, td in metal_tech.items():
        def _fmt_mom(v):
            return f"{v:+.1f}%" if v is not None else "—"

        sig_rows.append({
            "Metal":       name,
            "Price":       f"${td['price']:,.2f}",
            "vs EMA20":    f"{((td['price']/td['ema20']-1)*100):+.1f}%" if td.get("ema20") else "—",
            "vs EMA50":    f"{((td['price']/td['ema50']-1)*100):+.1f}%" if td.get("ema50") else "—",
            "5d mom":      _fmt_mom(td.get("mom_5d")),
            "20d mom":     _fmt_mom(td.get("mom_20d")),
            "% from 52wk high": f"{td['pct_from_hi']:+.1f}%",
            "Trend":       td["trend"],
            "Signal":      td["signal"],
        })

    if sig_rows:
        sig_df = pd.DataFrame(sig_rows)
        st.dataframe(sig_df, width='stretch', hide_index=True)
    else:
        st.info("Technical data unavailable — chart data still loading.")

    st.divider()

    # -----------------------------------------------------------------------
    # Advice
    # -----------------------------------------------------------------------

    st.subheader("Advice")

    adv_cols = st.columns(len(metal_tech) or 1)
    for col, (name, td) in zip(adv_cols, metal_tech.items()):
        sig = td["signal"]
        trend = td["trend"]
        ema20 = td["ema20"]
        price = td["price"]
        hi_52 = td["hi_52"]
        lo_52 = td["lo_52"]

        with col:
            st.markdown(f"**{name}**")
            if sig == "Bullish":
                st.success(f"Uptrend intact. Price above EMA20 (${ema20:,.0f}) and EMA50.")
                st.markdown(
                    f"- Support: EMA20 ~${ema20:,.0f}\n"
                    f"- 52wk high: ${hi_52:,.0f} ({td['pct_from_hi']:+.1f}%)\n"
                    f"- Pullbacks toward EMA20 are buying opportunities while trend holds."
                )
            elif sig == "Recovering":
                st.warning(f"Recovering — price above EMA20 but below EMA50.")
                st.markdown(
                    f"- Watch for EMA50 (${td['ema50']:,.0f}) reclaim as confirmation.\n"
                    f"- Support: EMA20 ~${ema20:,.0f}\n"
                    f"- Not yet a clean long — wait for EMA50 cross."
                )
            elif sig == "Watch":
                st.warning(f"Pulling back below EMA20 — trend under pressure.")
                st.markdown(
                    f"- Key level: EMA20 ~${ema20:,.0f} (now resistance)\n"
                    f"- EMA50 at ${td['ema50']:,.0f} is next support\n"
                    f"- Avoid new longs until EMA20 reclaimed."
                )
            else:
                st.warning("Downtrend — below both EMA20 and EMA50.")
                st.markdown(
                    f"- Avoid long exposure\n"
                    f"- EMA20 ~${ema20:,.0f} is now resistance\n"
                    f"- 52wk low: ${lo_52:,.0f} is next support\n"
                    f"- Wait for trend reversal confirmation."
                )

    st.divider()

    # -----------------------------------------------------------------------
    # Price chart
    # -----------------------------------------------------------------------

    st.subheader("1-year price chart")

    chart_options = {
        "Gold (GC=F)":           "GC=F",
        "Silver (SI=F)":         "SI=F",
        "Platinum (PL=F)":       "PL=F",
        "Palladium (PA=F)":      "PA=F",
        "GLD — SPDR Gold":       "GLD",
        "SLV — iShares Silver":  "SLV",
        "GDX — Gold Miners":     "GDX",
        "GDXJ — Jr Gold Miners": "GDXJ",
    }

    chart_label  = st.selectbox("Select instrument", list(chart_options.keys()), key="metal_chart_sel")
    chart_ticker = chart_options[chart_label]
    chart_df     = fetch_metal_chart(chart_ticker)

    if not chart_df.empty and "Close" in chart_df.columns:
        # Overlay EMA20 / EMA50 / EMA200 with colourblind-safe palette (Wong).
        # Order of colours matches the column order below.
        chart_df = chart_df.copy()
        chart_df["EMA20"]  = chart_df["Close"].ewm(span=20,  adjust=False).mean()
        chart_df["EMA50"]  = chart_df["Close"].ewm(span=50,  adjust=False).mean()
        chart_df["EMA200"] = chart_df["Close"].ewm(span=200, adjust=False).mean()
        st.line_chart(
            chart_df[["Close", "EMA20", "EMA50", "EMA200"]],
            color=["#000000", "#E69F00", "#56B4E9", "#CC79A7"],
            width='stretch',
        )
        latest_close = float(chart_df["Close"].iloc[-1])
        hi_52 = float(chart_df["High"].max()) if "High" in chart_df.columns else latest_close
        lo_52 = float(chart_df["Low"].min())  if "Low"  in chart_df.columns else latest_close
        ci1, ci2, ci3 = st.columns(3)
        ci1.metric("52wk high", f"${hi_52:,.2f}")
        ci2.metric("52wk low",  f"${lo_52:,.2f}")
        ci3.metric("% from high", f"{(latest_close/hi_52-1)*100:+.1f}%")
    else:
        st.info(f"No chart data for {chart_ticker}.")

    st.divider()

    # -----------------------------------------------------------------------
    # Holdings tracker
    # -----------------------------------------------------------------------

    st.subheader("Holdings")

    conn = get_conn()
    metal_df = pd.read_sql("SELECT * FROM metal_holdings ORDER BY id", conn)
    conn.close()

    metal_rows = []
    for _, h in metal_df.iterrows():
        if h["holding_type"] == "physical":
            fut_ticker = _METAL_FUTURES.get(h["metal"])
            px_usd   = metal_px.get(fut_ticker, {}).get("price", 0.0) if fut_ticker else 0.0
            prev_usd = metal_px.get(fut_ticker, {}).get("prev_close", 0.0) if fut_ticker else 0.0
        else:
            px_usd   = metal_px.get(h["metal"], {}).get("price", 0.0)
            prev_usd = metal_px.get(h["metal"], {}).get("prev_close", 0.0)

        qty          = h["quantity"]
        avg_buy_nzd  = h["avg_buy_price_nzd"]
        cost_nzd     = qty * avg_buy_nzd

        if px_usd and nzdusd_m:
            current_nzd_per_unit = px_usd / nzdusd_m
            current_nzd  = qty * current_nzd_per_unit
            pl_nzd       = current_nzd - cost_nzd
            pl_pct       = (current_nzd_per_unit - avg_buy_nzd) / avg_buy_nzd * 100 if avg_buy_nzd else 0.0
            today_pl_nzd = qty * (px_usd - prev_usd) / nzdusd_m if prev_usd else 0.0
        else:
            current_nzd = cost_nzd
            pl_nzd = pl_pct = today_pl_nzd = 0.0

        metal_rows.append({
            "id":            h["id"],
            "Metal/ETF":     h["metal"],
            "Type":          h["holding_type"],
            "Quantity":      qty,
            "Avg buy (NZD)": round(avg_buy_nzd, 4),
            "Cost (NZD)":    round(cost_nzd, 2),
            "Current (NZD)": round(current_nzd, 2),
            "P&L (NZD)":     round(pl_nzd, 2),
            "P&L %":         round(pl_pct, 2),
            "Today (NZD)":   round(today_pl_nzd, 2),
            "Notes":         h["notes"],
        })

    if metal_rows:
        m_cost  = sum(r["Cost (NZD)"]    for r in metal_rows)
        m_value = sum(r["Current (NZD)"] for r in metal_rows)
        m_pl    = sum(r["P&L (NZD)"]     for r in metal_rows)
        m_today = sum(r["Today (NZD)"]   for r in metal_rows)
        m_pl_pct = (m_pl / m_cost * 100) if m_cost else 0.0
        ms1, ms2, ms3, ms4 = st.columns(4)
        ms1.metric("Value",     f"NZD {m_value:,.2f}")
        ms2.metric("Today",     f"NZD {m_today:+,.2f}", delta=f"{m_today:+.2f}", delta_color="normal")
        ms3.metric("Total P&L", f"NZD {m_pl:+,.2f}",   delta=f"{m_pl_pct:+.2f}%", delta_color="normal")
        ms4.metric("NZD/USD",   f"{nzdusd_m:.4f}")
        st.divider()

    with st.expander("Add holding", expanded=False):
        with st.form("add_metal_form", clear_on_submit=True):
            mf1, mf2, mf3 = st.columns(3)
            holding_type_in = mf1.selectbox("Type", ["physical", "etf"])
            if holding_type_in == "physical":
                metal_in  = mf2.selectbox("Metal", list(_METAL_FUTURES.keys()))
                qty_label = "Quantity (oz)"
            else:
                metal_in  = mf2.selectbox("ETF", list(_METAL_ETFS.keys()))
                qty_label = "Shares"
            avg_nzd_in  = mf3.number_input("Avg buy price (NZD)", min_value=0.0, format="%.4f")
            qty_in      = st.number_input(qty_label, min_value=0.0, format="%.4f")
            metal_notes = st.text_area("Notes")
            m_submitted = st.form_submit_button("Add holding")
            if m_submitted:
                if qty_in <= 0 or avg_nzd_in <= 0:
                    st.error("Quantity and avg buy price required.")
                else:
                    conn = get_conn()
                    conn.execute(
                        "INSERT INTO metal_holdings (metal, holding_type, quantity, avg_buy_price_nzd, notes) VALUES (?,?,?,?,?)",
                        (metal_in, holding_type_in, qty_in, avg_nzd_in, metal_notes),
                    )
                    conn.commit()
                    conn.close()
                    qty_unit = "oz" if holding_type_in == "physical" else "shares"
                    st.success(f"Added {qty_in} {qty_unit} of {metal_in}")
                    st.cache_data.clear()
                    st.rerun()

    if metal_rows:
        display_metals = pd.DataFrame(metal_rows).drop(columns=["id"])
        st.dataframe(display_metals, width='stretch', hide_index=True,
            column_config={
                "P&L (NZD)":   st.column_config.NumberColumn("P&L (NZD)",   format="%.2f"),
                "P&L %":       st.column_config.NumberColumn("P&L %",       format="%.2f%%"),
                "Today (NZD)": st.column_config.NumberColumn("Today (NZD)", format="%.2f"),
            })

        with st.expander("Notes", expanded=False):
            for r in metal_rows:
                if r["Notes"]:
                    st.markdown(f"**{r['Metal/ETF']}** ({r['Type']}) — {r['Notes']}")

        with st.expander("Delete a holding", expanded=False):
            del_options = {f"#{r['id']}  {r['Metal/ETF']}  ({r['Type']}, {r['Quantity']} units)": r["id"] for r in metal_rows}
            del_label   = st.selectbox("Select holding", list(del_options.keys()), key="metal_del_sel")
            if st.button("Delete holding", type="secondary", key="metal_del_btn"):
                conn = get_conn()
                conn.execute("DELETE FROM metal_holdings WHERE id = ?", (del_options[del_label],))
                conn.commit()
                conn.close()
                st.cache_data.clear()
                st.rerun()
    else:
        st.info("No holdings recorded. Add one above.")

    st.caption(f"Spot prices: 5min  •  Charts: 1hr  •  NZD/USD {nzdusd_m:.4f}")
