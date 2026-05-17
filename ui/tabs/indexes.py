"""
ui/tabs/indexes.py — Macro & sector ETF dashboard.

Group selector (US Major / Sectors / Themes / Macro), per-symbol scan table
with relative strength vs SPY, plus a detail panel with chart and bias read.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


_INDEX_GROUPS = {
    "US Major": {
        "S&P 500 (SPY)":        "SPY",
        "Nasdaq-100 (QQQ)":     "QQQ",
        "Russell 2000 (IWM)":   "IWM",
        "Dow Jones (DIA)":      "DIA",
        "Total Market (VTI)":   "VTI",
    },
    "Sectors": {
        "Financials (XLF)":         "XLF",
        "Energy (XLE)":             "XLE",
        "Technology (XLK)":         "XLK",
        "Health Care (XLV)":        "XLV",
        "Industrials (XLI)":        "XLI",
        "Consumer Disc. (XLY)":     "XLY",
        "Consumer Staples (XLP)":   "XLP",
        "Utilities (XLU)":          "XLU",
        "Materials (XLB)":          "XLB",
        "Real Estate (XLRE)":       "XLRE",
        "Communications (XLC)":     "XLC",
    },
    "Themes": {
        "Semiconductors (SMH)":     "SMH",
        "Semis alt (SOXX)":         "SOXX",
        "Regional Banks (KRE)":     "KRE",
        "Biotech (XBI)":            "XBI",
        "Defense / Aerospace (ITA)":"ITA",
        "Innovation (ARKK)":        "ARKK",
        "Cyber (HACK)":             "HACK",
        "Clean Energy (ICLN)":      "ICLN",
        "Quantum (QTUM)":           "QTUM",
    },
    "Macro": {
        "Volatility (^VIX)":        "^VIX",
        "Gold Miners (GDX)":        "GDX",
        "Long Bonds (TLT)":         "TLT",
        "Dollar Index (UUP)":       "UUP",
        "EAFE intl (EFA)":          "EFA",
        "Emerging Markets (EEM)":   "EEM",
        "China (FXI)":              "FXI",
        "Japan (EWJ)":              "EWJ",
    },
}


@st.cache_data(ttl=300)
def _idx_fetch(symbol: str) -> dict:
    """Fetch 6mo OHLCV + indicators for an index/ETF symbol.
    VIX gets regime-based bias (>25 risk-off, <15 complacency)."""
    import yfinance as _yf
    try:
        data = _yf.Ticker(symbol).history(period="6mo", interval="1d")
        if data.empty or len(data) < 20:
            return {}
        close = data["Close"]
        price  = float(close.iloc[-1])
        prev   = float(close.iloc[-2])
        sma20  = float(close.rolling(20).mean().iloc[-1])
        sma50  = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else sma20

        delta = close.diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs    = gain / loss.replace(0, 1e-9)
        rsi_v = float((100 - 100 / (1 + rs)).iloc[-1])

        high, low = data["High"], data["Low"]
        prev_close = close.shift()
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_v = float(tr.rolling(14).mean().iloc[-1])

        change_pct = (price - prev) / prev * 100 if prev else 0
        high_20    = float(close.iloc[-20:].max())
        low_20     = float(close.iloc[-20:].min())

        if symbol == "^VIX":
            if   price >= 25:  bias = "⚠ Risk-off"
            elif price <= 15:  bias = "Complacency"
            else:              bias = "Normal"
        else:
            if   price > sma20 > sma50 and rsi_v > 50:  bias = "↗ Bullish"
            elif price < sma20 < sma50 and rsi_v < 50:  bias = "↘ Bearish"
            else:                                       bias = "→ Range"

        return {
            "price":        price,
            "change_pct":   change_pct,
            "rsi":          rsi_v,
            "atr":          atr_v,
            "sma20":        sma20,
            "sma50":        sma50,
            "high_20":      high_20,
            "low_20":       low_20,
            "bias":         bias,
            "close_series": close,
        }
    except Exception:
        return {}


def render() -> None:
    """Render the Indexes tab."""
    st.subheader("📊 Indexes & sector ETFs")
    st.caption(
        "Macro and sector context. Use this to answer: **is the move I'm "
        "seeing in NVDA a semis-wide move, a broad-market move, or NVDA-specific?** "
        "Sector ETFs are tradable; pure indexes (e.g. ^VIX, ^GSPC) are "
        "informational only."
    )

    group_choice = st.radio(
        "Group",
        list(_INDEX_GROUPS.keys()),
        index=0,
        horizontal=True,
        key="idx_group",
    )
    _universe = _INDEX_GROUPS[group_choice]

    _spy = _idx_fetch("SPY")
    _spy_change = _spy.get("change_pct", 0.0) if _spy else 0.0

    rows = []
    for _label, _sym in _universe.items():
        d = _idx_fetch(_sym)
        if not d:
            continue
        rs_vs_spy = d["change_pct"] - _spy_change if _sym != "SPY" else 0.0
        rows.append({
            "Index":      _label,
            "Symbol":     _sym,
            "Price":      d["price"],
            "Change %":   d["change_pct"],
            "vs SPY %":   rs_vs_spy,
            "RSI(14)":    d["rsi"],
            "vs SMA20 %": (d["price"] - d["sma20"]) / d["sma20"] * 100 if d["sma20"] else 0,
            "Bias":       d["bias"],
        })

    if rows:
        idx_df = pd.DataFrame(rows)
        st.dataframe(
            idx_df,
            width='stretch',
            hide_index=True,
            column_config={
                "Index":      st.column_config.TextColumn("Index"),
                "Symbol":     st.column_config.TextColumn("Symbol", width="small"),
                "Price":      st.column_config.NumberColumn("Price",      format="$%.2f"),
                "Change %":   st.column_config.NumberColumn("Change %",   format="%+.2f%%"),
                "vs SPY %":   st.column_config.NumberColumn(
                    "vs SPY %",
                    format="%+.2f%%",
                    help="Today's % move minus SPY's. Positive = outperforming the broad market.",
                ),
                "RSI(14)":    st.column_config.NumberColumn("RSI(14)",    format="%.1f"),
                "vs SMA20 %": st.column_config.NumberColumn("vs SMA20 %", format="%+.2f%%"),
                "Bias":       st.column_config.TextColumn("Bias", width="small"),
            },
        )
    else:
        st.error("Couldn't fetch index data from Yahoo. Try again in a minute.")

    st.divider()

    # -----------------------------------------------------------------------
    # Detail panel — one index at a time
    # -----------------------------------------------------------------------
    st.subheader("Detail")
    _flat = {label: sym for grp in _INDEX_GROUPS.values() for label, sym in grp.items()}
    detail_label = st.selectbox(
        "Choose an index / ETF",
        list(_flat.keys()),
        index=0,
        key="idx_detail",
    )
    if detail_label:
        sym = _flat[detail_label]
        d = _idx_fetch(sym)
        if d:
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Price", f"${d['price']:.2f}", f"{d['change_pct']:+.2f}%")
            mc2.metric("RSI(14)", f"{d['rsi']:.1f}")
            mc3.metric("ATR(14)", f"${d['atr']:.2f}")
            mc4.metric("Bias", d["bias"])
            mc5.metric(
                "20d high / low",
                f"${d['high_20']:.2f} / ${d['low_20']:.2f}",
            )

            # Chart with SMA20 + SMA50
            try:
                import altair as _ix_alt
                _series = d["close_series"]
                _ix_df = pd.DataFrame({
                    "Date":  _series.index,
                    "Close": _series.values,
                    "SMA20": _series.rolling(20).mean().values,
                    "SMA50": _series.rolling(50).mean().values,
                })
                try:
                    _ix_df["Date"] = _ix_df["Date"].dt.tz_localize(None)
                except Exception:
                    pass
                _long = _ix_df.melt(
                    id_vars="Date",
                    value_vars=["Close", "SMA20", "SMA50"],
                    var_name="Series",
                    value_name="Price",
                )
                _line = (
                    _ix_alt.Chart(_long)
                    .mark_line()
                    .encode(
                        x=_ix_alt.X("Date:T", axis=_ix_alt.Axis(title=None)),
                        y=_ix_alt.Y("Price:Q",
                                    axis=_ix_alt.Axis(title=f"{sym} price"),
                                    scale=_ix_alt.Scale(zero=False)),
                        color=_ix_alt.Color(
                            "Series:N",
                            scale=_ix_alt.Scale(
                                domain=["Close", "SMA20", "SMA50"],
                                range=["#1f77b4", "#ff7f0e", "#7f7f7f"],
                            ),
                            legend=_ix_alt.Legend(orient="top", title=None),
                        ),
                        tooltip=[
                            _ix_alt.Tooltip("Date:T"),
                            _ix_alt.Tooltip("Series:N"),
                            _ix_alt.Tooltip("Price:Q", format="$.2f"),
                        ],
                    )
                    .properties(height=320)
                    .interactive()
                )
                st.altair_chart(_line, width='stretch')
            except Exception:
                st.line_chart(d["close_series"])

            sma_dist = (d["price"] - d["sma20"]) / d["sma20"] * 100 if d["sma20"] else 0
            hi_dist  = (d["high_20"] - d["price"]) / d["price"] * 100
            lo_dist  = (d["price"]   - d["low_20"]) / d["price"] * 100
            st.markdown(
                f"**Where price sits:** {abs(sma_dist):.1f}% "
                f"{'above' if sma_dist > 0 else 'below'} SMA20 · "
                f"{hi_dist:.1f}% below 20-day high · "
                f"{lo_dist:.1f}% above 20-day low."
            )

            if sym == "^VIX":
                st.caption(
                    "_**VIX is the fear gauge.** Above 25 = risk-off (sell rallies, "
                    "small size, hedge). Below 15 = complacency (be wary of trend "
                    "trades — vol regime usually expands from here). 15-25 = normal._"
                )
        else:
            st.error(f"Couldn't fetch data for {detail_label}.")

    st.divider()

    with st.expander("ℹ️ How to use this tab"):
        st.markdown("""
**For every trade you consider in Lookup or Advice, glance at the relevant
sector here first:**

- Trading **NVDA**? Check **SMH** or **SOXX** to see if it's a sector move.
- Trading **SOFI / fintechs**? Check **XLF** (financials) and **KRE** (regional banks).
- Trading **defense / govtech (KTOS, LMT)**? Check **ITA**.
- Trading **biotech (XBI components)**? Check **XBI**.
- Trading **anything tech**? Cross-check with **QQQ** for broad-tech sentiment.

**Always glance at VIX before sizing up:**
- VIX above 25 → cut size, wider stops, prefer spreads to outrights.
- VIX below 15 → don't get complacent. Long-vol regimes usually start here.

**Sector-relative-strength** (RS) is a real edge — if SMH is up +0.3% but
NVDA is up +4%, that's stock-specific strength worth a closer look. If SMH
is up +3% and NVDA is up +3.2%, you're mostly just paying for sector beta.

_The current tab shows absolute moves. Future enhancement: per-row relative
strength vs SPY (Phase 12+ candidate)._
""")
