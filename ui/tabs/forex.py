"""
ui/tabs/forex.py — FOREX currency pair analytical screen.

Standalone tab — no shared app state. All data pulled live from yfinance.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


_FX_MAJORS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
}
_FX_CROSSES = {
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "AUDJPY": "AUDJPY=X",
    "EURAUD": "EURAUD=X",
    "CHFJPY": "CHFJPY=X",
}


@st.cache_data(ttl=300)
def _fx_fetch(yf_symbol: str) -> dict:
    import yfinance as _yf
    try:
        data = _yf.Ticker(yf_symbol).history(period="6mo", interval="1d")
        if data.empty or len(data) < 20:
            return {}
        close = data["Close"]
        price  = float(close.iloc[-1])
        prev   = float(close.iloc[-2])
        sma20  = float(close.rolling(20).mean().iloc[-1])
        sma50  = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else sma20

        # RSI(14) — Wilder's smoothing approximation via rolling mean
        delta = close.diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs    = gain / loss.replace(0, 1e-9)
        rsi_v = float((100 - 100 / (1 + rs)).iloc[-1])

        # ATR(14)
        high, low = data["High"], data["Low"]
        prev_close = close.shift()
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_v = float(tr.rolling(14).mean().iloc[-1])

        is_jpy        = "JPY" in yf_symbol
        pip_size      = 0.01 if is_jpy else 0.0001
        price_decimals = 3 if is_jpy else 5

        change_pct = (price - prev) / prev * 100 if prev else 0
        pip_move   = (price - prev) / pip_size if prev else 0
        atr_pips   = atr_v / pip_size
        high_20    = float(close.iloc[-20:].max())
        low_20     = float(close.iloc[-20:].min())

        if   price > sma20 > sma50 and rsi_v > 50:  bias = "↗ Bullish"
        elif price < sma20 < sma50 and rsi_v < 50:  bias = "↘ Bearish"
        else:                                       bias = "→ Range"

        return {
            "price":         price,
            "change_pct":    change_pct,
            "pip_move":      pip_move,
            "rsi":           rsi_v,
            "atr_pips":      atr_pips,
            "sma20":         sma20,
            "sma50":         sma50,
            "high_20":       high_20,
            "low_20":        low_20,
            "bias":          bias,
            "pip_size":      pip_size,
            "price_decimals": price_decimals,
            "close_series":  close,
        }
    except Exception:
        return {}


def _fx_fmt_price(p: float, decimals: int) -> str:
    return f"{p:.{decimals}f}"


def render() -> None:
    """Render the FOREX tab."""
    st.subheader("💱 FOREX — Currency pair analysis")
    st.warning(
        "**FX is leveraged trading** — most retail traders lose money (industry "
        "data shows 70-80% in any given quarter). This tab is an **analytical "
        "screen, not advice**. Volume-based signals (RVOL, breakouts on volume) "
        "don't apply to FX — there is no central exchange. See "
        "**Learn → Lesson 10: FOREX basics** for the full framework."
    )

    universe_choice = st.radio(
        "Pair set",
        ["Majors", "Crosses", "Both"],
        index=0,
        horizontal=True,
        key="fx_universe",
    )
    if universe_choice == "Majors":
        _universe = _FX_MAJORS
    elif universe_choice == "Crosses":
        _universe = _FX_CROSSES
    else:
        _universe = {**_FX_MAJORS, **_FX_CROSSES}

    rows = []
    for _name, _sym in _universe.items():
        d = _fx_fetch(_sym)
        if not d:
            continue
        rows.append({
            "Pair":       _name,
            "Price":      _fx_fmt_price(d["price"], d["price_decimals"]),
            "Change %":   d["change_pct"],
            "Pips today": d["pip_move"],
            "RSI(14)":    d["rsi"],
            "ATR pips":   d["atr_pips"],
            "Bias":       d["bias"],
        })

    if rows:
        fx_df = pd.DataFrame(rows)
        st.dataframe(
            fx_df,
            width='stretch',
            hide_index=True,
            column_config={
                "Change %":   st.column_config.NumberColumn("Change %", format="%+.2f%%"),
                "Pips today": st.column_config.NumberColumn("Pips today", format="%+.1f"),
                "RSI(14)":    st.column_config.NumberColumn("RSI(14)", format="%.1f"),
                "ATR pips":   st.column_config.NumberColumn("ATR pips", format="%.1f"),
            },
        )
    else:
        st.error(
            "Couldn't fetch FX data from Yahoo. Possible rate-limit or outage. "
            "Try again in a minute."
        )

    st.divider()

    # -----------------------------------------------------------------------
    # Pair detail
    # -----------------------------------------------------------------------
    st.subheader("Pair detail")
    detail_pair = st.selectbox(
        "Choose a pair",
        list({**_FX_MAJORS, **_FX_CROSSES}.keys()),
        index=0,
        key="fx_detail_pair",
    )
    if detail_pair:
        _sym = {**_FX_MAJORS, **_FX_CROSSES}[detail_pair]
        d = _fx_fetch(_sym)
        if d:
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric(
                "Price",
                _fx_fmt_price(d["price"], d["price_decimals"]),
                f"{d['change_pct']:+.2f}% / {d['pip_move']:+.1f} pips",
            )
            mc2.metric("RSI(14)", f"{d['rsi']:.1f}")
            mc3.metric("ATR (pips)", f"{d['atr_pips']:.1f}")
            mc4.metric("Bias", d["bias"])
            mc5.metric(
                "20d high / low",
                f"{_fx_fmt_price(d['high_20'], d['price_decimals'])} / "
                f"{_fx_fmt_price(d['low_20'], d['price_decimals'])}",
            )

            st.line_chart(d["close_series"])

            high_dist_pips = (d["high_20"] - d["price"]) / d["pip_size"]
            low_dist_pips  = (d["price"]   - d["low_20"]) / d["pip_size"]
            sma20_dist     = (d["price"]   - d["sma20"])  / d["pip_size"]

            st.markdown(
                f"**Where price sits:** {abs(high_dist_pips):.0f} pips "
                f"{'below' if high_dist_pips > 0 else 'above'} 20-day high · "
                f"{abs(low_dist_pips):.0f} pips "
                f"{'above' if low_dist_pips > 0 else 'below'} 20-day low · "
                f"{abs(sma20_dist):.0f} pips "
                f"{'above' if sma20_dist > 0 else 'below'} SMA20."
            )
            st.caption(
                f"_Daily ATR is **{d['atr_pips']:.0f} pips** — expect roughly that "
                f"much movement on a typical day. A stop that's tighter than "
                f"1×ATR will likely get hit by noise; one that's wider than 2×ATR "
                f"is risking more than the daily range. RSI above 70 = overbought, "
                f"below 30 = oversold. Bias is a simple price-vs-SMA20-vs-SMA50 "
                f"read; range-bound bias means mean-reversion strategies fit better "
                f"than trend-following._"
            )
        else:
            st.error(f"Couldn't fetch data for {detail_pair}.")

    st.divider()

    with st.expander("⚠️ Limitations and external resources you need", expanded=False):
        st.markdown(
            """
- **Yahoo FX is a delayed, aggregated feed.** Your broker's quote will
  differ; spreads are wider than what's shown. Use this for *analysis*, not
  for entry decisions.
- **No economic calendar here.** Check before trading:
  - [forexfactory.com](https://www.forexfactory.com) — most popular calendar, color-coded by impact
  - [investing.com/economic-calendar](https://www.investing.com/economic-calendar) — broader coverage
- **High-impact events move FX 50–100+ pips in seconds.** Don't hold positions
  through NFP (first Friday), CPI releases, FOMC, ECB, or BOE rate decisions
  unless that's the trade.
- **No carry / swap data.** Long AUDJPY or NZDJPY pays positive overnight
  rollover at your broker (varies). Short the same pairs and you pay.
- **No central bank statement parsing.** That's where the real macro moves
  come from. Read the actual statements if you trade through them.
- **Weekend gap risk.** FX is closed Friday 5pm ET → Sunday 5pm ET. Anything
  significant on the weekend (war, surprise rate cut) gaps the Sunday open
  past stops.
            """
        )
