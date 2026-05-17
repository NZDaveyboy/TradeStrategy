"""
ui/tabs/portfolio.py — Portfolio Builder tab.

Builds a "speculative" sleeve (high-RVOL momentum longs + crypto + GDXJ
in bull gold) and an "investment" sleeve (quality longs + GLD/SLV/GDX
in bull gold), with per-position sizing, risk flags, and a dream-sector
breakdown. All sleeve construction is pure — `_build_speculative` and
`_build_investment` take screener data + metal context and return lists.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import streamlit as st

from ui.data import (
    fetch_nzdusd,
    fetch_metal_prices,
    fetch_metal_technicals,
    fetch_market_context,
    METAL_FUTURES     as _METAL_FUTURES,
    METAL_ETFS        as _METAL_ETFS,
    ASSET_DRIVERS     as _ASSET_DRIVERS,
)
from ui.helpers import regime_label as _regime_label, driver_tags as _driver_tags_raw


def _driver_tags(ticker: str, screener_row: dict | None = None) -> list[str]:
    """Local wrapper that injects ASSET_DRIVERS into the pure helper."""
    return _driver_tags_raw(ticker, screener_row, _ASSET_DRIVERS)


# ---------------------------------------------------------------------------
# Portfolio construction engine
# ---------------------------------------------------------------------------

_INVESTMENT_GRADE_NAMES = {
    "NVDA","ASML","TSM","AAPL","MSFT","GOOGL","AMZN","META",
    "TSLA","ARM","AVGO","ORCL","CRM","ADBE","SHOP","NFLX",
    "GLD","SLV","GDX","SPY","QQQ",
}

_SPEC_EXCLUDE = {"SPY","QQQ"}   # index ETFs not speculative picks


def _build_speculative(screener_df: pd.DataFrame, metal_tech: dict, mctx: dict) -> list[dict]:
    """
    Select 4-6 speculative picks: high-RVOL momentum longs + best crypto.
    If gold is bullish, add GDXJ as a levered play.
    """
    picks = []

    if screener_df.empty:
        return picks

    # Long momentum equities — sort by RVOL, filter for quality
    eq = screener_df[
        (screener_df["direction"] == "long")
        & screener_df["setup_type"].str.contains("momentum", case=False, na=False)
        & (screener_df["tradescore"].fillna(0) >= 30)
        & ~screener_df["ticker"].str.endswith("-USD")
        & ~screener_df["ticker"].isin(_SPEC_EXCLUDE)
    ].sort_values(["rvol", "tradescore"], ascending=False)

    for _, row in eq.head(3).iterrows():
        price = float(row.get("price") or 0)
        stop  = float(row.get("stop_loss") or 0)
        risk_pct = ((price - stop) / price * 100) if price and stop and price > stop else None
        picks.append({
            "ticker":      row["ticker"],
            "display":     row["ticker"],
            "asset_class": "equity",
            "direction":   "long",
            "price":       price,
            "stop":        stop if stop else None,
            "risk_pct":    risk_pct,
            "tradescore":  float(row.get("tradescore") or 0),
            "rvol":        float(row.get("rvol") or 0),
            "rsi":         float(row.get("rsi") or 0),
            "setup_type":  row.get("setup_type", ""),
            "rationale":   (
                f"High-RVOL momentum breakout ({row.get('rvol', 0):.1f}x volume). "
                f"TradeScore {row.get('tradescore', 0):.0f}. RSI {row.get('rsi', 0):.0f} — not exhausted. "
                f"Entry on intraday confirmation above ${price:.2f}."
            ),
            "weight":      0.10,
            "hold":        "3–10 trading days",
            "exit":        f"Stop ${stop:.2f}" if stop else "EMA20 as trailing stop",
        })

    # Best crypto long
    crypto = screener_df[
        screener_df["ticker"].str.endswith("-USD")
        & (screener_df["direction"] == "long")
        & (screener_df["tradescore"].fillna(0) >= 30)
    ].sort_values("tradescore", ascending=False)

    btc_d = mctx.get("btc", {})
    crypto_ok = btc_d.get("above_ema20", False)

    for _, row in crypto.head(2 if crypto_ok else 1).iterrows():
        coin = row["ticker"].replace("-USD", "")
        price = float(row.get("price") or 0)
        picks.append({
            "ticker":      row["ticker"],
            "display":     coin,
            "asset_class": "crypto",
            "direction":   "long",
            "price":       price,
            "stop":        float(row.get("stop_loss") or 0) or None,
            "risk_pct":    None,
            "tradescore":  float(row.get("tradescore") or 0),
            "rvol":        float(row.get("rvol") or 0),
            "rsi":         float(row.get("rsi") or 0),
            "setup_type":  row.get("setup_type", ""),
            "rationale":   (
                f"{'BTC in uptrend — crypto risk-on environment. ' if crypto_ok else ''}"
                f"TradeScore {row.get('tradescore', 0):.0f}. "
                f"Digital asset with high beta to risk-on moves. Size small — volatility is 3–5x equities."
            ),
            "weight":      0.08,
            "hold":        "1–4 weeks",
            "exit":        "EMA20 break on daily chart",
        })

    # Junior miners if gold is bullish
    gold_signal = metal_tech.get("Gold", {}).get("signal", "")
    if gold_signal == "Bullish" and len(picks) < 6:
        gdxj_p = fetch_metal_prices().get("GDXJ", {}).get("price", 0)
        picks.append({
            "ticker":      "GDXJ",
            "display":     "GDXJ",
            "asset_class": "etf",
            "direction":   "long",
            "price":       gdxj_p,
            "stop":        None,
            "risk_pct":    None,
            "tradescore":  None,
            "rvol":        None,
            "rsi":         None,
            "setup_type":  "ETF",
            "rationale":   (
                "Gold in confirmed uptrend. Junior miners amplify gold moves 2–3x. "
                "GDXJ provides levered exposure without single-stock risk. "
                "Hold as long as gold stays above EMA20."
            ),
            "weight":      0.07,
            "hold":        "Weeks to months (follow gold trend)",
            "exit":        "Gold breaks EMA20",
        })

    return picks


def _build_investment(screener_df: pd.DataFrame, metal_tech: dict, mctx: dict) -> list[dict]:
    """
    Select 4–6 investment-grade picks: high-TradeScore quality names + gold allocation.
    """
    picks = []

    if not screener_df.empty:
        # Quality long setups — high TradeScore, established names
        quality = screener_df[
            (screener_df["direction"] == "long")
            & (screener_df["tradescore"].fillna(0) >= 40)
            & ~screener_df["ticker"].str.endswith("-USD")
            & screener_df["ticker"].isin(_INVESTMENT_GRADE_NAMES)
        ].sort_values("tradescore", ascending=False)

        for _, row in quality.head(4).iterrows():
            price = float(row.get("price") or 0)
            stop  = float(row.get("stop_loss") or 0)
            ts    = float(row.get("tradescore") or 0)
            picks.append({
                "ticker":      row["ticker"],
                "display":     row["ticker"],
                "asset_class": "equity",
                "direction":   "long",
                "price":       price,
                "stop":        stop if stop else None,
                "risk_pct":    ((price - stop) / price * 100) if price and stop and price > stop else None,
                "tradescore":  ts,
                "rvol":        float(row.get("rvol") or 0),
                "rsi":         float(row.get("rsi") or 0),
                "setup_type":  row.get("setup_type", ""),
                "rationale":   (
                    f"High-conviction setup. TradeScore {ts:.0f} — above the 45-point threshold for quality. "
                    f"Established name with institutional following. "
                    f"RSI {row.get('rsi', 0):.0f} — momentum without being extended. "
                    f"Size larger than speculative — stop is defined, thesis is durable."
                ),
                "weight":      0.20,
                "hold":        "2–8 weeks",
                "exit":        f"Close below EMA20 (stop ~${stop:.2f})" if stop else "EMA20 as trailing stop",
            })

    # Gold — always include if signal is bullish or neutral
    gold_td = metal_tech.get("Gold", {})
    gold_sig = gold_td.get("signal", "")
    if gold_sig in ("Bullish", "Recovering", "Watch") or not screener_df.empty:
        gold_p = fetch_metal_prices().get("GC=F", {}).get("price", 0)
        gld_p  = fetch_metal_prices().get("GLD",  {}).get("price", 0)
        ema20  = gold_td.get("ema20", 0)
        picks.append({
            "ticker":      "GC=F",
            "display":     "Gold (via GLD ETF)",
            "asset_class": "metal",
            "direction":   "long",
            "price":       gld_p or gold_p,
            "stop":        round(ema20 / (fetch_metal_prices().get("GC=F", {}).get("price", 1) or 1) * (gld_p or gold_p) * 0.98, 2) if ema20 and gld_p else None,
            "risk_pct":    None,
            "tradescore":  None,
            "rvol":        None,
            "rsi":         None,
            "setup_type":  "Store of value",
            "rationale":   (
                f"Gold signal: {gold_sig}. Core defensive allocation — not a trade, a position. "
                "Negative correlation to USD and equities in risk-off environments. "
                "Inflation hedge with no counterparty risk. "
                "Use GLD ETF for liquidity; hold as long as macro drivers support (USD weakness, rate uncertainty)."
            ),
            "weight":      0.20,
            "hold":        "Months — macro driven",
            "exit":        f"Gold breaks below EMA20 (~${ema20:,.0f})" if ema20 else "EMA20 break",
        })

    return picks


# ---------------------------------------------------------------------------
# Dream portfolios — curated best-in-class picks per sector
# ---------------------------------------------------------------------------

_DREAM_SECTORS: list[dict] = [
    {
        "name": "AI & Semiconductors",
        "thesis": "AI infrastructure buildout — picks-and-shovels exposure. NVDA leads compute, "
                  "AVGO custom silicon, TSM/ASML the manufacturing chokepoints.",
        "horizon": "12+ months",
        "picks": [
            {"ticker": "NVDA", "weight": 0.25, "role": "AI compute leader"},
            {"ticker": "AVGO", "weight": 0.20, "role": "Custom AI silicon + networking"},
            {"ticker": "TSM",  "weight": 0.20, "role": "Foundry monopoly on leading-edge nodes"},
            {"ticker": "ASML", "weight": 0.15, "role": "EUV lithography monopoly"},
            {"ticker": "ARM",  "weight": 0.10, "role": "Architecture royalties on every chip"},
            {"ticker": "AMD",  "weight": 0.10, "role": "Second-source AI compute"},
        ],
    },
    {
        "name": "Mega-cap Tech",
        "thesis": "Cash machines with durable moats. Each is a near-monopoly in its lane.",
        "horizon": "Multi-year",
        "picks": [
            {"ticker": "MSFT",  "weight": 0.25, "role": "Enterprise + Azure + AI distribution"},
            {"ticker": "GOOGL", "weight": 0.20, "role": "Search + YouTube + Cloud + DeepMind"},
            {"ticker": "META",  "weight": 0.20, "role": "Ads engine + AI infra ROIC"},
            {"ticker": "AMZN",  "weight": 0.20, "role": "AWS margin + retail + ads"},
            {"ticker": "AAPL",  "weight": 0.15, "role": "Services flywheel + capital return"},
        ],
    },
    {
        "name": "Software & Cloud",
        "thesis": "Sticky enterprise revenue with the AI distribution layer on top.",
        "horizon": "6-18 months",
        "picks": [
            {"ticker": "CRM",  "weight": 0.25, "role": "Enterprise CRM standard"},
            {"ticker": "ORCL", "weight": 0.25, "role": "Database + cloud for AI training"},
            {"ticker": "ADBE", "weight": 0.20, "role": "Creative + GenAI tools"},
            {"ticker": "NOW",  "weight": 0.20, "role": "Workflow automation"},
            {"ticker": "SHOP", "weight": 0.10, "role": "Commerce infrastructure"},
        ],
    },
    {
        "name": "Crypto Majors",
        "thesis": "Digital store of value (BTC) plus settlement / smart-contract platforms (ETH, SOL). "
                  "High beta to risk-on. Size small.",
        "horizon": "Cycle (months to years)",
        "picks": [
            {"ticker": "BTC-USD", "weight": 0.60, "role": "Reserve asset of crypto"},
            {"ticker": "ETH-USD", "weight": 0.30, "role": "Settlement layer for DeFi"},
            {"ticker": "SOL-USD", "weight": 0.10, "role": "High-throughput L1 — higher risk"},
        ],
    },
    {
        "name": "Precious Metals",
        "thesis": "Hedge against USD weakness, inflation, geopolitical risk. "
                  "Miners amplify metal moves 2-3x.",
        "horizon": "Macro driven",
        "picks": [
            {"ticker": "GLD",  "weight": 0.50, "role": "Gold core position"},
            {"ticker": "SLV",  "weight": 0.20, "role": "Silver — industrial + monetary"},
            {"ticker": "GDX",  "weight": 0.20, "role": "Senior miners — operational leverage"},
            {"ticker": "GDXJ", "weight": 0.10, "role": "Junior miners — highest beta to gold"},
        ],
    },
    {
        "name": "Mobility & EV",
        "thesis": "Energy / transport transition with autonomy optionality. Volatile — size accordingly.",
        "horizon": "Multi-year",
        "picks": [
            {"ticker": "TSLA", "weight": 1.00, "role": "EV + storage + autonomy bet"},
        ],
    },
    {
        "name": "High-Risk Speculative Small + Mid-caps",
        "thesis": (
            "The asymmetric-upside sleeve. Names with binary outcomes — most will "
            "underperform but one or two could 5-10x on technology/regulatory wins. "
            "**Size this entire sector at 5-10% of your total portfolio** — not "
            "retirement money. Themes: quantum compute, space launch + comms, "
            "advanced nuclear, defense drones, Bitcoin-leveraged miners, fintech "
            "disruption."
        ),
        "horizon": "12+ months — survivors-bias play",
        "picks": [
            {"ticker": "IONQ", "weight": 0.15, "role": "Quantum compute pure-play (trapped-ion)"},
            {"ticker": "RKLB", "weight": 0.15, "role": "Small-launch + space systems (mid-cap)"},
            {"ticker": "ASTS", "weight": 0.10, "role": "Space-based cellular broadband — binary"},
            {"ticker": "OKLO", "weight": 0.10, "role": "Advanced small-modular nuclear — pre-revenue"},
            {"ticker": "KTOS", "weight": 0.15, "role": "Defense drones + hypersonics + quantum"},
            {"ticker": "MARA", "weight": 0.10, "role": "Bitcoin mining leverage (high beta to BTC)"},
            {"ticker": "SMCI", "weight": 0.10, "role": "AI server / data center — recovery play"},
            {"ticker": "SOFI", "weight": 0.15, "role": "Fintech mid-cap — bank + lending + invest"},
        ],
    },
    {
        "name": "Broad Market",
        "thesis": "When you don't have a sector edge, own the market. Lowest-fee path to equity beta.",
        "horizon": "Indefinite",
        "picks": [
            {"ticker": "SPY", "weight": 0.50, "role": "S&P 500 core"},
            {"ticker": "QQQ", "weight": 0.50, "role": "Nasdaq 100 — tech tilt"},
        ],
    },
]


def _render_dream_sector(sector: dict, screener_df: pd.DataFrame, sector_nzd: float):
    """Render one sector's dream portfolio inside an expander."""
    name    = sector["name"]
    thesis  = sector["thesis"]
    horizon = sector["horizon"]
    picks   = sector["picks"]

    rows = []
    flagged: list[str] = []
    for p in picks:
        tkr = p["ticker"]
        ts = price = rsi_v = None

        if not screener_df.empty:
            match = screener_df[screener_df["ticker"] == tkr]
            if not match.empty:
                row = match.iloc[0]
                ts    = float(row.get("tradescore") or 0) or None
                price = float(row.get("price") or 0) or None
                rsi_v = float(row.get("rsi") or 0) or None

        if price is None:
            mp = fetch_metal_prices().get(tkr)
            if mp and mp.get("price"):
                price = float(mp["price"])

        if ts is not None and ts >= 30:
            flagged.append(tkr)

        rows.append({
            "Ticker":     tkr,
            "Role":       p["role"],
            "Weight %":   round(p["weight"] * 100, 1),
            "NZD":        round(sector_nzd * p["weight"], 0),
            "Price":      round(price, 2) if price else None,
            "TradeScore": round(ts, 0)    if ts    else None,
            "RSI":        round(rsi_v, 0) if rsi_v else None,
        })

    with st.expander(f"**{name}**  ·  {horizon}", expanded=False):
        st.markdown(
            f"<div style='font-size:0.8rem;color:#bbb;line-height:1.4;margin-bottom:6px'>{thesis}</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame(rows),
            width='stretch', hide_index=True,
            column_config={
                "Weight %":   st.column_config.NumberColumn("Weight", format="%.1f%%"),
                "NZD":        st.column_config.NumberColumn("NZD", format="NZD %.0f"),
                "Price":      st.column_config.NumberColumn("Price", format="$%.2f"),
                "TradeScore": st.column_config.NumberColumn("TS", format="%d"),
                "RSI":        st.column_config.NumberColumn("RSI", format="%d"),
            },
        )
        if flagged:
            st.markdown(
                f"<div style='font-size:0.72rem;color:#7fbf7f;margin-top:2px'>"
                f"On today's screener: {', '.join(flagged)}</div>",
                unsafe_allow_html=True,
            )




def render(get_conn: Callable, dates: list) -> None:
    """Render the Portfolio tab.

    Args:
      get_conn: callable returning a DB connection (no args).
    """

    nzdusd_port  = fetch_nzdusd()
    metal_px_port = fetch_metal_prices()
    metal_tech_port = fetch_metal_technicals()
    mctx_port    = fetch_market_context()

    # -----------------------------------------------------------------------
    # Macro regime
    # -----------------------------------------------------------------------

    spy_port = mctx_port.get("spy", {})
    btc_port = mctx_port.get("btc", {})
    usd_port = mctx_port.get("usd", {})
    tnx_port = mctx_port.get("tnx", {})

    rg1, rg2, rg3, rg4 = st.columns(4)
    rg1.metric("Equity regime",  _regime_label(mctx_port, "spy"),
               delta=f"{spy_port.get('chg', 0):+.2f}%" if spy_port.get("chg") is not None else None,
               delta_color="normal")
    rg2.metric("Crypto (BTC)",   _regime_label(mctx_port, "btc"),
               delta=f"{btc_port.get('chg', 0):+.2f}%" if btc_port.get("chg") is not None else None,
               delta_color="normal")
    rg3.metric("USD trend",      _regime_label(mctx_port, "usd"),
               delta=f"{usd_port.get('chg', 0):+.2f}%" if usd_port.get("chg") is not None else None,
               delta_color="inverse")
    rg4.metric("10yr yield",
               f"{tnx_port.get('price', 0):.2f}%" if tnx_port.get("price") else "—",
               delta=f"{tnx_port.get('chg', 0):+.2f}%" if tnx_port.get("chg") is not None else None,
               delta_color="inverse")

    st.divider()

    # -----------------------------------------------------------------------
    # Load today's screener
    # -----------------------------------------------------------------------

    if dates:
        conn = get_conn()
        today_screener = pd.read_sql(
            "SELECT * FROM results WHERE run_date = ? ORDER BY tradescore DESC",
            conn, params=(dates[0],),
        )
        conn.close()
        run_date_label = dates[0]
    else:
        today_screener = pd.DataFrame()
        run_date_label = "—"

    # Portfolio size input
    port_size_nzd = st.number_input(
        "Portfolio size (NZD)",
        min_value=1000.0, max_value=500000.0, value=10000.0, step=1000.0,
        help="Set this to your actual or hypothetical portfolio size. "
             "All position sizes and dollar amounts will scale to this."
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Build portfolios
    # -----------------------------------------------------------------------

    spec_picks = _build_speculative(today_screener, metal_tech_port, mctx_port)
    inv_picks  = _build_investment(today_screener,  metal_tech_port, mctx_port)

    def _render_pick_card(pick: dict, allocation_nzd: float, idx: int):
        ticker   = pick["display"]
        ac       = pick["asset_class"]
        price    = pick.get("price", 0)
        stop     = pick.get("stop")
        rationale = pick.get("rationale", "")
        hold     = pick.get("hold", "—")
        exit_rule = pick.get("exit", "—")
        ts       = pick.get("tradescore")
        rvol     = pick.get("rvol")
        rsi_v    = pick.get("rsi")
        risk_pct = pick.get("risk_pct")
        drivers  = _driver_tags(pick["ticker"])

        stats: list[str] = [f"Alloc <b>NZD {allocation_nzd:,.0f}</b>"]
        if price:
            stats.append(f"Px <b>${price:,.2f}</b>")
        if stop and price:
            stats.append(f"Stop <b>${stop:,.2f}</b> ({((stop/price-1)*100):+.1f}%)")
        elif stop:
            stats.append(f"Stop <b>${stop:,.2f}</b>")
        if ts is not None:
            stats.append(f"TS <b>{ts:.0f}</b>")
        if rvol is not None and rvol > 0:
            stats.append(f"RVOL <b>{rvol:.1f}x</b>")
        if rsi_v is not None and rsi_v > 0:
            stats.append(f"RSI <b>{rsi_v:.0f}</b>")

        tags_html = " ".join(
            f"<span style='font-size:0.66rem;background:#222;padding:1px 6px;"
            f"border-radius:3px;color:#ccc'>{t}</span>"
            for t in drivers
        )

        with st.container(border=True):
            st.markdown(
                f"<div style='font-size:0.95rem;line-height:1.25'>"
                f"<b>{ticker}</b> "
                f"<span style='color:#888;font-size:0.72rem'>{ac}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='font-size:0.76rem;color:#bbb;margin:2px 0 6px 0'>"
                + " · ".join(stats)
                + "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-size:0.8rem;line-height:1.4;margin-bottom:4px'>"
                f"<b>Why:</b> {rationale}"
                f"</div>",
                unsafe_allow_html=True,
            )
            if tags_html:
                st.markdown(
                    f"<div style='margin-top:2px'>{tags_html}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"<div style='font-size:0.72rem;color:#888;margin-top:4px'>"
                f"Hold: {hold} · Exit: {exit_rule}"
                f"</div>",
                unsafe_allow_html=True,
            )
            if risk_pct and stop and price:
                risk_nzd = allocation_nzd * risk_pct / 100
                st.markdown(
                    f"<div style='font-size:0.7rem;color:#888'>"
                    f"Risk at stop: {risk_pct:.1f}% = NZD {risk_nzd:,.0f}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    def _render_portfolio_section(title: str, caption: str, picks: list[dict], colour_hint: str, total_nzd: float):
        st.markdown(f"### {title}")
        st.caption(caption)

        if not picks:
            st.info("No qualifying picks today. Check back after the next screener run.")
            return

        total_weight = sum(p["weight"] for p in picks)
        # Normalise weights so they sum to 1
        for p in picks:
            p["_norm_weight"] = p["weight"] / total_weight if total_weight else 1 / len(picks)

        for i, pick in enumerate(picks):
            alloc = total_nzd * pick["_norm_weight"]
            _render_pick_card(pick, alloc, i)

        # Allocation summary
        st.markdown("**Allocation breakdown**")
        alloc_data = {p["display"]: round(total_nzd * p["_norm_weight"], 0) for p in picks}
        alloc_df = pd.DataFrame(list(alloc_data.items()), columns=["Position", "NZD"]).set_index("Position")
        st.bar_chart(alloc_df, width='stretch')

        # Asset class mix
        class_mix: dict[str, float] = {}
        for p in picks:
            ac = p["asset_class"]
            class_mix[ac] = class_mix.get(ac, 0) + p["_norm_weight"] * 100
        mix_str = "  |  ".join(f"{k.title()} {v:.0f}%" for k, v in class_mix.items())
        st.caption(f"Mix: {mix_str}")

    # -----------------------------------------------------------------------
    # Two-column layout
    # -----------------------------------------------------------------------

    spec_budget = port_size_nzd * 0.40   # 40% of portfolio to speculative
    inv_budget  = port_size_nzd * 0.60   # 60% to investment grade

    col_s, col_i = st.columns(2)

    with col_s:
        _render_portfolio_section(
            title="Speculative  (40%)",
            caption=(
                "Short-term, high-momentum positions. Higher risk, higher potential reward. "
                "Each position is sized so a stop-out costs 1–2% of total portfolio. "
                "These require active management — check daily."
            ),
            picks=spec_picks,
            colour_hint="orange",
            total_nzd=spec_budget,
        )

    with col_i:
        _render_portfolio_section(
            title="Investment Grade  (60%)",
            caption=(
                "Quality assets with a durable thesis. Hold weeks to months. "
                "Position sizes are larger because conviction is higher and stops are wider. "
                "Review weekly, not daily."
            ),
            picks=inv_picks,
            colour_hint="blue",
            total_nzd=inv_budget,
        )

    st.divider()

    # -----------------------------------------------------------------------
    # Combined portfolio view
    # -----------------------------------------------------------------------

    all_picks = spec_picks + inv_picks
    if all_picks:
        st.subheader("Combined portfolio")

        # Normalise within each portfolio
        spec_norm = sum(p.get("_norm_weight", 0) for p in spec_picks)
        inv_norm  = sum(p.get("_norm_weight", 0) for p in inv_picks)

        combined_rows = []
        for p in spec_picks:
            w = p.get("_norm_weight", 0)
            port_pct = (w / spec_norm * 0.40 * 100) if spec_norm else 0
            combined_rows.append({
                "Portfolio":    "Speculative",
                "Ticker":       p["display"],
                "Asset class":  p["asset_class"],
                "Hold period":  p.get("hold", "—"),
                "Port % ":      round(port_pct, 1),
                "NZD":          round(spec_budget * w, 0),
                "Exit rule":    p.get("exit", "—"),
            })
        for p in inv_picks:
            w = p.get("_norm_weight", 0)
            port_pct = (w / inv_norm * 0.60 * 100) if inv_norm else 0
            combined_rows.append({
                "Portfolio":    "Investment",
                "Ticker":       p["display"],
                "Asset class":  p["asset_class"],
                "Hold period":  p.get("hold", "—"),
                "Port % ":      round(port_pct, 1),
                "NZD":          round(inv_budget * w, 0),
                "Exit rule":    p.get("exit", "—"),
            })

        st.dataframe(
            pd.DataFrame(combined_rows),
            width='stretch', hide_index=True,
            column_config={
                "Port % ": st.column_config.NumberColumn("Port %", format="%.1f%%"),
                "NZD":     st.column_config.NumberColumn("NZD", format="NZD %.0f"),
            },
        )

        # Theme concentration
        st.divider()
        st.subheader("Theme concentration")
        st.caption("Identifies where your portfolio is thematically concentrated — useful for spotting correlation risk.")

        theme_counts: dict[str, list[str]] = {}
        for p in all_picks:
            for tag in _driver_tags(p["ticker"]):
                theme_counts.setdefault(tag, []).append(p["display"])

        multi = {t: tickers for t, tickers in theme_counts.items() if len(tickers) >= 2}
        single = {t: tickers for t, tickers in theme_counts.items() if len(tickers) == 1}

        if multi:
            st.markdown("**Concentrated themes (2+ positions):**")
            for theme, tickers in sorted(multi.items(), key=lambda x: -len(x[1])):
                st.markdown(f"- **{theme}**: {', '.join(tickers)}")
        if single:
            with st.expander("Single-position themes"):
                for theme, tickers in single.items():
                    st.markdown(f"- {theme}: {tickers[0]}")

    st.divider()

    # -----------------------------------------------------------------------
    # Dream portfolios — best-in-class per sector
    # -----------------------------------------------------------------------

    st.subheader("Dream portfolios by sector")
    st.caption(
        "Aspirational best-in-class picks per sector, regardless of today's screen. "
        "Pick a sector you have conviction in and use these as your core allocations within it. "
        "Tickers showing a TradeScore are also flagging on today's screener."
    )

    n_sectors = len(_DREAM_SECTORS)
    sector_alloc = port_size_nzd / n_sectors if n_sectors else 0
    st.caption(
        f"NZD figures below assume **equal-weight {n_sectors} sectors** "
        f"(NZD {sector_alloc:,.0f} per sector). Tilt to your conviction."
    )

    for sector in _DREAM_SECTORS:
        _render_dream_sector(sector, today_screener, sector_alloc)

    st.divider()

    # -----------------------------------------------------------------------
    # Key risks
    # -----------------------------------------------------------------------

    st.subheader("Key risks to watch")

    risks = []
    spy_d = mctx_port.get("spy", {})
    if not spy_d.get("above_ema50"):
        risks.append("Equities below EMA50 — momentum picks are swimming against the tide. Reduce speculative allocation.")
    if not btc_port.get("above_ema20"):
        risks.append("BTC below EMA20 — crypto risk-off. Halve crypto exposure or avoid.")
    if usd_port.get("above_ema50"):
        risks.append("USD strengthening — headwind for gold and commodities. Watch metal positions.")
    if tnx_port.get("price", 0) > 4.5:
        risks.append(f"10yr yield {tnx_port['price']:.2f}% — elevated rates compress growth multiples. Favour value over momentum.")
    if not risks:
        risks.append("No major red flags in the current macro data. Continue following position-level stops.")

    for r in risks:
        st.warning(r)

    st.caption(
        f"Built from screener run {run_date_label}  •  "
        f"Regime data refreshes every 5 min  •  "
        f"NZD/USD {nzdusd_port:.4f}"
    )
