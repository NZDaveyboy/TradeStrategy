"""
ui/tabs/options.py — Options chain + structured strategies tab.

Pulls live option chains, computes Greeks via core.options_math, and lets
the user explore call/put strategies for a given ticker. Cross-references
with the screener's recommendation and catalyst data when available.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

import pandas as pd
import streamlit as st

from core.options_math import bs_price, bs_greeks, _ncdf, _npdf
from core.recommendations import STRATEGY_DISPLAY, build_recommendation
from ui.data import (
    _provider as _provider,
    enrich_chain,
    fetch_nzdusd,
    get_chain,
    get_rv30,
    payoff_df,
)


def render(get_conn: Callable, dates: list) -> None:
    """Render the Options tab."""

    RISK_FREE = 0.045

    opt_sub = st.radio(
        "Section",
        ["Recommendations", "Chain & Position", "Strategy Builder", "Backtest"],
        horizontal=True,
    )

    # -----------------------------------------------------------------------
    # Shared ticker picker
    # -----------------------------------------------------------------------

    if dates:
        conn = get_conn()
        eq_tickers = pd.read_sql(
            "SELECT DISTINCT ticker FROM results WHERE run_date=? AND asset='equity'",
            conn, params=(dates[0],),
        )["ticker"].tolist()
        conn.close()
    else:
        eq_tickers = []

    ot1, ot2 = st.columns([2, 3])
    manual_t = ot1.text_input("Ticker", placeholder="META, INTC, NVDA …").strip().upper()
    pick_t   = ot2.selectbox("Or from today's screener", ["—"] + sorted(eq_tickers), key="opt_t")
    opt_ticker = manual_t or (pick_t if pick_t != "—" else "")

    st.divider()

    # =======================================================================
    # SECTION R — Recommendations
    # =======================================================================

    if opt_sub == "Recommendations":

        if not opt_ticker:
            st.info("Enter a ticker above to get an options strategy recommendation.")
        else:
            try:
                # Screener context for this ticker
                screener_row = None
                if dates:
                    conn = get_conn()
                    _sq = pd.read_sql(
                        "SELECT * FROM results WHERE run_date=? AND ticker=? LIMIT 1",
                        conn, params=(dates[0], opt_ticker),
                    )
                    conn.close()
                    screener_row = _sq.iloc[0].to_dict() if not _sq.empty else None

                spot     = _provider.get_quote(opt_ticker).last_price
                expiries = _provider.get_expiries(opt_ticker)
                rv30     = get_rv30(opt_ticker)

                if not expiries or not spot:
                    st.warning(f"No options data available for {opt_ticker}.")
                else:
                    # Best 30–45 DTE expiry
                    today_dt = datetime.now(timezone.utc).date()
                    best_exp, best_dte = None, None
                    for _e in expiries:
                        _dte = (datetime.strptime(_e, "%Y-%m-%d").date() - today_dt).days
                        if 25 <= _dte <= 55:
                            best_exp, best_dte = _e, _dte
                            break
                    if not best_exp:
                        best_exp = min(expiries, key=lambda _e: abs((datetime.strptime(_e, "%Y-%m-%d").date() - today_dt).days - 35))
                        best_dte = (datetime.strptime(best_exp, "%Y-%m-%d").date() - today_dt).days

                    calls_raw, puts_raw, _ = get_chain(opt_ticker, best_exp)

                    # ATM IV
                    _atm_rows = calls_raw[calls_raw["strike"].between(spot*0.95, spot*1.05) & (calls_raw["impliedVolatility"] > 0)]
                    atm_iv = float(_atm_rows["impliedVolatility"].mean()) if not _atm_rows.empty else (rv30 or 0.30)

                    direction  = screener_row.get("direction")  if screener_row else None
                    tradescore = float(screener_row.get("tradescore") or 0) if screener_row else None
                    setup_type = screener_row.get("setup_type") if screener_row else None

                    # Unified recommendation — same logic as Advice tab
                    _rec_row = screener_row if screener_row else {"ticker": opt_ticker, "price": spot}
                    # Catalyst overlay for the user-selected ticker (cached).
                    try:
                        from core.catalysts import compute_catalyst_score as _ccs_opt
                        _opt_catalyst = _ccs_opt(opt_ticker, float(_rec_row.get("price") or spot))
                    except Exception:
                        _opt_catalyst = None
                    rec = build_recommendation(
                        _rec_row,
                        atm_iv=atm_iv if atm_iv else None,
                        rv30=rv30     if rv30     else None,
                        catalyst=_opt_catalyst,
                    )
                    rec_strat = STRATEGY_DISPLAY.get(rec.strategy_name) if rec.is_actionable else None
                    rec_bias  = (
                        "Bullish" if direction == "long" else
                        "Bearish" if direction == "short" else
                        "Neutral / Pullback entry" if rec.strategy_name == "cash_secured_put" else None
                    )

                    # IV note from rec
                    if atm_iv:
                        _iv_pct = f"IV {atm_iv*100:.0f}%"
                        _rv_pct = f"30d RV {rv30*100:.0f}%" if rv30 else ""
                        if rec.iv_assessment == "expensive":
                            iv_note = f"{_iv_pct} is elevated vs {_rv_pct} — spread reduces vega exposure."
                        elif rec.iv_assessment == "cheap":
                            iv_note = f"{_iv_pct} is below {_rv_pct} — options are cheap, outright is the better play."
                        else:
                            iv_note = f"{_iv_pct}  |  {_rv_pct}" if rv30 else _iv_pct
                    else:
                        iv_note = rec.rationale

                    # Context metrics
                    if screener_row:
                        _sc1, _sc2, _sc3, _sc4 = st.columns(4)
                        _sc1.metric("Spot",       f"${spot:.2f}")
                        _sc2.metric("TradeScore", f"{tradescore:.0f}" if tradescore else "—")
                        _sc3.metric("Setup",      setup_type or "—")
                        _sc4.metric("Direction",  "🟢 Long" if direction == "long" else "🔴 Short" if direction == "short" else direction or "—")
                    else:
                        _sc1, _sc2 = st.columns(2)
                        _sc1.metric("Spot",   f"${spot:.2f}")
                        _sc2.metric("30d RV", f"{rv30*100:.1f}%" if rv30 else "—")
                        st.caption(f"{opt_ticker} not in today's screener — showing live options data only.")

                    st.divider()

                    if not rec_strat:
                        st.warning(rec.rationale or "No clear directional signal. Enter a ticker with a long or short setup.")
                        for _w in rec.warnings:
                            st.info(_w)
                    else:
                        call_strikes = sorted(calls_raw[calls_raw["bid"] > 0]["strike"].tolist())
                        put_strikes  = sorted(puts_raw[puts_raw["bid"] > 0]["strike"].tolist(), reverse=True)

                        atm_call_k = min(call_strikes, key=lambda k: abs(k - spot)) if call_strikes else spot
                        atm_put_k  = min(put_strikes,  key=lambda k: abs(k - spot)) if put_strikes  else spot
                        otm_call_k = min(call_strikes, key=lambda k: abs(k - spot*1.05)) if call_strikes else round(spot*1.05, 2)
                        otm_put_k  = min(put_strikes,  key=lambda k: abs(k - spot*0.95)) if put_strikes  else round(spot*0.95, 2)

                        def _mid(df, strike):
                            r = df[df["strike"] == strike]
                            if r.empty:
                                return 0.0
                            _r = r.iloc[0]
                            b, a = float(_r.get("bid", 0) or 0), float(_r.get("ask", 0) or 0)
                            return round((b + a) / 2 if a > 0 else float(_r.get("lastPrice", 0) or 0), 3)

                        nzdusd_r = fetch_nzdusd()
                        T_exp = best_dte / 365.0

                        # Contract selection — keyed on rec.strategy_name (snake_case)
                        _sn = rec.strategy_name

                        if _sn == "long_call":
                            k1, prem1 = atm_call_k, _mid(calls_raw, atm_call_k)
                            legs = [{"type":"call","strike":k1,"premium":prem1,"qty":1,"position":"long"}]
                            net, max_loss, max_profit = prem1, round(prem1*100,2), "Unlimited"
                            be_price = round(k1 + prem1, 2)
                            strike_desc = f"Strike ${k1:.2f} (ATM call)"

                        elif _sn == "bull_call_spread":
                            k1, k2 = atm_call_k, otm_call_k
                            p1, p2 = _mid(calls_raw, k1), _mid(calls_raw, k2)
                            net = round(p1 - p2, 3)
                            legs = [
                                {"type":"call","strike":k1,"premium":p1,"qty":1,"position":"long"},
                                {"type":"call","strike":k2,"premium":p2,"qty":1,"position":"short"},
                            ]
                            max_loss   = round(net * 100, 2)
                            max_profit = round(((k2 - k1) - net) * 100, 2)
                            be_price   = round(k1 + net, 2)
                            strike_desc = f"Buy ${k1:.2f} / Sell ${k2:.2f} call"

                        elif _sn == "long_put":
                            k1, prem1 = atm_put_k, _mid(puts_raw, atm_put_k)
                            legs = [{"type":"put","strike":k1,"premium":prem1,"qty":1,"position":"long"}]
                            net, max_loss, max_profit = prem1, round(prem1*100,2), round((k1-prem1)*100,2)
                            be_price = round(k1 - prem1, 2)
                            strike_desc = f"Strike ${k1:.2f} (ATM put)"

                        elif _sn == "bear_put_spread":
                            k1, k2 = atm_put_k, otm_put_k
                            p1, p2 = _mid(puts_raw, k1), _mid(puts_raw, k2)
                            net = round(p1 - p2, 3)
                            legs = [
                                {"type":"put","strike":k1,"premium":p1,"qty":1,"position":"long"},
                                {"type":"put","strike":k2,"premium":p2,"qty":1,"position":"short"},
                            ]
                            max_loss   = round(net * 100, 2)
                            max_profit = round(((k1 - k2) - net) * 100, 2)
                            be_price   = round(k1 - net, 2)
                            strike_desc = f"Buy ${k1:.2f} / Sell ${k2:.2f} put"

                        else:  # cash_secured_put
                            k1, prem1 = otm_put_k, _mid(puts_raw, otm_put_k)
                            legs = [{"type":"put","strike":k1,"premium":prem1,"qty":1,"position":"short"}]
                            net        = -prem1
                            max_loss   = round((k1 - prem1) * 100, 2)
                            max_profit = round(prem1 * 100, 2)
                            be_price   = round(k1 - prem1, 2)
                            strike_desc = f"Sell ${k1:.2f} put (~5% OTM)"

                        move_needed = abs(be_price - spot) / spot * 100
                        cost_nzd    = abs(net) * 100 / (nzdusd_r or 0.57)

                        st.subheader(f"Recommended: {rec_strat}")
                        if rec.invalidation_price:
                            st.caption(
                                f"{rec_bias}  ·  {opt_ticker}  ·  Expiry {best_exp} ({best_dte}d)  ·  {strike_desc}  "
                                f"·  Stop ${rec.invalidation_price:.2f}"
                            )
                        else:
                            st.caption(f"{rec_bias}  ·  {opt_ticker}  ·  Expiry {best_exp} ({best_dte}d)  ·  {strike_desc}")
                        st.info(iv_note)
                        if rec.warnings:
                            for _w in rec.warnings:
                                st.warning(_w)

                        _rm1, _rm2, _rm3, _rm4 = st.columns(4)
                        _rm1.metric("Net cost",    f"${abs(net):.3f}/share")
                        _rm2.metric("Total NZD",   f"NZD {cost_nzd:,.0f}", help="Max loss if option expires worthless")
                        _rm3.metric("Break-even",  f"${be_price:.2f}", delta=f"{((be_price/spot-1)*100):+.1f}% from spot")
                        _rm4.metric("Move needed", f"{move_needed:.1f}%")

                        _mc1, _mc2 = st.columns(2)
                        _mc1.metric("Max loss",   f"NZD {max_loss/(nzdusd_r or 0.57):,.0f}" if isinstance(max_loss, (int,float)) else str(max_loss))
                        _mc2.metric("Max profit", f"NZD {max_profit/(nzdusd_r or 0.57):,.0f}" if isinstance(max_profit, (int,float)) else str(max_profit))

                        _iv_g = atm_iv or 0.30
                        _g = bs_greeks(spot, legs[0]["strike"], T_exp, RISK_FREE, _iv_g, legs[0]["type"])
                        st.markdown(
                            f"Delta {_g['delta']:+.3f}  ·  moves ~${abs(_g['delta'])*100:.0f} per $1 stock move.  "
                            f"Theta {_g['theta']:.4f}  ·  costs ~NZD {abs(_g['theta'])*100/(nzdusd_r or 0.57):.2f}/day.  "
                            f"IV {atm_iv*100:.0f}%  ·  {best_dte}d to expiry."
                        )

                        st.subheader("Payoff at expiry")
                        _pnl = payoff_df(spot, legs)
                        st.line_chart(_pnl.set_index("Stock price"))
                        st.caption("P&L per share at expiry. Multiply by 100 × contracts for total.")

            except Exception as _e:
                st.error(f"Could not generate recommendation for {opt_ticker}: {_e}")

    # =======================================================================
    # SECTION A — Chain & Position
    # =======================================================================

    elif opt_sub == "Chain & Position":

        if not opt_ticker:
            st.info("Enter a ticker to load its options chain.")
        else:
            try:
                spot     = _provider.get_quote(opt_ticker).last_price
                expiries = _provider.get_expiries(opt_ticker)
                rv30     = get_rv30(opt_ticker)

                if not expiries:
                    st.warning(f"No options data for {opt_ticker}.")
                else:
                    # Spot + RV
                    mc1, mc2 = st.columns(2)
                    mc1.metric("Spot price", f"${spot:.2f}")
                    if rv30:
                        mc2.metric("30d Realised Vol", f"{rv30*100:.1f}%",
                                   help="Compare to option IV. IV >> RV = expensive options.")

                    # Expiry
                    today_dt = datetime.now(timezone.utc).date()
                    exp_opts = []
                    for e in expiries:
                        dte = (datetime.strptime(e, "%Y-%m-%d").date() - today_dt).days
                        tag = "weekly" if dte <= 14 else ("near" if dte <= 45 else ("mid" if dte <= 90 else "far"))
                        exp_opts.append((f"{e}  ({dte}d — {tag})", e))

                    sel_exp_label = st.selectbox("Expiry", [l for l,_ in exp_opts],
                        help="30–60 DTE = sweet spot for buying. Under 7d: theta accelerates sharply.")
                    sel_exp = dict(exp_opts)[sel_exp_label]
                    dte_days = (datetime.strptime(sel_exp, "%Y-%m-%d").date() - today_dt).days

                    # IV vs RV warning
                    calls_raw, puts_raw, _ = get_chain(opt_ticker, sel_exp)
                    atm_iv_rows = calls_raw[
                        (calls_raw["strike"].between(spot*0.95, spot*1.05)) &
                        (calls_raw["impliedVolatility"] > 0)
                    ]
                    atm_iv = float(atm_iv_rows["impliedVolatility"].mean()) if not atm_iv_rows.empty else None

                    if atm_iv and rv30:
                        if atm_iv > rv30 * 1.3:
                            st.warning(f"ATM IV {atm_iv*100:.0f}% is {((atm_iv/rv30-1)*100):.0f}% above 30d realised vol ({rv30*100:.0f}%) — options are expensive. IV often compresses after entry.")
                        elif atm_iv < rv30 * 0.85:
                            st.success(f"ATM IV {atm_iv*100:.0f}% is below 30d realised vol ({rv30*100:.0f}%) — options are relatively cheap.")
                        else:
                            st.info(f"ATM IV {atm_iv*100:.0f}%  |  30d RV {rv30*100:.0f}% — fairly priced.")

                    # Chain
                    opt_type_sel = st.radio("Type", ["Calls", "Puts"], horizontal=True, key="chain_type")
                    raw = calls_raw if opt_type_sel == "Calls" else puts_raw
                    otype = "call" if opt_type_sel == "Calls" else "put"
                    chain_df = enrich_chain(raw, spot, sel_exp, otype)

                    if chain_df.empty:
                        st.info("No liquid contracts for this expiry.")
                    else:
                        st.dataframe(chain_df, width='stretch', hide_index=True,
                            column_config={
                                "ITM":        st.column_config.CheckboxColumn("ITM"),
                                "IV %":       st.column_config.NumberColumn("IV %",      format="%.1f%%"),
                                "Theta/day":  st.column_config.NumberColumn("Theta/day", format="%.4f"),
                                "Vega/1%":    st.column_config.NumberColumn("Vega/1%",   format="%.4f"),
                                "Break-even": st.column_config.NumberColumn("Break-even",format="$%.2f"),
                            })

                        st.caption("**Delta** — moves per $1 stock move.  **Theta** — daily decay cost.  **Vega** — gain/loss per 1% IV change.")

                        st.divider()

                        # Position builder
                        st.subheader("Position builder")
                        nzdusd_o = fetch_nzdusd()
                        pb1, pb2 = st.columns(2)
                        sel_strike = pb1.selectbox("Strike", chain_df["Strike"].tolist(), key="pb_strike")
                        n_contracts = pb2.number_input("Contracts (1 = 100 shares)", 1, 50, 1, key="pb_c")

                        sel_row = chain_df[chain_df["Strike"] == sel_strike]
                        if not sel_row.empty:
                            s = sel_row.iloc[0]
                            prem = s["Mid"] if s["Mid"] > 0 else s["Ask"]
                            cost_usd = prem * 100 * n_contracts
                            cost_nzd = cost_usd / nzdusd_o if nzdusd_o else cost_usd
                            be = s["Break-even"]
                            move_pct = abs(be - spot) / spot * 100

                            r1c, r2c, r3c, r4c = st.columns(4)
                            r1c.metric("Premium (mid)", f"${prem:.3f}")
                            r2c.metric("Total cost", f"NZD {cost_nzd:,.0f}", help="Your max loss if option expires worthless.")
                            r3c.metric("Break-even", f"${be:.2f}")
                            r4c.metric("Move needed", f"{move_pct:.1f}%")

                            st.markdown(
                                f"Delta {s['Delta']:+.3f} — option moves ~${abs(s['Delta'])*100*n_contracts:.0f} per $1 stock move.  "
                                f"Theta {s['Theta/day']:.4f} — costs ~NZD {abs(s['Theta/day'])*100*n_contracts/nzdusd_o:.2f}/day.  "
                                f"IV {s['IV %']:.1f}%  |  {dte_days}d to expiry."
                            )
            except Exception as e:
                st.error(f"Could not load options for {opt_ticker}: {e}")

    # =======================================================================
    # SECTION B — Strategy Builder
    # =======================================================================

    elif opt_sub == "Strategy Builder":

        STRATEGIES_DEF = {
            "Long Call": {
                "desc": "Bullish. Buy one call. Profits if stock rises above break-even. Max loss = premium paid.",
                "legs": 1, "bias": "bullish",
                "when": "Strong directional conviction upward. IV is low/fair. At least 30 DTE.",
            },
            "Bull Call Spread": {
                "desc": "Bullish with reduced cost. Buy lower-strike call, sell higher-strike call. Caps both risk and profit.",
                "legs": 2, "bias": "bullish",
                "when": "Moderately bullish. IV is high (spread reduces vega exposure). Want to cut cost.",
            },
            "Long Put": {
                "desc": "Bearish. Buy one put. Profits if stock falls below break-even. Max loss = premium paid.",
                "legs": 1, "bias": "bearish",
                "when": "Strong conviction downward (e.g. heading into bad earnings). IV is low.",
            },
            "Bear Put Spread": {
                "desc": "Bearish with reduced cost. Buy higher-strike put, sell lower-strike put.",
                "legs": 2, "bias": "bearish",
                "when": "Moderately bearish. IV is elevated. Want to reduce premium outlay.",
            },
            "Cash-Secured Put": {
                "desc": "Income / stock entry strategy. Sell a put, collect premium. If stock falls to strike you buy the shares at a discount.",
                "legs": 1, "bias": "neutral-bullish",
                "when": "Happy to own the stock at the strike price. IV is high (premium collection).",
            },
            "Covered Call": {
                "desc": "Income on existing position. Sell a call against shares you own. Caps upside, reduces cost basis.",
                "legs": 1, "bias": "neutral",
                "when": "Already long the stock. Want income. Expect sideways to slight upside.",
            },
        }

        strat_name = st.selectbox("Strategy", list(STRATEGIES_DEF.keys()))
        sdef = STRATEGIES_DEF[strat_name]

        st.info(f"**{strat_name}** — {sdef['desc']}\n\n**Use when:** {sdef['when']}")

        if not opt_ticker:
            st.warning("Enter a ticker above to build this strategy.")
        else:
            try:
                spot     = _provider.get_quote(opt_ticker).last_price
                expiries = _provider.get_expiries(opt_ticker)
                rv30     = get_rv30(opt_ticker)

                if not expiries:
                    st.warning(f"No options data for {opt_ticker}.")
                else:
                    today_dt = datetime.now(timezone.utc).date()
                    exp_opts = [(f"{e}  ({(datetime.strptime(e,'%Y-%m-%d').date()-today_dt).days}d)", e) for e in expiries]
                    sel_exp_label = st.selectbox("Expiry", [l for l,_ in exp_opts], key="sb_exp")
                    sel_exp = dict(exp_opts)[sel_exp_label]
                    dte_days = (datetime.strptime(sel_exp, "%Y-%m-%d").date() - today_dt).days
                    T = max(dte_days / 365.0, 0)
                    rv = rv30 or 0.30

                    calls_raw, puts_raw, _ = get_chain(opt_ticker, sel_exp)

                    nzdusd_s = fetch_nzdusd()
                    legs = []

                    if strat_name == "Long Call":
                        strikes = sorted(calls_raw[calls_raw["bid"] > 0]["strike"].tolist())
                        k1 = st.selectbox("Strike", strikes, index=min(len(strikes)//2, len(strikes)-1), key="lc_k1")
                        row1 = calls_raw[calls_raw["strike"] == k1].iloc[0]
                        prem1 = float((row1.get("bid",0) + row1.get("ask",0)) / 2 or row1.get("lastPrice",0))
                        legs = [{"type":"call","strike":k1,"premium":prem1,"qty":1,"position":"long"}]

                    elif strat_name == "Bull Call Spread":
                        strikes = sorted(calls_raw[calls_raw["bid"] > 0]["strike"].tolist())
                        sb1, sb2 = st.columns(2)
                        k1 = sb1.selectbox("Buy strike (lower)", strikes, key="bcs_k1")
                        k2_opts = [s for s in strikes if s > k1]
                        if k2_opts:
                            k2 = sb2.selectbox("Sell strike (higher)", k2_opts, key="bcs_k2")
                            r1b = calls_raw[calls_raw["strike"]==k1].iloc[0]
                            r2b = calls_raw[calls_raw["strike"]==k2].iloc[0]
                            p1 = float((r1b.get("bid",0)+r1b.get("ask",0))/2 or r1b.get("lastPrice",0))
                            p2 = float((r2b.get("bid",0)+r2b.get("ask",0))/2 or r2b.get("lastPrice",0))
                            legs = [
                                {"type":"call","strike":k1,"premium":p1,"qty":1,"position":"long"},
                                {"type":"call","strike":k2,"premium":p2,"qty":1,"position":"short"},
                            ]

                    elif strat_name == "Long Put":
                        strikes = sorted(puts_raw[puts_raw["bid"] > 0]["strike"].tolist(), reverse=True)
                        k1 = st.selectbox("Strike", strikes, key="lp_k1")
                        row1 = puts_raw[puts_raw["strike"]==k1].iloc[0]
                        prem1 = float((row1.get("bid",0)+row1.get("ask",0))/2 or row1.get("lastPrice",0))
                        legs = [{"type":"put","strike":k1,"premium":prem1,"qty":1,"position":"long"}]

                    elif strat_name == "Bear Put Spread":
                        strikes = sorted(puts_raw[puts_raw["bid"] > 0]["strike"].tolist(), reverse=True)
                        bp1, bp2 = st.columns(2)
                        k1 = bp1.selectbox("Buy strike (higher)", strikes, key="bps_k1")
                        k2_opts = [s for s in strikes if s < k1]
                        if k2_opts:
                            k2 = bp2.selectbox("Sell strike (lower)", k2_opts, key="bps_k2")
                            r1p = puts_raw[puts_raw["strike"]==k1].iloc[0]
                            r2p = puts_raw[puts_raw["strike"]==k2].iloc[0]
                            p1 = float((r1p.get("bid",0)+r1p.get("ask",0))/2 or r1p.get("lastPrice",0))
                            p2 = float((r2p.get("bid",0)+r2p.get("ask",0))/2 or r2p.get("lastPrice",0))
                            legs = [
                                {"type":"put","strike":k1,"premium":p1,"qty":1,"position":"long"},
                                {"type":"put","strike":k2,"premium":p2,"qty":1,"position":"short"},
                            ]

                    elif strat_name == "Cash-Secured Put":
                        strikes = sorted(puts_raw[puts_raw["bid"] > 0]["strike"].tolist(), reverse=True)
                        k1 = st.selectbox("Strike to sell", strikes, key="csp_k1")
                        row1 = puts_raw[puts_raw["strike"]==k1].iloc[0]
                        prem1 = float((row1.get("bid",0)+row1.get("ask",0))/2 or row1.get("lastPrice",0))
                        legs = [{"type":"put","strike":k1,"premium":prem1,"qty":1,"position":"short"}]

                    elif strat_name == "Covered Call":
                        strikes = sorted(calls_raw[calls_raw["bid"] > 0]["strike"].tolist())
                        k1 = st.selectbox("Strike to sell", strikes, key="cc_k1")
                        row1 = calls_raw[calls_raw["strike"]==k1].iloc[0]
                        prem1 = float((row1.get("bid",0)+row1.get("ask",0))/2 or row1.get("lastPrice",0))
                        legs = [{"type":"call","strike":k1,"premium":prem1,"qty":1,"position":"short"}]

                    if legs:
                        net_debit = sum(
                            (l["premium"] if l["position"]=="long" else -l["premium"]) * l["qty"]
                            for l in legs
                        )
                        max_profit = max_loss = None

                        if strat_name == "Long Call":
                            max_loss   = round(net_debit * 100, 2)
                            max_profit = "Unlimited"
                            be_price   = round(legs[0]["strike"] + net_debit, 2)
                        elif strat_name == "Bull Call Spread":
                            width = legs[1]["strike"] - legs[0]["strike"]
                            max_profit = round((width - net_debit) * 100, 2)
                            max_loss   = round(net_debit * 100, 2)
                            be_price   = round(legs[0]["strike"] + net_debit, 2)
                        elif strat_name == "Long Put":
                            max_loss   = round(net_debit * 100, 2)
                            max_profit = round((legs[0]["strike"] - net_debit) * 100, 2)
                            be_price   = round(legs[0]["strike"] - net_debit, 2)
                        elif strat_name == "Bear Put Spread":
                            width = legs[0]["strike"] - legs[1]["strike"]
                            max_profit = round((width - net_debit) * 100, 2)
                            max_loss   = round(net_debit * 100, 2)
                            be_price   = round(legs[0]["strike"] - net_debit, 2)
                        elif strat_name == "Cash-Secured Put":
                            max_profit = round(abs(net_debit) * 100, 2)
                            max_loss   = round((legs[0]["strike"] - abs(net_debit)) * 100, 2)
                            be_price   = round(legs[0]["strike"] - abs(net_debit), 2)
                        elif strat_name == "Covered Call":
                            max_profit = round((legs[0]["strike"] - spot + abs(net_debit)) * 100, 2)
                            max_loss   = "Unlimited downside on shares"
                            be_price   = round(spot - abs(net_debit), 2)

                        n_contracts_s = st.number_input("Contracts", 1, 50, 1, key="sb_contracts")
                        cost_nzd = abs(net_debit) * 100 * n_contracts_s / (nzdusd_s or 0.57)

                        sm1, sm2, sm3, sm4 = st.columns(4)
                        sm1.metric("Net debit/credit", f"${net_debit:+.3f}")
                        sm2.metric("Total cost", f"NZD {cost_nzd:,.0f}")
                        sm3.metric("Max loss", f"${max_loss}" if isinstance(max_loss, str) else f"${max_loss:,.0f}")
                        sm4.metric("Max profit", f"${max_profit}" if isinstance(max_profit, str) else f"${max_profit:,.0f}")

                        st.metric("Break-even at expiry", f"${be_price:.2f}",
                                  delta=f"{((be_price/spot-1)*100):+.1f}% from spot")

                        # Payoff diagram
                        st.subheader("Payoff at expiry")
                        pnl = payoff_df(spot, legs)
                        pnl_display = pnl.set_index("Stock price")
                        st.line_chart(pnl_display)
                        st.caption("Shows profit/loss per share at expiry across a ±30% price range. Multiply by 100 × contracts for total P&L.")

            except Exception as e:
                st.error(f"Strategy builder error for {opt_ticker}: {e}")

    # =======================================================================
    # SECTION C — Backtest
    # =======================================================================

    else:  # Backtest

        conn = get_conn()
        bt_opt_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='backtest_options'"
        ).fetchone()
        bt_opt = pd.read_sql("SELECT * FROM backtest_options", conn) if bt_opt_exists else pd.DataFrame()
        conn.close()

        if bt_opt.empty:
            st.info("No options backtest data yet.")
            st.code("python3 options_backtest.py", language="bash")
            st.markdown(
                "This simulates buying ATM and OTM calls on each screener pick using "
                "Black-Scholes with 30-day realised volatility as the IV input. "
                "Run it after each session alongside `backtest_v2.py`."
            )
            st.warning(
                "**Important:** IV crush is not modelled. Simulated returns assume IV stays constant. "
                "Real options bought into high-IV spikes will perform worse than shown here."
            )
        else:
            bt_opt_fwd = bt_opt.dropna(subset=["return_1d"])

            # Top metrics
            bm1, bm2, bm3 = st.columns(3)
            bm1.metric("Simulated trades",  len(bt_opt_fwd))
            bm2.metric("Avg return 1d (ATM 30d)",
                round(bt_opt_fwd[bt_opt_fwd["strategy_name"]=="atm_call_30d"]["return_1d"].mean(), 1),
                help="Average % return on ATM 30DTE calls held 1 day.")
            bm3.metric("Win rate 1d (ATM 30d)",
                f"{(bt_opt_fwd[bt_opt_fwd['strategy_name']=='atm_call_30d']['return_1d']>0).mean()*100:.0f}%")

            st.warning(
                "IV crush not modelled. These returns assume implied volatility stays constant "
                "after entry. In practice, buying options into high-RVOL moves often results in "
                "IV compression that erodes returns even when the stock moves in your favour."
            )

            st.divider()

            # Return by strategy + score
            st.subheader("Return by strategy and score")
            summary = (
                bt_opt_fwd.groupby(["strategy_name", "screener_score"])
                .agg(
                    trades    =("return_1d", "count"),
                    avg_1d    =("return_1d", "mean"),
                    avg_3d    =("return_3d", "mean"),
                    avg_5d    =("return_5d", "mean"),
                    win_rate  =("return_1d", lambda x: (x>0).mean()*100),
                )
                .round(1)
                .reset_index()
            )
            summary.columns = ["Strategy", "Score", "Trades", "Avg 1d %", "Avg 3d %", "Avg 5d %", "Win rate %"]
            st.dataframe(summary, width='stretch', hide_index=True)

            st.divider()

            # Equity vs Options comparison for same picks
            st.subheader("Equity vs options — same screener picks")
            conn = get_conn()
            eq_bt = pd.read_sql(
                "SELECT run_date, ticker, score, return_1d AS eq_1d, return_3d AS eq_3d FROM backtest WHERE return_1d IS NOT NULL",
                conn,
            )
            conn.close()

            if not eq_bt.empty and not bt_opt_fwd.empty:
                atm = bt_opt_fwd[bt_opt_fwd["strategy_name"]=="atm_call_30d"][
                    ["run_date","ticker","return_1d","return_3d"]
                ].rename(columns={"return_1d":"opt_1d","return_3d":"opt_3d"})
                comp = eq_bt.merge(atm, on=["run_date","ticker"], how="inner")
                if not comp.empty:
                    comp_display = comp[["ticker","run_date","score","eq_1d","opt_1d","eq_3d","opt_3d"]].copy()
                    comp_display.columns = ["Ticker","Date","Score","Equity 1d %","Option 1d %","Equity 3d %","Option 3d %"]
                    comp_display = comp_display.sort_values("Option 1d %", ascending=False)
                    st.dataframe(comp_display, width='stretch', hide_index=True,
                        column_config={
                            "Equity 1d %":  st.column_config.NumberColumn(format="%+.1f%%"),
                            "Option 1d %":  st.column_config.NumberColumn(format="%+.1f%%"),
                            "Equity 3d %":  st.column_config.NumberColumn(format="%+.1f%%"),
                            "Option 3d %":  st.column_config.NumberColumn(format="%+.1f%%"),
                        })
                    st.caption("Option returns are simulated (Black-Scholes, constant IV). Use for directional comparison only.")

            st.divider()

            # Full trade log
            st.subheader("Full options trade log")
            bf1, bf2 = st.columns(2)
            filt_strat = bf1.selectbox("Strategy", ["All"] + sorted(bt_opt_fwd["strategy_name"].unique()), key="opt_bt_strat")
            filt_score = bf2.slider("Min score", 0, 4, 0, key="opt_bt_score")
            filtered_opt = bt_opt_fwd.copy()
            if filt_strat != "All":
                filtered_opt = filtered_opt[filtered_opt["strategy_name"]==filt_strat]
            filtered_opt = filtered_opt[filtered_opt["screener_score"] >= filt_score]
            filtered_opt = filtered_opt.sort_values(["run_date","screener_score"], ascending=[False,False])

            log_cols = ["run_date","ticker","screener_score","strategy_name",
                        "entry_stock_px","strike","entry_iv","entry_opt_px","entry_delta",
                        "return_1d","return_3d","return_5d","return_10d"]
            log_cols = [c for c in log_cols if c in filtered_opt.columns]
            st.caption(f"{len(filtered_opt)} simulated trades")
            st.dataframe(filtered_opt[log_cols], width='stretch', hide_index=True,
                column_config={
                    "return_1d":  st.column_config.NumberColumn("1d %",  format="%+.1f%%"),
                    "return_3d":  st.column_config.NumberColumn("3d %",  format="%+.1f%%"),
                    "return_5d":  st.column_config.NumberColumn("5d %",  format="%+.1f%%"),
                    "return_10d": st.column_config.NumberColumn("10d %", format="%+.1f%%"),
                    "entry_iv":   st.column_config.NumberColumn("IV",    format="%.1%%"),
                    "screener_score": st.column_config.NumberColumn("Score", format="%d/4"),
                })
