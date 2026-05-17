"""
ui/tabs/dashboard.py — Single-viewport morning view.

Shows top opportunities, regime banner, macro context, key alerts.
The opportunity detail dialog and the picker function are exported so
other tabs that want to surface the same opportunity cards can reuse them.
"""

from __future__ import annotations

import json
from typing import Callable

import pandas as pd
import streamlit as st

from core.db import get_conn
from core.recommendations import STRATEGY_DISPLAY, build_recommendation
from core.sec_edgar import get_recent_filings
from core.setups import compute_trade_setup
from ui.data import (
    _provider as _provider,
    fetch_nzdusd,
    fetch_prices,
    fetch_metal_prices,
    fetch_metal_technicals,
    fetch_market_context,
    ASSET_DRIVERS,
)
from ui.helpers import regime_label, driver_tags


# -------------------------------------------------------------------------
# Shared helpers — exported for other tabs that want to render opportunity cards
# -------------------------------------------------------------------------


@st.dialog("📊 Opportunity Details", width="large")
def show_opportunity_detail(row: dict):
    ticker = row["ticker"]

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader(ticker)
        st.caption(row.get("strategy", ""))
    with col2:
        st.metric("TradeScore", f"{row.get('tradescore', 0) or 0:.0f}")
    with col3:
        # conviction is now the setup_type label
        raw_ex = row.get("explain") or "{}"
        try:
            _ex = json.loads(raw_ex) if isinstance(raw_ex, str) else raw_ex
        except Exception:
            _ex = {}
        _conviction = _ex.get("conviction") or _ex.get("setup_type") or row.get("conviction") or "—"
        st.metric("Setup", _conviction)

    st.divider()

    with st.expander("Score breakdown", expanded=True):
        raw_explain = row.get("explain") or "{}"
        try:
            explain = json.loads(raw_explain) if isinstance(raw_explain, str) else raw_explain
        except Exception:
            explain = {}

        # New scorer exposes named sub-scores at top level
        ms  = explain.get("momentum_score")
        ee  = explain.get("early_entry")
        er  = explain.get("extension_risk")
        lq  = explain.get("liquidity")
        rat = explain.get("rationale") or row.get("rationale")

        if ms is not None:
            st.caption("**Sub-scores  (MomentumScore + EarlyEntry + Liquidity − ExtensionRisk)**")
            # Max values: MS=25, EE=25, LQ=15, ER=20
            _sub_rows = [
                ("Momentum",       ms,  25, False),
                ("Early Entry",    ee,  25, False),
                ("Liquidity",      lq,  15, False),
                ("Extension Risk", er,  20, True),   # penalty — higher is worse
            ]
            for label, val, cap, is_penalty in _sub_rows:
                if val is None:
                    continue
                pct = min(float(val) / cap, 1.0)
                prefix = "⚠️ " if is_penalty else ""
                suffix = " (penalty)" if is_penalty else ""
                st.progress(pct, text=f"{prefix}{label}{suffix}: {val:.1f} / {cap}")
            if rat:
                st.caption(f"**Rationale:** {rat}")
        else:
            # Fallback: old flat component format (pre-rewrite rows)
            components = explain.get("components", {})
            if components and all(isinstance(v, (int, float)) for v in components.values()):
                st.caption("**Signal contributions**")
                for k, v in components.items():
                    if k.startswith("penalty"):
                        label = k.replace("penalty_", "").replace("_", " ").title()
                        st.progress(min(float(v), 1.0), text=f"⚠️ {label}: -{v:.2f}")
                    elif v > 0:
                        st.progress(min(float(v), 1.0),
                                    text=f"{k.replace('_', ' ').title()}: {v:.2f}")
            else:
                st.info("Run the screener to generate score breakdown.")

    st.divider()

    with st.expander("Trade setup", expanded=True):
        # compute_trade_setup uses price/vwap/ema9/ema20/atr/day_high/day_low
        # day_high/day_low aren't stored in DB — use price as fallback so setup
        # still produces a direction; entry will be slightly off but close enough
        setup_row = dict(row)
        setup_row.setdefault("day_high", row.get("price", 0))
        setup_row.setdefault("day_low",  row.get("price", 0))
        setup_row.setdefault("conviction", _conviction)

        try:
            setup = compute_trade_setup(setup_row)
            direction = setup.direction
            entry     = setup.entry     if setup.direction != "neutral" else None
            stop      = setup.stop      if setup.direction != "neutral" else row.get("stop_loss")
            target    = setup.target    if setup.direction != "neutral" else None
            rr        = setup.rr        if setup.direction != "neutral" else None
            rat       = setup.rationale
        except Exception:
            direction, entry, stop, target, rr = "—", None, row.get("stop_loss"), None, None
            rat = ""

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Direction",
                  "🟢 Long" if direction == "long" else
                  "🔴 Short" if direction == "short" else "—")
        c2.metric("Entry",  f"${entry:.2f}"  if entry  else "—")
        c3.metric("Stop",   f"${stop:.2f}"   if stop   else "—")
        c4.metric("Target", f"${target:.2f}" if target else "—")
        if rr:
            st.caption(f"Risk/Reward: {rr:.1f}:1  ·  {rat}")

    st.divider()

    with st.expander("Recent alerts", expanded=False):
        try:
            conn = get_conn()
            alert_df = pd.read_sql_query(
                "SELECT triggered_at, alert_type, value, price, rvol "
                "FROM alerts WHERE ticker = ? ORDER BY triggered_at DESC LIMIT 5",
                conn, params=(ticker,)
            )
            conn.close()
            if not alert_df.empty:
                st.dataframe(alert_df, hide_index=True, width='stretch')
            else:
                st.caption("No recent alerts for this ticker.")
        except Exception as e:
            st.caption(f"Could not load alerts: {e}")

    with st.expander("📄 Recent SEC filings", expanded=False):
        try:
            filings = get_recent_filings(ticker)
        except Exception:
            filings = None
        if filings is None:
            st.caption("Could not load filings.")
        elif filings:
            for f in filings:
                st.markdown(
                    f"**{f['form']}** · {f['filed']} · "
                    f"[{f.get('description', 'View filing')}]({f['url']})"
                )
        else:
            st.caption("No recent filings found or not applicable "
                       "(crypto assets have no SEC filings).")

    st.markdown(
        f"[View on Finviz](https://finviz.com/quote.ashx?t={ticker}) · "
        f"[Yahoo Finance](https://finance.yahoo.com/quote/{ticker})"
    )


def pick_top_opportunities(df: pd.DataFrame, n: int = 7,
                           direction: str = "long") -> pd.DataFrame:
    """
    Select best trade candidates with sector diversity.
    direction: "long" | "short" | "both"
    """
    required = {"ticker", "rvol", "price", "score"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    filtered = df[
        (df["rvol"] >= 1.8) &
        (df["price"] >= 2.0) &
        (df["score"] >= 2)
    ].copy()

    # Filter by direction if the column exists
    if direction != "both" and "direction" in filtered.columns:
        filtered = filtered[filtered["direction"] == direction]

    score_col = "tradescore" if "tradescore" in filtered.columns else "score"
    filtered = filtered.sort_values(score_col, ascending=False)

    picked = []
    sector_counts: dict = {}
    for _, row in filtered.iterrows():
        sector = row.get("sector") or "Unknown"
        if sector_counts.get(sector, 0) >= 2:
            continue
        picked.append(row)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(picked) >= n:
            break

    return pd.DataFrame(picked) if picked else pd.DataFrame()





def render(
    *,
    get_conn: Callable,
    screener_ready: bool,
    dates: list,
    selected_date,
) -> None:
    """Render the Dashboard tab."""

    st.subheader("📋 Today's dashboard")
    _dash_col_head, _dash_col_btn = st.columns([3, 1])
    with _dash_col_head:
        st.caption(
            f"_Generated {pd.Timestamp.today():%A %b %d %Y, %H:%M}_ — "
            "the morning view. Refresh button below clears caches."
        )
    with _dash_col_btn:
        if st.button("🔄 Refresh data", width='stretch', key="dash_refresh"):
            st.cache_data.clear()
            st.rerun()

    # ── Reusable tile (small, theme-friendly) ─────────────────────────────
    # NB: single-line HTML — multi-line `unsafe_allow_html` blocks inside
    # Streamlit's markdown renderer can have blank lines interpreted as
    # paragraph breaks, which leaks the trailing </div> as literal text.
    def _dash_tile(col, label: str, value: str, sub: str = "", value_color: str | None = None):
        color_style = f"color:{value_color};" if value_color else ""
        sub_html = f'<div style="font-size:10px;opacity:0.55;margin-top:2px">{sub}</div>' if sub else ""
        html = (
            f'<div style="background:rgba(128,128,128,0.10);border:1px solid rgba(128,128,128,0.15);'
            f'border-radius:8px;padding:10px 12px;line-height:1.25;min-height:62px">'
            f'<div style="font-size:10px;text-transform:uppercase;letter-spacing:0.06em;opacity:0.65;font-weight:500">{label}</div>'
            f'<div style="font-size:16px;font-weight:600;{color_style}margin-top:3px">{value}</div>'
            f'{sub_html}'
            f'</div>'
        )
        col.markdown(html, unsafe_allow_html=True)

    @st.cache_data(ttl=300)
    def _dash_quote(symbol: str) -> dict:
        """Latest close + 1-day change for a ticker. Fast, 5-min cache."""
        import yfinance as _yf
        try:
            data = _yf.Ticker(symbol).history(period="5d", interval="1d")
            if data.empty or len(data) < 2:
                return {}
            price = float(data["Close"].iloc[-1])
            prev  = float(data["Close"].iloc[-2])
            return {"price": price, "change_pct": (price - prev) / prev * 100.0}
        except Exception:
            return {}

    def _dash_color(v: float) -> str:
        return "#2ca02c" if v > 0 else ("#d62728" if v < 0 else "#888")

    # ── Row 1: Market regime ───────────────────────────────────────────────
    st.markdown("### 🌐 Market regime")
    _r1c1, _r1c2, _r1c3, _r1c4 = st.columns(4)
    for _col, _sym, _label in [
        (_r1c1, "SPY",  "S&P 500 (SPY)"),
        (_r1c2, "QQQ",  "Nasdaq-100 (QQQ)"),
        (_r1c3, "IWM",  "Russell 2000 (IWM)"),
        (_r1c4, "^VIX", "VIX"),
    ]:
        _q = _dash_quote(_sym)
        if not _q:
            _dash_tile(_col, _label, "—")
            continue
        if _sym == "^VIX":
            _vix = _q["price"]
            _regime = (
                "⚠ Risk-off"   if _vix >= 25 else
                "Complacency"  if _vix <= 15 else
                "Normal"
            )
            _dash_tile(_col, _label, f"{_vix:.2f}", _regime)
        else:
            _dash_tile(
                _col, _label,
                f"${_q['price']:,.2f}",
                f"{_q['change_pct']:+.2f}% today",
                _dash_color(_q["change_pct"]),
            )

    # ── Row 2: two-column split (Actionable | Catalysts) ──────────────────
    st.divider()
    _action_col, _cat_col = st.columns(2)

    # ── LEFT: Actionable today (from Quantum Ecosystem signals) ───────────
    @st.cache_data(ttl=1800, show_spinner=False)
    def _dash_quantum_signals():
        """Run the Quantum Ecosystem classifier — 30-min cache."""
        from datetime import date as _date, timedelta as _td
        from core.quantum import (
            load_universe, fetch_prices, IndexBuilder, classify_constituents,
        )
        try:
            uni = load_universe()
            tickers = [c.ticker for c in uni.all_companies()] + list(uni.benchmarks)
            today = _date.today()
            start = today - _td(days=400)   # ~16 months
            prices = fetch_prices(tickers, start, today)
            if prices.empty:
                return None
            uni_tickers   = [c.ticker for c in uni.all_companies()]
            uni_prices    = prices[[t for t in uni_tickers if t in prices.columns]]
            result = IndexBuilder(uni, prices).build_ecosystem(
                pd.Timestamp(start), pd.Timestamp(today),
            )
            return classify_constituents(result, uni_prices, uni)
        except Exception:
            return None

    with _action_col:
        st.markdown("### ⚡ Actionable today")
        _sig_df = _dash_quantum_signals()
        if _sig_df is None or _sig_df.empty:
            st.info("Quantum signals unavailable. Click Refresh data.")
        else:
            _buys  = _sig_df[_sig_df["Signal"] == "BUY"].head(3)
            _sells = _sig_df[_sig_df["Signal"] == "SELL"].head(2)
            _watch = _sig_df[_sig_df["Signal"] == "WATCH"].head(2)

            if not _buys.empty:
                st.markdown("**🟢 Top BUY signals (Quantum Ecosystem)**")
                for _, r in _buys.iterrows():
                    st.markdown(
                        f"- **{r['Ticker']}** — _{r['Reason']}_  "
                        f"(Conv {r['Conviction']:.1f}, Wt {r['Weight %']:.1f}%, "
                        f"1m {r['1m %']:+.1f}%)"
                    )
            if not _sells.empty:
                st.markdown("**🔴 SELL signals**")
                for _, r in _sells.iterrows():
                    st.markdown(
                        f"- **{r['Ticker']}** — _{r['Reason']}_  "
                        f"(Held {r['Held return %']:+.1f}%, 1m {r['1m %']:+.1f}%)"
                    )
            if not _watch.empty:
                st.markdown("**⏰ WATCH (overbought / awaiting setup)**")
                for _, r in _watch.iterrows():
                    st.markdown(
                        f"- **{r['Ticker']}** — _{r['Reason']}_"
                    )

    # ── RIGHT: This week's catalysts ──────────────────────────────────────
    @st.cache_data(ttl=1800, show_spinner=False)
    def _dash_catalysts():
        """Earnings within 7d + recent 8-Ks for the Quantum universe."""
        from core.quantum import load_universe
        from core.catalysts import get_next_earnings
        from core.sec_edgar import get_recent_filings

        uni = load_universe()
        upcoming_earnings: list[dict] = []
        recent_8ks: list[dict] = []
        for c in uni.all_companies():
            t = c.ticker
            try:
                e = get_next_earnings(t)
                if e and e.get("days_to") is not None and 0 <= e["days_to"] <= 14:
                    upcoming_earnings.append({
                        "Ticker": t, "Date": e["date"],
                        "Days": e["days_to"],
                        "EPS est": e.get("eps_estimate") or 0,
                    })
            except Exception:
                pass
            try:
                fs = get_recent_filings(t, limit=3)
                for f in fs:
                    fd = f.get("filed", "")
                    # last 14 days, 8-K only (most material)
                    if fd and (pd.Timestamp.today() - pd.Timestamp(fd)).days <= 14 and f.get("form") == "8-K":
                        recent_8ks.append({
                            "Ticker": t, "Date": fd,
                            "Form":   f.get("form"),
                            "URL":    f.get("url"),
                            "Label":  f.get("items_label", ""),  # highlight-reel label
                        })
            except Exception:
                pass
        return {
            "earnings": sorted(upcoming_earnings, key=lambda x: x["Days"])[:6],
            "filings":  sorted(recent_8ks,        key=lambda x: x["Date"], reverse=True)[:6],
        }

    with _cat_col:
        st.markdown("### 📅 This week's catalysts")
        _cat = _dash_catalysts()
        _e_list = _cat.get("earnings", [])
        _f_list = _cat.get("filings", [])

        if _e_list:
            st.markdown("**📊 Earnings in next 14 days**")
            for r in _e_list:
                eps_str = f" · est ${r['EPS est']:.2f}" if r["EPS est"] else ""
                st.markdown(f"- **{r['Ticker']}** — {r['Date']} (in {r['Days']}d){eps_str}")
        else:
            st.markdown("**📊 Earnings in next 14 days**")
            st.caption("_No upcoming earnings on the Quantum universe._")

        if _f_list:
            st.markdown("**📜 Recent 8-K filings (last 14d)**")
            for r in _f_list:
                # Highlight-reel format: ticker — date — what's in it — link
                label = r.get("Label") or "8-K"
                if r.get("URL"):
                    st.markdown(f"- **{r['Ticker']}** — {r['Date']} — {label} [→]({r['URL']})")
                else:
                    st.markdown(f"- **{r['Ticker']}** — {r['Date']} — {label}")
        else:
            st.markdown("**📜 Recent 8-K filings (last 14d)**")
            st.caption("_No recent 8-K filings on the Quantum universe._")

    # ── Row 3: Quantum index status ───────────────────────────────────────
    st.divider()
    st.markdown("### ⚛️ Quantum index status (Ecosystem, last 90 days)")

    @st.cache_data(ttl=1800, show_spinner=False)
    def _dash_quantum_status():
        from datetime import date as _date, timedelta as _td
        from core.quantum import (
            load_universe, fetch_prices, IndexBuilder, run_full_backtest,
        )
        try:
            uni = load_universe()
            tickers = [c.ticker for c in uni.all_companies()] + list(uni.benchmarks)
            today = _date.today()
            start = today - _td(days=120)
            prices = fetch_prices(tickers, start, today)
            uni_prices = prices[[c.ticker for c in uni.all_companies() if c.ticker in prices.columns]]
            bench_prices = prices[[b for b in uni.benchmarks if b in prices.columns]]
            r = IndexBuilder(uni, prices).build_ecosystem(pd.Timestamp(start), pd.Timestamp(today))
            bt = run_full_backtest(r, uni_prices, bench_prices)
            level_today = float(r.levels.iloc[-1])
            level_prev  = float(r.levels.iloc[-2]) if len(r.levels) >= 2 else level_today
            move = (level_today / level_prev - 1.0) * 100.0 if level_prev else 0.0
            return {
                "level": level_today, "move_1d": move,
                "total_pct": bt["stats"].total_return_pct,
                "conc": bt["concentration"],
            }
        except Exception:
            return None

    _qstatus = _dash_quantum_status()
    if _qstatus:
        _q1, _q2, _q3, _q4 = st.columns(4)
        _dash_tile(_q1, "Ecosystem level",
                   f"{_qstatus['level']:.1f}",
                   f"{_qstatus['move_1d']:+.2f}% today",
                   _dash_color(_qstatus["move_1d"]))
        _dash_tile(_q2, "90-day return", f"{_qstatus['total_pct']:+.1f}%",
                   "", _dash_color(_qstatus["total_pct"]))
        _dash_tile(_q3, "HHI", f"{_qstatus['conc']['hhi']:.3f}",
                   _qstatus["conc"]["diversification_label"])
        _dash_tile(_q4, "Top-3 share", f"{_qstatus['conc']['top3_share_pct']:.1f}%",
                   "of attribution")
    else:
        st.info("Quantum status unavailable.")

    # ── Row 4: Sector ribbon ──────────────────────────────────────────────
    st.divider()
    st.markdown("### 🏛️ Sector ribbon (today's move)")
    _SECTORS = ["SMH", "XLF", "XLK", "XLE", "XLI", "XLV", "QTUM", "SOXX", "KRE", "ITA"]
    _scols = st.columns(len(_SECTORS))
    for _col, _sym in zip(_scols, _SECTORS):
        _q = _dash_quote(_sym)
        if not _q:
            _dash_tile(_col, _sym, "—")
            continue
        _dash_tile(_col, _sym,
                   f"{_q['change_pct']:+.2f}%",
                   f"${_q['price']:,.2f}",
                   _dash_color(_q["change_pct"]))
