import json
import math
import os
import sqlite3
from datetime import date, datetime

from core.sec_edgar import get_recent_filings
from core.setups import compute_trade_setup

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

DB_PATH = os.path.join(os.path.dirname(__file__), "screener.db")

st.set_page_config(page_title="TradeStrategy", layout="wide")
st.title("TradeStrategy")


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_conn():
    return sqlite3.connect(DB_PATH)


def init_tracker_tables():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker         TEXT    NOT NULL,
            trade_type     TEXT    NOT NULL,
            entry_date     TEXT    NOT NULL,
            entry_price    REAL    NOT NULL,
            shares         REAL    NOT NULL,
            position_nzd   REAL    NOT NULL,
            notes          TEXT    DEFAULT '',
            is_open        INTEGER DEFAULT 1
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS crypto_holdings (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            coin             TEXT    NOT NULL,
            amount           REAL    NOT NULL,
            avg_buy_price_nzd REAL   NOT NULL,
            notes            TEXT    DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            triggered_at TEXT NOT NULL,
            scan_date    TEXT NOT NULL,
            scan_window  TEXT NOT NULL,
            ticker       TEXT NOT NULL,
            alert_type   TEXT NOT NULL,
            value        REAL NOT NULL,
            price        REAL,
            change_pct   REAL,
            rvol         REAL,
            gap_pct      REAL
        )
    """)
    conn.commit()
    conn.close()


init_tracker_tables()


# ---------------------------------------------------------------------------
# Price fetching (cached 5 min)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400)
def get_company_info(ticker: str) -> dict:
    """Returns basic company description from yfinance. Cached 24h."""
    if ticker.endswith("-USD"):
        return {}
    try:
        info = yf.Ticker(ticker).info
        return {
            "name":    info.get("longName") or info.get("shortName") or ticker,
            "sector":  info.get("sector") or "",
            "industry":info.get("industry") or "",
            "summary": info.get("longBusinessSummary") or "",
            "website": info.get("website") or "",
        }
    except Exception:
        return {}


@st.cache_data(ttl=300)
def fetch_nzdusd() -> float:
    try:
        rate = yf.Ticker("NZDUSD=X").fast_info["last_price"]
        return float(rate) if rate else 0.57
    except Exception:
        return 0.57


@st.cache_data(ttl=300)
def fetch_prices(tickers: tuple) -> dict:
    """Returns {ticker: {"price": float, "prev_close": float}}."""
    result = {}
    if not tickers:
        return result
    for ticker in tickers:
        try:
            fi = yf.Ticker(ticker).fast_info
            result[ticker] = {
                "price":      float(fi.get("last_price") or fi.get("regularMarketPrice") or 0),
                "prev_close": float(fi.get("previous_close") or fi.get("regularMarketPreviousClose") or 0),
            }
        except Exception:
            result[ticker] = {"price": 0.0, "prev_close": 0.0}
    return result


# ---------------------------------------------------------------------------
# Screener data (must exist before tabs so sidebar can read it)
# ---------------------------------------------------------------------------

screener_ready = os.path.exists(DB_PATH)
dates = []
if screener_ready:
    conn = get_conn()
    dates = pd.read_sql(
        "SELECT DISTINCT run_date FROM results ORDER BY run_date DESC", conn
    )["run_date"].tolist()
    conn.close()

# ---------------------------------------------------------------------------
# Sidebar — screener filters only
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Screener filters")

    if not dates:
        st.info("No screener data yet. Run `python run.py` first.")
        selected_date = None
        strategy = "All"
        asset_filter = "All"
        min_score = 0
        min_change = 0
        min_rvol = 0.0
    else:
        selected_date = st.selectbox("Date", dates)

        conn = get_conn()
        strategies = pd.read_sql(
            "SELECT DISTINCT strategy FROM results WHERE run_date = ?",
            conn, params=(selected_date,)
        )["strategy"].tolist()
        conn.close()

        strategy_options = ["All"] + sorted(strategies)
        strategy = st.selectbox("Strategy", strategy_options)

        if strategy == "momentum":
            st.info(
                "**Momentum criteria**\n"
                "- Market cap < $2B\n"
                "- Float < 50M shares\n"
                "- RVOL ≥ 2×\n"
                "- Day change ≥ +5%"
            )

        asset_filter = st.selectbox("Asset type", ["All", "equity", "crypto"])
        min_score = st.slider(
            "Min score (0–4)", 0, 4, 0,
            help=(
                "Each stock is scored 0–4 based on four signals:\n"
                "- MACD above its signal line\n"
                "- Price above all three EMAs (9 / 20 / 200)\n"
                "- Price above VWAP\n"
                "- Rising 3-day volume trend\n\n"
                "Set to 3 or 4 to see only the strongest setups."
            ),
        )
        default_change = 5 if strategy == "momentum" else 0
        default_rvol   = 2.0 if strategy == "momentum" else 0.0
        min_change = st.slider(
            "Min change %", 0, 100, default_change,
            help=(
                "Minimum percentage price change on the day compared to the previous close.\n\n"
                "Higher values surface stocks with stronger intraday momentum. "
                "Momentum strategy defaults to 5%."
            ),
        )
        min_rvol = st.slider(
            "Min RVOL", 0.0, 20.0, default_rvol, 0.5,
            help=(
                "Relative Volume — today's volume divided by the stock's average daily volume.\n\n"
                "2× means twice the usual volume. High RVOL signals unusual activity "
                "and potential breakout conditions. Momentum strategy defaults to 2×."
            ),
        )

    st.divider()
    with st.expander("How to use"):
        st.markdown(
            """
**Getting started**

1. Run the screener from your terminal to populate data:
   ```
   cd ~/TradeStrategy
   python3 run.py
   ```
2. Come back here and select the date from the **Date** dropdown.

**Tabs**

- **Screener** — ranked list of candidates from the last scan. Use the filters above to narrow results. Score 3–4 means all major signals align.
- **Trade Tracker** — log your open positions (equities and crypto). Live P&L updates every 5 minutes. All values in NZD.
- **Alerts** — pre-market gap and RVOL spikes captured by `scan_premarket.py`.

**Filters**

| Filter | What it does |
|---|---|
| Strategy | Filter by ticker watchlist (AI, Tech, Crypto, Momentum, or Finviz gainers) |
| Asset type | Equities or crypto |
| Min score | 0–4 signal strength — higher is more selective |
| Min change % | Day's price move vs previous close |
| Min RVOL | Volume relative to average — flags unusual activity |

**Scheduling** — add cron jobs to catch moves early and run the daily scan before the US open:
```
# Daily screener — 9am ET (2am NZT)
0 2 * * 1-5 cd ~/TradeStrategy && python3 run.py

# Intraday — every 15 min during US hours (catches volume spikes early)
*/15 14-21 * * 1-5 cd ~/TradeStrategy && python3 scan_intraday.py

# Crypto intraday — every 30 min, 24/7
*/30 * * * * cd ~/TradeStrategy && python3 scan_intraday.py --crypto-only
```
            """
        )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_screener, tab_tracker, tab_alerts, tab_backtest, tab_advice, tab_options, tab_learn = st.tabs(["Screener", "Trade Tracker", "Alerts", "Backtest", "Advice", "Options", "Learn"])


# ---------------------------------------------------------------------------
# Screener — opportunity detail dialog
# ---------------------------------------------------------------------------

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
                st.dataframe(alert_df, hide_index=True, use_container_width=True)
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


# ===========================================================================
# TAB 1 — Screener
# ===========================================================================

with tab_screener:
    if not dates or selected_date is None:
        st.info("No screener data yet. Run `python run.py` first.")
    else:
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
                    st.metric(
                        label=f"**{opp['ticker']}**",
                        value=f"{opp[score_col]:.0f}",
                        delta=f"{opp['change_pct']:.2f}%",
                    )
                    st.caption(
                        f"{_conviction}  ·  "
                        f"RVOL {opp.get('rvol', 0):.1f}x  ·  "
                        f"{opp.get('strategy', '')}"
                    )
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
                use_container_width=True,
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
            st.dataframe(history, use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 2 — Trade Tracker
# ===========================================================================

with tab_tracker:

    nzdusd = fetch_nzdusd()

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------

    conn = get_conn()
    trades_df = pd.read_sql(
        "SELECT * FROM trades WHERE is_open = 1 ORDER BY entry_date DESC", conn
    )
    crypto_df = pd.read_sql(
        "SELECT * FROM crypto_holdings ORDER BY id", conn
    )
    conn.close()

    # -----------------------------------------------------------------------
    # Fetch live prices for open trades
    # -----------------------------------------------------------------------

    equity_tickers = tuple(trades_df["ticker"].unique()) if not trades_df.empty else ()
    crypto_coins   = tuple(
        (c + "-USD") for c in crypto_df["coin"].unique()
    ) if not crypto_df.empty else ()

    all_price_tickers = equity_tickers + crypto_coins
    prices = fetch_prices(all_price_tickers) if all_price_tickers else {}

    # -----------------------------------------------------------------------
    # Build P&L rows for open trades
    # -----------------------------------------------------------------------

    trade_rows = []
    for _, t in trades_df.iterrows():
        px = prices.get(t["ticker"], {})
        current_usd  = px.get("price", 0.0)
        prev_usd     = px.get("prev_close", 0.0)
        shares       = t["shares"]
        cost_nzd     = t["position_nzd"]

        if current_usd and nzdusd:
            current_nzd  = shares * current_usd / nzdusd
            pl_nzd       = current_nzd - cost_nzd
            pl_pct       = (current_usd - t["entry_price"]) / t["entry_price"] * 100
            today_pl_nzd = shares * (current_usd - prev_usd) / nzdusd if prev_usd else 0.0
        else:
            current_nzd = cost_nzd
            pl_nzd = pl_pct = today_pl_nzd = 0.0

        trade_rows.append({
            "id":           t["id"],
            "Ticker":       t["ticker"],
            "Type":         t["trade_type"],
            "Entry date":   t["entry_date"],
            "Entry price":  t["entry_price"],
            "Shares":       round(shares, 4),
            "Cost (NZD)":   round(cost_nzd, 2),
            "Current (NZD)":round(current_nzd, 2),
            "P&L (NZD)":    round(pl_nzd, 2),
            "P&L %":        round(pl_pct, 2),
            "Today (NZD)":  round(today_pl_nzd, 2),
            "Notes":        t["notes"],
        })

    # -----------------------------------------------------------------------
    # Build P&L rows for crypto holdings
    # -----------------------------------------------------------------------

    crypto_rows = []
    for _, h in crypto_df.iterrows():
        yf_ticker = h["coin"] + "-USD"
        px = prices.get(yf_ticker, {})
        current_usd   = px.get("price", 0.0)
        prev_usd      = px.get("prev_close", 0.0)
        amount        = h["amount"]
        avg_buy_nzd   = h["avg_buy_price_nzd"]
        cost_nzd      = amount * avg_buy_nzd

        if current_usd and nzdusd:
            current_nzd  = amount * current_usd / nzdusd
            pl_nzd       = current_nzd - cost_nzd
            current_nzd_per_unit = current_usd / nzdusd
            pl_pct       = (current_nzd_per_unit - avg_buy_nzd) / avg_buy_nzd * 100 if avg_buy_nzd else 0
            today_pl_nzd = amount * (current_usd - prev_usd) / nzdusd if prev_usd else 0.0
        else:
            current_nzd = cost_nzd
            pl_nzd = pl_pct = today_pl_nzd = 0.0

        crypto_rows.append({
            "id":              h["id"],
            "Coin":            h["coin"],
            "Amount":          amount,
            "Avg buy (NZD)":   round(avg_buy_nzd, 4),
            "Cost (NZD)":      round(cost_nzd, 2),
            "Current (NZD)":   round(current_nzd, 2),
            "P&L (NZD)":       round(pl_nzd, 2),
            "P&L %":           round(pl_pct, 2),
            "Today (NZD)":     round(today_pl_nzd, 2),
            "Notes":           h["notes"],
        })

    # -----------------------------------------------------------------------
    # Summary metrics
    # -----------------------------------------------------------------------

    total_cost    = sum(r["Cost (NZD)"]    for r in trade_rows) \
                  + sum(r["Cost (NZD)"]    for r in crypto_rows)
    total_value   = sum(r["Current (NZD)"] for r in trade_rows) \
                  + sum(r["Current (NZD)"] for r in crypto_rows)
    total_pl      = sum(r["P&L (NZD)"]     for r in trade_rows) \
                  + sum(r["P&L (NZD)"]     for r in crypto_rows)
    total_today   = sum(r["Today (NZD)"]   for r in trade_rows) \
                  + sum(r["Today (NZD)"]   for r in crypto_rows)
    total_pl_pct  = (total_pl / total_cost * 100) if total_cost else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio value", f"NZD {total_value:,.2f}")
    c2.metric(
        "Today's P&L",
        f"NZD {total_today:+,.2f}",
        delta=f"{total_today:+.2f}",
        delta_color="normal",
    )
    c3.metric(
        "Total P&L",
        f"NZD {total_pl:+,.2f}",
        delta=f"{total_pl_pct:+.2f}%",
        delta_color="normal",
    )
    c4.metric("NZD/USD rate", f"{nzdusd:.4f}")

    st.divider()

    # -----------------------------------------------------------------------
    # Open trades
    # -----------------------------------------------------------------------

    st.subheader("Open trades")

    with st.expander("Add trade", expanded=False):
        with st.form("add_trade_form", clear_on_submit=True):
            fc1, fc2, fc3 = st.columns(3)
            ticker_in    = fc1.text_input("Ticker (e.g. AAPL, BTC-USD)").strip().upper()
            trade_type   = fc2.selectbox("Type", ["equity", "options", "crypto"])
            entry_date   = fc3.date_input("Entry date", value=date.today())

            fd1, fd2 = st.columns(2)
            entry_price  = fd1.number_input("Entry price (USD)", min_value=0.0, format="%.4f")
            position_nzd = fd2.number_input("Position size (NZD)", min_value=0.0, format="%.2f")

            notes_in = st.text_area("Thesis / notes")
            submitted = st.form_submit_button("Add trade")

            if submitted:
                if not ticker_in or entry_price <= 0 or position_nzd <= 0:
                    st.error("Ticker, entry price, and position size are all required.")
                else:
                    shares_calc = (position_nzd * nzdusd) / entry_price
                    conn = get_conn()
                    conn.execute(
                        """INSERT INTO trades
                           (ticker, trade_type, entry_date, entry_price, shares, position_nzd, notes)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (ticker_in, trade_type, str(entry_date),
                         entry_price, shares_calc, position_nzd, notes_in),
                    )
                    conn.commit()
                    conn.close()
                    st.success(f"Added {ticker_in} — {shares_calc:.4f} shares @ ${entry_price:.4f}")
                    st.cache_data.clear()
                    st.rerun()

    if trade_rows:
        display_trades = pd.DataFrame(trade_rows).drop(columns=["id"])
        st.dataframe(
            display_trades,
            use_container_width=True,
            hide_index=True,
            column_config={
                "P&L (NZD)": st.column_config.NumberColumn("P&L (NZD)", format="%.2f"),
                "P&L %":     st.column_config.NumberColumn("P&L %",     format="%.2f%%"),
                "Today (NZD)": st.column_config.NumberColumn("Today (NZD)", format="%.2f"),
            },
        )

        with st.expander("Notes", expanded=False):
            for r in trade_rows:
                if r["Notes"]:
                    st.markdown(f"**{r['Ticker']}** — {r['Notes']}")

        with st.expander("Close or delete a trade", expanded=False):
            trade_options = {
                f"#{r['id']}  {r['Ticker']}  ({r['Entry date']})": r["id"]
                for r in trade_rows
            }
            selected_label = st.selectbox("Select trade", list(trade_options.keys()))
            selected_id    = trade_options[selected_label]
            col_close, col_del = st.columns(2)

            if col_close.button("Mark closed"):
                conn = get_conn()
                conn.execute("UPDATE trades SET is_open = 0 WHERE id = ?", (selected_id,))
                conn.commit()
                conn.close()
                st.cache_data.clear()
                st.rerun()

            if col_del.button("Delete permanently", type="secondary"):
                conn = get_conn()
                conn.execute("DELETE FROM trades WHERE id = ?", (selected_id,))
                conn.commit()
                conn.close()
                st.cache_data.clear()
                st.rerun()
    else:
        st.info("No open trades. Add one above.")

    st.divider()

    # -----------------------------------------------------------------------
    # Crypto holdings
    # -----------------------------------------------------------------------

    st.subheader("Crypto holdings")

    with st.expander("Add holding", expanded=False):
        with st.form("add_crypto_form", clear_on_submit=True):
            cc1, cc2, cc3 = st.columns(3)
            coin_in        = cc1.text_input("Coin (e.g. BTC, ETH, SOL)").strip().upper()
            amount_in      = cc2.number_input("Amount held", min_value=0.0, format="%.6f")
            avg_buy_nzd_in = cc3.number_input("Avg buy price (NZD)", min_value=0.0, format="%.4f")
            crypto_notes   = st.text_area("Notes")
            c_submitted    = st.form_submit_button("Add holding")

            if c_submitted:
                if not coin_in or amount_in <= 0 or avg_buy_nzd_in <= 0:
                    st.error("Coin, amount, and avg buy price are all required.")
                else:
                    conn = get_conn()
                    conn.execute(
                        """INSERT INTO crypto_holdings
                           (coin, amount, avg_buy_price_nzd, notes)
                           VALUES (?, ?, ?, ?)""",
                        (coin_in, amount_in, avg_buy_nzd_in, crypto_notes),
                    )
                    conn.commit()
                    conn.close()
                    st.success(f"Added {amount_in} {coin_in}")
                    st.cache_data.clear()
                    st.rerun()

    if crypto_rows:
        display_crypto = pd.DataFrame(crypto_rows).drop(columns=["id"])
        st.dataframe(
            display_crypto,
            use_container_width=True,
            hide_index=True,
            column_config={
                "P&L (NZD)":   st.column_config.NumberColumn("P&L (NZD)",   format="%.2f"),
                "P&L %":       st.column_config.NumberColumn("P&L %",       format="%.2f%%"),
                "Today (NZD)": st.column_config.NumberColumn("Today (NZD)", format="%.2f"),
            },
        )

        with st.expander("Notes", expanded=False):
            for r in crypto_rows:
                if r["Notes"]:
                    st.markdown(f"**{r['Coin']}** — {r['Notes']}")

        with st.expander("Delete a holding", expanded=False):
            holding_options = {
                f"#{r['id']}  {r['Coin']}  ({r['Amount']} units)": r["id"]
                for r in crypto_rows
            }
            selected_label_c = st.selectbox("Select holding", list(holding_options.keys()))
            selected_id_c    = holding_options[selected_label_c]
            if st.button("Delete holding", type="secondary"):
                conn = get_conn()
                conn.execute("DELETE FROM crypto_holdings WHERE id = ?", (selected_id_c,))
                conn.commit()
                conn.close()
                st.cache_data.clear()
                st.rerun()
    else:
        st.info("No crypto holdings. Add one above.")

    st.caption(f"Prices refresh every 5 min  •  NZD/USD {nzdusd:.4f}  •  All values in NZD")


# ===========================================================================
# TAB 3 — Alerts
# ===========================================================================

with tab_alerts:

    ALERT_LABELS = {
        "rvol":     "RVOL (daily)",
        "rvol_15m": "RVOL 15m",
        "change":   "Change",
        "gap_up":   "Gap Up",
        "gap_down": "Gap Down",
    }

    ALERTS_CSV = os.path.join(os.path.dirname(__file__), "alerts.csv")

    if not os.path.exists(ALERTS_CSV):
        st.info("No alerts yet. The scanner will create alerts.csv on first run.")
    else:
        all_alerts = pd.read_csv(ALERTS_CSV)

        if all_alerts.empty:
            st.info("No alerts yet.")
        else:
            today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
            today_df  = all_alerts[all_alerts["scan_date"] == today_str]

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Alerts today",            len(today_df))
            m2.metric("Tickers flagged today",   today_df["ticker"].nunique())
            m3.metric("Total alerts (all time)", len(all_alerts))
            m4.metric("Last scan",               all_alerts["triggered_at"].iloc[-1])

            st.divider()

            # Filters
            f1, f2, f3 = st.columns(3)
            date_filter   = f1.selectbox("Date",   ["All"] + sorted(all_alerts["scan_date"].unique().tolist(), reverse=True))
            ticker_filter = f2.selectbox("Ticker", ["All"] + sorted(all_alerts["ticker"].unique().tolist()))
            type_filter   = f3.selectbox("Alert type", ["All", "rvol", "rvol_15m", "change", "gap_up", "gap_down"])

            filtered = all_alerts.copy()
            if date_filter   != "All":
                filtered = filtered[filtered["scan_date"]  == date_filter]
            if ticker_filter != "All":
                filtered = filtered[filtered["ticker"]     == ticker_filter]
            if type_filter   != "All":
                filtered = filtered[filtered["alert_type"] == type_filter]
            filtered = filtered.iloc[::-1].reset_index(drop=True)  # newest first

            st.caption(f"{len(filtered)} alert(s) matching filters")

            if filtered.empty:
                st.info("No alerts match the current filters.")
            else:
                filtered["Alert"] = filtered["alert_type"].map(ALERT_LABELS).fillna(filtered["alert_type"])

                display_df = filtered[[
                    "scan_date", "scan_window", "ticker",
                    "Alert", "value", "price",
                    "change_pct", "rvol", "gap_pct",
                ]].copy()
                display_df.columns = [
                    "Date", "Window", "Ticker",
                    "Alert", "Value", "Price",
                    "Change %", "RVOL", "Gap %",
                ]

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Change %": st.column_config.NumberColumn("Change %", format="%.2f%%"),
                        "RVOL":     st.column_config.NumberColumn("RVOL",     format="%.2fx"),
                        "Gap %":    st.column_config.NumberColumn("Gap %",    format="%.2f%%"),
                        "Price":    st.column_config.NumberColumn("Price",    format="$%.4f"),
                        "Value":    st.column_config.NumberColumn("Value",    format="%.2f"),
                    },
                )

                if len(filtered) > 1:
                    st.subheader("Alert breakdown")
                    breakdown = (
                        filtered.groupby("Alert")
                        .size()
                        .reset_index(name="count")
                        .set_index("Alert")
                    )
                    st.bar_chart(breakdown["count"])

            with st.expander("All-time ticker frequency"):
                freq = (
                    all_alerts.groupby("ticker")
                    .agg(
                        total_alerts=("alert_type", "count"),
                        days_flagged=("scan_date",  "nunique"),
                        last_flagged=("scan_date",  "max"),
                    )
                    .sort_values("total_alerts", ascending=False)
                    .reset_index()
                )
                st.dataframe(freq, use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 4 — Backtest
# ===========================================================================

with tab_backtest:

    conn = get_conn()
    bt_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='backtest'"
    ).fetchone()
    bt_df = pd.read_sql("SELECT * FROM backtest", conn) if bt_exists else pd.DataFrame()
    conn.close()

    if bt_df.empty:
        st.info("No backtest data yet. Run `python3 backtest.py` to populate.")
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

        st.dataframe(score_summary, use_container_width=True, hide_index=True)

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
        st.dataframe(strat_summary, use_container_width=True, hide_index=True)

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
            use_container_width=True,
            hide_index=True,
            column_config={
                "return_1d":  st.column_config.NumberColumn("1d %",  format="%+.2f%%"),
                "return_3d":  st.column_config.NumberColumn("3d %",  format="%+.2f%%"),
                "return_5d":  st.column_config.NumberColumn("5d %",  format="%+.2f%%"),
                "return_10d": st.column_config.NumberColumn("10d %", format="%+.2f%%"),
                "score":      st.column_config.NumberColumn("Score",  format="%d/4"),
            },
        )


# ===========================================================================
# TAB 5 — Advice
# ===========================================================================

with tab_advice:

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def signal_reasons(row: pd.Series) -> list[str]:
        """Plain-English reasons why this stock scored what it scored."""
        reasons = []
        is_crypto = str(row.get("ticker", "")).endswith("-USD")

        if row.get("macd", 0) > row.get("macd_signal", 0):
            reasons.append("MACD crossed above signal — momentum turning up")

        if is_crypto:
            if row.get("ema9", 0) > row.get("ema20", 0):
                reasons.append("EMA9 above EMA20 — short-term trend is bullish")
            if row.get("rvol", 0) >= 1.5:
                reasons.append(f"RVOL {row['rvol']:.1f}× — above-average participation")
            rsi = row.get("rsi", 50)
            if 40 <= rsi <= 75:
                reasons.append(f"RSI {rsi:.0f} — in momentum zone, not yet exhausted")
            elif rsi > 75:
                reasons.append(f"RSI {rsi:.0f} — overbought, risk of pullback")
        else:
            ema9, ema20, ema200 = row.get("ema9", 0), row.get("ema20", 0), row.get("ema200", 0)
            if ema9 > ema20 > ema200:
                reasons.append("EMA9 > EMA20 > EMA200 — trend aligned across all timeframes")
            elif ema9 > ema20:
                reasons.append("EMA9 above EMA20 — short-term trend bullish but below 200")
            if row.get("price", 0) > row.get("vwap", 0):
                reasons.append("Price above VWAP — buyers in control")
            if row.get("volume_trend_up") == 1:
                reasons.append("3-day volume trend rising — institutional accumulation signal")
            if row.get("rvol", 0) >= 3:
                reasons.append(f"RVOL {row['rvol']:.1f}× — heavy unusual volume, something is moving this")
            elif row.get("rvol", 0) >= 1.5:
                reasons.append(f"RVOL {row['rvol']:.1f}× — above-average volume")

        return reasons

    def entry_advice(row: pd.Series) -> str:
        is_crypto = str(row.get("ticker", "")).endswith("-USD")
        if is_crypto:
            return (
                "Wait for 15m RVOL to spike above 1.5× before entering — "
                "run scan_intraday.py to catch the signal. Enter on a 15m candle close above the current price."
            )
        return (
            "Watch the first 15-minute candle after open. Enter on a break above its high "
            "with volume confirming (RVOL ≥ 2× on the intraday scan). "
            "Do not chase if the stock has already moved more than 5% before you enter."
        )

    def sizing_advice(row: pd.Series, risk_nzd: float, nzdusd: float) -> str:
        price     = row.get("price", 0)
        stop      = row.get("stop_loss", 0)
        if not price or not stop or price <= stop:
            return "Stop loss not calculable — skip position sizing."
        stop_dist_usd = price - stop
        stop_dist_nzd = stop_dist_usd / nzdusd if nzdusd else stop_dist_usd
        if stop_dist_nzd <= 0:
            return "Stop distance is zero — do not trade."
        shares    = risk_nzd / stop_dist_nzd
        cost_nzd  = shares * price / nzdusd if nzdusd else shares * price
        target    = round(price + 2 * stop_dist_usd, 4)   # 2:1 R/R
        return (
            f"Risk NZD {risk_nzd:.0f} → **{shares:.1f} shares** "
            f"(position ≈ NZD {cost_nzd:,.0f})  |  "
            f"Stop: ${stop:.4f}  |  Target (2:1): ${target:.4f}"
        )

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------

    if not dates:
        st.info("No screener data yet. Run `python3 run.py` first.")
    else:
        latest_date = dates[0]

        conn = get_conn()
        today_df = pd.read_sql(
            "SELECT * FROM results WHERE run_date = ? ORDER BY score DESC, change_pct DESC",
            conn, params=(latest_date,)
        )

        bt_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='backtest'"
        ).fetchone()
        missed_df = pd.read_sql(
            """
            SELECT b.run_date, b.ticker, b.strategy, b.score,
                   b.entry_price, b.return_1d, b.return_3d, b.return_5d
            FROM backtest b
            WHERE b.score >= 3
              AND b.return_1d IS NOT NULL
              AND b.run_date < ?
            ORDER BY b.return_1d DESC
            """,
            conn, params=(latest_date,)
        ) if bt_exists else pd.DataFrame()
        conn.close()

        nzdusd = fetch_nzdusd()

        # -----------------------------------------------------------------------
        # Position size risk input
        # -----------------------------------------------------------------------

        st.subheader("Risk per trade")
        risk_nzd = st.number_input(
            "How much NZD are you willing to lose if this trade hits stop?",
            min_value=10.0, max_value=10000.0, value=150.0, step=10.0,
            help="This is your maximum loss per trade, not your position size. "
                 "Position size is calculated from this and the stop distance."
        )

        st.divider()

        # -----------------------------------------------------------------------
        # Top picks — score 3 and 4
        # -----------------------------------------------------------------------

        top_picks = today_df[today_df["score"] >= 3]

        st.subheader(f"Today's top picks  —  {latest_date}")

        if top_picks.empty:
            st.warning("No stocks scored 3 or higher today. Check back after the next screener run.")
        else:
            for _, row in top_picks.iterrows():
                ticker = row["ticker"]
                score  = int(row["score"])
                stars  = "★" * score + "☆" * (4 - score)

                with st.container(border=True):
                    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
                    c1.markdown(f"### {ticker}  `{stars}`")
                    c2.metric("Score",    f"{score}/4")
                    c3.metric("Change",   f"{row['change_pct']:+.2f}%")
                    c4.metric("RVOL",     f"{row['rvol']:.1f}×")

                    reasons = signal_reasons(row)
                    if reasons:
                        st.markdown("**Why it scored:**")
                        for r in reasons:
                            st.markdown(f"- {r}")

                    st.markdown("**Entry:**")
                    st.markdown(entry_advice(row))

                    st.markdown("**Position size:**")
                    st.markdown(sizing_advice(row, risk_nzd, nzdusd))

                    with st.expander("Full indicators"):
                        ind_cols = ["price", "stop_loss", "rsi", "rvol",
                                    "ema9", "ema20", "ema200", "macd", "macd_signal", "vwap"]
                        ind_cols = [c for c in ind_cols if c in row.index]
                        st.dataframe(
                            pd.DataFrame(row[ind_cols]).T,
                            use_container_width=True, hide_index=True,
                        )

                    with st.expander(f"About {ticker}"):
                        co = get_company_info(ticker)
                        if co:
                            st.markdown(f"**{co['name']}**")
                            if co["sector"] or co["industry"]:
                                st.caption(f"{co['sector']}  ·  {co['industry']}")
                            if co["summary"]:
                                # Trim to first 3 sentences
                                sentences = co["summary"].split(". ")
                                brief = ". ".join(sentences[:3]).strip()
                                if not brief.endswith("."):
                                    brief += "."
                                st.markdown(brief)
                            if co["website"]:
                                st.markdown(co["website"])
                        else:
                            st.caption("No company data available.")

        st.divider()

        # -----------------------------------------------------------------------
        # What you missed — previous score 3-4 picks with outcomes
        # -----------------------------------------------------------------------

        st.subheader("What you missed")
        st.caption("Previous screener runs that scored 3 or higher — and what happened next.")

        if missed_df.empty:
            st.info("No historical high-score picks with forward returns yet. Run `python3 backtest.py` after each session.")
        else:
            for _, row in missed_df.iterrows():
                r1  = row.get("return_1d")
                r3  = row.get("return_3d")
                r5  = row.get("return_5d")
                direction = "up" if (r1 or 0) > 0 else "down"
                color     = "green" if direction == "up" else "red"

                with st.container(border=True):
                    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
                    c1.markdown(f"**{row['ticker']}**  `{row['run_date']}`  score {int(row['score'])}/4")
                    c2.metric("Entry",  f"${row['entry_price']:.2f}")
                    c3.metric("1d",     f"{r1:+.1f}%" if r1 is not None else "—")
                    c4.metric("3d",     f"{r3:+.1f}%" if r3 is not None else "—")
                    c5.metric("5d",     f"{r5:+.1f}%" if r5 is not None else "—")




# ===========================================================================
# TAB 6 — Options
# ===========================================================================

# ---------------------------------------------------------------------------
# Black-Scholes (no scipy)
# ---------------------------------------------------------------------------

def _ncdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_price(S, K, T, r, sigma, opt="call") -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0) if opt == "call" else max(K - S, 0)
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if opt == "call":
            return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
        return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)
    except Exception:
        return 0.0

def bs_greeks(S, K, T, r, sigma, opt="call") -> dict:
    zero = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return zero
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        pdf1  = _npdf(d1)
        gamma = pdf1 / (S * sigma * math.sqrt(T))
        vega  = S * pdf1 * math.sqrt(T) / 100
        if opt == "call":
            delta = _ncdf(d1)
            theta = (-(S * pdf1 * sigma) / (2 * math.sqrt(T))
                     - r * K * math.exp(-r * T) * _ncdf(d2)) / 365
        else:
            delta = _ncdf(d1) - 1
            theta = (-(S * pdf1 * sigma) / (2 * math.sqrt(T))
                     + r * K * math.exp(-r * T) * _ncdf(-d2)) / 365
        return {"delta": round(delta,3), "gamma": round(gamma,5),
                "theta": round(theta,4), "vega": round(vega,4)}
    except Exception:
        return zero

@st.cache_data(ttl=300)
def get_chain(ticker: str, expiry: str):
    tk    = yf.Ticker(ticker)
    chain = tk.option_chain(expiry)
    spot  = float(tk.fast_info.get("last_price") or 0)
    return chain.calls, chain.puts, spot

@st.cache_data(ttl=3600)
def get_rv30(ticker: str) -> float | None:
    try:
        hist = yf.Ticker(ticker).history(period="90d", interval="1d")
        if len(hist) < 31:
            return None
        lr = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        return float(lr.tail(30).std() * math.sqrt(252))
    except Exception:
        return None

def enrich_chain(df, spot, expiry_str, opt_type, r=0.045):
    today  = datetime.utcnow().date()
    exp_dt = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    T      = max((exp_dt - today).days, 0) / 365.0
    rows = []
    for _, row in df.iterrows():
        K   = float(row["strike"])
        iv  = float(row["impliedVolatility"]) if row.get("impliedVolatility", 0) > 0 else 0.0
        bid = float(row.get("bid") or 0)
        ask = float(row.get("ask") or 0)
        mid = round((bid + ask) / 2, 3) if ask > 0 else 0.0
        prem = mid or float(row.get("lastPrice") or 0)
        g   = bs_greeks(spot, K, T, r, iv, opt_type)
        be  = round(K + prem, 2) if opt_type == "call" else round(K - prem, 2)
        rows.append({
            "Strike": K, "ITM": (spot > K) if opt_type == "call" else (spot < K),
            "Bid": round(bid,3), "Ask": round(ask,3), "Mid": mid,
            "IV %": round(iv*100, 1),
            "Delta": g["delta"], "Gamma": g["gamma"],
            "Theta/day": g["theta"], "Vega/1%": g["vega"],
            "OI": int(row.get("openInterest") or 0),
            "Volume": int(row.get("volume") or 0),
            "Break-even": be,
        })
    out = pd.DataFrame(rows)
    return out[(out["Bid"] > 0) | (out["OI"] > 0)].reset_index(drop=True)

def payoff_df(spot, legs, price_range_pct=0.30):
    """
    legs: list of dicts — {type, strike, premium, qty, position}
      type: 'call' or 'put'
      position: 'long' (+1) or 'short' (-1)
      qty: number of contracts
    Returns DataFrame {Stock price, P&L (per share)}
    """
    lo = spot * (1 - price_range_pct)
    hi = spot * (1 + price_range_pct)
    prices = np.linspace(lo, hi, 200)
    total_pnl = np.zeros(len(prices))
    for leg in legs:
        K      = leg["strike"]
        prem   = leg["premium"]
        pos    = 1 if leg["position"] == "long" else -1
        qty    = leg.get("qty", 1)
        if leg["type"] == "call":
            intrinsic = np.maximum(prices - K, 0)
        else:
            intrinsic = np.maximum(K - prices, 0)
        total_pnl += pos * qty * (intrinsic - prem)
    return pd.DataFrame({"Stock price": prices, "P&L per share": total_pnl})


with tab_options:

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

                tk       = yf.Ticker(opt_ticker)
                spot     = float(tk.fast_info.get("last_price") or 0)
                expiries = tk.options
                rv30     = get_rv30(opt_ticker)

                if not expiries or not spot:
                    st.warning(f"No options data available for {opt_ticker}.")
                else:
                    # Best 30–45 DTE expiry
                    today_dt = datetime.utcnow().date()
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
                    setup_type = screener_row.get("setup_type") if screener_row else None
                    tradescore = float(screener_row.get("tradescore") or 0) if screener_row else None

                    iv_expensive = bool(rv30 and atm_iv > rv30 * 1.3)
                    iv_cheap     = bool(rv30 and atm_iv < rv30 * 0.85)

                    # Strategy selection
                    if direction == "long":
                        rec_strat = "Bull Call Spread" if iv_expensive else "Long Call"
                        rec_bias  = "Bullish"
                        if iv_expensive:
                            iv_note = f"IV {atm_iv*100:.0f}% is elevated vs 30d RV {rv30*100:.0f}% — spread reduces vega exposure and cuts premium cost."
                        elif iv_cheap:
                            iv_note = f"IV {atm_iv*100:.0f}% is below 30d RV {rv30*100:.0f}% — options are cheap, outright call is the better play."
                        else:
                            iv_note = f"IV {atm_iv*100:.0f}% is fair vs 30d RV {rv30*100:.0f}% — outright call is fine."
                    elif direction == "short":
                        rec_strat = "Bear Put Spread" if iv_expensive else "Long Put"
                        rec_bias  = "Bearish"
                        if iv_expensive:
                            iv_note = f"IV {atm_iv*100:.0f}% elevated vs RV {rv30*100:.0f}% — spread reduces cost vs outright put."
                        else:
                            iv_note = f"IV {atm_iv*100:.0f}% fair/cheap vs RV {rv30*100:.0f}% — outright put captures full downside."
                    elif setup_type in {"Strong but extended", "Overextended", "Extended downside move", "Strong downside setup"}:
                        rec_strat = "Cash-Secured Put"
                        rec_bias  = "Neutral / Pullback entry"
                        iv_note   = (
                            f"IV {atm_iv*100:.0f}% is elevated — good for premium collection."
                            if iv_expensive else
                            f"IV {atm_iv*100:.0f}% is fair."
                        )
                    else:
                        rec_strat = None
                        rec_bias  = None
                        iv_note   = f"IV {atm_iv*100:.0f}%  |  30d RV {rv30*100:.0f}%" if rv30 else f"IV {atm_iv*100:.0f}%"

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
                        st.warning("No clear directional signal from the screener. Enter a ticker that has a long or short setup.")
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

                        if rec_strat == "Long Call":
                            k1, prem1 = atm_call_k, _mid(calls_raw, atm_call_k)
                            legs = [{"type":"call","strike":k1,"premium":prem1,"qty":1,"position":"long"}]
                            net, max_loss, max_profit = prem1, round(prem1*100,2), "Unlimited"
                            be_price = round(k1 + prem1, 2)
                            strike_desc = f"Strike ${k1:.2f} (ATM call)"

                        elif rec_strat == "Bull Call Spread":
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

                        elif rec_strat == "Long Put":
                            k1, prem1 = atm_put_k, _mid(puts_raw, atm_put_k)
                            legs = [{"type":"put","strike":k1,"premium":prem1,"qty":1,"position":"long"}]
                            net, max_loss, max_profit = prem1, round(prem1*100,2), round((k1-prem1)*100,2)
                            be_price = round(k1 - prem1, 2)
                            strike_desc = f"Strike ${k1:.2f} (ATM put)"

                        elif rec_strat == "Bear Put Spread":
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

                        else:  # Cash-Secured Put
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
                        st.caption(f"{rec_bias}  ·  {opt_ticker}  ·  Expiry {best_exp} ({best_dte}d)  ·  {strike_desc}")
                        st.info(iv_note)

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
                tk      = yf.Ticker(opt_ticker)
                expiries = tk.options
                spot    = float(tk.fast_info.get("last_price") or 0)
                rv30    = get_rv30(opt_ticker)

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
                    today_dt = datetime.utcnow().date()
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
                        st.dataframe(chain_df, use_container_width=True, hide_index=True,
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
                tk       = yf.Ticker(opt_ticker)
                expiries = tk.options
                spot     = float(tk.fast_info.get("last_price") or 0)
                rv30     = get_rv30(opt_ticker)

                if not expiries:
                    st.warning(f"No options data for {opt_ticker}.")
                else:
                    today_dt = datetime.utcnow().date()
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
                "Run it after each session alongside `backtest.py`."
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
            st.dataframe(summary, use_container_width=True, hide_index=True)

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
                    st.dataframe(comp_display, use_container_width=True, hide_index=True,
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
            st.dataframe(filtered_opt[log_cols], use_container_width=True, hide_index=True,
                column_config={
                    "return_1d":  st.column_config.NumberColumn("1d %",  format="%+.1f%%"),
                    "return_3d":  st.column_config.NumberColumn("3d %",  format="%+.1f%%"),
                    "return_5d":  st.column_config.NumberColumn("5d %",  format="%+.1f%%"),
                    "return_10d": st.column_config.NumberColumn("10d %", format="%+.1f%%"),
                    "entry_iv":   st.column_config.NumberColumn("IV",    format="%.1%%"),
                    "screener_score": st.column_config.NumberColumn("Score", format="%d/4"),
                })


# ===========================================================================
# TAB 7 — Learn
# ===========================================================================

with tab_learn:

    st.subheader("Options fundamentals")
    st.caption("Each lesson uses live data from your watchlist and the positions you hold.")

    lesson = st.selectbox("Choose a lesson", [
        "1. What is an option?",
        "2. Calls vs Puts",
        "3. The Greeks — Delta",
        "4. The Greeks — Theta (time decay)",
        "5. The Greeks — Vega (implied volatility)",
        "6. IV crush — the most common way to lose money",
        "7. Strategies and when to use them",
        "8. Position sizing and risk management",
        "9. The most common mistakes",
    ])

    RISK_FREE_L = 0.045

    # -----------------------------------------------------------------------
    # Shared live example data (INTC — one of Dave's positions)
    # -----------------------------------------------------------------------

    @st.cache_data(ttl=600)
    def learn_example():
        try:
            tk    = yf.Ticker("INTC")
            spot  = float(tk.fast_info.get("last_price") or 62)
            expiries = tk.options
            # Pick ~30 DTE expiry
            today_dt = datetime.utcnow().date()
            exp = None
            for e in expiries:
                dte = (datetime.strptime(e, "%Y-%m-%d").date() - today_dt).days
                if 20 <= dte <= 50:
                    exp = e
                    break
            exp = exp or expiries[1]
            chain = tk.option_chain(exp)
            dte   = (datetime.strptime(exp, "%Y-%m-%d").date() - today_dt).days
            # Find ATM call
            calls = chain.calls
            calls = calls[calls["bid"] > 0]
            atm   = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]].iloc[0]
            K     = float(atm["strike"])
            iv    = float(atm["impliedVolatility"])
            bid   = float(atm.get("bid", 0))
            ask   = float(atm.get("ask", 0))
            mid   = round((bid + ask) / 2, 3)
            return {"spot": spot, "exp": exp, "dte": dte, "K": K, "iv": iv, "mid": mid}
        except Exception:
            return {"spot": 62.0, "exp": "2026-05-15", "dte": 35, "K": 62.0, "iv": 0.45, "mid": 3.20}

    ex = learn_example()
    S, K, T_ex, iv_ex, mid_ex = ex["spot"], ex["K"], ex["dte"]/365, ex["iv"], ex["mid"]

    st.divider()

    # =======================================================================
    if lesson == "1. What is an option?":
    # =======================================================================

        st.markdown("""
An option is a **contract** that gives you the right — but not the obligation — to buy or sell a stock at a specific price, before a specific date.

You pay a **premium** upfront. That premium is your maximum loss. The stock moves in your favour, the option gains value. It moves against you, the option loses value — but you can never lose more than what you paid.

**Three things define every option:**

| Term | What it means |
|---|---|
| **Strike price** | The price at which you have the right to buy or sell |
| **Expiry date** | The date the contract expires — after this it's worthless |
| **Premium** | What you pay to buy the contract (your max loss) |

**Options vs buying stock directly:**
""")

        nzdusd_l = fetch_nzdusd()
        shares_direct = round((1000 * nzdusd_l) / S, 4)
        cost_option   = round(mid_ex * 100, 2)
        cost_nzd_opt  = round(cost_option / nzdusd_l, 2)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Buy INTC stock directly**")
            st.markdown(f"- Spend NZD 1,000 → get **{shares_direct} shares**")
            st.markdown(f"- Stock goes up 10% → you make NZD {1000*0.10:.0f}")
            st.markdown(f"- Stock goes to zero → you lose NZD 1,000")

        with col2:
            st.markdown("**Buy 1 ATM call option on INTC**")
            st.markdown(f"- Spend NZD {cost_nzd_opt:.0f} → control **100 shares**")
            st.markdown(f"- Stock goes up 10% → option might gain 40–60%")
            st.markdown(f"- Option expires worthless → you lose NZD {cost_nzd_opt:.0f} only")

        st.info(f"Live example: INTC at ${S:.2f}. ATM call (strike ${K:.2f}, expiry {ex['exp']}) costs ${mid_ex:.3f} per share = **${mid_ex*100:.2f} per contract** (NZD {cost_nzd_opt:.0f}).")

        st.markdown("""
**Key rule:** Options buyers have **limited loss, unlimited upside**.
Options sellers have **limited upside (the premium), unlimited risk**. Start as a buyer.
""")

    # =======================================================================
    elif lesson == "2. Calls vs Puts":
    # =======================================================================

        st.markdown("""
**Call option** — you think the stock is going UP.
Gives you the right to *buy* shares at the strike price.

**Put option** — you think the stock is going DOWN.
Gives you the right to *sell* shares at the strike price.
""")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Call")
            st.markdown(f"""
- You buy a call on INTC at strike **${K:.2f}**
- If INTC rises to **${K*1.15:.2f}** (+15%), your call is worth at least **${max(K*1.15-K,0):.2f}** in intrinsic value
- If INTC stays below **${K:.2f}** at expiry → expires worthless, you lose the premium
- **Use when:** Bullish. RVOL spiking. Score 3–4 on screener.
""")
        with c2:
            st.markdown("### Put")
            st.markdown(f"""
- You buy a put on INTC at strike **${K:.2f}**
- If INTC drops to **${K*0.85:.2f}** (−15%), your put is worth at least **${max(K-K*0.85,0):.2f}** in intrinsic value
- If INTC stays above **${K:.2f}** at expiry → expires worthless
- **Use when:** Bearish. Bad earnings expected. Hedging an existing long position.
""")

        st.divider()
        st.markdown("**Intrinsic vs extrinsic value**")
        st.markdown(f"""
An option's premium has two parts:

- **Intrinsic value** — the profit if you exercised right now. For a call at ${K:.2f} with stock at ${S:.2f}: ${max(S-K,0):.2f}
- **Extrinsic (time) value** — what you pay for time + volatility. This is **{mid_ex - max(S-K,0):.3f}** of your {mid_ex:.3f} premium.

All extrinsic value goes to zero at expiry. That's why time works against option buyers.
""")

        iv_pct = round(iv_ex * 100, 1)
        st.info(f"Live: INTC ATM call premium = ${mid_ex:.3f}. Intrinsic = ${max(S-K,0):.2f}. Time value = ${mid_ex - max(S-K,0):.3f}. Current IV = {iv_pct}%.")

    # =======================================================================
    elif lesson == "3. The Greeks — Delta":
    # =======================================================================

        g = bs_greeks(S, K, T_ex, RISK_FREE_L, iv_ex, "call")
        delta = g["delta"]

        st.markdown(f"""
**Delta** tells you how much the option price moves for every $1 the stock moves.

INTC ATM call delta = **{delta:.3f}**

This means:
- Stock goes up $1 → option gains **${delta:.3f}** per share → **${delta*100:.2f} per contract**
- Stock goes up $5 → option gains approximately **${delta*5:.2f}** per share
- Stock drops $1 → option loses **${delta:.3f}** per share

**Delta also tells you the approximate probability the option expires in the money.**
Delta {delta:.2f} ≈ {delta*100:.0f}% chance of expiring with value.

**Delta by strike:**
""")

        delta_rows = []
        for moneyness, label in [(0.90,"Deep ITM"), (0.95,"ITM"), (1.00,"ATM"), (1.05,"OTM"), (1.10,"Deep OTM")]:
            Kx = round(S * moneyness, 2)
            gx = bs_greeks(S, Kx, T_ex, RISK_FREE_L, iv_ex, "call")
            delta_rows.append({
                "Type": label, "Strike": f"${Kx:.2f}",
                "Delta": gx["delta"],
                "Approx prob ITM": f"{gx['delta']*100:.0f}%",
                "Move per $1 stock (per contract)": f"${gx['delta']*100:.2f}",
            })
        st.dataframe(pd.DataFrame(delta_rows), use_container_width=True, hide_index=True)

        st.markdown("""
**What delta to choose?**
- **0.70+ (deep ITM):** Moves almost like owning stock. Expensive. Lower % return but safer.
- **0.50 (ATM):** Balanced. Most common starting point.
- **0.30 (OTM):** Cheap. Needs a bigger move. Higher % return if it works, more often expires worthless.
- **< 0.20 (far OTM):** Lottery ticket. Rarely pays off. Avoid until you understand options well.
""")

    # =======================================================================
    elif lesson == "4. The Greeks — Theta (time decay)":
    # =======================================================================

        g     = bs_greeks(S, K, T_ex, RISK_FREE_L, iv_ex, "call")
        theta = g["theta"]

        st.markdown(f"""
**Theta** is the daily cost of holding an option. Every day that passes, the option loses this much value — even if the stock doesn't move.

INTC ATM call theta = **{theta:.4f}** per share per day = **${abs(theta)*100:.2f} per contract per day**

Over {ex['dte']} days to expiry, that's **${abs(theta)*100*ex['dte']:.2f}** in total time decay — which is most of your premium.

Theta accelerates. It's slow far from expiry and rapid in the last 2 weeks.
""")

        # Theta decay chart
        decay_rows = []
        for days_left in range(ex["dte"], 0, -1):
            T_temp = days_left / 365.0
            px = bs_price(S, K, T_temp, RISK_FREE_L, iv_ex, "call")
            decay_rows.append({"Days to expiry": days_left, "Option value ($)": round(px, 4)})
        decay_df = pd.DataFrame(decay_rows).set_index("Days to expiry").sort_index()
        st.line_chart(decay_df)
        st.caption(f"INTC ATM call (strike ${K:.2f}, IV {iv_ex*100:.0f}%) — value over time assuming stock stays at ${S:.2f}. "
                   "The curve accelerates downward as expiry approaches.")

        st.markdown("""
**Rules of thumb:**
- Hold options for **short periods** when buying — theta is working against you every day
- Don't hold options into the last 2 weeks unless you're very confident
- Sellers (cash-secured puts, covered calls) *benefit* from theta — it's working for them
""")

    # =======================================================================
    elif lesson == "5. The Greeks — Vega (implied volatility)":
    # =======================================================================

        g    = bs_greeks(S, K, T_ex, RISK_FREE_L, iv_ex, "call")
        vega = g["vega"]

        st.markdown(f"""
**Vega** tells you how much the option price changes for every 1% change in implied volatility (IV).

INTC ATM call vega = **{vega:.4f}** per share = **${vega*100:.2f} per contract** per 1% IV move.

If IV rises from {iv_ex*100:.0f}% to {iv_ex*100+5:.0f}% (up 5%), option gains **${vega*5*100:.2f}** per contract — even if the stock doesn't move.
If IV drops from {iv_ex*100:.0f}% to {iv_ex*100-10:.0f}% (down 10%), option loses **${vega*10*100:.2f}** per contract.
""")

        # IV sensitivity chart
        iv_rows = []
        for iv_pct in range(10, 120, 5):
            px = bs_price(S, K, T_ex, RISK_FREE_L, iv_pct/100, "call")
            iv_rows.append({"IV %": iv_pct, "Option value ($)": round(px, 4)})
        iv_df = pd.DataFrame(iv_rows).set_index("IV %")
        st.line_chart(iv_df)
        st.caption(f"INTC ATM call (strike ${K:.2f}, {ex['dte']}d to expiry) — value at different IV levels, stock held at ${S:.2f}.")

        st.markdown(f"""
**Current INTC IV: {iv_ex*100:.0f}%**

High IV = expensive options. Low IV = cheap options.

**The rule:** Buy options when IV is low. Sell options when IV is high.

The Options tab shows you 30d Realised Vol vs ATM IV for any ticker. 
If IV is significantly above realised vol, options are expensive — consider a spread instead of an outright buy.
""")

    # =======================================================================
    elif lesson == "6. IV crush — the most common way to lose money":
    # =======================================================================

        st.markdown("""
**IV crush** happens when implied volatility collapses after a known event — usually earnings.

Before earnings, IV inflates because nobody knows what will happen. Option prices rise.
After earnings, the uncertainty resolves. IV collapses — sometimes by 30–50% in one day.

**The result:** You buy a call before earnings. The stock goes UP 5%. But your call loses value because IV dropped 40%.

This is the most common way beginners lose money on options.
""")

        # Show the effect numerically
        pre_iv  = min(iv_ex * 2.0, 1.5)
        post_iv = iv_ex * 0.6
        st.markdown(f"**INTC example (hypothetical earnings scenario):**")

        crush_rows = []
        for label, s_move, iv_used in [
            ("Stock flat, pre-earnings IV",      S,       pre_iv),
            ("Stock +5%, IV crushes post-earn",  S*1.05,  post_iv),
            ("Stock +10%, IV crushes post-earn", S*1.10,  post_iv),
            ("Stock +15%, IV crushes post-earn", S*1.15,  post_iv),
        ]:
            px = bs_price(s_move, K, T_ex, RISK_FREE_L, iv_used, "call")
            ret = (px / mid_ex - 1) * 100
            crush_rows.append({
                "Scenario":    label,
                "Stock price": f"${s_move:.2f}",
                "IV":          f"{iv_used*100:.0f}%",
                "Option value":f"${px:.3f}",
                "Return vs entry": f"{ret:+.1f}%",
            })
        st.dataframe(pd.DataFrame(crush_rows), use_container_width=True, hide_index=True)

        st.markdown(f"Entry price: ${mid_ex:.3f} at IV {iv_ex*100:.0f}%.")

        st.warning("Stock +5% but option loses money. This is IV crush in action.")

        st.markdown("""
**How to avoid it:**
1. Check the IV vs Realised Vol on the Options tab before buying
2. Avoid buying options in the week before earnings unless you have strong conviction and understand the IV risk
3. Use spreads instead of outright buys when IV is elevated — the spread reduces your vega exposure
4. The intraday scanner (scan_intraday.py) fires on RVOL spikes — if you see a spike *after* an earnings gap, IV is already compressing. Enter cautiously.
""")

    # =======================================================================
    elif lesson == "7. Strategies and when to use them":
    # =======================================================================

        st.markdown("Select a strategy to see its payoff and when to use it.")

        strat_pick = st.selectbox("Strategy", [
            "Long Call", "Bull Call Spread", "Long Put",
            "Cash-Secured Put", "Covered Call",
        ], key="learn_strat")

        guides = {
            "Long Call": {
                "when":     "Strong bullish conviction. Score 3–4 on screener. IV is low or fair (< 30d RV).",
                "risk":     "Limited — premium paid only.",
                "reward":   "Unlimited.",
                "avoid":    "Before earnings (IV crush). When IV >> realised vol. Far OTM strikes.",
                "legs":     [{"type":"call","strike":K,"premium":mid_ex,"qty":1,"position":"long"}],
            },
            "Bull Call Spread": {
                "when":     "Bullish but IV is high. Buying the spread reduces your vega risk vs a naked call.",
                "risk":     "Net debit paid.",
                "reward":   "Capped at the spread width minus net debit.",
                "avoid":    "When you have very high conviction — the spread caps your upside.",
                "legs":     [
                    {"type":"call","strike":K,       "premium":mid_ex,    "qty":1,"position":"long"},
                    {"type":"call","strike":K*1.05,  "premium":mid_ex*0.4,"qty":1,"position":"short"},
                ],
            },
            "Long Put": {
                "when":     "Bearish conviction. Expecting a drop. Or hedging existing long positions.",
                "risk":     "Premium paid.",
                "reward":   "Capped at strike price (stock can't go below zero).",
                "avoid":    "After a stock has already dropped significantly — put premium will be high.",
                "legs":     [{"type":"put","strike":K,"premium":mid_ex,"qty":1,"position":"long"}],
            },
            "Cash-Secured Put": {
                "when":     "Happy to buy the stock at the strike price. IV is high (collect rich premium). Good entry strategy.",
                "risk":     "Assigned stock at strike minus premium collected. Same as buying stock at a discount.",
                "reward":   "Premium collected if stock stays above strike.",
                "avoid":    "On stocks you do NOT want to own if assigned.",
                "legs":     [{"type":"put","strike":K,"premium":mid_ex,"qty":1,"position":"short"}],
            },
            "Covered Call": {
                "when":     "Already long the stock (like your META or INTC positions). Want income. Expect sideways to slight upside.",
                "risk":     "Caps your upside if stock rallies above strike. Still exposed to downside on shares.",
                "reward":   "Premium collected. Reduces your cost basis.",
                "avoid":    "If you think the stock is about to make a big move up — you'll miss it.",
                "legs":     [{"type":"call","strike":K*1.05,"premium":mid_ex*0.4,"qty":1,"position":"short"}],
            },
        }

        g = guides[strat_pick]
        st.markdown(f"**When to use:** {g['when']}")
        st.markdown(f"**Max risk:** {g['risk']}  |  **Max reward:** {g['reward']}")
        st.markdown(f"**Avoid when:** {g['avoid']}")

        pnl = payoff_df(S, g["legs"])
        st.line_chart(pnl.set_index("Stock price"))
        st.caption(f"Payoff at expiry. Spot = ${S:.2f}, strike = ${K:.2f}. Horizontal axis = stock price range.")

        net = sum((l["premium"] if l["position"]=="long" else -l["premium"]) for l in g["legs"])
        st.markdown(f"Net cost/credit: **${net:+.3f}** per share = **${net*100:+.2f}** per contract.")

    # =======================================================================
    elif lesson == "8. Position sizing and risk management":
    # =======================================================================

        st.markdown("""
**The rule that determines whether you survive long enough to get good:**

Never risk more than 2–5% of your total portfolio on a single options trade.

Options can go to zero. That is not a tail risk — it is a normal outcome on losing trades.
If you size correctly, a string of losses doesn't wipe you out.
""")

        nzdusd_l = fetch_nzdusd()
        port_nzd = st.number_input("Your total trading portfolio (NZD)", 1000.0, 500000.0, 5000.0, 500.0)
        risk_pct = st.slider("Max risk per trade (%)", 1, 10, 3)

        max_risk_nzd  = port_nzd * risk_pct / 100
        max_contracts = max(1, int(max_risk_nzd / (mid_ex * 100 / nzdusd_l)))

        st.markdown(f"""
**Portfolio:** NZD {port_nzd:,.0f}
**Max risk per trade ({risk_pct}%):** NZD {max_risk_nzd:,.0f}
**INTC ATM call costs:** NZD {mid_ex*100/nzdusd_l:,.0f} per contract (your max loss per contract)
**Max contracts:** {max_contracts}
""")

        st.success(f"At {risk_pct}% risk, you can buy up to **{max_contracts} contract(s)** on INTC without breaking position sizing rules.")

        st.markdown("""
**Exit rules — set these before you enter:**

| Situation | Action |
|---|---|
| Option gains 50–100% | Take profit — the math says taking 50% winners consistently beats holding for 100% |
| Option loses 50% | Cut the loss — the remaining value rarely recovers, and theta keeps eroding it |
| 7 days to expiry | Close or roll — gamma and theta are extreme in the final week |
| The thesis is wrong | Exit immediately — don't hold hoping it reverses |

**The discipline gap:** Most losses in options come from not following exit rules, not from picking the wrong direction.
""")

    # =======================================================================
    elif lesson == "9. The most common mistakes":
    # =======================================================================

        st.markdown("These are the moves that cost most beginners their first account.")

        mistakes = [
            {
                "title": "Buying far OTM weeklies",
                "why":   "They look cheap. $50 for a contract feels like a lottery ticket. They expire worthless 80%+ of the time. "
                         "The probability of a stock making a 15% move in 5 days is very low.",
                "fix":   "Start with ATM options, 30–45 DTE. Delta 0.40–0.60. More expensive but far more likely to have value at expiry.",
            },
            {
                "title": "Buying calls right before earnings",
                "why":   "IV inflates before earnings. You overpay. Even if the stock moves your way, IV crush can erase the gain. "
                         "See the IV crush lesson.",
                "fix":   "Either enter before IV inflates (1–2 weeks before earnings), or use a spread to reduce vega risk.",
            },
            {
                "title": "Ignoring theta on long holds",
                "why":   "Buying a 30 DTE call and holding it for 25 days while the stock goes sideways. "
                         "Theta has eaten most of your premium even though you were 'right' directionally.",
                "fix":   "Options need to move quickly. If the stock isn't moving in 7–10 days, reassess. Don't hold hoping.",
            },
            {
                "title": "No exit plan",
                "why":   "Entering with no defined profit target or stop loss. Watching a 60% winner turn into a 30% loser.",
                "fix":   "Set your exit levels before you enter: take 50–80% profit, cut at 50% loss. Use the position builder in the Options tab.",
            },
            {
                "title": "Oversizing — putting too much into one trade",
                "why":   "One bad trade wipes 30% of the account. Emotionally devastating. Leads to revenge trading.",
                "fix":   "Lesson 8 covers this. Max 2–5% of portfolio per trade.",
            },
            {
                "title": "Confusing 'cheap' with 'good value'",
                "why":   "A $0.20 option is not cheap if it needs a 25% move to profit. Price means nothing without context.",
                "fix":   "Always check the break-even price and the move needed. These are shown in the Chain & Position section.",
            },
        ]

        for m in mistakes:
            with st.expander(f"❌  {m['title']}"):
                st.markdown(f"**Why it happens:** {m['why']}")
                st.markdown(f"**Fix:** {m['fix']}")

        st.divider()
        st.markdown("""
**The honest summary:**

Options are not a shortcut to fast money. They are a tool. Used correctly — right sizing, right strategy for the IV environment, defined exits — they let you express a directional view with capped downside and leveraged upside.

The screener tells you *what* to watch. The intraday scanner tells you *when* activity is building.
Options let you act on that signal with less capital at risk than buying shares outright.

That's the edge. Build it slowly.
""")
