import os
import sqlite3
from datetime import date

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
        min_score = st.slider("Min score (0–4)", 0, 4, 0)
        default_change = 5 if strategy == "momentum" else 0
        default_rvol   = 2.0 if strategy == "momentum" else 0.0
        min_change = st.slider("Min change %", 0, 100, default_change)
        min_rvol   = st.slider("Min RVOL", 0.0, 20.0, default_rvol, 0.5)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_screener, tab_tracker, tab_alerts = st.tabs(["Screener", "Trade Tracker", "Alerts"])


# ===========================================================================
# TAB 1 — Screener
# ===========================================================================

with tab_screener:
    if not dates or selected_date is None:
        st.info("No screener data yet. Run `python run.py` first.")
    else:
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
            pl_pct       = (current_usd / nzdusd - avg_buy_nzd) / avg_buy_nzd * 100
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

    ALERT_COLORS = {
        "rvol":     "#ff6b35",
        "change":   "#4ecdc4",
        "gap_up":   "#2ecc71",
        "gap_down": "#e74c3c",
    }
    ALERT_LABELS = {
        "rvol":     "RVOL",
        "change":   "Change",
        "gap_up":   "Gap Up",
        "gap_down": "Gap Down",
    }

    conn = get_conn()
    alerts_exist = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='alerts'"
    ).fetchone()[0]

    if not alerts_exist:
        conn.close()
        st.info("No alerts table yet. The scanner will create it on first run.")
    else:
        # Summary counts
        today_str = pd.Timestamp.now().strftime("%Y-%m-%d")

        total_alerts = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        today_alerts = conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE scan_date = ?", (today_str,)
        ).fetchone()[0]
        distinct_tickers_today = conn.execute(
            "SELECT COUNT(DISTINCT ticker) FROM alerts WHERE scan_date = ?", (today_str,)
        ).fetchone()[0]
        last_scan = conn.execute(
            "SELECT triggered_at FROM alerts ORDER BY id DESC LIMIT 1"
        ).fetchone()
        last_scan_str = last_scan[0] if last_scan else "Never"

        conn.close()

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Alerts today",         today_alerts)
        m2.metric("Tickers flagged today", distinct_tickers_today)
        m3.metric("Total alerts (all time)", total_alerts)
        m4.metric("Last scan", last_scan_str)

        st.divider()

        # Filters
        conn = get_conn()
        all_dates = pd.read_sql(
            "SELECT DISTINCT scan_date FROM alerts ORDER BY scan_date DESC", conn
        )["scan_date"].tolist()
        all_tickers = pd.read_sql(
            "SELECT DISTINCT ticker FROM alerts ORDER BY ticker", conn
        )["ticker"].tolist()
        conn.close()

        f1, f2, f3 = st.columns(3)
        date_filter   = f1.selectbox("Date", ["All"] + all_dates)
        ticker_filter = f2.selectbox("Ticker", ["All"] + all_tickers)
        type_filter   = f3.selectbox(
            "Alert type", ["All", "rvol", "change", "gap_up", "gap_down"]
        )

        # Load filtered alerts
        query  = "SELECT * FROM alerts WHERE 1=1"
        params: list = []
        if date_filter != "All":
            query += " AND scan_date = ?"
            params.append(date_filter)
        if ticker_filter != "All":
            query += " AND ticker = ?"
            params.append(ticker_filter)
        if type_filter != "All":
            query += " AND alert_type = ?"
            params.append(type_filter)
        query += " ORDER BY id DESC"

        conn = get_conn()
        alerts_df = pd.read_sql(query, conn, params=params)
        conn.close()

        st.caption(f"{len(alerts_df)} alert(s) matching filters")

        if alerts_df.empty:
            st.info("No alerts match the current filters.")
        else:
            # Friendly label column
            alerts_df["type_label"] = alerts_df["alert_type"].map(ALERT_LABELS).fillna(alerts_df["alert_type"])

            display_df = alerts_df[[
                "scan_date", "scan_window", "ticker",
                "type_label", "value", "price",
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

            # Alert type breakdown chart
            if len(alerts_df) > 1:
                st.subheader("Alert breakdown")
                breakdown = (
                    alerts_df.groupby("type_label")
                    .size()
                    .reset_index(name="count")
                    .set_index("type_label")
                )
                st.bar_chart(breakdown["count"])

        with st.expander("All-time ticker frequency"):
            conn = get_conn()
            freq_df = pd.read_sql(
                """
                SELECT ticker,
                       COUNT(*)                          AS total_alerts,
                       COUNT(DISTINCT scan_date)         AS days_flagged,
                       MAX(scan_date)                    AS last_flagged
                FROM alerts
                GROUP BY ticker
                ORDER BY total_alerts DESC
                """,
                conn,
            )
            conn.close()
            st.dataframe(freq_df, use_container_width=True, hide_index=True)
