import os
import sqlite3

import pandas as pd
import streamlit as st

DB_PATH = os.path.join(os.path.dirname(__file__), "screener.db")

st.set_page_config(page_title="TradeStrategy Screener", layout="wide")
st.title("TradeStrategy Screener")

if not os.path.exists(DB_PATH):
    st.error("No data yet. Run `python run.py` first.")
    st.stop()

conn = sqlite3.connect(DB_PATH)
dates = pd.read_sql(
    "SELECT DISTINCT run_date FROM results ORDER BY run_date DESC", conn
)["run_date"].tolist()
conn.close()

if not dates:
    st.warning("No results in database yet.")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.header("Filters")
    selected_date = st.selectbox("Date", dates)

    conn = sqlite3.connect(DB_PATH)
    strategies = pd.read_sql(
        "SELECT DISTINCT strategy FROM results WHERE run_date = ?",
        conn, params=(selected_date,)
    )["strategy"].tolist()
    conn.close()

    strategy_options = ["All"] + sorted(strategies)
    strategy = st.selectbox("Strategy", strategy_options)
    asset_filter = st.selectbox("Asset type", ["All", "equity", "crypto"])
    min_score = st.slider("Min score (0–4)", 0, 4, 0)
    min_change = st.slider("Min change %", 0, 100, 0)
    min_rvol = st.slider("Min RVOL", 0.0, 20.0, 0.0, 0.5)

# --- Load data ---
conn = sqlite3.connect(DB_PATH)
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

# --- Display ---
st.caption(f"{len(df)} candidates  •  {selected_date}")

if df.empty:
    st.info("No stocks match the current filters.")
else:
    st.dataframe(
        df[[
            "ticker", "score", "strategy", "asset",
            "price", "change_pct", "rvol", "rsi",
            "ema9", "ema20", "ema200",
            "macd", "macd_signal", "vwap",
            "stop_loss", "volume_trend_up",
        ]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "score":           st.column_config.NumberColumn("Score", format="%d/4"),
            "change_pct":      st.column_config.NumberColumn("Change %", format="%.2f%%"),
            "rvol":            st.column_config.NumberColumn("RVOL", format="%.2fx"),
            "volume_trend_up": st.column_config.CheckboxColumn("Vol↑"),
        },
    )

    st.subheader("Top movers")
    st.bar_chart(df.set_index("ticker")["change_pct"].head(20))

# --- Historical summary ---
with st.expander("Run history"):
    conn = sqlite3.connect(DB_PATH)
    history = pd.read_sql(
        """
        SELECT run_date,
               COUNT(*)                        AS candidates,
               ROUND(AVG(score), 1)            AS avg_score,
               ROUND(MAX(change_pct), 1)       AS best_change_pct
        FROM results
        GROUP BY run_date
        ORDER BY run_date DESC
        """,
        conn,
    )
    conn.close()
    st.dataframe(history, use_container_width=True, hide_index=True)
