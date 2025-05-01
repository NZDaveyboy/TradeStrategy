import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="📈 Strategy Selector", layout="wide")

# --- Strategy dropdown ---
st.sidebar.title("Strategy Selector")
strategy = st.sidebar.selectbox(
    "Choose Strategy:",
    ("General Screener", "AI Supply Chain", "Top 10 Tech", "Crypto Screener")
)

ticker_files = {
    "General Screener": "tickers.txt",
    "AI Supply Chain": "tickers_ai.txt",
    "Top 10 Tech": "tickers_tech.txt",
    "Crypto Screener": "tickers_crypto.txt"
}

selected_file = ticker_files.get(strategy)

# Load enriched data
if os.path.exists("screened_stocks_enriched.csv"):
    df = pd.read_csv("screened_stocks_enriched.csv")
else:
    df = pd.DataFrame()

# Filter by selected tickers
if selected_file and os.path.exists(selected_file):
    with open(selected_file, "r") as f:
        selected_tickers = [line.strip() for line in f]
    df = df[df["Ticker"].isin(selected_tickers)]

# Convert to numeric to prevent type issues
numeric_cols = [
    "Change%", "RVOL", "Last_Close", "EMA_9", "EMA_20",
    "EMA_200", "VWAP", "MACD", "MACD_Signal"
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# --- Sidebar Filters ---
st.sidebar.header("🔎 Screener Filters")
min_change = st.sidebar.slider("Minimum % Change", 0, 100, 5, 1)
min_rvol = st.sidebar.slider("Minimum RVOL", 0.0, 20.0, 3.0, 0.1)

# --- Main Display ---
st.title(f"{strategy} Strategy Screener")
st.caption("Auto-updated with enrichment • Powered by NZDaveyboy 🚀")

if not df.empty:
    df = df[(df["Change%"] >= min_change) & (df["RVOL"] >= min_rvol)]

    def score_row(row):
        score = 0
        if row["MACD"] > row["MACD_Signal"]:
            score += 1
        if row["EMA_9"] > row["EMA_20"] > row["EMA_200"]:
            score += 1
        if row["Last_Close"] > row["VWAP"]:
            score += 1
        if row.get("Volume_Trend_Up", 0) == 1:
            score += 1
        return score

    df["Score"] = df.apply(score_row, axis=1)
    df = df.sort_values(by="Score", ascending=False)

    st.subheader("🔍 Screener Results (Ranked by Score)")
    st.dataframe(
        df[[
            "Ticker", "Score", "Change%", "RVOL", "Last_Close",
            "EMA_9", "EMA_20", "EMA_200", "VWAP",
            "MACD", "MACD_Signal", "Volume_Trend_Up", "Asset"
        ]],
        use_container_width=True
    )

    st.subheader("🚀 Top Movers")
    st.bar_chart(df.set_index("Ticker")["Change%"])
else:
    st.warning("⚠️ No stocks match current filter settings.")
