import streamlit as st
import pandas as pd
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="📈 Trading Strategy Screener",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR ---

# App header
st.sidebar.markdown("## 🎛 Strategy Selector")

# Strategy selector
strategy = st.sidebar.selectbox(
    "Choose Strategy:",
    ("General Screener", "AI Supply Chain", "Top 10 Tech")
)

# File map
ticker_files = {
    "General Screener": "tickers.txt",
    "AI Supply Chain": "tickers_ai.txt",
    "Top 10 Tech": "tickers_tech.txt"
}
tickers_file = ticker_files.get(strategy, "tickers.txt")

# --- Screener Filters ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Screener Filters")
min_change = st.sidebar.slider("Minimum % Change", 0, 100, 0, 1)
min_rvol = st.sidebar.slider("Minimum RVOL", 0.0, 20.0, 0.0, 0.1)

# --- Technical Filters ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Technical Filters")
filter_macd_cross = st.sidebar.checkbox("MACD > Signal Line")
filter_ema_stack = st.sidebar.checkbox("EMA Stacked (9 > 20 > 200)")
filter_vwap = st.sidebar.checkbox("Close > VWAP")
filter_volume = st.sidebar.checkbox("Volume Trend Up")

# --- MAIN AREA ---
st.title(f"📊 {strategy} Strategy Screener")
st.caption("Auto-updated with enrichment • Powered by NZDaveyboy 🚀")

csv_path = "screened_stocks_enriched.csv"

try:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Convert key columns to numeric to avoid type comparison issues
numeric_cols = [
    "Change%", "RVOL", "Last_Close", "EMA_9", "EMA_20",
    "EMA_200", "VWAP", "MACD", "MACD_Signal"
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # Apply strategy-specific ticker filter
        if os.path.exists(tickers_file):
            with open(tickers_file, "r") as file:
                selected_tickers = [line.strip() for line in file]
            df = df[df["Ticker"].isin(selected_tickers)]

        # Screener filter logic
        filtered_df = df[(df["Change%"] >= min_change) & (df["RVOL"] >= min_rvol)]

        # Apply technical filters
        if filter_macd_cross:
            filtered_df = filtered_df[filtered_df["MACD"] > filtered_df["MACD_Signal"]]
        if filter_ema_stack:
            filtered_df = filtered_df[
                (filtered_df["EMA_9"] > filtered_df["EMA_20"]) &
                (filtered_df["EMA_20"] > filtered_df["EMA_200"])
            ]
        if filter_vwap:
            filtered_df = filtered_df[filtered_df["Last_Close"] > filtered_df["VWAP"]]
        if filter_volume:
            filtered_df = filtered_df[filtered_df["Volume_Trend_Up"] == 1]

        # Display results
        st.subheader("🔎 Screener Results")
        if not filtered_df.empty:
            st.dataframe(
                filtered_df[
                    [
                        "Ticker", "Change%", "RVOL", "Last_Close",
                        "EMA_9", "EMA_20", "EMA_200", "VWAP",
                        "MACD", "MACD_Signal", "Volume_Trend_Up"
                    ]
                ],
                use_container_width=True
            )

            st.subheader("🚀 Top Movers")
            st.bar_chart(filtered_df.set_index("Ticker")["Change%"])
        else:
            st.warning("⚠️ No stocks match the current filter settings.")
    else:
        st.error(f"❌ Enriched data file not found: `{csv_path}`")

except Exception as e:
    st.error(f"❌ Unexpected error: {type(e).__name__} — {e}")
