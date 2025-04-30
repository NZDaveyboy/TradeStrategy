import streamlit as st
import pandas as pd
import os

# --- CONFIG ---
st.set_page_config(
    page_title="📈 Trading Strategy Screener",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR ---
st.sidebar.title("🎛 Strategy Selector")

strategy = st.sidebar.selectbox(
    "Choose Strategy:",
    ("General Screener", "AI Supply Chain", "Top 10 Tech")
)

ticker_files = {
    "General Screener": "tickers.txt",
    "AI Supply Chain": "tickers_ai.txt",
    "Top 10 Tech": "tickers_tech.txt"
}

tickers_file = ticker_files.get(strategy, "tickers.txt")

st.sidebar.header("🔍 Screener Filters")
min_change = st.sidebar.slider("Minimum % Change", 0, 100, 0, 1)
min_rvol = st.sidebar.slider("Minimum RVOL", 0.0, 20.0, 0.0, 0.1)

# --- TECHNICAL FILTERS ---
st.sidebar.header("📊 Technical Filters")
filter_macd_cross = st.sidebar.checkbox("MACD > Signal Line", value=False)
filter_ema_stack = st.sidebar.checkbox("EMA Stacked (9 > 20 > 200)", value=False)
filter_vwap = st.sidebar.checkbox("Close > VWAP", value=False)
filter_volume = st.sidebar.checkbox("Volume Trend Up", value=False)

# --- MAIN AREA ---
st.title(f"📊 {strategy} Strategy Screener")
st.caption("Auto-updated with enrichment • Powered by NZDaveyboy 🚀")

csv_path = "screened_stocks_enriched.csv"

try:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        if os.path.exists(tickers_file):
            with open(tickers_file, "r") as file:
                selected_tickers = [line.strip() for line in file]
            df = df[df["Ticker"].isin(selected_tickers)]

        # Basic filters
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

        # Show results
        st.subheader("🔎 Screener Results (Full View)")
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
        st.error(f"❌ Enriched data file not found: {csv_path}")

except Exception as e:
    st.error(f"❌ Unexpected error: {type(e).__name__} — {e}")
