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

st.sidebar.markdown("## 🎛 Strategy Selector")

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

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Screener Filters")
min_change = st.sidebar.slider("Minimum % Change", 0, 100, 5, 1)
min_rvol = st.sidebar.slider("Minimum RVOL", 0.0, 20.0, 3.0, 0.1)

# --- MAIN AREA ---

st.title(f"📊 {strategy} Strategy Screener")
st.caption("Auto-updated with scoring • Powered by NZDaveyboy 🚀")

csv_path = "screened_stocks_enriched.csv"

try:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # Ensure numeric types
        numeric_cols = [
            "Change%", "RVOL", "Last_Close", "EMA_9", "EMA_20",
            "EMA_200", "VWAP", "MACD", "MACD_Signal"
        ]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        # Filter by strategy-specific tickers
        if os.path.exists(tickers_file):
            with open(tickers_file, "r") as file:
                selected_tickers = [line.strip() for line in file]
            df = df[df["Ticker"].isin(selected_tickers)]

        # Initial screening based on % change and RVOL
        filtered_df = df[(df["Change%"] >= min_change) & (df["RVOL"] >= min_rvol)]

        # Score each row from 0–4
        def score_row(row):
            score = 0
            if row["MACD"] > row["MACD_Signal"]:
                score += 1
            if row["EMA_9"] > row["EMA_20"] > row["EMA_200"]:
                score += 1
            if row["Last_Close"] > row["VWAP"]:
                score += 1
            if row["Volume_Trend_Up"] == 1:
                score += 1
            return score

        filtered_df["Score"] = filtered_df.apply(score_row, axis=1)
        filtered_df = filtered_df.sort_values(by="Score", ascending=False)

        # --- DISPLAY ---
        st.subheader("🔎 Screener Results (Ranked by Score)")

        if not filtered_df.empty:
            st.dataframe(
                filtered_df[
                    [
                        "Ticker", "Score", "Change%", "RVOL", "Last_Close",
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
