import streamlit as st
import pandas as pd
import os

# --- CONFIG ---
st.set_page_config(
    page_title="🧠 Unified Trading Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR ---
st.sidebar.title("🎛 Select Your Strategy")

strategy = st.sidebar.selectbox(
    "Choose Strategy:",
    ("General Screener", "AI Supply Chain", "Top 10 Tech")
)

ticker_files = {
    "General Screener": "tickers.txt",
    "AI Supply Chain": "tickers_ai.txt",
    "Top 10 Tech": "tickers_tech.txt"  # You can add this file
}

tickers_file = ticker_files.get(strategy, "tickers.txt")

st.sidebar.header("🔍 Filters")
min_change = st.sidebar.slider("Minimum % Change", 0, 100, 10, 1)
min_rvol = st.sidebar.slider("Minimum RVOL", 0.0, 20.0, 3.0, 0.1)

# --- MAIN ---
st.title(f"📈 {strategy} Dashboard")
st.caption("Auto-updated daily • Built by NZDaveyboy 🚀")

# Load stock screener output
csv_path = "outputs/screened_stocks_intraday.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

    # Filter by selected tickers
    if os.path.exists(tickers_file):
        with open(tickers_file, "r") as file:
            selected_tickers = [line.strip() for line in file]
        df = df[df["Ticker"].isin(selected_tickers)]

    # Apply user filters
    filtered_df = df[(df["Change%"] >= min_change) & (df["RVOL"] >= min_rvol)]

    st.success(f"✅ {len(filtered_df)} stocks match your filters.")

    st.subheader("🔎 Screener Results")
    st.dataframe(filtered_df, use_container_width=True)

    if not filtered_df.empty:
        st.subheader("🚀 Top Movers")
        st.bar_chart(filtered_df.set_index("Ticker")["Change%"])
    else:
        st.warning("⚠️ No matching stocks for your strategy and filters today.")
else:
    st.warning("⚠️ No results file found yet. Please check again after the next automation run.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Created with ❤️ by NZDaveyboy • Live from GitHub")
