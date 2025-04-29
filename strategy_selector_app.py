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

st.sidebar.header("🔍 Filter Settings")
min_change = st.sidebar.slider("Minimum % Change", 0, 100, 0, 1)
min_rvol = st.sidebar.slider("Minimum RVOL", 0.0, 20.0, 0.0, 0.1)

# --- MAIN AREA ---
st.title(f"📈 {strategy}")
st.caption("Auto-updated daily • Powered by NZDaveyboy 🚀")

csv_path = "screened_stocks_enriched.csv"

# --- Try loading data
try:
    st.info(f"📂 Trying to load {csv_path}...")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.success(f"✅ Loaded CSV successfully!")

        st.write("🔍 CSV Columns Detected:", df.columns.tolist())
        st.dataframe(df)

        if os.path.exists(tickers_file):
            with open(tickers_file, "r") as file:
                selected_tickers = [line.strip() for line in file]

            df = df[df["Ticker"].isin(selected_tickers)]
            st.info(f"✅ Filtered {len(df)} rows matching selected tickers.")

        filtered_df = df[(df["Change%"] >= min_change) & (df["RVOL"] >= min_rvol)]

        if not filtered_df.empty:
            st.subheader("🔎 Screener Results")
            st.dataframe(filtered_df, use_container_width=True)
            st.subheader("🚀 Top Movers")
            st.bar_chart(filtered_df.set_index("Ticker")["Change%"])
        else:
            st.warning("⚠️ No stocks meet your current filter settings.")
    else:
        st.error(f"❌ File {csv_path} not found.")

except Exception as e:
    st.error(f"❌ Unexpected error: {e}")

# --- Footer
st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ by NZDaveyboy")
