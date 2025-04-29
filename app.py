import streamlit as st
import pandas as pd
import os

# Set page config
st.set_page_config(
    page_title="📈 Trade Strategy Screener Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Dashboard Settings")
    st.markdown("Tune your filters below 👇")

# --- Main Page ---
st.title("📈 Trade Strategy Screener Dashboard")
st.caption("Welcome to your live, auto-updating trade setup dashboard! 🚀")

st.markdown("""
This dashboard scans stocks based on your trading strategy:
- RVOL above 3
- Price gains over 10%
- EMA alignment (9 > 20 > 200)
- RSI < 70
- ATR-based dynamic stop

The stocks shown here are refreshed daily — ready for you to review!
""")

# Load CSV
csv_path = "outputs/screened_stocks_intraday.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

    st.success(f"✅ Loaded {len(df)} screened stocks today!")

    # --- Filters ---
    st.sidebar.header("📊 Filter Stocks")
    min_change = st.sidebar.slider("Minimum % Change", min_value=0, max_value=100, value=10, step=1)
    min_rvol = st.sidebar.slider("Minimum RVOL", min_value=0.0, max_value=20.0, value=3.0, step=0.1)

    filtered_df = df[(df["Change%"] >= min_change) & (df["RVOL"] >= min_rvol)]

    # --- Dataframe ---
    st.subheader("🔎 Screener Results")
    st.dataframe(filtered_df, use_container_width=True)

    # --- Chart ---
    if not filtered_df.empty:
        st.subheader("🚀 Top Movers by % Change")
        top_movers = filtered_df.sort_values(by="Change%", ascending=False)
        st.bar_chart(top_movers.set_index("Ticker")["Change%"])
    else:
        st.warning("⚠️ No stocks meet your filter criteria today.")
else:
    st.warning("⚠️ No results file found. Please check again after the next automation run.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption("Powered by NZDaveyboy 🚀 | Auto-updated daily.")
