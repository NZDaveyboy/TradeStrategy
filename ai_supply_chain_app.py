import streamlit as st
import pandas as pd
import os

# Config
st.set_page_config(
    page_title="🤖 AI Supply Chain Screener",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 AI Supply Chain Screener Dashboard")
st.caption("Daily trade signals across the core AI infrastructure and chip ecosystem.")

st.markdown("""
This dashboard scans a fixed list of AI-related stocks involved in:
- GPUs & Accelerators (NVDA, AMD)
- Foundries (TSM, ASML)
- AI Infrastructure (AVGO, SMCI, INTC, MU)
- Software/Model Platforms (PLTR, AI)

Filtering Criteria:
- RVOL ≥ 3  
- Price gain ≥ 10%  
- EMA alignment (9 > 20 > 200)  
- RSI < 70
""")

# Load file
csv_path = "outputs/screened_stocks_intraday.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

    # Filter to AI tickers only
    ai_tickers = ['NVDA', 'AMD', 'TSM', 'SMCI', 'ASML', 'AVGO', 'PLTR', 'INTC', 'MU', 'AI']
    df = df[df["Ticker"].isin(ai_tickers)]

    st.sidebar.header("🔍 Filters")
    min_change = st.sidebar.slider("Minimum % Change", 0, 100, 10, 1)
    min_rvol = st.sidebar.slider("Minimum RVOL", 0.0, 20.0, 3.0, 0.1)

    filtered = df[(df["Change%"] >= min_change) & (df["RVOL"] >= min_rvol)]

    st.success(f"✅ {len(filtered)} matching AI stocks found.")

    st.dataframe(filtered, use_container_width=True)

    if not filtered.empty:
        st.subheader("🚀 Top AI Movers Today")
        st.bar_chart(filtered.set_index("Ticker")["Change%"])
    else:
        st.warning("No AI tickers meet the current filter criteria.")
else:
    st.warning("⚠️ No CSV results file found. Please check after the next automation run.")

st.sidebar.caption("Auto-updated daily • Built by NZDaveyboy")
