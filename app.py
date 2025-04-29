import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Trade Strategy Screener Results", layout="wide")

st.title("📈 Trade Strategy Screener Results")
st.caption("Powered by automation - NZDaveyboy 🚀")

csv_path = "outputs/screened_stocks_intraday.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

    st.success(f"Loaded {len(df)} screened stocks.")

    # Filters
    min_change = st.slider("Minimum % Change", min_value=0, max_value=100, value=10)
    min_rvol = st.slider("Minimum RVOL", min_value=0.0, max_value=20.0, value=3.0, step=0.1)

    filtered_df = df[(df["Change%"] >= min_change) & (df["RVOL"] >= min_rvol)]

    st.dataframe(filtered_df, use_container_width=True)

    if not filtered_df.empty:
        st.subheader("Top Movers by % Change")
        top_movers = filtered_df.sort_values(by="Change%", ascending=False)
        st.bar_chart(top_movers.set_index("Ticker")["Change%"])
else:
    st.warning("No results file found yet. Please check again after next automation run.")

st.sidebar.info("📩 Results updated daily via GitHub Actions.")
