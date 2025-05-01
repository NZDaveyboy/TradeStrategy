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
df["Change%"] = pd.to_numeric(df["Change%"], errors="coerce")
df["RVOL"] = pd.to_numeric(df["RVOL"], errors="coerce")

# --- Sidebar Filters ---
st.sidebar.header("🔎 Screener Filters")
min_change = st.sidebar.slider("Minimum % Change", 0, 100, 0, 1)
min_rvol = st.sidebar.slider("Minimum RVOL", 0.0, 20.0, 0.0, 0.1)

st.sidebar.header("📊 Technical Filters")
filter_macd_cross = st.sidebar.checkbox("MACD > Signal Line", value=False)
filter_ema_stack = st.sidebar.checkbox("EMA Stacked (9 > 20 > 200)", value=False)
filter_vwap = st.sidebar.checkbox("Close > VWAP", value=False)
filter_volume = st.sidebar.checkbox("Volume Trend Up", value=False)

# --- Apply filters ---
if not df.empty:
    df = df[df["Change%"] >= min_change]
    df = df[df["RVOL"] >= min_rvol]

    if filter_macd_cross:
        df = df[df["MACD"] > df["MACD_Signal"]]
    if filter_ema_stack:
        df = df[(df["EMA_9"] > df["EMA_20"]) & (df["EMA_20"] > df["EMA_200"])]
    if filter_vwap:
        df = df[df["Last_Close"] > df["VWAP"]]
    if filter_volume:
        df = df[df["Volume_Trend_Up"] == 1]

    df = df.sort_values(by="Change%", ascending=False)

# --- Main Display ---
st.title(f"{strategy} Strategy Screener")

st.caption("Auto-updated with enrichment • Powered by NZDaveyboy 🚀")

if not df.empty:
    st.subheader("🔍 Screener Results")
    st.dataframe(df, use_container_width=True)
    st.subheader("🚀 Top Movers")
    st.bar_chart(df.set_index("Ticker")["Change%"])
else:
    st.warning("⚠️ No stocks match current filter settings.")


# === enrich_screener.py ===

import yfinance as yf
import pandas as pd
import numpy as np
import os

# Load tickers from all sources
files = ["tickers.txt", "tickers_ai.txt", "tickers_tech.txt", "tickers_crypto.txt"]
tickers = []
for file in files:
    if os.path.exists(file):
        with open(file, "r") as f:
            tickers += [line.strip() for line in f if line.strip()]

tickers = sorted(set(tickers))

rows = []
for ticker in tickers:
    try:
        data = yf.download(ticker, period="20d", interval="1d", progress=False)
        if len(data) < 15:
            continue

        # Indicators
        data["EMA_9"] = data["Close"].ewm(span=9).mean()
        data["EMA_20"] = data["Close"].ewm(span=20).mean()
        data["EMA_200"] = data["Close"].ewm(span=200).mean()
        data["VWAP"] = (data["High"] + data["Low"] + data["Close"]) / 3
        exp1 = data["Close"].ewm(span=12, adjust=False).mean()
        exp2 = data["Close"].ewm(span=26, adjust=False).mean()
        data["MACD"] = exp1 - exp2
        data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

        last = data.iloc[-1]
        change = (last["Close"] / data.iloc[-2]["Close"] - 1) * 100
        rvol = last["Volume"] / data["Volume"].tail(15).mean() if "Volume" in data.columns else 1

        is_crypto = ticker.endswith("-USD")
        volume_up = int(not is_crypto and data["Volume"].rolling(3).mean().iloc[-1] > data["Volume"].rolling(3).mean().iloc[-4])

        rows.append({
            "Ticker": ticker,
            "Change%": round(change, 2),
            "RVOL": round(rvol, 2),
            "Last_Close": round(last["Close"], 2),
            "EMA_9": round(last["EMA_9"], 2),
            "EMA_20": round(last["EMA_20"], 2),
            "EMA_200": round(last["EMA_200"], 2),
            "VWAP": round(last["VWAP"], 2),
            "MACD": round(last["MACD"], 4),
            "MACD_Signal": round(last["MACD_Signal"], 4),
            "Volume_Trend_Up": volume_up,
            "Asset": "crypto" if is_crypto else "equity"
        })

    except Exception as e:
        print(f"Error processing {ticker}: {e}")

if rows:
    pd.DataFrame(rows).to_csv("screened_stocks_enriched.csv", index=False)
    print("✅ Screener enrichment complete.")
else:
    print("❌ No valid tickers enriched.")
