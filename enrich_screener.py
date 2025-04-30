import yfinance as yf
import pandas as pd
import numpy as np
import time

# File paths
input_file = "screened_stocks_intraday.csv"
output_file = "screened_stocks_enriched.csv"

# Load original screener data
try:
    df_input = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"❌ Input file not found: {input_file}")
    exit(1)

# Load tickers from all strategy files
ticker_files = ["tickers.txt", "tickers_ai.txt", "tickers_tech.txt"]
all_tickers = set()

for file in ticker_files:
    try:
        with open(file, "r") as f:
            tickers = [line.strip() for line in f.readlines() if line.strip()]
            all_tickers.update(tickers)
    except FileNotFoundError:
        print(f"⚠️ File not found: {file} — skipping.")

all_tickers = sorted(all_tickers)
print(f"🧠 Unique tickers to enrich: {all_tickers}")

# Enrich each ticker
enriched_rows = []

for ticker in all_tickers:
    try:
        data = yf.download(ticker, period="15d", interval="1d", progress=False)

        if data is None or data.shape[0] < 15:
            print(f"⚠️ Not enough data for {ticker} — skipping.")
            continue

        close = data["Close"]
        volume = data["Volume"]

        ema_9 = close.ewm(span=9).mean().iloc[-1]
        ema_20 = close.ewm(span=20).mean().iloc[-1]
        ema_200 = close.ewm(span=200).mean().iloc[-1] if len(close) >= 200 else np.nan

        vwap = (data["Close"] * data["Volume"]).sum() / data["Volume"].sum()

        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()

        vol_trend = 1 if volume.iloc[-1] > volume.tail(15).mean() else 0

        enriched_rows.append({
            "Ticker": ticker,
            "Last_Close": float(close.iloc[-1]),
            "EMA_9": ema_9,
            "EMA_20": ema_20,
            "EMA_200": ema_200,
            "VWAP": vwap,
            "MACD": macd.iloc[-1],
            "MACD_Signal": signal.iloc[-1],
            "Volume_Trend_Up": vol_trend
        })

        print(f"✅ Enriched {ticker}")

        time.sleep(3)  # Avoid Yahoo rate limits

    except Exception as e:
        print(f"❌ Error processing {ticker}: {type(e).__name__} — {e}")

# Finalize enrichment
df_enriched = pd.DataFrame(enriched_rows)

if df_enriched.empty:
    print("❌ No enriched data generated. Possibly due to rate limits or API errors.")
    exit(1)

# Merge with original data (RVOL, Change%)
df_final = pd.merge(df_input, df_enriched, on="Ticker", how="inner")
df_final.to_csv(output_file, index=False)

print(f"✅ Screener enrichment complete. Saved to {output_file}")
