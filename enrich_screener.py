import yfinance as yf
import pandas as pd
import numpy as np

# Load the original screener CSV
input_file = "screened_stocks_intraday.csv"
output_file = "screened_stocks_enriched.csv"

df_input = pd.read_csv(input_file)

# Combine all tickers from strategy files
ticker_files = ["tickers.txt", "tickers_ai.txt", "tickers_tech.txt"]
all_tickers = set()

for file in ticker_files:
    try:
        with open(file, "r") as f:
            tickers = [line.strip() for line in f.readlines() if line.strip()]
            all_tickers.update(tickers)
    except FileNotFoundError:
        print(f"⚠️ File not found: {file} — skipping.")

print(f"🧠 Unique tickers to enrich: {sorted(all_tickers)}")

# Enrich each ticker
enriched_rows = []

for ticker in all_tickers:
    try:
        data = yf.download(ticker, period="15d", interval="1d", progress=False)

        if data.empty or len(data) < 15:
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

        vol_trend = 1 if volume[-1] > volume[-15:].mean() else 0

        enriched_rows.append({
            "Ticker": ticker,
            "Last_Close": close.iloc[-1],
            "EMA_9": ema_9,
            "EMA_20": ema_20,
            "EMA_200": ema_200,
            "VWAP": vwap,
            "MACD": macd.iloc[-1],
            "MACD_Signal": signal.iloc[-1],
            "Volume_Trend_Up": vol_trend
        })

    except Exception as e:
        print(f"❌ Error processing {ticker}: {e}")

# Create enrichment DataFrame
df_enriched = pd.DataFrame(enriched_rows)

# Merge with original screener (Change% + RVOL)
df_final = pd.merge(df_input, df_enriched, on="Ticker", how="inner")
df_final.to_csv(output_file, index=False)

print(f"✅ Enrichment complete. Saved to {output_file}")
