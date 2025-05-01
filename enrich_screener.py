import yfinance as yf
import pandas as pd
import os

# Define ticker source files
files = ["tickers.txt", "tickers_ai.txt", "tickers_tech.txt", "tickers_crypto.txt"]
tickers = []

# Load tickers from files
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
            print(f"Skipping {ticker}: not enough data")
            continue

        # Calculate indicators
        data["EMA_9"] = data["Close"].ewm(span=9).mean()
        data["EMA_20"] = data["Close"].ewm(span=20).mean()
        data["EMA_200"] = data["Close"].ewm(span=200).mean()
        data["VWAP"] = (data["High"] + data["Low"] + data["Close"]) / 3

        exp1 = data["Close"].ewm(span=12, adjust=False).mean()
        exp2 = data["Close"].ewm(span=26, adjust=False).mean()
        data["MACD"] = exp1 - exp2
        data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

        # Extract values
        last_close = data["Close"].iloc[-1]
        previous_close = data["Close"].iloc[-2]
        change = (last_close / previous_close - 1) * 100

        rvol = (
            data["Volume"].iloc[-1] / data["Volume"].tail(15).mean()
            if "Volume" in data.columns else 1
        )

        is_crypto = ticker.endswith("-USD")
        volume_up = int(
            not is_crypto and
            data["Volume"].rolling(3).mean().iloc[-1] >
            data["Volume"].rolling(3).mean().iloc[-4]
        )

        row = {
            "Ticker": ticker,
            "Change%": round(change, 2),
            "RVOL": round(rvol, 2),
            "Last_Close": round(last_close, 2),
            "EMA_9": round(data["EMA_9"].iloc[-1], 2),
            "EMA_20": round(data["EMA_20"].iloc[-1], 2),
            "EMA_200": round(data["EMA_200"].iloc[-1], 2),
            "VWAP": round(data["VWAP"].iloc[-1], 2),
            "MACD": round(data["MACD"].iloc[-1], 4),
            "MACD_Signal": round(data["MACD_Signal"].iloc[-1], 4),
            "Volume_Trend_Up": volume_up,
            "Asset": "crypto" if is_crypto else "equity"
        }

        rows.append(row)

    except Exception as e:
        print(f"❌ Error processing {ticker}: {e}")

# Export results
if rows:
    pd.DataFrame(rows).to_csv("screened_stocks_enriched.csv", index=False)
    print("✅ Screener enrichment complete.")
else:
    print("⚠️ No valid tickers enriched.")
