import yfinance as yf
import pandas as pd
import numpy as np
import os

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
