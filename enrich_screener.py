import yfinance as yf
import pandas as pd
import numpy as np
import os

input_file = "screened_stocks_intraday.csv"
output_file = "screened_stocks_enriched.csv"

if not os.path.exists(input_file):
    raise Exception(f"Input file {input_file} not found!")

df = pd.read_csv(input_file)

enriched_rows = []

for ticker in df["Ticker"]:
    try:
        stock_data = yf.download(ticker, period="5d", interval="5m", progress=False)

        if stock_data.empty:
            continue

        stock_data["EMA_9"] = stock_data["Close"].ewm(span=9, adjust=False).mean()
        stock_data["EMA_20"] = stock_data["Close"].ewm(span=20, adjust=False).mean()
        stock_data["EMA_200"] = stock_data["Close"].ewm(span=200, adjust=False).mean()
        stock_data["VWAP"] = (stock_data["Volume"] * (stock_data["High"] + stock_data["Low"]) / 2).cumsum() / stock_data["Volume"].cumsum()
        stock_data["MACD"] = stock_data["Close"].ewm(span=12, adjust=False).mean() - stock_data["Close"].ewm(span=26, adjust=False).mean()
        stock_data["MACD_Signal"] = stock_data["MACD"].ewm(span=9, adjust=False).mean()

        last_volume = stock_data["Volume"].iloc[-1]
        avg_volume = stock_data["Volume"].tail(15).mean()
        volume_trend_up = int(last_volume > avg_volume)

        last_close = stock_data["Close"].iloc[-1]
        last_vwap = stock_data["VWAP"].iloc[-1]
        ema_9 = stock_data["EMA_9"].iloc[-1]
        ema_20 = stock_data["EMA_20"].iloc[-1]
        ema_200 = stock_data["EMA_200"].iloc[-1]
        macd = stock_data["MACD"].iloc[-1]
        macd_signal = stock_data["MACD_Signal"].iloc[-1]

        enriched_rows.append({
            "Ticker": ticker,
            "Last_Close": last_close,
            "EMA_9": ema_9,
            "EMA_20": ema_20,
            "EMA_200": ema_200,
            "VWAP": last_vwap,
            "MACD": macd,
            "MACD_Signal": macd_signal,
            "Volume_Trend_Up": volume_trend_up,
        })

    except Exception as e:
        print(f"Error processing {ticker}: {e}")

enriched_df = pd.DataFrame(enriched_rows)
final_df = pd.merge(df, enriched_df, on="Ticker", how="inner")
final_df.to_csv(output_file, index=False)

print(f"✅ Screener enrichment complete! Saved to {output_file}.")
