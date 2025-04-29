import yfinance as yf
import pandas as pd
import os

def load_tickers(file_path):
    with open(file_path, "r") as f:
        tickers = [line.strip().upper() for line in f.readlines() if line.strip()]
    return tickers

def calculate_atr(data, period=14):
    data['H-L'] = data['High'] - data['Low']
    data['H-PC'] = abs(data['High'] - data['Close'].shift(1))
    data['L-PC'] = abs(data['Low'] - data['Close'].shift(1))
    tr = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    atr = tr.rolling(period).mean()
    return atr.iloc[-1]

def screen_stocks(tickers):
    candidates = []
    mode = "intraday"
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="20d", interval="1d")

            if data.shape[0] < 15:
                continue

            rvol = data["Volume"].iloc[-1] / data["Volume"].iloc[:-1].mean()
            price = data["Close"].iloc[-1]
            change = (price / data["Close"].iloc[-2] - 1) * 100

            ema9 = data["Close"].ewm(span=9, adjust=False).mean().iloc[-1]
            ema20 = data["Close"].ewm(span=20, adjust=False).mean().iloc[-1]
            ema200 = data["Close"].ewm(span=200, adjust=False).mean().iloc[-1]

            delta = data["Close"].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            avg_gain = up.rolling(14).mean().iloc[-1]
            avg_loss = down.rolling(14).mean().iloc[-1]
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
            rsi = 100 - (100 / (1 + rs))

            atr = calculate_atr(data)

            if (
                rvol >= 3
                and 1 <= price <= 30
                and change >= 10
                and ema9 > ema20 > ema200
                and rsi < 70
            ):
                candidates.append({
                    "Ticker": ticker,
                    "Price": round(price, 2),
                    "RVOL": round(rvol, 2),
                    "Change%": round(change, 2),
                    "EMA9": round(ema9, 2),
                    "EMA20": round(ema20, 2),
                    "EMA200": round(ema200, 2),
                    "RSI": round(rsi, 2),
                    "ATR": round(atr, 2),
                    "Suggested_Stop": round(price - (1.5 * atr), 2)
                })

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    return pd.DataFrame(candidates)

if __name__ == "__main__":
    print("Starting Headless Intraday Screener...")
    ticker_file = "tickers.txt"
    tickers = load_tickers(ticker_file)
    print(f"Loaded {len(tickers)} tickers.")

    results = screen_stocks(tickers)

    if results.empty:
        print("\nNo high-probability setups found today.")
    else:
        output_filename = "outputs/screened_stocks_intraday.csv"
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        results.to_csv(output_filename, index=False)
        print(f"\nResults saved to {output_filename}")