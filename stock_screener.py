import yfinance as yf
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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

def screen_stocks(tickers, mode="intraday"):
    candidates = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="20d", interval="1d")

            if data.shape[0] < 15:
                continue

            rvol = data["Volume"].iloc[-1] / data["Volume"].iloc[:-1].mean()
            price = data["Close"].iloc[-1]

            if mode == "intraday":
                change = (price / data["Close"].iloc[-2] - 1) * 100
            elif mode == "weekly":
                change = (price / data["Close"].iloc[-6] - 1) * 100
            else:
                raise ValueError("Mode must be 'intraday' or 'weekly'.")

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

def send_email(subject, body, to_email, from_email, from_password):
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, from_password)
        server.send_message(msg)
        server.quit()
        print("Email alert sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

if __name__ == "__main__":
    print("Welcome to the All-in-One Stock Screener!")
    print("Choose scanning mode:")
    print("1 - Intraday Movers (Today +10%)")
    print("2 - Weekly Movers (5-day +10%)")
    mode_choice = input("Enter 1 or 2: ")

    mode = "intraday" if mode_choice == "1" else "weekly"

    ticker_file = input("Enter path to ticker file (e.g., tickers.txt): ").strip()

    tickers = load_tickers(ticker_file)
    print(f"Loaded {len(tickers)} tickers.")

    results = screen_stocks(tickers, mode=mode)

    if results.empty:
        print("\nNo high-probability setups found today.")
    else:
        print("\nScreened Stocks:")
        print(results.to_string(index=False))

        output_filename = f"screened_stocks_{mode}.csv"
        results.to_csv(output_filename, index=False)
        print(f"\nResults saved to {output_filename}")

        EMAIL_ALERT = input("Send email alert? (y/n): ").strip().lower()
        if EMAIL_ALERT == 'y':
            from_email = input("Enter your email address (Gmail recommended): ").strip()
            from_password = input("Enter your email app password: ").strip()
            to_email = input("Enter destination email address: ").strip()

            email_body = f"High-probability setups found!\n\n{results.to_string(index=False)}"
            send_email(
                subject=f"Stock Screener Alert: {mode.capitalize()} Mode",
                body=email_body,
                to_email=to_email,
                from_email=from_email,
                from_password=from_password
            )