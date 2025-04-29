# Stock Screener - Intraday and Weekly Movers

This Python script screens for high-probability stock trading setups based on:
- Relative Volume (RVOL)
- Price Change (Intraday or Weekly)
- EMA Alignment (9 > 20 > 200)
- RSI < 70 (not overbought)
- ATR-based dynamic stop-loss suggestion
- VWAP awareness
- Catalyst consideration (manual)

## Features
- Choose Intraday or Weekly Scan
- Load tickers automatically from a text file
- Export results to CSV
- Optional email alerts for setups found

## How to Use
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare your `tickers.txt` with one ticker per line.
3. Run the script:
   ```bash
   python stock_screener.py
   ```
4. Follow prompts for scan type and optional email alerts.

## Files
- `stock_screener.py`: Main script.
- `tickers.txt`: List of tickers to scan.
- `requirements.txt`: Python dependencies.

---
Created with ❤️ by YourName