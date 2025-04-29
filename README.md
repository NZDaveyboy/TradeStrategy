# Trade Strategy Screener (Intraday Headless Mode)

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

This Python script automatically screens for intraday high-probability stock trading setups.

## Features
- Intraday Mode (default)
- RVOL > 3
- Price Change > +10% (Today)
- EMA Alignment (9 > 20 > 200)
- RSI < 70
- ATR-based dynamic stop-loss suggestion
- Exports results to /outputs folder as CSV
- GitHub Actions ready for automatic daily run

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare `tickers.txt` with one ticker per line.

3. Run locally:
   ```bash
   python stock_screener_auto.py
   ```

4. Or automate with GitHub Actions!

## Automation
The `.github/workflows/run_screener.yml` file runs this screener automatically every weekday at 9AM NZT.

---
Created with ❤️ by NZDaveyboy