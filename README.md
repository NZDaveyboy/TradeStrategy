# Trade Strategy Screener (Full Auto: Tickers + Intraday)

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

## What This Project Does
- Auto-scrapes Top Gainers daily (via Finviz)
- Auto-updates `tickers.txt`
- Screens for intraday momentum setups
- Dynamic ATR-based stop suggestion
- Exports results to CSV
- GitHub Actions runs everything daily at 9AM NZT

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run locally:
   ```bash
   python update_tickers.py
   python stock_screener_auto.py
   ```

3. Or automate everything via GitHub Actions!

## Automation
- `.github/workflows/run_screener.yml` schedules full pipeline
- No manual input needed after setup

---
Created with ❤️ by NZDaveyboy