# TradeStrategy

> A focused, explainable decision workbench for self-directed traders who
> screen momentum and event-driven setups, express them with shares or
> options, and want a tighter daily workflow than generic charting platforms
> provide.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-277%20passing-brightgreen)]()

TradeStrategy answers six questions about every ticker:

1. **Why is this on my screen?** — TradeScore (0–65 composite of momentum, early-entry, liquidity, news catalyst, extension risk)
2. **Why now?** — CatalystScore (0–100) from earnings, analyst actions, news, insider buying, SEC filings, federal contracts
3. **What is the risk?** — explicit invalidation level (EMA20 ± ½ ATR), stop-distance cap, IV assessment
4. **How should I express it?** — one of five options strategies (Long Call, Bull Call Spread, Long Put, Bear Put Spread, Cash-Secured Put) or "wait"
5. **What happened when I traded similar setups before?** — formal backtest engine with walk-forward validation
6. **What did I learn?** — analytics tab with annualised Sharpe / Sortino, drawdown, monthly heatmap, downloadable QuantStats tearsheet

It is **not** a charting replacement, an auto-trader, or a generic screener.
It is a **decision workbench** — its job is to surface specific, defensible
reasoning, not to maximise feature breadth.

---

## Quick start

```bash
# 1. Clone + Python 3.10+ environment
git clone <repo> && cd TradeStrategy
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# edit .env with your Telegram token + SEC user-agent if needed

# 3. (Optional) populate the screener database
python run.py          # one screener cycle — needs ~5 minutes

# 4. Launch the Streamlit app
streamlit run app.py
```

The app opens at `http://localhost:8501` with 11 tabs.

---

## The 11 tabs

| Tab | What it does | Use when |
|---|---|---|
| **Screener** | Today's top opportunities ranked by TradeScore, with detail dialogs for each | You want today's watchlist |
| **Lookup** | Free-form ticker analysis — type any symbol, get the full TradeScore + CatalystScore + Recommendation + price chart + 6 catalyst expanders (earnings, news, analyst actions, insider activity, SEC filings, federal contracts) | You have a specific name in mind (SOFI, NVDA, etc.) |
| **Backtest** | Per-ticker simulation results with QuantStats-style analytics: equity curve, drawdown, monthly heatmap, returns histogram. Downloadable HTML tearsheet. | You want to validate (or kill) a strategy idea |
| **Advice** | Options-strategy recommendations for the screener's top picks, with the same recommendation engine the Lookup tab uses | You've decided to trade — pick which to express |
| **Options** | Live IV vs RV30 analysis, contract pricing, strategy recommendation with the same engine | You're sizing a specific options trade |
| **Learn** | 11 lessons including a comprehensive **Glossary (Lesson 0)** covering every visible UI term, options fundamentals, and a FOREX basics primer | You don't recognise a term, or someone (e.g. your kid) is learning |
| **Metals** | Gold / silver / platinum / palladium futures context | You want macro context on precious metals |
| **FOREX** | Major + cross currency pair scan with RSI, ATR, trend bias. **No fake recommendation engine** — analysis only. | You're tracking FX as macro context (or actively trading it) |
| **Indexes** | Sector ETFs + macro indexes (SPY/QQQ/IWM/SMH/XLF/VIX/etc.) with bias + 6mo charts. VIX-specific risk-regime rules. | You're checking whether a move is sector-wide or stock-specific |
| **Quantum** | Three custom quantum technology indexes (Pure Play / Ecosystem / Barbell) with conviction scoring, drawdown, concentration analysis, ex-top counterfactuals, correlation vs QTUM | You're researching the quantum theme as a focused bet |
| **Portfolio** | Suggested allocation across theme sectors based on screener output | You want a single sanity-checked allocation view |

---

## Architecture overview

```
TradeStrategy/
├── app.py                    Streamlit UI — all 9 tabs
├── run.py                    Screener engine: pulls OHLCV, computes indicators, scores
├── backtest_v2.py            Per-ticker backtest runner (uses core/backtest_engine.py)
├── send_brief.py             Telegram daily brief
├── scan_premarket.py         Pre-market gap + RVOL scanner
├── scan_intraday.py          Intraday momentum scanner
│
├── core/
│   ├── tradescore.py         TradeScore composite (0–65) — 5 sub-scores
│   ├── recommendations.py    Recommendation engine — entry, stop, target, strategy, IV
│   ├── setups.py             Setup classifier (clean_breakout, extended, etc.)
│   ├── catalysts.py          Phase 10 catalyst layer (earnings, news, analyst, insider)
│   ├── sec_edgar.py          SEC EDGAR filings fetcher
│   ├── edgar_rss.py          SEC EDGAR RSS watcher for early-signal detection
│   ├── usaspending.py        Federal contract awards via USAspending.gov
│   ├── theme_watchlist.py    Theme-based ticker grouping
│   ├── backtest_engine.py    Phase 7 formal backtest engine wrapping backtesting.py
│   ├── analytics.py          Phase 8 analytics: annualised Sharpe/Sortino, drawdown, monthly heatmap, QuantStats HTML tearsheet
│   ├── db.py                 SQLite connection management
│   └── research/             Phase 9 research mode: parameter sweeps, walk-forward
│
├── providers/                Phase 5 data-provider abstraction
│   ├── base.py               MarketDataProvider + TickerDiscoveryProvider ABCs
│   ├── yfinance_provider.py  yfinance wrapper (OHLCV, quotes, fundamentals, options)
│   └── scraped_provider.py   FinvizDiscoveryProvider (HTML scraping isolated here)
│
├── data/                     Typed schemas
│   ├── __init__.py
│   └── models.py             Quote, Fundamentals (strict) + OHLCVBar, OptionContract (docs)
│
├── tests/                    277 tests across 13 test files
└── .github/workflows/        CI for tests + scheduled scanners
```

**Read order for new contributors**: `core/tradescore.py` → `core/recommendations.py` → `core/catalysts.py` → `app.py` (Lookup tab section is the cleanest example of how the layers compose).

---

## Configuration

All configuration is via environment variables in `.env`. See `.env.example`
for the full list with comments. Minimum to run:

- **`TELEGRAM_BOT_TOKEN`** + **`TELEGRAM_CHAT_ID`** — required only if you use `send_brief.py`. The app itself runs without Telegram.

Strongly recommended:

- **`SEC_USER_AGENT`** — SEC requires a real user-agent with contact info for the EDGAR API. Default values exist but you should override with your own (e.g. `"YourName your@email.com"`) before deploying.

---

## Running the screener

The screener runs offline (no Streamlit needed). One cycle pulls Finviz
top gainers + curated lists, fetches OHLCV from yfinance, scores
everything via TradeScore, and writes to `screener.db`:

```bash
python run.py           # one full screener pass
python scan_premarket.py # pre-market gap + RVOL scanner
python scan_intraday.py  # intraday momentum scanner
```

The Streamlit app reads from `screener.db` for the Screener / Backtest /
Advice / Options / Portfolio tabs. The Lookup tab does not need the
database — it fetches each ticker on demand.

---

## Backtesting

```bash
python backtest_v2.py             # backtests every ticker in the screener output
python options_backtest.py        # backtests the options strategy decisions
```

Results write to the `backtest_v2` table. The Streamlit Backtest tab
renders them with the full QuantStats-style analytics.

---

## Testing

```bash
pytest                            # all 277 tests
pytest tests/test_tradescore.py   # TradeScore math
pytest tests/test_recommendations.py  # entry/stop/target rules + catalyst overlay
pytest tests/test_analytics.py    # annualised Sharpe/Sortino + heatmap math
pytest tests/test_catalysts.py    # insider classification
pytest tests/test_models.py       # data schemas
pytest tests/test_providers.py    # provider mocks
```

CI runs `tests/` on every push (see `.github/workflows/ci.yml`).

---

## Design principles

1. **Sharper and more defensible, not broader.** The app does not aim to be a Bloomberg replacement. It aims to make the trader's "should I trade this and how" decision more explicit, faster, and more reviewable than poking around TradingView.
2. **Every score has component traceability.** No black-box. Every TradeScore and CatalystScore decomposes into named sub-scores; every Recommendation includes a plain-English rationale + explicit warnings.
3. **Asymmetric scoring where the data is asymmetric.** Insider buying is signal; insider selling is mostly noise — the scoring reflects this directly rather than treating buys and sells symmetrically.
4. **No fake recommendations.** When the data doesn't support a directional view (FOREX, crypto, low-coverage tickers), the app says so explicitly rather than inventing one.
5. **Tests for math-critical paths.** Sharpe annualisation, recommendation rules, options pricing, insider classification, monthly returns aggregation — all have dedicated tests.

---

## Roadmap status

See `tradestrategy-roadmap.md` for the full phase plan. Snapshot:

| Phase | Title | Status |
|---|---|---|
| 5  | Data provider abstraction        | ✅ Done |
| 6  | Test suite                       | ✅ Done |
| 7  | Formal backtest engine           | ✅ Done |
| 8  | Analytics and tearsheets         | ✅ Done |
| 9  | Research mode                    | ✅ Done |
| 10 | Catalyst layer                   | ✅ Done |
| 11 | README and docs rewrite          | ✅ Done (this file) |

Phase 12+ candidates documented at the bottom of the roadmap doc.

---

## License

MIT. See `LICENSE`.
