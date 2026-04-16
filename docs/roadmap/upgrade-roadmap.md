# TradeStrategy — Upgrade Roadmap

Save this file to `docs/roadmap/upgrade-roadmap.md` in the repo.
Claude Code should read this before any implementation work.

---

## What TradeStrategy is

A Streamlit trader decision workbench for self-directed momentum and options-oriented traders.
It already has: custom TradeScore engine (0–100), conviction labels, Top Opportunities cards,
trade setups (entry/stop/target/R:R), SEC filings via EDGAR, alerts, trade tracker, options
pricing/strategy tools, backtest views, and a Telegram daily brief.

The right direction is **sharper and more defensible** — not broader.

---

## Positioning

Own this niche:
> A focused, explainable decision workbench for self-directed traders who screen momentum
> and event-driven setups, express them with shares or options, and want a tighter daily
> workflow than generic charting platforms provide.

The product should answer six questions clearly:
1. Why is this on my screen?
2. Why now?
3. What is the risk?
4. How should I express it?
5. What happened when I traded similar setups before?
6. What did I learn?

Do NOT copy: Lean's full-platform ambition, Nautilus's execution complexity, OpenBB's breadth,
or crypto bot patterns.

---

## Key weaknesses to fix (in priority order)

1. Data stack is fragile — yfinance + Finviz scraping + SQLite + CSV alerts *(provider abstraction delivered; underlying sources unchanged)*
2. No real news/catalyst layer — EDGAR RSS watcher delivered; earnings/headlines still pending *(Phase 10)*
3. No test suite ✅ — 144 tests as of April 2026
4. Backtesting is product-driven, not evidence-driven — formal engine delivered; research mode in progress *(Phase 9)*
5. Analytics layer too light ✅ — equity curve, Sharpe/Sortino, win rate, score buckets delivered
6. README undersells the product *(Phase 11)*
7. No packaging boundary — still reads like a personal codebase *(Phase 11)*

---

## Status — April 2026

Phases 1–8 complete. Phase 9 in progress.

Additional deliverables outside the original phase plan:
- **Recommendation engine** (`core/recommendations.py`): unified strategy engine used by both
  Advice and Options tabs. Fixes stop anchor (EMA20±ATR, not cumulative VWAP), extension check
  ordering, and cross-tab consistency via explicit `iv_mode` parameter.
- **EDGAR RSS watcher** (`core/edgar_rss.py`): polls SEC EDGAR Atom feeds (8-K, S-1, SC 13G),
  matches against theme watchlist and screener universe. Early Signals panel in Screener tab.
- **Theme watchlist** (`core/theme_watchlist.py`): curated AI Infrastructure list with
  session-level overrides. Watchlist badges on opportunity cards. Watchlist section in Telegram brief.
- **TradeScore extraction** (`core/tradescore.py`): scoring engine lifted from `run.py` into
  its own module so research mode and tests can import it without the full screener.

---

## Phase plan

### Phase 5 — Data provider abstraction ✅
Wrap yfinance and scraping behind provider interfaces.
Delivered: `providers/yfinance_provider.py` (YFinanceProvider, FinvizDiscoveryProvider),
provider ABC, typed return contracts. Tests in `tests/test_providers.py`.

### Phase 6 — Test suite ✅
pytest coverage for TradeScore, backtest engine, options pricing, providers,
recommendations engine, EDGAR RSS, theme watchlist, analytics.
144 tests passing as of April 2026.

### Phase 7 — Formal backtest engine ✅
`core/backtest_engine.py` + `backtest_v2.py`. backtesting.py-based strategy simulation:
market entry next bar open, EMA20±ATR stop, configurable hold period and commission.
Tests in `tests/test_backtest_engine.py`.

### Phase 8 — Analytics and tearsheets ✅
`core/analytics.py`: equity curve, Sharpe/Sortino, win rate by setup, score-bucket
performance. Analytics section in Streamlit Backtest tab. Tests in `tests/test_analytics.py`.

### Phase 9 — Research mode (in progress)
Parameter sweeps for TradeScore thresholds, RVOL cutoffs, RSI bands, stop multiples.
Walk-forward validation. CLI entry point (`research_mode.py`).
Scoring engine extracted to `core/tradescore.py` in preparation.
Goal: answer "does TradeScore > 55 actually outperform > 45?"

### Phase 10 — Catalyst layer (highest differentiation value)
Real catalyst context for every setup:
- Earnings dates
- Recent SEC filings (partially delivered via EDGAR RSS watcher)
- Major headline ingestion
- Analyst action flags
- Event tags
- Simple catalyst score separate from price/volume score
- Plain-English explanation in cards and modal

### Phase 11 — README and docs rewrite
README should match the actual product.
Add `.env.example`, one-command local startup, architecture overview.

---

## Best reference repos (study, don't copy wholesale)

| Repo | What to learn from it |
|---|---|
| OpenBB | Architecture and data interface design |
| backtesting.py | Strategy abstraction and backtest engine patterns |
| vectorbt | Vectorized research, parameter sweeps, signal cohort analysis |
| QuantStats | Tearsheet design, analytics API, performance reporting |
| PyBroker | Walk-forward evaluation, cache-aware research |

Licensing notes:
- vectorbt is Apache 2.0 + Commons Clause — re-implement ideas, don't copy code directly
- OpenBB is AGPL — use as dependency or reference, not as code to merge in

---

## What Codex attempted (Phase 0 / architecture pass)

A Codex sandbox run created an initial package structure:
- `tradestrategy/data/database.py`
- `tradestrategy/screening/opportunities.py`
- `tradestrategy/options/pricing.py`
- `tradestrategy/app/advice.py`
- `tradestrategy/analytics/__init__.py` (scaffold only)
- `tradestrategy/backtesting/__init__.py` (scaffold only)
- `docs/architecture/current-state.md`
- `docs/architecture/target-state.md`
- Initial test files

That work exists only in the Codex sandbox — it was never pushed to the repo.
Claude Code should implement this structure directly in the actual repo rather than
trying to retrieve the Codex output.

---

## Operating rules for Claude Code

- Preserve trader workbench product direction
- Keep Streamlit as the main UI
- Preserve all existing user-facing behaviour unless a phase explicitly changes it
- Prefer modular Python, typed models, small reviewable changes
- Add tests for changed logic
- Update docs for new modules and CLI commands
- Do not merge external repos wholesale — use ideas and patterns selectively
- End each task with: changed files / what changed and why / local test steps / risks
