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

1. Data stack is fragile — yfinance + Finviz scraping + SQLite + CSV alerts
2. No real news/catalyst layer — only a stub in the scoring model
3. No test suite — zero credibility for open-source
4. Backtesting is product-driven, not evidence-driven
5. Analytics layer too light — no tearsheets, no Sharpe/Sortino, no drawdown reporting
6. README undersells the product
7. No packaging boundary — still reads like a personal codebase

---

## Phase plan

### Phase 5 — Data provider abstraction (next)
Wrap yfinance and scraping behind provider interfaces.
No user-facing changes. No scoring changes. No UI changes.
Goal: future data source swaps don't break the app.

Deliverables:
- `tradestrategy/data/providers/base.py` — typed provider interface
- `tradestrategy/data/providers/yfinance_provider.py` — wraps current yfinance calls
- `tradestrategy/data/providers/scraped_provider.py` — wraps current scraping calls
- `tradestrategy/data/models.py` — typed schemas for OHLCV, quotes, company info, option chains
- Tests for provider contracts

### Phase 6 — Test suite
Add pytest coverage for: TradeScore correctness, backtest reproducibility, options math,
provider contracts, daily brief generation, DB migration.
Add GitHub Actions CI workflow.

### Phase 7 — Formal backtest engine
Replace ad hoc backtest logic with a strategy abstraction inspired by backtesting.py.
Support: configurable commissions, slippage, position sizing, stop/target handling,
date-range replay, parameterized rules.
Screener outputs should feed directly into formal strategy replays.

### Phase 8 — Analytics and tearsheets (QuantStats-style)
Add to backtest outputs: equity curve, drawdowns, monthly returns heatmap,
Sharpe/Sortino/CAGR/volatility, win rate, expectancy, payoff ratio,
setup-type performance, score-bucket performance.
Add exportable HTML report. Connect summary views to Streamlit.

### Phase 9 — Research mode
Parameter sweeps for TradeScore thresholds, RVOL cutoffs, RSI bands, stop multiples.
Fast comparative research. Walk-forward validation. CLI entry point.
Goal: answer "does TradeScore > 55 actually outperform > 45?"

### Phase 10 — Catalyst layer (highest differentiation value)
Real catalyst context for every setup:
- Earnings dates
- Recent SEC filings (already partially built)
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
