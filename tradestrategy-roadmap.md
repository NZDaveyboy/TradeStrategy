# TradeStrategy — Upgrade Roadmap

Save this file to `docs/roadmap/upgrade-roadmap.md` in the repo.
Claude Code should read this before any implementation work.

**Last updated:** 2026-05-14 by Claude Code session.

---

## Current status (at a glance)

| Phase | Title | Status |
|---|---|---|
| 5  | Data provider abstraction        | ✅ **Done** (this session) |
| 6  | Test suite                       | ✅ **Done** — 277 tests passing, CI workflows in place |
| 7  | Formal backtest engine           | ✅ **Done** — `core/backtest_engine.py` + `backtest_v2.py` |
| 8  | Analytics and tearsheets         | ✅ **Done** (this session) |
| 9  | Research mode                    | ✅ **Done** — `core/research/` with sweep, walk_forward, rescore, compare |
| 10 | Catalyst layer                   | ✅ **Done** (this session) |
| 11 | README and docs rewrite          | ⬜ **Not started** — next priority |

### Out-of-original-scope work also completed this session

- **Lookup tab** — free-form ticker analysis with full TradeScore + Recommendation + CatalystScore + price chart + 6 catalyst expanders. Replaces the removed Trade Tracker + Alerts tabs.
- **FOREX tab** — major/cross pair scan with bias, plus Learn → Lesson 10 covering pip/lot/leverage/spread/carry fundamentals.
- **Glossary** — Learn → Lesson 0 covering every visible UI term in 10 sections (TradeScore subscores, indicators, setups, recommendation fields, Greeks, strategies, SEC filings, gov contracts, backtest analytics, CatalystScore).
- **Price chart on recommendations** — 6-month daily Close + EMA20 + EMA50 with dashed Entry/Stop/Target rules and dollar labels.
- **Catalyst tags flow into the Recommendation rationale** — Advice + Options + Lookup tabs now show binary-event warnings (e.g. "⚠ Earnings in 4 days") automatically.

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

1. ~~Data stack is fragile — yfinance + Finviz scraping + SQLite + CSV alerts~~ — **partially addressed** (Phase 5 provider abstraction done; alerts.csv layer removed when Alerts tab was deleted)
2. ~~No real news/catalyst layer — only a stub in the scoring model~~ — **addressed** (Phase 10: earnings, news, analyst, insider, USAspending, SEC filings — all wired into Lookup + Recommendation)
3. ~~No test suite — zero credibility for open-source~~ — **addressed** (277 tests across 13 test files, CI workflows for unit tests + screener + premarket scan)
4. ~~Backtesting is product-driven, not evidence-driven~~ — **partially addressed** (Phase 7 + 9 done — formal engine, walk-forward, parameter sweeps. Still need users actually running research-mode sweeps regularly.)
5. ~~Analytics layer too light — no tearsheets, no Sharpe/Sortino, no drawdown reporting~~ — **addressed** (Phase 8: properly annualised Sharpe/Sortino, drawdown chart, monthly heatmap, returns histogram, downloadable QuantStats HTML tearsheet)
6. **README undersells the product** — still the biggest remaining gap. Next phase.
7. No packaging boundary — still reads like a personal codebase — out of scope for the original 11 phases; could be Phase 12+.

---

## Phase plan

### ✅ Phase 5 — Data provider abstraction
Wrap yfinance and scraping behind provider interfaces.
No user-facing changes. No scoring changes. No UI changes.
Goal: future data source swaps don't break the app.

**Delivered:**
- `providers/base.py` — typed provider ABCs (MarketDataProvider, TickerDiscoveryProvider); re-exports schemas from `data.models` for backward compat
- `providers/yfinance_provider.py` — now yfinance-only (Finviz extracted)
- `providers/scraped_provider.py` — `FinvizDiscoveryProvider` lives here, separated from yfinance to isolate HTML-scraping deps
- `data/models.py` — strict `Quote` and `Fundamentals`, plus documentation-grade `OHLCVBar` / `OptionContract` / `OptionChain` schemas (DataFrame column contracts for now; future strict-typing planned)
- `data/__init__.py` — re-exports all schemas
- Tests: 16 in `tests/test_providers.py`, 13 in `tests/test_models.py`

**Note on namespace:** roadmap originally specified `tradestrategy/data/providers/` package; actual repo uses flat `providers/` + `data/` because restructuring the whole namespace would have touched ~25 call sites for no behavioural benefit. Path can be migrated in a later refactor if a packaging boundary (Phase 12) is added.

### ✅ Phase 6 — Test suite
**Delivered:**
- 277 tests across 13 test files: tradescore, recommendations, analytics, catalysts, backtest_engine, walk_forward, providers, models, db, edgar_rss, options_pricing, theme_watchlist, rescore_fidelity, research_sweep
- CI workflows in `.github/workflows/`: `ci.yml`, `premarket_scan.yml`, `screener.yml`, `test_screener_in_actions.yml`
- All math-critical code (annualised Sharpe/Sortino, recommendation rules, options pricing) has dedicated test files

**Coverage gap:** no `pytest --cov` report committed yet. Worth running once and committing the coverage threshold to CI.

### ✅ Phase 7 — Formal backtest engine
**Delivered:**
- `core/backtest_engine.py` — strategy abstraction wrapping `backtesting.py`
- `backtest_v2.py` — uses the engine; saves per-ticker results to `backtest_v2` table
- `options_backtest.py` — companion options strategy backtest
- Configurable commissions, position sizing, stop/target handling, max_hold_days

**Legacy:** old `backtest.py` at repo root still exists but is unused. Consider deleting in a small cleanup PR.

### ✅ Phase 8 — Analytics and tearsheets (QuantStats-style)
**Delivered (this session):**
- **Properly annualised Sharpe and Sortino** — was previously `mean/std` (not annualised, no risk-free rate); now `((mean − rf_per_period) / std) × √N` with configurable `periods_per_year` and `risk_free_rate`
- **Avg per-ticker Sharpe** — mean of saved per-ticker annualised Sharpes from `backtesting.py`, surfaced as a cross-check
- **Drawdown chart** — Altair area chart paired with equity curve
- **Monthly returns heatmap** — Altair year × month grid with green/red colour scale and numeric labels
- **Returns distribution histogram** — bins of per-bet returns with zero-line marker
- **Downloadable QuantStats HTML tearsheet** — full report with rolling Sharpe, drawdown periods, percentile ranks
- 11 new tests in `test_analytics.py` covering annualisation math, drawdown invariants, monthly-table grouping

### ✅ Phase 9 — Research mode
**Delivered:**
- `research_mode.py` + full `core/research/` package: `sweep.py`, `walk_forward.py`, `params.py`, `rescore.py`, `compare.py`, `storage.py`
- Parameter sweeps over TradeScore thresholds, RVOL cutoffs, RSI bands, stop multiples
- Walk-forward validation with separate train/test windows
- Tests for sweep storage, walk-forward, rescore fidelity

### ✅ Phase 10 — Catalyst layer (highest differentiation value)
**Delivered (this session):**

All seven original deliverables:
- ✅ **Earnings dates** — `get_next_earnings`, `get_recent_earnings_history`
- ✅ **Recent SEC filings** — `core/sec_edgar.py` already had the fetcher; now wired into Lookup tab
- ✅ **Major headline ingestion** — `get_recent_news` (Yahoo Finance feed)
- ✅ **Analyst action flags** — `get_analyst_actions` (upgrades, downgrades, target raises/cuts, consensus rating)
- ✅ **Event tags** — plain-English colour-coded strings (🟢/🔴/⚪) like "4-quarter beat streak", "12 price target cuts in 90d", "⚠ Earnings in 4 days — binary event risk"
- ✅ **CatalystScore (0–100)** — composite separate from TradeScore. Baseline 50; 8 component types from earnings, analysts, news, insider activity push it up/down. Hard-clamped to [0, 100]
- ✅ **Plain-English explanation in cards/modal** — catalyst tags now flow into `build_recommendation()` rationale and warnings; ⚠ tags are promoted to warnings automatically; Advice + Options + Lookup tabs all show consistent rationale

**Extension beyond original scope:**
- ✅ **USAspending federal contracts** — `core/usaspending.py` adds a "Recent federal contracts" expander to Lookup for any ticker that's a federal contractor (auto-detected via yfinance longName)
- ✅ **Form 4 insider trade scoring** — `get_insider_activity` parses yfinance's pre-classified data; asymmetric scoring (buying is signal, selling mostly noise); insider buy clusters score +15 (gold-tier signal)

### ⬜ Phase 11 — README and docs rewrite (next)
README should match the actual product. Currently 919 bytes — undersized for what the app now contains.

**Deliverables:**
- New README with: positioning, screenshots of each tab, one-command setup, architecture overview, contribution notes
- `.env.example` with all environment variables documented
- Update CLAUDE Code operating-rules section to reflect completed phases
- Optional: `docs/architecture/current-state.md` to replace the Codex-attempted file (which was never pushed)

---

## Phase 12+ candidates (discovered during execution)

Items that came up during Phase 5–10 implementation but weren't in the
original 11-phase plan. Recorded here so they don't get lost.

### Code hygiene
- **Delete legacy `backtest.py`** at repo root — superseded by `backtest_v2.py` and `core/backtest_engine.py`. Currently dead code that confuses readers.
- **Renumber `# TAB X` comments** in `app.py` — gaps from removed Trade Tracker (#2) and Alerts (#3) tabs left the numbering as 1, 2 (Lookup), 4, 5, 6, 7, 8, 8b (FOREX), 9. Purely cosmetic but worth fixing on a code-tidy pass.
- **Commit a `pytest --cov` baseline** — currently no coverage report committed; threshold can then be enforced in CI.

### Catalyst layer extensions
- **Form 4 XML parsing direct from SEC** — currently using yfinance's pre-classified `.insider_transactions` (P / S / A / M codes mapped to Purchase / Sale / Conversion etc). Going direct to SEC EDGAR XML would give finer detail (10b5-1 vs discretionary, derivative vs non-derivative) but requires bs4/lxml XML parsing per filing.
- **News sentiment scoring** — currently counts news items as "intensity" (coverage signal). Adding direction (bullish / bearish) via keyword or small embedding model would let CatalystScore use news to push score up/down, not just signal activity.
- **Earnings-drift (PEAD) backtest** — the data is now in `core/catalysts.py`. A dedicated PEAD strategy run via the formal backtest engine would validate whether to lean on earnings-surprise drift as a primary signal.

### Data layer
- **Convert provider DataFrames to typed records** — `OHLCVBar` / `OptionContract` exist in `data/models.py` as documentation schemas. Providers still return DataFrames. Migration would force strict-typing through the call chain but breaks many downstream callers.
- **Packaging boundary** — restructure flat `core/`, `providers/`, `data/` into a single `tradestrategy/` namespace package with `__init__.py` exports. Originally specced in the Phase 5 deliverables list under `tradestrategy/data/providers/`. Out-of-scope for the analytics work but worth considering before open-sourcing.

### Process / docs
- **Daily-brief Telegram modernisation** — `send_brief.py` exists but predates the catalyst layer; doesn't include CatalystScore in the brief.
- **Daily-brief includes Lookup-tab metrics** — same idea.
- **Versioned schema migrations** for `screener.db` (Trade Tracker and Alerts tables are now orphaned but still exist).

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

## What Codex attempted (Phase 0 / architecture pass) — historical

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

That work existed only in the Codex sandbox — never pushed.

**Resolution:** the equivalent functionality has been built directly in
the flat repo layout (`core/`, `providers/`, `data/`) without the
`tradestrategy/` namespace package. If a packaging boundary is added in
Phase 12+, the Codex layout becomes a useful reference for the target
structure.

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
