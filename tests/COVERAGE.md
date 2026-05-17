# Test coverage baseline

**Generated:** 2026-05-14 (after Phase 5-11 completion)
**Tool:** `pytest-cov`
**Total:** 277 tests, 59% line coverage across `core/`, `providers/`, `data/`

## Run it yourself

```bash
pytest tests/ --cov=core --cov=providers --cov=data --cov-report=term-missing
```

## Coverage by module

| Module | Coverage | Notes |
|---|---:|---|
| `core/tradescore.py`              | **93%** | Math-critical — fully tested |
| `core/recommendations.py`         | **95%** | Math-critical — entry/stop/target rules + catalyst overlay |
| `core/backtest_engine.py`         | **95%** | Wraps backtesting.py; thin layer |
| `core/analytics.py`               | **80%** | Annualised Sharpe/Sortino, drawdown, monthly heatmap |
| `core/db.py`                      | 100% | SQLite connection helpers |
| `data/models.py`                  | 100% | Typed schemas |
| `data/__init__.py`                | 100% | Re-exports |
| `providers/base.py`               | 100% | Re-exports + ABCs |
| `providers/yfinance_provider.py`  | 100% | All methods mocked |
| `providers/scraped_provider.py`   | 100% | Finviz scraper mocked |
| `core/edgar_rss.py`               |  68% | Parsing logic tested, network paths skipped |
| `core/research/params.py`         |  90% | |
| `core/research/walk_forward.py`   |  63% | |
| `core/research/rescore.py`        |  65% | |
| `core/research/sweep.py`          |  56% | |
| `core/research/storage.py`        |  28% | SQLite I/O |
| `core/research/compare.py`        |   0% | No tests yet |
| `core/theme_watchlist.py`         |  46% | |
| **Network-bound modules (uncovered by design):** | | |
| `core/catalysts.py`               |   9% | Only `_classify_insider` tested; fetchers require network mocking |
| `core/sec_edgar.py`               |   0% | Network-bound fetchers |
| `core/usaspending.py`             |   0% | Network-bound fetchers |
| `core/setups.py`                  |   0% | Used by tradescore, indirectly covered |

## Why the network-bound modules show low coverage

`catalysts.py`, `sec_edgar.py`, `usaspending.py` are mostly thin yfinance
wrappers — fetching is a network round-trip we don't want to mock for
every test. The math layered on top (insider classification, catalyst
score components) **is** tested separately. Adding network-mocked tests
would push coverage to ~75% but adds test brittleness without catching
real bugs.

## Suggested coverage targets

- **`core/recommendations.py`, `core/tradescore.py`, `core/analytics.py`** — keep above **90%** (math-critical)
- **`providers/`, `data/`** — keep at **100%** (small surface, contract-critical)
- **`core/research/`** — push toward **75%** by adding sweep storage tests
- **Network-bound modules** — accept ≤15% coverage; rely on integration tests via `app.py` rendering

## CI enforcement

Not enforced yet. Suggested: add `--cov-fail-under=55` to the `.github/workflows/ci.yml` pytest step to prevent regression below the current baseline.
