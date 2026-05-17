import json
import math
import os
from datetime import date, datetime, timezone

# Load .env from the project root so ANTHROPIC_API_KEY (and the existing
# Telegram tokens) are available to the Streamlit app process.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass  # dotenv optional — env vars can also be set in the shell

from core.db import get_connection
from core.edgar_rss import poll_early_signals
from core.recommendations import STRATEGY_DISPLAY, build_recommendation
from core.sec_edgar import get_recent_filings
from core.setups import compute_trade_setup
from core.theme_watchlist import is_on_watchlist

import numpy as np
import pandas as pd
import streamlit as st

from providers.yfinance_provider import YFinanceProvider

_provider = YFinanceProvider()

DB_PATH = os.path.join(os.path.dirname(__file__), "screener.db")


@st.cache_data(ttl=120)
def _fetch_early_signals(tickers_key: tuple[str, ...]) -> list[dict]:
    """Cached EDGAR RSS poll. Returns list of dicts (EarlySignal fields) for easy serialisation."""
    signals = poll_early_signals(screener_tickers=list(tickers_key))
    return [
        {
            "ticker":      s.ticker,
            "company":     s.company,
            "filing_type": s.filing_type,
            "filed_at":    s.filed_at.strftime("%Y-%m-%d %H:%M UTC"),
            "url":         s.url,
            "match_source": s.match_source,
        }
        for s in signals
    ]


st.set_page_config(page_title="TradeStrategy", layout="wide")
st.title("TradeStrategy")


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_conn():
    return get_connection(DB_PATH)


def init_tracker_tables():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker         TEXT    NOT NULL,
            trade_type     TEXT    NOT NULL,
            entry_date     TEXT    NOT NULL,
            entry_price    REAL    NOT NULL,
            shares         REAL    NOT NULL,
            position_nzd   REAL    NOT NULL,
            notes          TEXT    DEFAULT '',
            is_open        INTEGER DEFAULT 1
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS crypto_holdings (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            coin             TEXT    NOT NULL,
            amount           REAL    NOT NULL,
            avg_buy_price_nzd REAL   NOT NULL,
            notes            TEXT    DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            triggered_at TEXT NOT NULL,
            scan_date    TEXT NOT NULL,
            scan_window  TEXT NOT NULL,
            ticker       TEXT NOT NULL,
            alert_type   TEXT NOT NULL,
            value        REAL NOT NULL,
            price        REAL,
            change_pct   REAL,
            rvol         REAL,
            gap_pct      REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metal_holdings (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            metal             TEXT    NOT NULL,
            holding_type      TEXT    NOT NULL DEFAULT 'physical',
            quantity          REAL    NOT NULL,
            avg_buy_price_nzd REAL    NOT NULL,
            notes             TEXT    DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolios (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_type    TEXT    NOT NULL,
            ticker            TEXT    NOT NULL,
            asset_class       TEXT    NOT NULL,
            quantity          REAL    NOT NULL,
            avg_buy_price_nzd REAL    NOT NULL,
            thesis            TEXT    DEFAULT '',
            notes             TEXT    DEFAULT '',
            added_date        TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()


init_tracker_tables()


# ---------------------------------------------------------------------------
# Price fetching (cached 5 min)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400)
def get_company_info(ticker: str) -> dict:
    """Returns basic company description from yfinance. Cached 24h."""
    if ticker.endswith("-USD"):
        return {}
    try:
        fund = _provider.get_fundamentals(ticker)
        return {
            "name":    fund.name,
            "sector":  fund.sector,
            "industry":fund.industry,
            "summary": fund.summary,
            "website": fund.website,
        }
    except Exception:
        return {}


# fetch_nzdusd, fetch_prices moved to ui/data.py (imported at top of file).


# ---------------------------------------------------------------------------
# Screener data (must exist before tabs so sidebar can read it)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def _load_dates() -> list[str]:
    try:
        conn = get_conn()
        result = pd.read_sql(
            "SELECT DISTINCT run_date FROM results ORDER BY run_date DESC", conn
        )["run_date"].tolist()
        conn.close()
        return result
    except Exception:
        return []  # results table doesn't exist yet — fresh deploy or empty DB


dates = _load_dates()
screener_ready = bool(dates)


from core.peers import PEER_MAP, fetch_peer_fundamentals_raw
# Pure display formatters used by multiple tabs — see ui/helpers.py for docs.
# Aliased back to their original underscore names so we don't need to rename
# every call site during the Phase 1 extraction.
from ui.helpers import (
    format_holder_value as _format_holder_value,
    qoq_change_label   as _qoq_change_label,
    fmt_usd_compact    as _fmt_usd_compact,
    regime_label       as _regime_label,
    driver_tags        as _driver_tags_raw,
)


def _driver_tags(ticker: str, screener_row: dict | None = None) -> list[str]:
    """Local wrapper that injects the ASSET_DRIVERS constant — keeps
    call sites in app.py / tab modules unchanged."""
    from ui.data import ASSET_DRIVERS
    return _driver_tags_raw(ticker, screener_row, ASSET_DRIVERS)
# Streamlit-cached data fetchers (see ui/data.py). Aliased to the original
# names so existing call sites keep working unchanged.
from ui.data import (
    fetch_peer_fundamentals,
    fetch_company_info,
    fetch_institutional_data,
    fetch_intraday_bars      as _fetch_intraday_bars,
    cached_pile_in_scan      as _cached_pile_in_scan,
    fetch_nzdusd,
    fetch_prices,
    fetch_metal_prices,
    fetch_metal_chart,
    fetch_metal_technicals,
    fetch_market_context,
    METAL_FUTURES            as _METAL_FUTURES,
    METAL_FUTURES_REV        as _METAL_FUTURES_REV,
    METAL_ETFS               as _METAL_ETFS,
    ALL_METAL_TICKERS        as _ALL_METAL_TICKERS,
    ASSET_DRIVERS            as _ASSET_DRIVERS,
)


# fetch_peer_fundamentals, fetch_company_info, _fetch_intraday_bars moved to
# ui/data.py (imported at top of file).


# _render_live_intraday fragment moved to ui/tabs/lookup.py


# Options helpers (get_chain, get_rv30, enrich_chain, payoff_df) moved to
# ui/data.py so tab modules can import them directly.

# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Screener filters")

    if not dates:
        st.info("No screener data yet. Run `python run.py` first.")
        selected_date = None
        strategy = "All"
        asset_filter = "All"
        min_score = 0
        min_change = 0
        min_rvol = 0.0
        setup_type_filter: list[str] = []
    else:
        selected_date = st.selectbox("Date", dates)

        conn = get_conn()
        strategies = pd.read_sql(
            "SELECT DISTINCT strategy FROM results WHERE run_date = ?",
            conn, params=(selected_date,)
        )["strategy"].tolist()
        setup_types = pd.read_sql(
            "SELECT DISTINCT setup_type FROM results "
            "WHERE run_date = ? AND setup_type IS NOT NULL AND setup_type <> '' "
            "ORDER BY setup_type",
            conn, params=(selected_date,)
        )["setup_type"].tolist()
        conn.close()

        strategy_options = ["All"] + sorted(strategies)
        strategy = st.selectbox("Strategy", strategy_options)

        if strategy == "momentum":
            st.info(
                "**Momentum criteria**\n"
                "- Market cap < $2B\n"
                "- Float < 50M shares\n"
                "- RVOL ≥ 2×\n"
                "- Day change ≥ +5%"
            )

        asset_filter = st.selectbox("Asset type", ["All", "equity", "crypto"])
        min_score = st.slider(
            "Min score (0–4)", 0, 4, 0,
            help=(
                "Each stock is scored 0–4 based on four signals:\n"
                "- MACD above its signal line\n"
                "- Price above all three EMAs (9 / 20 / 200)\n"
                "- Price above VWAP\n"
                "- Rising 3-day volume trend\n\n"
                "Set to 3 or 4 to see only the strongest setups."
            ),
        )
        default_change = 5 if strategy == "momentum" else 0
        default_rvol   = 2.0 if strategy == "momentum" else 0.0
        min_change = st.slider(
            "Min change %", 0, 100, default_change,
            help=(
                "Minimum percentage price change on the day compared to the previous close.\n\n"
                "Higher values surface stocks with stronger intraday momentum. "
                "Momentum strategy defaults to 5%."
            ),
        )
        min_rvol = st.slider(
            "Min RVOL", 0.0, 20.0, default_rvol, 0.5,
            help=(
                "Relative Volume — today's volume divided by the stock's average daily volume.\n\n"
                "2× means twice the usual volume. High RVOL signals unusual activity "
                "and potential breakout conditions. Momentum strategy defaults to 2×."
            ),
        )

        # Setup-type multi-select. Empty selection = all (no filter applied).
        setup_type_filter = st.multiselect(
            "Setup type",
            options=setup_types,
            default=setup_types,
            help=(
                "Filter by the screener's setup classification. Common values:\n"
                "- **Emerging momentum / Early breakout** — actionable longs\n"
                "- **Momentum watchlist** — building setups not yet actionable\n"
                "- **Strong but extended / Overextended** — wait for pullback\n"
                "- **Emerging weakness / Bearish breakdown** — short candidates\n"
                "- **Avoid / Low quality** — skip\n\n"
                "Deselect a setup type to hide it from the Screener table."
            ),
        )

    st.divider()
    with st.expander("How to use"):
        st.markdown(
            """
**Getting started**

1. Run the screener from your terminal to populate data:
   ```
   cd ~/TradeStrategy
   python3 run.py
   ```
2. Come back here and select the date from the **Date** dropdown.

**Tabs**

- **Screener** — ranked list of candidates from the last scan. Use the filters above to narrow results. Score 3–4 means all major signals align.
- **Trade Tracker** — log your open positions (equities and crypto). Live P&L updates every 5 minutes. All values in NZD.
- **Alerts** — pre-market gap and RVOL spikes captured by `scan_premarket.py`.

**Filters**

| Filter | What it does |
|---|---|
| Strategy | Filter by ticker watchlist (AI, Tech, Crypto, Momentum, or Finviz gainers) |
| Asset type | Equities or crypto |
| Min score | 0–4 signal strength — higher is more selective |
| Min change % | Day's price move vs previous close |
| Min RVOL | Volume relative to average — flags unusual activity |
| Setup type | Restrict to one or more setup classifications (deselect to hide) |

**Scheduling** — add cron jobs to catch moves early and run the daily scan before the US open:
```
# Daily screener — 9am ET (2am NZT)
0 2 * * 1-5 cd ~/TradeStrategy && python3 run.py

# Intraday — every 15 min during US hours (catches volume spikes early)
*/15 14-21 * * 1-5 cd ~/TradeStrategy && python3 scan_intraday.py

# Crypto intraday — every 30 min, 24/7
*/30 * * * * cd ~/TradeStrategy && python3 scan_intraday.py --crypto-only
```
            """
        )


# ---------------------------------------------------------------------------
# Tabs — lazy dispatch (only the active view renders per rerun)
# ---------------------------------------------------------------------------

from ui.tabs import (
    dashboard    as _tab_dashboard,
    screener     as _tab_screener,
    lookup       as _tab_lookup,
    backtest     as _tab_backtest,
    advice       as _tab_advice,
    options      as _tab_options,
    learn        as _tab_learn,
    metals       as _tab_metals,
    forex        as _tab_forex,
    indexes      as _tab_indexes,
    quantum      as _tab_quantum,
    portfolio    as _tab_portfolio,
    smart_money  as _tab_smart_money,
    copilot      as _tab_copilot,
)

TAB_LABELS = [
    "Dashboard", "Screener", "Lookup", "Backtest", "Advice",
    "Options", "Learn", "Metals", "FOREX", "Indexes",
    "Quantum", "Portfolio", "💰 Smart Money", "🤖 Copilot",
]

active_tab = st.segmented_control(
    "View",
    options=TAB_LABELS,
    default="Dashboard",
    label_visibility="collapsed",
    key="active_tab",
) or "Dashboard"

if active_tab == "Dashboard":
    _tab_dashboard.render(
        get_conn=get_conn,
        screener_ready=screener_ready,
        dates=dates,
        selected_date=selected_date,
    )
elif active_tab == "Screener":
    _tab_screener.render(
        get_conn=get_conn,
        dates=dates,
        selected_date=selected_date,
        strategy=strategy,
        asset_filter=asset_filter,
        min_score=min_score,
        min_change=min_change,
        min_rvol=min_rvol,
        setup_type_filter=setup_type_filter,
        setup_types=setup_types,
        fetch_early_signals=_fetch_early_signals,
    )
elif active_tab == "Lookup":
    _tab_lookup.render()
elif active_tab == "Backtest":
    _tab_backtest.render(get_conn=get_conn)
elif active_tab == "Advice":
    _tab_advice.render(get_conn=get_conn, strategy=strategy, dates=dates)
elif active_tab == "Options":
    _tab_options.render(get_conn=get_conn, dates=dates)
elif active_tab == "Learn":
    _tab_learn.render()
elif active_tab == "Metals":
    _tab_metals.render(regime_label=_regime_label, driver_tags=_driver_tags)
elif active_tab == "FOREX":
    _tab_forex.render()
elif active_tab == "Indexes":
    _tab_indexes.render()
elif active_tab == "Quantum":
    _tab_quantum.render()
elif active_tab == "Portfolio":
    _tab_portfolio.render(get_conn=get_conn, dates=dates)
elif active_tab == "💰 Smart Money":
    _tab_smart_money.render(get_conn=get_conn, screener_ready=screener_ready)
elif active_tab == "🤖 Copilot":
    _tab_copilot.render(DB_PATH)
