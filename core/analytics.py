"""
core/analytics.py — Portfolio-level analytics for backtest_v2 results.

Reads from the backtest_v2 table (created by backtest_v2.py), joins with
results for setup_type / tradescore, then returns DataFrames and dicts
ready for the Streamlit UI.

Sharpe and Sortino are properly annualised:

    annualised_Sharpe  = ((mean_per_period − rf_per_period) / std_per_period) × √N
    annualised_Sortino = ((mean_per_period − rf_per_period) / downside_std)   × √N

Where `mean_per_period` is the mean of the per-bet returns from the equity
curve, `N` is the assumed deployment frequency (`periods_per_year`, default
252 = one bet per trading day), and rf is the risk-free rate prorated per
period. `downside_std` is the standard deviation of returns below zero
(MAR = 0).

Two Sharpes are returned:
  - `sharpe`              — annualised Sharpe of the sequence of bet returns
  - `avg_strategy_sharpe` — mean of the already-annualised per-ticker
                            Sharpes saved by backtest_v2.py (via
                            backtesting.py's internal annualisation)

`portfolio_stats` also returns `periods_per_year` and `risk_free_rate` so
the UI can display the assumptions used.

Max drawdown comes from the cumulative equity curve via quantstats.
"""

from __future__ import annotations

import math
import os

import pandas as pd

from core.db import get_connection
import quantstats as qs

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "screener.db")

SCORE_BUCKETS = [(0, 40), (40, 60), (60, 80), (80, 100)]
DEFAULT_PERIODS_PER_YEAR = 252.0   # one bet per trading day — documented assumption
DEFAULT_RISK_FREE_RATE   = 0.045    # 4.5% annual, matches RISK_FREE_L in app.py


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_v2_data(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Load backtest_v2 joined with aggregated results for setup_type / tradescore.

    Filters to rows where error IS NULL and n_trades > 0. Returns an empty
    DataFrame if the backtest_v2 table does not exist yet.

    Extra columns added on join:
        setup_type      — modal setup_type for that ticker across all results rows
        avg_tradescore  — mean tradescore for that ticker across all results rows
    """
    if not os.path.exists(db_path):
        return pd.DataFrame()

    conn = get_connection(db_path)
    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='backtest_v2'"
        ).fetchone()
        if not exists:
            return pd.DataFrame()

        v2 = pd.read_sql("SELECT * FROM backtest_v2", conn)

        has_results = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='results'"
        ).fetchone()
        if has_results:
            res = pd.read_sql(
                "SELECT ticker, setup_type, tradescore FROM results",
                conn,
            )
            if not res.empty:
                def _mode(x):
                    m = x.dropna().mode()
                    return m.iloc[0] if not m.empty else None

                res_agg = (
                    res.groupby("ticker")
                    .agg(
                        setup_type    =("setup_type",  _mode),
                        avg_tradescore=("tradescore",  "mean"),
                    )
                    .reset_index()
                )
                v2 = v2.merge(res_agg, on="ticker", how="left")
            else:
                v2["setup_type"]     = None
                v2["avg_tradescore"] = float("nan")
        else:
            v2["setup_type"]     = None
            v2["avg_tradescore"] = float("nan")

    finally:
        conn.close()

    # Keep only usable rows
    v2 = v2[v2["error"].isna() & (v2["n_trades"] > 0)].copy()
    for col in ("return_pct", "win_rate", "sharpe", "max_drawdown", "avg_trade_pct"):
        if col in v2.columns:
            v2[col] = pd.to_numeric(v2[col], errors="coerce")
    v2 = v2.dropna(subset=["return_pct"])
    return v2.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _returns_series(df: pd.DataFrame) -> pd.Series:
    """Per-ticker return_pct as decimal fractions, ordered by run_at if present."""
    if "run_at" in df.columns:
        df = df.sort_values("run_at")
    return (df["return_pct"] / 100.0).reset_index(drop=True)


def _safe_float(v) -> float:
    try:
        f = float(v)
        return float("nan") if math.isnan(f) else f
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def equity_curve(df: pd.DataFrame) -> pd.Series:
    """
    Cumulative equity curve from per-ticker return_pct, starting at 1.0.

    Ordered by run_at when that column is present so the curve follows
    chronological backtest order.
    """
    if df.empty:
        return pd.Series(dtype=float)
    rets  = _returns_series(df)
    curve = (1 + rets).cumprod()
    return curve.reset_index(drop=True)


def dated_returns_series(
    df: pd.DataFrame,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> pd.Series:
    """
    Per-bet returns as a date-indexed Series suitable for time-series analytics.

    If `run_at` timestamps span more than 30 days, we use the real timestamps.
    Otherwise we synthesise one observation per trading day, ending today, so
    monthly aggregations and tearsheet plots have a time axis to bucket on.

    The annualisation factor in portfolio_stats already assumes one bet per
    period — this function makes that assumption visible in the time index
    rather than implicit in the math.
    """
    if df.empty:
        return pd.Series(dtype=float, name="returns")

    rets = _returns_series(df)
    if rets.empty:
        return pd.Series(dtype=float, name="returns")

    # Try to use real timestamps when they span a meaningful range
    if "run_at" in df.columns:
        if "run_at" in df.columns:
            sorted_df = df.sort_values("run_at")
        else:
            sorted_df = df
        ts = pd.to_datetime(sorted_df["run_at"], errors="coerce")
        if ts.notna().sum() >= 2:
            valid_ts = ts.dropna()
            span_days = (valid_ts.max() - valid_ts.min()).total_seconds() / 86400.0
            if span_days > 30:
                rets_ordered = _returns_series(sorted_df)
                # Use timestamps directly; collisions are fine (heatmap groups
                # by month). Build a fresh DatetimeIndex (named "date") so
                # downstream groupby([index.year, index.month]) doesn't
                # collide with the source column name "run_at".
                idx = pd.DatetimeIndex(
                    pd.to_datetime(sorted_df["run_at"], errors="coerce").values,
                    name="date",
                )
                return pd.Series(
                    rets_ordered.values,
                    index=idx,
                    name="returns",
                ).dropna()

    # Synthetic fallback: one bet per trading day, ending today
    today = pd.Timestamp.today().normalize()
    bdays = pd.bdate_range(end=today, periods=len(rets))
    return pd.Series(rets.values, index=bdays, name="returns")


def drawdown_series(df: pd.DataFrame) -> pd.Series:
    """
    Drawdown % at each point along the equity curve.

    Drawdown_t = (equity_t / cummax(equity)_t) − 1, expressed in percent.
    Always ≤ 0. Used for the drawdown chart in the Backtest tab.
    """
    if df.empty:
        return pd.Series(dtype=float, name="drawdown_pct")
    curve = equity_curve(df)
    if curve.empty:
        return curve.rename("drawdown_pct")
    roll_max = curve.cummax()
    dd = ((curve / roll_max) - 1.0) * 100.0
    return dd.rename("drawdown_pct")


def monthly_returns_table(
    df: pd.DataFrame,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> pd.DataFrame:
    """
    Aggregate returns by calendar (Year, Month) for the QuantStats-style heatmap.

    Returns a DataFrame with columns: Year, Month (1-12), Month_label (Jan-Dec),
    Return (%, compounded within the month), N_obs.

    Built from `dated_returns_series` so synthetic-time-axis backtests still
    produce a meaningful heatmap (last N bets bucketed by their synthesised
    months).
    """
    if df.empty:
        return pd.DataFrame(columns=["Year", "Month", "Month_label", "Return", "N_obs"])

    s = dated_returns_series(df, periods_per_year)
    if s.empty:
        return pd.DataFrame(columns=["Year", "Month", "Month_label", "Return", "N_obs"])

    # Compound returns within each (year, month). Use explicit pd.Series
    # group-keys with names so reset_index produces clean column names.
    yr = pd.Series(s.index.year,  index=s.index, name="Year")
    mo = pd.Series(s.index.month, index=s.index, name="Month")
    grouped = (1.0 + s).groupby([yr, mo])
    monthly = grouped.prod() - 1.0
    counts  = grouped.size()

    out = monthly.reset_index().rename(columns={0: "Return", monthly.name or "returns": "Return"})
    out["N_obs"]       = counts.values
    out["Return"]      = out["Return"] * 100.0
    out["Month_label"] = out["Month"].apply(
        lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"][int(m) - 1]
    )
    return out[["Year", "Month", "Month_label", "Return", "N_obs"]]


def quantstats_tearsheet_html(
    df: pd.DataFrame,
    title: str = "TradeStrategy Backtest",
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> bytes | None:
    """
    Generate a full QuantStats HTML tearsheet from the backtest results.

    Builds a date-indexed returns Series from `df`, runs `quantstats.reports.html`
    against it, reads the resulting file, and returns the bytes. Returns None
    on failure or when there are too few observations to render meaningfully
    (< 10).

    The caller (Backtest tab) uses this with `st.download_button` to offer
    the report as a download.
    """
    if df.empty:
        return None

    import os
    import tempfile

    s = dated_returns_series(df, periods_per_year)
    if s.empty or len(s) < 10:
        return None

    # quantstats writes to a file path; we read it back as bytes.
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as tf:
            tmp_path = tf.name
    except Exception:
        return None

    try:
        qs.reports.html(s, output=tmp_path, title=title)
        try:
            with open(tmp_path, "rb") as f:
                return f.read()
        except Exception:
            return None
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def portfolio_stats(
    df: pd.DataFrame,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
    risk_free_rate:   float = DEFAULT_RISK_FREE_RATE,
) -> dict:
    """
    Aggregate portfolio-level stats from per-ticker backtest_v2 rows.

    Parameters
    ----------
    df : DataFrame
        Output of `load_v2_data()`. One row per ticker.
    periods_per_year : float
        Assumed deployment frequency for annualisation. 252 = one bet
        per trading day (default). Override to match how often the
        strategy would actually be deployed in production.
    risk_free_rate : float
        Annual risk-free rate (e.g. 0.045 = 4.5%). Prorated per period
        for the excess-return calculation.

    Returns
    -------
    dict with:
        total_trades, n_tickers,
        win_rate (%), avg_return (%), expectancy (%),
        sharpe              — annualised Sharpe of bet-return sequence (excess)
        sortino             — annualised Sortino (excess / downside std)
        max_drawdown (%),
        avg_strategy_sharpe — mean of saved per-ticker Sharpes (already
                              annualised by backtesting.py); float("nan")
                              if the column is missing or all-NaN
        periods_per_year    — the assumption used (returned for transparency)
        risk_free_rate      — the assumption used
    """
    base = {
        "total_trades":         0,
        "n_tickers":            0,
        "win_rate":             float("nan"),
        "avg_return":           float("nan"),
        "expectancy":           float("nan"),
        "sharpe":               float("nan"),
        "sortino":              float("nan"),
        "max_drawdown":         float("nan"),
        "avg_strategy_sharpe":  float("nan"),
        "periods_per_year":     periods_per_year,
        "risk_free_rate":       risk_free_rate,
    }
    if df.empty:
        return base

    returns = _returns_series(df)   # per-bet returns as decimals
    curve   = equity_curve(df)

    std     = returns.std()
    neg     = returns[returns < 0]
    neg_std = neg.std() if len(neg) > 1 else float("nan")

    # ── Annualised Sharpe / Sortino from excess returns ────────────────────
    sqrt_n            = math.sqrt(periods_per_year) if periods_per_year > 0 else float("nan")
    rf_per_period     = risk_free_rate / periods_per_year if periods_per_year > 0 else 0.0
    excess_mean       = returns.mean() - rf_per_period

    if std > 0 and not math.isnan(sqrt_n):
        sharpe = _safe_float(excess_mean / std * sqrt_n)
    else:
        sharpe = float("nan")

    if (not math.isnan(neg_std)) and neg_std > 0 and not math.isnan(sqrt_n):
        sortino = _safe_float(excess_mean / neg_std * sqrt_n)
    else:
        sortino = float("nan")

    # ── Avg of per-ticker Sharpes (already annualised by backtesting.py) ──
    if "sharpe" in df.columns and df["sharpe"].notna().any():
        avg_strategy_sharpe = _safe_float(
            pd.to_numeric(df["sharpe"], errors="coerce").dropna().mean()
        )
    else:
        avg_strategy_sharpe = float("nan")

    # ── Expectancy: avg_win × win_rate + avg_loss × loss_rate ─────────────
    wins   = returns[returns > 0]
    losses = returns[returns < 0]
    wr     = len(wins) / len(returns) if len(returns) else 0.0
    lr     = 1.0 - wr
    avg_w  = wins.mean()   if len(wins)   else 0.0
    avg_l  = losses.mean() if len(losses) else 0.0
    expect = avg_w * wr + avg_l * lr

    # ── Max drawdown via quantstats (takes price/equity series) ────────────
    try:
        max_dd = _safe_float(qs.stats.max_drawdown(curve)) * 100
    except Exception:
        roll_max = curve.cummax()
        dd       = (curve / roll_max) - 1
        max_dd   = _safe_float(dd.min()) * 100

    return {
        "total_trades":         int(df["n_trades"].sum()),
        "n_tickers":            len(df),
        "win_rate":             wr * 100,
        "avg_return":           _safe_float(returns.mean()) * 100,
        "expectancy":           expect * 100,
        "sharpe":               sharpe,
        "sortino":              sortino,
        "max_drawdown":         max_dd,
        "avg_strategy_sharpe":  avg_strategy_sharpe,
        "periods_per_year":     periods_per_year,
        "risk_free_rate":       risk_free_rate,
    }


def win_rate_by_setup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Win rate and avg return grouped by setup_type.

    Returns a DataFrame with columns:
        Setup Type, Tickers, Avg Return %, Win Rate %, Total Trades
    Sorted by Win Rate % descending. Returns empty DataFrame if no
    setup_type data is present.
    """
    if df.empty or "setup_type" not in df.columns:
        return pd.DataFrame()

    sub = df.dropna(subset=["setup_type"])
    if sub.empty:
        return pd.DataFrame()

    result = (
        sub.groupby("setup_type")
        .agg(
            Tickers      =("ticker",     "count"),
            avg_return   =("return_pct", "mean"),
            win_rate     =("return_pct", lambda x: (x > 0).mean() * 100),
            total_trades =("n_trades",   "sum"),
        )
        .round(1)
        .reset_index()
    )
    result = result.rename(columns={
        "setup_type":  "Setup Type",
        "avg_return":  "Avg Return %",
        "win_rate":    "Win Rate %",
        "total_trades":"Total Trades",
    })
    return result.sort_values("Win Rate %", ascending=False).reset_index(drop=True)


def performance_by_score_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Avg return and win rate for tradescore buckets: 0-40, 40-60, 60-80, 80-100.

    Uses avg_tradescore joined from the results table. Returns empty DataFrame
    if tradescore data is not present.
    """
    if df.empty or "avg_tradescore" not in df.columns:
        return pd.DataFrame()

    sub = df.dropna(subset=["avg_tradescore"]).copy()
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for lo, hi in SCORE_BUCKETS:
        mask = (sub["avg_tradescore"] >= lo) & (sub["avg_tradescore"] < hi)
        grp  = sub[mask]
        if grp.empty:
            continue
        avg_trade = (
            round(float(grp["avg_trade_pct"].mean()), 1)
            if "avg_trade_pct" in grp.columns and grp["avg_trade_pct"].notna().any()
            else float("nan")
        )
        rows.append({
            "Score Bucket":  f"{lo}\u2013{hi}",
            "Tickers":       len(grp),
            "Total Trades":  int(grp["n_trades"].sum()),
            "Avg Return %":  round(float(grp["return_pct"].mean()), 1),
            "Win Rate %":    round(float((grp["return_pct"] > 0).mean() * 100), 1),
            "Avg Trade %":   avg_trade,
        })
    return pd.DataFrame(rows)
