"""
src/backtest.py — Performance + risk metrics over an index time series.

Computes the standard analytics a trader cares about:
  - Total return, CAGR, annualised volatility, Sharpe (rf=0)
  - Max drawdown + drawdown duration
  - Per-constituent contribution (best/worst names by $ contribution to index)
  - Rolling correlation with benchmarks
  - Static correlation matrix

All metrics annualise at 252 trading days/year (the industry default for
US equity backtests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from core.quantum.index import IndexResult


TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestStats:
    total_return_pct:    float
    cagr_pct:            float
    annual_vol_pct:      float
    sharpe:              float
    max_drawdown_pct:    float
    max_drawdown_days:   int
    n_days:              int
    start_date:          str
    end_date:            str

    def to_dict(self) -> dict:
        return {
            "Total return %":      round(self.total_return_pct, 2),
            "CAGR %":              round(self.cagr_pct, 2),
            "Annual vol %":        round(self.annual_vol_pct, 2),
            "Sharpe (rf=0)":       round(self.sharpe, 3),
            "Max drawdown %":      round(self.max_drawdown_pct, 2),
            "Max DD duration (d)": self.max_drawdown_days,
            "Trading days":        self.n_days,
            "Start":               self.start_date,
            "End":                 self.end_date,
        }


# ---------------------------------------------------------------------------
# Headline stats
# ---------------------------------------------------------------------------

def compute_stats(levels: pd.Series) -> BacktestStats:
    """Annualised headline stats from an index level series (starts at 100)."""
    if levels.empty or len(levels) < 2:
        return BacktestStats(0, 0, 0, 0, 0, 0, 0, "", "")

    rets = levels.pct_change().dropna()
    total_return = (levels.iloc[-1] / levels.iloc[0] - 1.0) * 100.0
    years = max(len(rets) / TRADING_DAYS_PER_YEAR, 1e-9)
    cagr = ((levels.iloc[-1] / levels.iloc[0]) ** (1 / years) - 1.0) * 100.0
    vol = rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0
    sharpe = (rets.mean() * TRADING_DAYS_PER_YEAR) / (rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if rets.std() > 0 else 0.0

    dd = drawdown_series(levels)
    max_dd = dd.min() if not dd.empty else 0.0
    max_dd_dur = _max_drawdown_duration(dd)

    return BacktestStats(
        total_return_pct=total_return,
        cagr_pct=cagr,
        annual_vol_pct=vol,
        sharpe=sharpe,
        max_drawdown_pct=max_dd,
        max_drawdown_days=max_dd_dur,
        n_days=len(rets),
        start_date=str(levels.index[0].date()),
        end_date=str(levels.index[-1].date()),
    )


def drawdown_series(levels: pd.Series) -> pd.Series:
    """Drawdown % at each point. Always ≤ 0."""
    if levels.empty:
        return pd.Series(dtype=float)
    roll_max = levels.cummax()
    return (levels / roll_max - 1.0) * 100.0


def _max_drawdown_duration(dd: pd.Series) -> int:
    """Longest consecutive run of days the index was in drawdown."""
    if dd.empty:
        return 0
    in_dd = dd < -1e-9
    if not in_dd.any():
        return 0
    max_run = run = 0
    for v in in_dd:
        run = run + 1 if v else 0
        max_run = max(max_run, run)
    return max_run


# ---------------------------------------------------------------------------
# Contributor analysis
# ---------------------------------------------------------------------------

def full_attribution(
    result: IndexResult,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Per-ticker attribution, all rows, sorted desc by Contribution %.

    Three different "return" columns surface the math honestly:

      Total return %    Full-period price return: first-valid → last-day.
      Held return %     Price return *during the period the index actually held
                        the position* (first held date → last held date). This
                        is the return the index had real exposure to.
      Contribution %    Σ_t( weight_{i,t} × return_{i,t} ) × 100 — the linear
                        (arithmetic) Brinson attribution used in industry
                        portfolio reports. For highly volatile names this
                        CAN DIVERGE from intuition: a stock with a hugely
                        negative Held return can show positive Contribution
                        because arithmetic returns sum differently than
                        compound returns. This is "volatility pumping" /
                        arithmetic attribution drift — not a bug, but worth
                        cross-checking against Held return %.

    Contribution share % is each ticker's |Contribution %| as a percent of
    the gross sum. Sums to 100% across all tickers.
    """
    cols = [
        "Ticker", "Total return %", "Held return %", "Held days",
        "Avg weight %", "Contribution %", "Contribution share %",
    ]
    if result.levels.empty or result.weights.empty:
        return pd.DataFrame(columns=cols)

    rets = prices.pct_change()
    rows = []
    for t in result.weights.columns:
        if t not in rets.columns:
            continue
        w_series = result.weights[t].reindex(result.levels.index).fillna(0.0)
        r_series = rets[t].reindex(result.levels.index).fillna(0.0)
        contrib_total_pct = float((w_series * r_series).sum() * 100.0)
        avg_weight = float(w_series.mean() * 100.0)

        # Full-window price return
        prices_t = prices[t].reindex(result.levels.index).dropna()
        if len(prices_t) >= 2:
            t_ret = float((prices_t.iloc[-1] / prices_t.iloc[0] - 1.0) * 100.0)
        else:
            t_ret = float("nan")

        # Held-period return: only over days the index had non-zero weight
        held_dates = w_series[w_series > 1e-9].index
        held_days = int(len(held_dates))
        held_ret = float("nan")
        if held_days >= 2:
            try:
                p_first = float(prices.at[held_dates[0], t])
                p_last  = float(prices.at[held_dates[-1], t])
                if pd.notna(p_first) and pd.notna(p_last) and p_first > 0:
                    held_ret = (p_last / p_first - 1.0) * 100.0
            except (KeyError, ValueError):
                pass

        rows.append({
            "Ticker":         t,
            "Total return %": round(t_ret, 2)    if pd.notna(t_ret)    else float("nan"),
            "Held return %":  round(held_ret, 2) if pd.notna(held_ret) else float("nan"),
            "Held days":      held_days,
            "Avg weight %":   round(avg_weight, 2),
            "Contribution %": round(contrib_total_pct, 2),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        df["Contribution share %"] = []
        return df[cols]

    # Share is computed on contribution magnitude so a name that detracts a
    # lot is recognized as "concentration of impact", not zero. Drop NaN
    # contributions before computing.
    total_abs = df["Contribution %"].abs().sum()
    if total_abs > 0:
        df["Contribution share %"] = (df["Contribution %"].abs() / total_abs * 100).round(2)
    else:
        df["Contribution share %"] = 0.0

    return df.sort_values("Contribution %", ascending=False).reset_index(drop=True)[cols]


def contributor_analysis(
    result: IndexResult,
    prices: pd.DataFrame,
    top_n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper — top_n best and worst contributors.

    Built on top of `full_attribution`. Returns (best_df, worst_df).
    """
    df = full_attribution(result, prices)
    if df.empty:
        return df, df
    # Drop the share column from the convenience output (it's noisy here)
    df = df[["Ticker", "Total return %", "Avg weight %", "Contribution %"]]
    best  = df.head(top_n).reset_index(drop=True)
    worst = df.tail(top_n).iloc[::-1].reset_index(drop=True)
    return best, worst


# ---------------------------------------------------------------------------
# Signal classifier — rules-based BUY / WATCH / HOLD / SELL per constituent
# ---------------------------------------------------------------------------

SIGNAL_BUY   = "BUY"
SIGNAL_WATCH = "WATCH"
SIGNAL_HOLD  = "HOLD"
SIGNAL_SELL  = "SELL"

# Sort order when grouping by signal for display
_SIGNAL_ORDER = {SIGNAL_BUY: 0, SIGNAL_WATCH: 1, SIGNAL_HOLD: 2, SIGNAL_SELL: 3}


def _rsi_14(s: pd.Series, period: int = 14) -> float:
    """Wilder-style RSI computed from a daily close series.

    Edge cases:
      - Insufficient data (< period + 1 rows): returns 50 (neutral)
      - Strictly monotonic uptrend (loss = 0, gain > 0): returns 100 (max)
      - Strictly monotonic downtrend (gain = 0, loss > 0): returns 0
      - Flat series (gain = 0, loss = 0): returns 50
    """
    if s is None or len(s) < period + 1:
        return 50.0
    delta = s.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()

    last_gain = float(gain.iloc[-1]) if pd.notna(gain.iloc[-1]) else 0.0
    last_loss = float(loss.iloc[-1]) if pd.notna(loss.iloc[-1]) else 0.0

    if last_loss == 0.0:
        return 100.0 if last_gain > 0.0 else 50.0
    rs = last_gain / last_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def _classify_one(
    *,
    score: float,
    held_return: float,
    ret_1m: float,
    ret_3m: float,
    rsi: float,
) -> tuple[str, str]:
    """Apply the BUY/WATCH/HOLD/SELL rules to one constituent. Returns (signal, reason).

    Rules evaluated in priority order:

      1. Capital impairment with no recent rescue → SELL
      2. Significant drawdown + still falling → SELL
      3. Very strong 1-month momentum + not yet exhausted → BUY
      4. High conviction (≥2.0) + positive momentum → BUY
      5. Sustained 3-month momentum → BUY
      6. Overbought (RSI > 75) → WATCH (wait for pullback)
      7. Top conviction in a flat tape → WATCH (wait for setup)
      8. Default → HOLD
    """
    # 1. Major capital impairment, no recovery yet
    if held_return <= -50.0 and ret_1m < 5.0:
        return SIGNAL_SELL, f"Held return {held_return:+.0f}% — significant impairment, no recent recovery"

    # 2. Already underwater + still falling
    if held_return < -25.0 and ret_1m < -5.0:
        return SIGNAL_SELL, f"Down {held_return:+.0f}% and still falling ({ret_1m:+.0f}% in 1m)"

    # 3. Very strong recent momentum (and not extremely overbought)
    if ret_1m > 15.0 and rsi < 80.0:
        return SIGNAL_BUY, f"Strong 1-month momentum {ret_1m:+.0f}%"

    # 4. High conviction with positive momentum
    if score >= 2.0 and ret_1m > 3.0 and rsi < 75.0:
        return SIGNAL_BUY, f"High conviction ({score:.1f}) with positive momentum ({ret_1m:+.0f}% 1m)"

    # 5. Sustained 3-month momentum
    if ret_3m > 25.0 and ret_1m >= 0.0 and rsi < 80.0:
        return SIGNAL_BUY, f"Sustained 3-month momentum {ret_3m:+.0f}%"

    # 6. Overbought — wait for pullback
    if rsi > 75.0:
        return SIGNAL_WATCH, f"Overbought (RSI {rsi:.0f}) — wait for pullback"

    # 7. Top conviction but no momentum yet
    if score >= 2.5 and abs(ret_1m) < 5.0:
        return SIGNAL_WATCH, f"Top conviction ({score:.1f}), no momentum signal yet"

    # 8. Default
    return SIGNAL_HOLD, "Position in line — no action signal"


def classify_constituents(
    result: IndexResult,
    prices: pd.DataFrame,
    universe,
) -> pd.DataFrame:
    """Per-constituent BUY/WATCH/HOLD/SELL classification.

    Uses:
      - Conviction score (from `compute_final_scores`)
      - Held return % (price return over index-held period)
      - 1-month and 3-month price return (~22 and ~66 trading days)
      - RSI(14)
      - Current weight

    Returns a DataFrame with columns:
      Ticker, Company, Category, Signal, Reason, Conviction,
      Weight %, Held return %, 1m %, 3m %, RSI(14)

    Sorted by Signal priority (BUY → WATCH → HOLD → SELL), then by
    descending Weight % within each bucket.
    """
    from core.quantum.scoring import compute_final_scores

    cols = [
        "Ticker", "Company", "Category", "Signal", "Reason",
        "Conviction", "Weight %", "Held return %", "1m %", "3m %", "RSI(14)",
    ]
    if result.levels.empty or result.weights.empty or not result.constituents:
        return pd.DataFrame(columns=cols)

    score_map = compute_final_scores(result.constituents)
    name_map  = {c.ticker: c.company_name for c in result.constituents}
    cat_map   = {c.ticker: c.category     for c in result.constituents}

    rows = []
    for c in result.constituents:
        t = c.ticker
        if t not in prices.columns:
            continue
        p = prices[t].dropna()
        if p.empty:
            continue

        # Recent price moves
        last  = float(p.iloc[-1])
        ret_1m = float((last / float(p.iloc[-22]) - 1.0) * 100.0) if len(p) >= 22 else 0.0
        ret_3m = float((last / float(p.iloc[-66]) - 1.0) * 100.0) if len(p) >= 66 else 0.0
        rsi_v  = _rsi_14(p)

        # Held return over index hold period
        w_series = result.weights.get(t)
        cur_w   = 0.0
        held_ret = 0.0
        if w_series is not None and not w_series.empty:
            cur_w = float(w_series.iloc[-1] * 100.0)
            held = w_series[w_series > 1e-9]
            if len(held) >= 2:
                try:
                    p_first = float(prices.at[held.index[0], t])
                    p_last  = float(prices.at[held.index[-1], t])
                    if pd.notna(p_first) and pd.notna(p_last) and p_first > 0:
                        held_ret = (p_last / p_first - 1.0) * 100.0
                except (KeyError, ValueError):
                    pass

        score = float(score_map.get(t, 0.0))
        signal, reason = _classify_one(
            score=score,
            held_return=held_ret,
            ret_1m=ret_1m,
            ret_3m=ret_3m,
            rsi=rsi_v,
        )

        rows.append({
            "Ticker":         t,
            "Company":        name_map.get(t, t),
            "Category":       cat_map.get(t, "").replace("_", " ").title(),
            "Signal":         signal,
            "Reason":         reason,
            "Conviction":     round(score, 2),
            "Weight %":       round(cur_w, 2),
            "Held return %":  round(held_ret, 2),
            "1m %":           round(ret_1m, 2),
            "3m %":           round(ret_3m, 2),
            "RSI(14)":        round(rsi_v, 1),
        })

    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df

    # Sort by signal priority, then weight desc
    df["_sig_order"] = df["Signal"].map(_SIGNAL_ORDER).fillna(99)
    df = df.sort_values(["_sig_order", "Weight %"], ascending=[True, False]).drop(columns=["_sig_order"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Concentration metrics — "how dependent is the index on a few names?"
# ---------------------------------------------------------------------------

def concentration_metrics(
    result: IndexResult,
    prices: pd.DataFrame,
) -> dict:
    """Quantify how concentrated the index return is in a few names.

    Returns a dict with:
      n_constituents      Number of constituents that contributed (any non-zero)
      top1_share_pct      Share of total |contribution| from the top 1 name
      top3_share_pct      Same, top 3 names
      top5_share_pct      Same, top 5 names
      hhi                 Herfindahl-Hirschman Index on contribution shares
                          (squared shares as fractions, range 0..1).
                          <0.15 = diversified, 0.15-0.25 = moderate, >0.25 = concentrated.
      diversification_label  Plain-English label of the HHI bucket

    HHI is computed on the *gross* contribution shares (absolute values) so
    a single name that drove a huge negative also counts as concentration.
    """
    attrib = full_attribution(result, prices)
    if attrib.empty:
        return {
            "n_constituents":          0,
            "top1_share_pct":          0.0,
            "top3_share_pct":          0.0,
            "top5_share_pct":          0.0,
            "hhi":                     0.0,
            "diversification_label":   "—",
        }

    shares = attrib["Contribution share %"].to_numpy() / 100.0   # fraction
    # Sort descending so top-N is contiguous at the start
    shares_sorted = sorted(shares, reverse=True)

    top1 = shares_sorted[0] * 100 if len(shares_sorted) >= 1 else 0.0
    top3 = sum(shares_sorted[:3]) * 100 if len(shares_sorted) >= 1 else 0.0
    top5 = sum(shares_sorted[:5]) * 100 if len(shares_sorted) >= 1 else 0.0
    hhi  = float(sum(s ** 2 for s in shares))

    if hhi < 0.15:
        label = "Diversified"
    elif hhi < 0.25:
        label = "Moderately concentrated"
    else:
        label = "Highly concentrated"

    n_contrib = int((attrib["Contribution %"].abs() > 0).sum())

    return {
        "n_constituents":         n_contrib,
        "top1_share_pct":         round(top1, 2),
        "top3_share_pct":         round(top3, 2),
        "top5_share_pct":         round(top5, 2),
        "hhi":                    round(hhi, 4),
        "diversification_label":  label,
    }


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def correlation_matrix(
    index_levels: pd.Series,
    benchmark_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Pearson correlation of daily returns: index vs each benchmark column.

    `benchmark_prices` should have one column per benchmark ticker, with a
    DatetimeIndex aligned (or alignable) to `index_levels`.
    """
    if index_levels.empty or benchmark_prices.empty:
        return pd.DataFrame()
    idx_rets = index_levels.pct_change()
    bench_rets = benchmark_prices.pct_change()
    combined = pd.concat(
        [idx_rets.rename("INDEX"), bench_rets],
        axis=1,
    ).dropna()
    if combined.empty:
        return pd.DataFrame()
    return combined.corr().round(3)


def rolling_correlation(
    index_levels: pd.Series,
    benchmark_prices: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """60-day rolling correlation of the index vs each benchmark."""
    if index_levels.empty or benchmark_prices.empty:
        return pd.DataFrame()
    idx_rets = index_levels.pct_change().dropna()
    out: dict[str, pd.Series] = {}
    for col in benchmark_prices.columns:
        br = benchmark_prices[col].pct_change()
        aligned = pd.concat([idx_rets, br], axis=1).dropna()
        if len(aligned) < window:
            continue
        out[col] = aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Public convenience: full backtest bundle
# ---------------------------------------------------------------------------

def run_full_backtest(
    result: IndexResult,
    universe_prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame | None = None,
) -> dict:
    """Convenience wrapper — produce every metric in one call.

    Returns a dict of named pieces:
      stats               BacktestStats
      drawdown            pd.Series of drawdown %
      best_contributors   pd.DataFrame
      worst_contributors  pd.DataFrame
      corr_matrix         pd.DataFrame (or None)
      rolling_corr        pd.DataFrame (or None)
    """
    out: dict = {
        "stats":              compute_stats(result.levels),
        "drawdown":           drawdown_series(result.levels),
    }
    best, worst = contributor_analysis(result, universe_prices)
    out["best_contributors"]  = best
    out["worst_contributors"] = worst
    out["full_attribution"]   = full_attribution(result, universe_prices)
    out["concentration"]      = concentration_metrics(result, universe_prices)
    if benchmark_prices is not None and not benchmark_prices.empty:
        out["corr_matrix"]   = correlation_matrix(result.levels, benchmark_prices)
        out["rolling_corr"]  = rolling_correlation(result.levels, benchmark_prices)
    else:
        out["corr_matrix"]   = None
        out["rolling_corr"]  = None
    return out
