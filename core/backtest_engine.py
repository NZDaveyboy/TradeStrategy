"""
core/backtest_engine.py — Formal strategy backtest engine.

Wraps backtesting.py's Strategy / Backtest classes around screener signals.
All market logic lives here; entry points (backtest_v2.py, tests) stay thin.

Key design decisions:
  - Entry:  market order on next bar's open after signal date
  - Stop:   stop-loss from screener stop_loss column (passed per signal)
  - Target: take-profit price (optional, None → time exit only)
  - Exit:   forced close after max_hold_days trading bars (default 10)
  - Size:   fraction of equity per trade (default 0.95)
  - One position at a time — strategy skips new signals while in a trade
"""

from __future__ import annotations

import math

import pandas as pd
from backtesting import Backtest, Strategy

from providers.yfinance_provider import YFinanceProvider

_provider = YFinanceProvider()


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class ScreenerStrategy(Strategy):
    """
    Executes screener signals as market orders on the next bar's open.

    Parameters — injected via Backtest.run(**kwargs):
        signal_dates  dict  {date_str: {"stop": float|None, "target": float|None}}
                            Keys are YYYY-MM-DD strings matching the OHLCV index.
        max_hold_days int   Trading bars before forced time-based exit (default 10).
        size          float Fraction of available equity per trade (default 0.95).
    """

    signal_dates:  dict  = {}
    max_hold_days: int   = 10
    size:          float = 0.95

    def init(self):
        pass   # indicators would be added here in future extensions

    def next(self):
        today = str(self.data.index[-1].date())

        # ── Time-based exit ──────────────────────────────────────────────
        if self.position:
            trade     = self.trades[0]
            bars_held = int((self.data.index > trade.entry_time).sum())
            if bars_held >= self.max_hold_days:
                trade.close()
                return

        # ── Entry ────────────────────────────────────────────────────────
        if not self.position and today in self.signal_dates:
            sig    = self.signal_dates[today]
            stop   = sig.get("stop")
            target = sig.get("target")
            self.buy(
                size=self.size,
                sl=stop   if stop   and stop   > 0 else None,
                tp=target if target and target > 0 else None,
            )


# ---------------------------------------------------------------------------
# Data helper
# ---------------------------------------------------------------------------

def _fetch_data(ticker: str, signals: list[dict]) -> pd.DataFrame:
    """Fetch daily OHLCV covering all signal dates plus a forward buffer."""
    dates = sorted(s["date"] for s in signals)
    start = (pd.Timestamp(dates[0])  - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end   = (pd.Timestamp(dates[-1]) + pd.Timedelta(days=30)).strftime("%Y-%m-%d")

    data = _provider.get_ohlcv_range(ticker, start=start, end=end)
    if data.empty:
        return data

    # backtesting.py requires a tz-naive DatetimeIndex
    if data.index.tz is not None:
        data.index = data.index.tz_convert(None)

    return data[["Open", "High", "Low", "Close", "Volume"]]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_backtest(
    ticker: str,
    signals: list[dict],
    *,
    cash: float = 10_000,
    commission: float = 0.001,
    max_hold_days: int = 10,
    size: float = 0.95,
) -> dict:
    """
    Run a strategy backtest for a single ticker.

    signals is a list of dicts, one per screener run date:
        {
            "date":       str          YYYY-MM-DD (screener run_date)
            "stop":       float|None   stop-loss price (None → no hard stop)
            "target":     float|None   take-profit price (None → time exit only)
            # optional metadata — not used by the engine:
            "tradescore": float|None
            "setup_type": str|None
            "direction":  str|None
        }

    Returns a dict with keys:
        ticker, n_signals, n_trades,
        return_pct, sharpe, max_drawdown, win_rate, avg_trade_pct,
        trades (pd.DataFrame|None), equity_curve (pd.DataFrame|None),
        error (str|None)
    """
    base: dict = {
        "ticker":        ticker,
        "n_signals":     len(signals),
        "n_trades":      0,
        "return_pct":    float("nan"),
        "sharpe":        float("nan"),
        "max_drawdown":  float("nan"),
        "win_rate":      float("nan"),
        "avg_trade_pct": float("nan"),
        "trades":        None,
        "equity_curve":  None,
        "error":         None,
    }

    if not signals:
        base["error"] = "no signals"
        return base

    try:
        data = _fetch_data(ticker, signals)
    except Exception as exc:
        base["error"] = f"data fetch failed: {exc}"
        return base

    if data.empty or len(data) < 2:
        base["error"] = "insufficient price data"
        return base

    signal_dates = {
        s["date"]: {"stop": s.get("stop"), "target": s.get("target")}
        for s in signals
    }

    try:
        bt = Backtest(
            data,
            ScreenerStrategy,
            cash=cash,
            commission=commission,
            trade_on_close=False,   # enter next bar's open — no lookahead bias
            finalize_trades=True,   # close any open trade at end so it appears in stats
        )
        stats = bt.run(
            signal_dates=signal_dates,
            max_hold_days=max_hold_days,
            size=size,
        )
    except Exception as exc:
        base["error"] = f"backtest failed: {exc}"
        return base

    def _f(v) -> float:
        """Return float, converting NaN to nan (not None) for uniform handling."""
        try:
            f = float(v)
            return f
        except (TypeError, ValueError):
            return float("nan")

    return {
        **base,
        "n_trades":      int(stats["# Trades"]),
        "return_pct":    _f(stats["Return [%]"]),
        "sharpe":        _f(stats.get("Sharpe Ratio")),
        "max_drawdown":  _f(stats["Max. Drawdown [%]"]),
        "win_rate":      _f(stats.get("Win Rate [%]")),
        "avg_trade_pct": _f(stats.get("Avg. Trade [%]")),
        "trades":        stats["_trades"],
        "equity_curve":  stats["_equity_curve"],
        "error":         None,
    }
