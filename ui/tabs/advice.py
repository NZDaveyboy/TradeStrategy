"""
ui/tabs/advice.py — Recommendation engine card view.

For each top-scoring screener row, builds a structured Recommendation
(entry / stop / target / strategy / rationale / warnings) and renders
it as a card. Optionally pulls live catalyst signals from `core.catalysts`.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import streamlit as st

from core.recommendations import STRATEGY_DISPLAY, build_recommendation
from ui.data import _provider as _provider, fetch_company_info as get_company_info, fetch_nzdusd


def render(get_conn: Callable, strategy: str, dates: list) -> None:
    """Render the Advice tab.

    Args:
      get_conn: callable returning a DB connection (no args).
      strategy: sidebar selection (\"All\" or a strategy name).
      dates:    list of available run_dates from the screener DB.
    """

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _format_rec_markdown(rec) -> str:
        """Format a Recommendation as Advice-tab markdown."""
        if rec.setup_type == "crypto_no_options":
            return "_Options are not available for crypto — use directional position sizing above._"

        if not rec.is_actionable or rec.strategy_name == "wait":
            header = {
                "extended":           "**Extended move — wait for a pullback**",
                "pullback_candidate": "**Stop too wide — wait for a better entry**",
                "liquidity_concern":  "**Low liquidity — avoid options**",
            }.get(rec.setup_type, "**No clear directional edge — wait**")
            lines = [header, "", rec.rationale]
            if rec.invalidation_price:
                lines.append(
                    f"\n_Watch: pullback toward ${rec.invalidation_price:.2f} "
                    "(EMA20 − ½ ATR) before reassessing._"
                )
            if rec.warnings:
                lines.append("")
                for w in rec.warnings:
                    lines.append(f"⚠️ {w}")
            lines.append("\n_Options tab → Recommendations for exact contract pricing._")
            return "\n".join(lines)

        display   = STRATEGY_DISPLAY.get(rec.strategy_name, rec.strategy_name)
        bias      = "Bullish" if rec.direction == "long" else "Bearish"
        delta_ref = "~0.50" if rec.direction == "long" else "~−0.50"
        side      = "below" if rec.direction == "long" else "above"

        lines = [
            f"**{bias} — {display}**",
            "",
            f"- **Strategy:** {display}. "
            + (
                "Spread chosen to reduce vega cost — IV is elevated."
                if rec.iv_assessment == "expensive" else
                "IV is fair/cheap — outright option captures full move."
                if rec.iv_assessment in ("fair", "cheap") else
                "Check IV vs 30d RV on the Options tab before choosing outright vs spread."
            ),
            "- **Expiry:** 30–45 DTE",
            f"- **Strike:** ATM near ${rec.entry_reference:.2f}. Target delta {delta_ref}.",
            f"- **Exit thesis invalidated:** {side} ${rec.invalidation_price:.2f} "
            "(EMA20 ± ½ ATR). Close the option.",
            f"- **Profit target:** ${rec.target_price:.2f} (2R). Take 50–80% profit — don't hold to expiry.",
        ]
        if rec.warnings:
            lines.append("")
            for w in rec.warnings:
                lines.append(f"⚠️ {w}")
        lines += ["", "_Options tab → Recommendations for exact contract pricing._"]
        return "\n".join(lines)

    def signal_reasons(row: pd.Series) -> list[str]:
        """Plain-English reasons why this stock scored what it scored."""
        reasons = []
        is_crypto = str(row.get("ticker", "")).endswith("-USD")

        if row.get("macd", 0) > row.get("macd_signal", 0):
            reasons.append("MACD crossed above signal — momentum turning up")

        if is_crypto:
            if row.get("ema9", 0) > row.get("ema20", 0):
                reasons.append("EMA9 above EMA20 — short-term trend is bullish")
            if row.get("rvol", 0) >= 1.5:
                reasons.append(f"RVOL {row['rvol']:.1f}× — above-average participation")
            rsi = row.get("rsi", 50)
            if 40 <= rsi <= 75:
                reasons.append(f"RSI {rsi:.0f} — in momentum zone, not yet exhausted")
            elif rsi > 75:
                reasons.append(f"RSI {rsi:.0f} — overbought, risk of pullback")
        else:
            ema9, ema20, ema200 = row.get("ema9", 0), row.get("ema20", 0), row.get("ema200", 0)
            if ema9 > ema20 > ema200:
                reasons.append("EMA9 > EMA20 > EMA200 — trend aligned across all timeframes")
            elif ema9 > ema20:
                reasons.append("EMA9 above EMA20 — short-term trend bullish but below 200")
            if row.get("price", 0) > row.get("vwap", 0):
                reasons.append("Price above VWAP — buyers in control")
            if row.get("volume_trend_up") == 1:
                reasons.append("3-day volume trend rising — institutional accumulation signal")
            if row.get("rvol", 0) >= 3:
                reasons.append(f"RVOL {row['rvol']:.1f}× — heavy unusual volume, something is moving this")
            elif row.get("rvol", 0) >= 1.5:
                reasons.append(f"RVOL {row['rvol']:.1f}× — above-average volume")

        return reasons

    def entry_advice(row: pd.Series) -> str:
        is_crypto = str(row.get("ticker", "")).endswith("-USD")
        if is_crypto:
            return (
                "Wait for 15m RVOL to spike above 1.5× before entering — "
                "run scan_intraday.py to catch the signal. Enter on a 15m candle close above the current price."
            )
        return (
            "Watch the first 15-minute candle after open. Enter on a break above its high "
            "with volume confirming (RVOL ≥ 2× on the intraday scan). "
            "Do not chase if the stock has already moved more than 5% before you enter."
        )

    def sizing_advice(row: pd.Series, risk_nzd: float, nzdusd: float) -> str:
        price     = row.get("price", 0)
        stop      = row.get("stop_loss", 0)
        if not price or not stop or price <= stop:
            return "Stop loss not calculable — skip position sizing."
        stop_dist_usd = price - stop
        stop_dist_nzd = stop_dist_usd / nzdusd if nzdusd else stop_dist_usd
        if stop_dist_nzd <= 0:
            return "Stop distance is zero — do not trade."
        shares    = risk_nzd / stop_dist_nzd
        cost_nzd  = shares * price / nzdusd if nzdusd else shares * price
        target    = round(price + 2 * stop_dist_usd, 4)   # 2:1 R/R
        return (
            f"Risk NZD {risk_nzd:.0f} → **{shares:.1f} shares** "
            f"(position ≈ NZD {cost_nzd:,.0f})  |  "
            f"Stop: ${stop:.4f}  |  Target (2:1): ${target:.4f}"
        )

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------

    if not dates:
        st.info("No screener data yet. Run `python3 run.py` first.")
    else:
        latest_date = dates[0]

        conn = get_conn()
        today_df = pd.read_sql(
            "SELECT * FROM results WHERE run_date = ? ORDER BY tradescore DESC, change_pct DESC",
            conn, params=(latest_date,)
        )

        bt_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='backtest'"
        ).fetchone()
        missed_df = pd.read_sql(
            """
            SELECT b.run_date, b.ticker, b.strategy, b.score,
                   b.entry_price, b.return_1d, b.return_3d, b.return_5d
            FROM backtest b
            WHERE b.score >= 3
              AND b.return_1d IS NOT NULL
              AND b.run_date < ?
            ORDER BY b.return_1d DESC
            """,
            conn, params=(latest_date,)
        ) if bt_exists else pd.DataFrame()
        conn.close()

        nzdusd = fetch_nzdusd()

        # -----------------------------------------------------------------------
        # Position size risk input
        # -----------------------------------------------------------------------

        st.subheader("Risk per trade")
        risk_nzd = st.number_input(
            "How much NZD are you willing to lose if this trade hits stop?",
            min_value=10.0, max_value=10000.0, value=150.0, step=10.0,
            help="This is your maximum loss per trade, not your position size. "
                 "Position size is calculated from this and the stop distance."
        )

        st.divider()

        # -----------------------------------------------------------------------
        # Top picks — score 3 and 4
        # -----------------------------------------------------------------------

        # Sort by tradescore; include any row with a direction signal
        top_picks = today_df[today_df["direction"].isin(["long", "short", "neutral"])].copy()
        if "tradescore" in top_picks.columns:
            top_picks = top_picks.sort_values("tradescore", ascending=False)

        # Cap rendered picks — each card triggers per-ticker yfinance/SEC fetches,
        # so unbounded iteration over 50+ picks is the cold-load bottleneck.
        n_avail = len(top_picks)
        if n_avail > 0:
            n_show = st.slider(
                "Picks to display",
                min_value=1,
                max_value=min(n_avail, 50),
                value=min(10, n_avail),
                help="Higher values fetch company info and catalysts for more tickers — slower first render.",
            )
            top_picks = top_picks.head(n_show)

        st.subheader(f"Today's top picks  —  {latest_date}")

        if top_picks.empty:
            st.warning("No directional setups today. Check back after the next screener run.")
        else:
            for _, row in top_picks.iterrows():
                ticker = row["ticker"]
                score  = int(row["score"])
                stars  = "★" * score + "☆" * (4 - score)

                with st.container(border=True):
                    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
                    c1.markdown(f"### {ticker}  `{stars}`")
                    c2.metric("Score",    f"{score}/4")
                    c3.metric("Change",   f"{row['change_pct']:+.2f}%")
                    c4.metric("RVOL",     f"{row['rvol']:.1f}×")

                    reasons = signal_reasons(row)
                    if reasons:
                        st.markdown("**Why it scored:**")
                        for r in reasons:
                            st.markdown(f"- {r}")

                    st.markdown("**Entry:**")
                    st.markdown(entry_advice(row))

                    st.markdown("**Position size:**")
                    st.markdown(sizing_advice(row, risk_nzd, nzdusd))

                    with st.expander("Full indicators"):
                        ind_cols = ["price", "stop_loss", "rsi", "rvol",
                                    "ema9", "ema20", "ema200", "macd", "macd_signal", "vwap"]
                        ind_cols = [c for c in ind_cols if c in row.index]
                        st.dataframe(
                            pd.DataFrame(row[ind_cols]).T,
                            width='stretch', hide_index=True,
                        )

                    with st.expander(f"About {ticker}"):
                        co = get_company_info(ticker)
                        if co:
                            st.markdown(f"**{co['name']}**")
                            if co["sector"] or co["industry"]:
                                st.caption(f"{co['sector']}  ·  {co['industry']}")
                            if co["summary"]:
                                # Trim to first 3 sentences
                                sentences = co["summary"].split(". ")
                                brief = ". ".join(sentences[:3]).strip()
                                if not brief.endswith("."):
                                    brief += "."
                                st.markdown(brief)
                            if co["website"]:
                                st.markdown(co["website"])
                        else:
                            st.caption("No company data available.")

        st.divider()

        # -----------------------------------------------------------------------
        # Options recommendations — separate section
        # -----------------------------------------------------------------------

        st.subheader("Options recommendations")
        st.caption(
            "Options strategy for each top pick, based on directional setup and IV environment. "
            "Check the Options tab for exact contract pricing."
        )

        equity_picks = top_picks[~top_picks["ticker"].str.endswith("-USD")]
        crypto_picks = top_picks[top_picks["ticker"].str.endswith("-USD")]

        if equity_picks.empty:
            st.info("No equity picks today — no options recommendations to show.")
        else:
            # Fetch catalyst for each pick. Cached via @st.cache_data, so the
            # first render of the day is slow (~3-5s per ticker) but
            # subsequent renders are instant.
            from core.catalysts import compute_catalyst_score as _ccs_advice
            with st.spinner("Fetching catalyst context for picks…"):
                _advice_catalysts = {
                    _r["ticker"]: _ccs_advice(_r["ticker"], float(_r.get("price") or 0))
                    for _, _r in equity_picks.iterrows()
                }
            for _, row in equity_picks.iterrows():
                ticker    = row["ticker"]
                direction = str(row.get("direction") or "")
                score     = int(row.get("score") or 0)
                stars     = "★" * score + "☆" * (4 - score)
                rec       = build_recommendation(
                    row.to_dict(),
                    iv_mode="fallback",
                    catalyst=_advice_catalysts.get(ticker),
                )

                with st.container(border=True):
                    oc1, oc2, oc3, oc4 = st.columns([2, 1, 1, 1])
                    oc1.markdown(f"### {ticker}  `{stars}`")
                    oc2.metric("Direction",
                               "🟢 Long" if direction == "long" else
                               "🔴 Short" if direction == "short" else direction or "—")
                    oc3.metric("Price", f"${row['price']:.2f}")
                    oc4.metric("Stop", f"${rec.invalidation_price:.2f}" if rec.invalidation_price else "—")
                    st.markdown(_format_rec_markdown(rec))

        if not crypto_picks.empty:
            st.caption("Crypto picks: " + ", ".join(crypto_picks["ticker"].tolist()) + " — options not available for crypto.")

        st.divider()

        # -----------------------------------------------------------------------
        # What you missed — previous score 3-4 picks with outcomes
        # -----------------------------------------------------------------------

        st.subheader("What you missed")
        st.caption("Previous screener runs that scored 3 or higher — and what happened next.")

        if missed_df.empty:
            st.info("No historical high-score picks with forward returns yet. Run `python3 backtest_v2.py` after each session.")
        else:
            for _, row in missed_df.iterrows():
                r1  = row.get("return_1d")
                r3  = row.get("return_3d")
                r5  = row.get("return_5d")
                direction = "up" if (r1 or 0) > 0 else "down"
                color     = "green" if direction == "up" else "red"

                with st.container(border=True):
                    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
                    c1.markdown(f"**{row['ticker']}**  `{row['run_date']}`  score {int(row['score'])}/4")
                    c2.metric("Entry",  f"${row['entry_price']:.2f}")
                    c3.metric("1d",     f"{r1:+.1f}%" if r1 is not None else "—")
                    c4.metric("3d",     f"{r3:+.1f}%" if r3 is not None else "—")
                    c5.metric("5d",     f"{r5:+.1f}%" if r5 is not None else "—")
