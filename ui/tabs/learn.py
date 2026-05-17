"""
ui/tabs/learn.py — Options fundamentals education tab.

10 lessons covering options basics, Greeks, IV crush, strategies, position
sizing, common mistakes, and FOREX basics. Uses live data (INTC option
chain) as a shared example throughout.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from core.options_math import bs_price, bs_greeks, _ncdf, _npdf
from ui.data import _provider as _provider, fetch_nzdusd, payoff_df


def render() -> None:
    """Render the Learn tab."""

    st.subheader("Options fundamentals")
    st.caption("Each lesson uses live data from your watchlist and the positions you hold.")

    lesson = st.selectbox("Choose a lesson", [
        "0. Glossary — every term explained",
        "1. What is an option?",
        "2. Calls vs Puts",
        "3. The Greeks — Delta",
        "4. The Greeks — Theta (time decay)",
        "5. The Greeks — Vega (implied volatility)",
        "6. IV crush — the most common way to lose money",
        "7. Strategies and when to use them",
        "8. Position sizing and risk management",
        "9. The most common mistakes",
        "10. FOREX basics (different game from stocks)",
    ])

    RISK_FREE_L = 0.045

    # -----------------------------------------------------------------------
    # Shared live example data (INTC — one of Dave's positions)
    # -----------------------------------------------------------------------

    @st.cache_data(ttl=600)
    def learn_example():
        try:
            spot     = _provider.get_quote("INTC").last_price or 62
            expiries = _provider.get_expiries("INTC")
            # Pick ~30 DTE expiry
            today_dt = datetime.now(timezone.utc).date()
            exp = None
            for e in expiries:
                dte = (datetime.strptime(e, "%Y-%m-%d").date() - today_dt).days
                if 20 <= dte <= 50:
                    exp = e
                    break
            exp = exp or expiries[1]
            calls_df, _puts_df = _provider.get_option_chain("INTC", exp)
            dte   = (datetime.strptime(exp, "%Y-%m-%d").date() - today_dt).days
            # Find ATM call
            calls = calls_df
            calls = calls[calls["bid"] > 0]
            atm   = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]].iloc[0]
            K     = float(atm["strike"])
            iv    = float(atm["impliedVolatility"])
            bid   = float(atm.get("bid", 0))
            ask   = float(atm.get("ask", 0))
            mid   = round((bid + ask) / 2, 3)
            return {"spot": spot, "exp": exp, "dte": dte, "K": K, "iv": iv, "mid": mid}
        except Exception:
            return {"spot": 62.0, "exp": "2026-05-15", "dte": 35, "K": 62.0, "iv": 0.45, "mid": 3.20}

    ex = learn_example()
    S, K, T_ex, iv_ex, mid_ex = ex["spot"], ex["K"], ex["dte"]/365, ex["iv"], ex["mid"]

    st.divider()

    # =======================================================================
    if lesson == "0. Glossary — every term explained":
    # =======================================================================

        st.markdown("""
    This is the dictionary for every term you'll see in this app. The point is
    to demystify the jargon — once you know what each thing measures, you can
    read the Screener, Lookup, Advice, and Options tabs without guessing.

    The glossary is organised into eight sections. Read top to bottom once;
    after that, treat it as a reference.
    """)

        with st.expander("📊 1. The TradeScore — how this app scores stocks", expanded=False):
            st.markdown("""
    **TradeScore (0–65)** is this app's composite score for whether a stock is
    worth a trade *right now*. Higher is better. It's the sum of four positive
    sub-scores minus one penalty:

    `TradeScore = Momentum + EarlyEntry + Liquidity + NewsCatalyst − ExtensionRisk`

    | Sub-score | Max | What it measures | High score means |
    |---|---|---|---|
    | **MomentumScore** | 25 | Today's RVOL, today's % change, MACD vs signal | Stock is moving up, on strong volume, with momentum confirmed |
    | **EarlyEntryScore** | 25 | RSI in the 52–68 sweet spot, price near EMA20, breakout from a base | You're catching it early — not chasing a top |
    | **LiquidityScore** | varies | Dollar volume and volume consistency | You can actually get in and out without huge slippage |
    | **NewsCatalystScore** | varies | Stub for now (Phase 10 expansion) | Recent material news / catalyst |
    | **ExtensionRiskScore** | up to −20 | RSI > 70, single-day overshoot, ATR multiples above VWAP, 5-session run-up | Stock is overextended — buying here is buying the top |

    A score of **45+** is "interesting." **55+** is the threshold where the
    research mode (Phase 9) starts to find statistical edge. Below **30** is
    usually "don't bother."

    **Conviction labels** map the score to a one-word verdict: `low`, `medium`,
    `high`, `very high`.
    """)

        with st.expander("📈 2. Technical indicators — what each number means", expanded=False):
            st.markdown("""
    These are the standard inputs to the TradeScore. They all come from price
    and volume — nothing else.

    | Term | Formula / source | What it tells you |
    |---|---|---|
    | **Price** | Latest close | Where the stock is now |
    | **Change %** | (Today − yesterday) / yesterday × 100 | Today's move |
    | **RVOL** (Relative Volume) | Today's volume ÷ average daily volume | 1.0× = normal, 2.0× = busy, 5.0× = something is happening |
    | **RSI(14)** | 14-day Relative Strength Index | 0–100. Below 30 = oversold (buyers usually step in). Above 70 = overbought (sellers usually step in). 50 = neutral. |
    | **EMA9 / EMA20 / EMA200** | 9/20/200-day Exponential Moving Average | Short-term, medium-term, long-term trend lines. Price above EMA = uptrend; below = downtrend. |
    | **MACD / MACD Signal** | 12-day EMA − 26-day EMA, plus a 9-day EMA of that | MACD above signal = bullish momentum. MACD crossing below signal = warning. |
    | **ATR(14)** | 14-day Average True Range | The typical daily price range in dollars. Used to size stops. |
    | **VWAP** | Volume-Weighted Average Price (intraday) | The "fair price" weighted by where most volume traded. Price above VWAP = bulls control today. |
    | **Dollar Volume** | Price × volume | Total $ traded today. Need at least $5–10M/day to trade without moving the price yourself. |
    | **Float shares** | Shares available to public | Low float (<50M) + volume = explosive moves. Caution. |
    | **Change 5d** | 5-session price change % | How extended the move is. >15% in 5 days is a yellow flag. |
    """)

        with st.expander("🎯 3. Setups and direction — how the app labels a chart", expanded=False):
            st.markdown("""
    After scoring, the app classifies the chart into one of these **setups**:

    | Setup | Plain English | What to do |
    |---|---|---|
    | **clean_breakout** | Strong trend, sensible stop, IV not crazy | The cleanest entries. Trade with conviction. |
    | **emerging_momentum** | Trend forming but TradeScore not yet high | Watch list. Could become a clean breakout. |
    | **extended** | Price has run too far above the trend anchor | Wait for a pullback. Buying here is buying the top. |
    | **pullback_candidate** | Direction is clear but the stop is too wide to risk now | Wait for a tighter setup. |
    | **no_edge** | Score is too low or direction is mixed | Skip. |
    | **liquidity_concern** | Float or dollar volume too small | Avoid options here (wide spreads will eat profit). |
    | **crypto_no_options** | Crypto ticker (BTC-USD etc.) | Use spot exposure, not options. |

    **Direction** is `long`, `short`, or `neutral`. Long when price > VWAP and
    EMA9 > EMA20 (uptrend, momentum aligned). Short when both flip. Neutral
    when mixed — no clean directional signal.
    """)

        with st.expander("💡 4. The Recommendation — what each field means", expanded=False):
            st.markdown("""
    Built by `core/recommendations.py`. The Advice, Options, and Lookup tabs
    all show the same fields:

    | Field | What it is |
    |---|---|
    | **Recommendation Category** | `actionable` (trade it), `watchlist` (wait), or `avoid` |
    | **Strategy Name** | The specific options structure: Long Call, Bull Call Spread, Long Put, Bear Put Spread, Cash-Secured Put, or "wait" |
    | **Entry Reference** | Current price — your entry guide. Not a limit order. |
    | **Invalidation Price (Stop)** | The price that says "thesis is wrong, close the trade." Computed as EMA20 ± ½ × ATR. |
    | **Target Price** | 2× the risk. If stop is $1 below entry, target is $2 above. |
    | **Risk:Reward (R:R)** | Reward ÷ risk. 2.0 means you risk $1 to make $2. Don't trade below 1.5. |
    | **IV Assessment** | `cheap`, `fair`, `expensive`, or `unavailable`. Tells you whether options are overpriced right now. |
    | **Rationale** | Plain-English reasoning specific to *this* ticker (not a template). |
    | **Warnings** | Yellow flags: tight stop, low float, earnings tomorrow, etc. |
    | **is_actionable** | Boolean — is this trade-ready right now? |

    **Why this matters:** the app never gives a recommendation without a stop
    and a target. If you can't define the stop, you can't size the trade.
    """)

        with st.expander("📐 5. The Greeks (options) — the four risk dials", expanded=False):
            st.markdown("""
    The Greeks measure how an option's price reacts to four different changes.
    Treat them as **dials** that you can read to understand a trade's risk.

    | Greek | What it answers | Practical reading |
    |---|---|---|
    | **Delta** | If the stock moves $1, how much does the option move? | 0.50 = the option mirrors half the stock move. 0.90 = nearly stock-equivalent (deep ITM). 0.10 = lottery ticket (deep OTM). |
    | **Gamma** | How fast does Delta change? | High Gamma = the option's behaviour changes fast as price moves. Most pronounced near the strike. |
    | **Theta** | How much value does the option lose per day from time decay? | Always negative for buyers. Bigger as expiry approaches. The "rent" you pay to hold the option. |
    | **Vega** | How much does the option's price change if IV moves 1%? | Bigger for longer-dated options. Why "buying IV high, selling IV low" hurts. |
    | **Rho** | Sensitivity to interest rates. | Mostly ignore unless trading very long-dated LEAPS. |

    **The takeaway:** never buy an option without knowing your Delta (your
    real directional exposure) and Theta (what you pay each day).
    """)

        with st.expander("🛠️ 6. Options strategies — the five the app uses", expanded=False):
            st.markdown("""
    The app recommends one of five strategies (or "wait"). Each has a clear
    "when to use" rule.

    | Strategy | When to use | Max loss | Max gain |
    |---|---|---|---|
    | **Long Call** | Bullish + IV cheap or fair | Premium paid | Unlimited |
    | **Bull Call Spread** | Bullish + IV expensive (defined risk) | Net debit | Spread width − net debit |
    | **Long Put** | Bearish + IV cheap or fair | Premium paid | Strike − premium |
    | **Bear Put Spread** | Bearish + IV expensive | Net debit | Spread width − net debit |
    | **Cash-Secured Put** | Want to own the stock at a lower price | Strike × 100 if assigned | Premium received |
    | **Wait** | No setup meets criteria — sit on hands | $0 | $0 (this is a feature, not a failure) |

    **Two rules baked in:**
    - All recommendations target **30–45 DTE** (days to expiry) — sweet spot for
      Theta efficiency and time to be right.
    - Take profit at **50–80%** of max gain. Don't hold to expiry.
    """)

        with st.expander("📜 7. SEC filings — what each form means", expanded=False):
            st.markdown("""
    The Lookup tab shows recent SEC filings. Here's what each form is.

    | Form | Meaning | Trader relevance |
    |---|---|---|
    | **8-K** | "Material event" — anything important happens (M&A, contract win, CEO change, lawsuit) | Read these. The most catalyst-relevant filing. |
    | **10-Q** | Quarterly earnings + financials | Read these around earnings. Especially the segment results. |
    | **10-K** | Annual report | The long version. Most useful for understanding the business model and risk factors. |
    | **S-1** | Registration for IPO or secondary offering | Issuance = dilution. Usually a headwind. |
    | **Form 4** | Insider trade (officer/director/10%+ holder) | **Buying clusters are a signal. Selling is mostly noise.** Transaction code `P` = open-market purchase (bullish), `S` = sale. |

    **Where to read them:** the link in the Lookup tab takes you straight to
    the SEC document. EDGAR (sec.gov) is the source — always free, always
    authoritative.
    """)

        with st.expander("🏛️ 8. Government contracts — what USAspending shows", expanded=False):
            st.markdown("""
    For defense, govtech, and IT-services tickers, the Lookup tab shows recent
    federal contract awards from USAspending.gov. For consumer or non-contractor
    names (SOFI, NVDA, etc.) this section is empty — that's correct.

    | Term | Meaning |
    |---|---|
    | **Action Date** | The day the federal obligation was signed |
    | **Amount** | The dollar value of that specific action (not the full contract value) |
    | **Kind** | `NEW` = new award. `MOD P000X` = modification or option exercise on an existing contract. |
    | **Agency** | Who is paying (DoD, DOE, DHS, HHS, NASA, etc.) |
    | **Recipient** | The exact subsidiary name USAspending uses |

    **Caveats — read these:**
    - USAspending publishes contract data **30–90 days after the award is
      signed**, so this is a **lagged signal**, not a real-time alert.
    - Most contracts are **modifications** to existing deals, not new wins. New
      awards historically moved stocks more than modifications.
    - **Non-DoD agencies** (DOE, Treasury, DHS) showed stronger post-event drift
      in backtest than DoD — the DoD daily press release is well front-run.
    - **Materiality matters.** A $50M contract is news for a $200M micro-cap
      and noise for a $200B mega-cap. Always check the amount vs the company's
      market cap.

    **This is a catalyst signal, not a strategy.** Use it alongside TradeScore,
    SEC filings, and your own judgement.
    """)

        with st.expander("📐 9. Backtest analytics — Sharpe, Sortino, drawdown", expanded=False):
            st.markdown("""
    The Backtest tab summary shows four risk/return metrics. Read them together,
    not in isolation — each has a known weakness.

    | Metric | Formula | What high means | Watch out for |
    |---|---|---|---|
    | **Sharpe (ann.)** | (mean_excess / std) × √N | Returns are consistently positive relative to volatility | A few big wins can inflate Sharpe; doesn't distinguish upside vs downside volatility |
    | **Sortino (ann.)** | (mean_excess / downside_std) × √N | Strategy avoids deep losses, not just any volatility | Less reference data — interpret cautiously below 1.0 |
    | **Avg per-ticker Sharpe** | mean of saved per-ticker Sharpes | Individual ticker strategies are mostly profitable on their own | Doesn't account for cross-ticker correlation — could double-count macro exposure |
    | **Max drawdown** | min(equity_curve / cummax(equity_curve) − 1) | Strategy never lost more than this peak-to-trough | Historical only; future drawdowns can be larger. Most strategies should expect 1.5–2× backtest max in live use. |

    ### How annualisation works in this app

    The app stores **one row per ticker** after a backtest, not per trade. The
    Sharpe calculation treats the equity curve as a sequence of bet outcomes,
    then annualises by √N where N = `periods_per_year` (default 252 — one bet
    per trading day).

    If your actual deployment frequency differs, override the parameter:

    ```python
    from core.analytics import portfolio_stats
    stats = portfolio_stats(df, periods_per_year=52, risk_free_rate=0.045)
    ```

    Common settings:
    - **252** — one bet per trading day (intraday/swing default)
    - **52** — one bet per week (weekly swing)
    - **12** — one bet per month (positional)
    - **4** — one bet per quarter (very long hold)

    ### Risk-free rate

    Default is **4.5% annual**, prorated per period for the excess return.
    Used by both Sharpe and Sortino. Override via the `risk_free_rate`
    parameter if you want pre-Fed-rate-cut comparisons (e.g., 2% for 2021,
    0% for 2020).

    ### What these metrics don't tell you

    - **Path-dependence.** Same Sharpe can come from steady gains or a few
      spikes followed by long flat periods. Always look at the equity curve.
    - **Survivorship.** If the backtest universe excludes delisted tickers,
      metrics are optimistic.
    - **Slippage.** The backtest engine subtracts commission but doesn't model
      realistic spread/slippage for thinly-traded names.
    - **Regime.** A strategy with Sharpe 2.0 in 2020 momentum could be -1.0
      in 2022 mean-reversion. Test across multiple regimes (Phase 9 research
      mode).
    """)

        with st.expander("🎯 10. CatalystScore — the second score that matters", expanded=False):
            st.markdown("""
    **CatalystScore (0–100)** is the answer to the question:
    *"What's happening to this company in the world right now?"*

    TradeScore measures **price/volume mechanics** — is the chart in a tradeable
    setup? CatalystScore measures **the news/earnings/analyst environment** —
    is there a real reason this should move? **They can disagree.** When they
    do, that's the most useful information you'll get from this app.

    The two scores point you to four different situations:

    | TradeScore | CatalystScore | Read |
    |---|---|---|
    | High | High | Clean technical setup + supportive catalysts. Highest-conviction trades. |
    | High | Low | The chart is right but nothing fundamental is happening — momentum trade only, expect mean reversion. |
    | Low | High | A catalyst is building but price hasn't priced it in yet — patient watchlist. |
    | Low | Low | Skip. |

    **Baseline = 50.** Positive catalysts push the score up; negative catalysts
    push it down. The score is clamped to 0–100.

    ### What contributes to CatalystScore

    | Component | Range | What it captures |
    |---|---|---|
    | **Earnings beat/miss (last quarter)** | −15 to +15 | Did the last report beat or miss EPS estimates? Big surprises (>10%) score higher. |
    | **Beat / miss streak** | −10 to +10 | Pattern across last 4 quarters. A 4-quarter beat streak earns the bonus; 2+ misses in a row penalises. |
    | **Imminent earnings risk** | −15 to 0 | Earnings within 7 days = −15 (binary event risk). 8–14 days = −8. 15–30 days = −3. |
    | **Analyst upgrade momentum** | −15 to +15 | Net upgrades vs downgrades in last 90 days. |
    | **Price target movement** | −12 to +12 | Net target raises vs cuts in last 90 days. |
    | **Consensus rating** | −8 to +8 | "Strong buy"/"buy" = +8. "Sell"/"strong sell" = −8. Hold = neutral. |
    | **Consensus target upside** | −10 to +15 | If mean target is ≥30% above current price = +15. If target is ≥15% below = −10. |
    | **News intensity** | 0 to +5 | Number of Yahoo Finance news items in last 7 days. Coverage signal, not direction. |
    | **Insider buying** | −5 to +15 | Form 4 trades in last 90d. Asymmetric: buying is signal, selling is mostly noise (see below). |

    ### Insider buying scoring — asymmetric on purpose

    Insiders sell for many reasons (tax, diversification, 10b5-1 plans,
    funding life events). They buy for **one** reason — they think the
    stock will rise. The scoring reflects this:

    | Pattern | Score | When you see it |
    |---|---:|---|
    | 3+ unique buyers, ≥$250k total in 90d | **+15** | "Insider buy cluster" — the strongest catalyst signal in the layer |
    | 2 buyers, ≥$100k total | +8 | Moderate cluster |
    | 1 buyer, ≥$500k single buy | +6 | Big single-insider buy (e.g., CEO putting their own money in) |
    | 1 buyer, ≥$50k | +3 | Routine single purchase |
    | 3+ sellers, ≥$500k total, no buyers | −5 | Heavy lopsided selling (only penalised case) |
    | Anything else | 0 | Noise / normal |

    A "cluster" of insider buying within a 90-day window is one of the most
    durable signals in academic literature — it precedes price gains
    significantly more often than chance, especially in small-caps. Selling
    clusters do NOT predict losses with the same reliability, which is why
    sell penalties are small and require lopsided activity.

    ### What CatalystScore does NOT measure

    - **Direction.** A high CatalystScore can be bullish OR bearish (10 target cuts and a miss is "high catalyst, bearish"). Read the **tags** for direction.
    - **Quality of news.** It counts headlines, not sentiment. A "company sued" headline and a "company raises guidance" headline both count as news.
    - **Macro events.** Fed meetings, CPI, geopolitical events aren't ticker-specific.

    ### How to read the tags

    Each tag is **plain-English specific** to this ticker. Examples you'll see:

    - 🟢 `Strong earnings beat last quarter (+19.3% surprise)` — bullish
    - 🟢 `4-quarter beat streak` — pattern
    - 🟢 `Mean analyst target $21.10 = +38% upside (consensus from 20 analysts)` — implied upside
    - 🔴 `Earnings miss last quarter (-4.2% surprise)` — bearish
    - 🔴 `12 price target cuts in 90d` — bearish analyst momentum
    - 🔴 `⚠ Earnings in 4 days — binary event risk` — event risk warning
    - ⚪ `Consensus rating: hold (20 analysts)` — neutral

    Tags surface **why** the score is what it is. If you ever look at a
    CatalystScore and can't explain it, you're missing context — read the tags.
    """)

        st.divider()
        st.caption(
            "_Glossary maintained alongside the codebase. If you see a term in "
            "the app that isn't explained here, flag it — every visible field "
            "in the UI should appear in one of the sections above._"
        )

    # =======================================================================
    elif lesson == "1. What is an option?":
    # =======================================================================

        st.markdown("""
    An option is a **contract** that gives you the right — but not the obligation — to buy or sell a stock at a specific price, before a specific date.

    You pay a **premium** upfront. That premium is your maximum loss. The stock moves in your favour, the option gains value. It moves against you, the option loses value — but you can never lose more than what you paid.

    **Three things define every option:**

    | Term | What it means |
    |---|---|
    | **Strike price** | The price at which you have the right to buy or sell |
    | **Expiry date** | The date the contract expires — after this it's worthless |
    | **Premium** | What you pay to buy the contract (your max loss) |

    **Options vs buying stock directly:**
    """)

        nzdusd_l = fetch_nzdusd()
        shares_direct = round((1000 * nzdusd_l) / S, 4)
        cost_option   = round(mid_ex * 100, 2)
        cost_nzd_opt  = round(cost_option / nzdusd_l, 2)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Buy INTC stock directly**")
            st.markdown(f"- Spend NZD 1,000 → get **{shares_direct} shares**")
            st.markdown(f"- Stock goes up 10% → you make NZD {1000*0.10:.0f}")
            st.markdown(f"- Stock goes to zero → you lose NZD 1,000")

        with col2:
            st.markdown("**Buy 1 ATM call option on INTC**")
            st.markdown(f"- Spend NZD {cost_nzd_opt:.0f} → control **100 shares**")
            st.markdown(f"- Stock goes up 10% → option might gain 40–60%")
            st.markdown(f"- Option expires worthless → you lose NZD {cost_nzd_opt:.0f} only")

        st.info(f"Live example: INTC at ${S:.2f}. ATM call (strike ${K:.2f}, expiry {ex['exp']}) costs ${mid_ex:.3f} per share = **${mid_ex*100:.2f} per contract** (NZD {cost_nzd_opt:.0f}).")

        st.markdown("""
    **Key rule:** Options buyers have **limited loss, unlimited upside**.
    Options sellers have **limited upside (the premium), unlimited risk**. Start as a buyer.
    """)

    # =======================================================================
    elif lesson == "2. Calls vs Puts":
    # =======================================================================

        st.markdown("""
    **Call option** — you think the stock is going UP.
    Gives you the right to *buy* shares at the strike price.

    **Put option** — you think the stock is going DOWN.
    Gives you the right to *sell* shares at the strike price.
    """)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Call")
            st.markdown(f"""
    - You buy a call on INTC at strike **${K:.2f}**
    - If INTC rises to **${K*1.15:.2f}** (+15%), your call is worth at least **${max(K*1.15-K,0):.2f}** in intrinsic value
    - If INTC stays below **${K:.2f}** at expiry → expires worthless, you lose the premium
    - **Use when:** Bullish. RVOL spiking. Score 3–4 on screener.
    """)
        with c2:
            st.markdown("### Put")
            st.markdown(f"""
    - You buy a put on INTC at strike **${K:.2f}**
    - If INTC drops to **${K*0.85:.2f}** (−15%), your put is worth at least **${max(K-K*0.85,0):.2f}** in intrinsic value
    - If INTC stays above **${K:.2f}** at expiry → expires worthless
    - **Use when:** Bearish. Bad earnings expected. Hedging an existing long position.
    """)

        st.divider()
        st.markdown("**Intrinsic vs extrinsic value**")
        st.markdown(f"""
    An option's premium has two parts:

    - **Intrinsic value** — the profit if you exercised right now. For a call at ${K:.2f} with stock at ${S:.2f}: ${max(S-K,0):.2f}
    - **Extrinsic (time) value** — what you pay for time + volatility. This is **{mid_ex - max(S-K,0):.3f}** of your {mid_ex:.3f} premium.

    All extrinsic value goes to zero at expiry. That's why time works against option buyers.
    """)

        iv_pct = round(iv_ex * 100, 1)
        st.info(f"Live: INTC ATM call premium = ${mid_ex:.3f}. Intrinsic = ${max(S-K,0):.2f}. Time value = ${mid_ex - max(S-K,0):.3f}. Current IV = {iv_pct}%.")

    # =======================================================================
    elif lesson == "3. The Greeks — Delta":
    # =======================================================================

        g = bs_greeks(S, K, T_ex, RISK_FREE_L, iv_ex, "call")
        delta = g["delta"]

        st.markdown(f"""
    **Delta** tells you how much the option price moves for every $1 the stock moves.

    INTC ATM call delta = **{delta:.3f}**

    This means:
    - Stock goes up $1 → option gains **${delta:.3f}** per share → **${delta*100:.2f} per contract**
    - Stock goes up $5 → option gains approximately **${delta*5:.2f}** per share
    - Stock drops $1 → option loses **${delta:.3f}** per share

    **Delta also tells you the approximate probability the option expires in the money.**
    Delta {delta:.2f} ≈ {delta*100:.0f}% chance of expiring with value.

    **Delta by strike:**
    """)

        delta_rows = []
        for moneyness, label in [(0.90,"Deep ITM"), (0.95,"ITM"), (1.00,"ATM"), (1.05,"OTM"), (1.10,"Deep OTM")]:
            Kx = round(S * moneyness, 2)
            gx = bs_greeks(S, Kx, T_ex, RISK_FREE_L, iv_ex, "call")
            delta_rows.append({
                "Type": label, "Strike": f"${Kx:.2f}",
                "Delta": gx["delta"],
                "Approx prob ITM": f"{gx['delta']*100:.0f}%",
                "Move per $1 stock (per contract)": f"${gx['delta']*100:.2f}",
            })
        st.dataframe(pd.DataFrame(delta_rows), width='stretch', hide_index=True)

        st.markdown("""
    **What delta to choose?**
    - **0.70+ (deep ITM):** Moves almost like owning stock. Expensive. Lower % return but safer.
    - **0.50 (ATM):** Balanced. Most common starting point.
    - **0.30 (OTM):** Cheap. Needs a bigger move. Higher % return if it works, more often expires worthless.
    - **< 0.20 (far OTM):** Lottery ticket. Rarely pays off. Avoid until you understand options well.
    """)

    # =======================================================================
    elif lesson == "4. The Greeks — Theta (time decay)":
    # =======================================================================

        g     = bs_greeks(S, K, T_ex, RISK_FREE_L, iv_ex, "call")
        theta = g["theta"]

        st.markdown(f"""
    **Theta** is the daily cost of holding an option. Every day that passes, the option loses this much value — even if the stock doesn't move.

    INTC ATM call theta = **{theta:.4f}** per share per day = **${abs(theta)*100:.2f} per contract per day**

    Over {ex['dte']} days to expiry, that's **${abs(theta)*100*ex['dte']:.2f}** in total time decay — which is most of your premium.

    Theta accelerates. It's slow far from expiry and rapid in the last 2 weeks.
    """)

        # Theta decay chart
        decay_rows = []
        for days_left in range(ex["dte"], 0, -1):
            T_temp = days_left / 365.0
            px = bs_price(S, K, T_temp, RISK_FREE_L, iv_ex, "call")
            decay_rows.append({"Days to expiry": days_left, "Option value ($)": round(px, 4)})
        decay_df = pd.DataFrame(decay_rows).set_index("Days to expiry").sort_index()
        st.line_chart(decay_df)
        st.caption(f"INTC ATM call (strike ${K:.2f}, IV {iv_ex*100:.0f}%) — value over time assuming stock stays at ${S:.2f}. "
                   "The curve accelerates downward as expiry approaches.")

        st.markdown("""
    **Rules of thumb:**
    - Hold options for **short periods** when buying — theta is working against you every day
    - Don't hold options into the last 2 weeks unless you're very confident
    - Sellers (cash-secured puts, covered calls) *benefit* from theta — it's working for them
    """)

    # =======================================================================
    elif lesson == "5. The Greeks — Vega (implied volatility)":
    # =======================================================================

        g    = bs_greeks(S, K, T_ex, RISK_FREE_L, iv_ex, "call")
        vega = g["vega"]

        st.markdown(f"""
    **Vega** tells you how much the option price changes for every 1% change in implied volatility (IV).

    INTC ATM call vega = **{vega:.4f}** per share = **${vega*100:.2f} per contract** per 1% IV move.

    If IV rises from {iv_ex*100:.0f}% to {iv_ex*100+5:.0f}% (up 5%), option gains **${vega*5*100:.2f}** per contract — even if the stock doesn't move.
    If IV drops from {iv_ex*100:.0f}% to {iv_ex*100-10:.0f}% (down 10%), option loses **${vega*10*100:.2f}** per contract.
    """)

        # IV sensitivity chart
        iv_rows = []
        for iv_pct in range(10, 120, 5):
            px = bs_price(S, K, T_ex, RISK_FREE_L, iv_pct/100, "call")
            iv_rows.append({"IV %": iv_pct, "Option value ($)": round(px, 4)})
        iv_df = pd.DataFrame(iv_rows).set_index("IV %")
        st.line_chart(iv_df)
        st.caption(f"INTC ATM call (strike ${K:.2f}, {ex['dte']}d to expiry) — value at different IV levels, stock held at ${S:.2f}.")

        st.markdown(f"""
    **Current INTC IV: {iv_ex*100:.0f}%**

    High IV = expensive options. Low IV = cheap options.

    **The rule:** Buy options when IV is low. Sell options when IV is high.

    The Options tab shows you 30d Realised Vol vs ATM IV for any ticker. 
    If IV is significantly above realised vol, options are expensive — consider a spread instead of an outright buy.
    """)

    # =======================================================================
    elif lesson == "6. IV crush — the most common way to lose money":
    # =======================================================================

        st.markdown("""
    **IV crush** happens when implied volatility collapses after a known event — usually earnings.

    Before earnings, IV inflates because nobody knows what will happen. Option prices rise.
    After earnings, the uncertainty resolves. IV collapses — sometimes by 30–50% in one day.

    **The result:** You buy a call before earnings. The stock goes UP 5%. But your call loses value because IV dropped 40%.

    This is the most common way beginners lose money on options.
    """)

        # Show the effect numerically
        pre_iv  = min(iv_ex * 2.0, 1.5)
        post_iv = iv_ex * 0.6
        st.markdown(f"**INTC example (hypothetical earnings scenario):**")

        crush_rows = []
        for label, s_move, iv_used in [
            ("Stock flat, pre-earnings IV",      S,       pre_iv),
            ("Stock +5%, IV crushes post-earn",  S*1.05,  post_iv),
            ("Stock +10%, IV crushes post-earn", S*1.10,  post_iv),
            ("Stock +15%, IV crushes post-earn", S*1.15,  post_iv),
        ]:
            px = bs_price(s_move, K, T_ex, RISK_FREE_L, iv_used, "call")
            ret = (px / mid_ex - 1) * 100
            crush_rows.append({
                "Scenario":    label,
                "Stock price": f"${s_move:.2f}",
                "IV":          f"{iv_used*100:.0f}%",
                "Option value":f"${px:.3f}",
                "Return vs entry": f"{ret:+.1f}%",
            })
        st.dataframe(pd.DataFrame(crush_rows), width='stretch', hide_index=True)

        st.markdown(f"Entry price: ${mid_ex:.3f} at IV {iv_ex*100:.0f}%.")

        st.warning("Stock +5% but option loses money. This is IV crush in action.")

        st.markdown("""
    **How to avoid it:**
    1. Check the IV vs Realised Vol on the Options tab before buying
    2. Avoid buying options in the week before earnings unless you have strong conviction and understand the IV risk
    3. Use spreads instead of outright buys when IV is elevated — the spread reduces your vega exposure
    4. The intraday scanner (scan_intraday.py) fires on RVOL spikes — if you see a spike *after* an earnings gap, IV is already compressing. Enter cautiously.
    """)

    # =======================================================================
    elif lesson == "7. Strategies and when to use them":
    # =======================================================================

        st.markdown("Select a strategy to see its payoff and when to use it.")

        strat_pick = st.selectbox("Strategy", [
            "Long Call", "Bull Call Spread", "Long Put",
            "Cash-Secured Put", "Covered Call",
        ], key="learn_strat")

        guides = {
            "Long Call": {
                "when":     "Strong bullish conviction. Score 3–4 on screener. IV is low or fair (< 30d RV).",
                "risk":     "Limited — premium paid only.",
                "reward":   "Unlimited.",
                "avoid":    "Before earnings (IV crush). When IV >> realised vol. Far OTM strikes.",
                "legs":     [{"type":"call","strike":K,"premium":mid_ex,"qty":1,"position":"long"}],
            },
            "Bull Call Spread": {
                "when":     "Bullish but IV is high. Buying the spread reduces your vega risk vs a naked call.",
                "risk":     "Net debit paid.",
                "reward":   "Capped at the spread width minus net debit.",
                "avoid":    "When you have very high conviction — the spread caps your upside.",
                "legs":     [
                    {"type":"call","strike":K,       "premium":mid_ex,    "qty":1,"position":"long"},
                    {"type":"call","strike":K*1.05,  "premium":mid_ex*0.4,"qty":1,"position":"short"},
                ],
            },
            "Long Put": {
                "when":     "Bearish conviction. Expecting a drop. Or hedging existing long positions.",
                "risk":     "Premium paid.",
                "reward":   "Capped at strike price (stock can't go below zero).",
                "avoid":    "After a stock has already dropped significantly — put premium will be high.",
                "legs":     [{"type":"put","strike":K,"premium":mid_ex,"qty":1,"position":"long"}],
            },
            "Cash-Secured Put": {
                "when":     "Happy to buy the stock at the strike price. IV is high (collect rich premium). Good entry strategy.",
                "risk":     "Assigned stock at strike minus premium collected. Same as buying stock at a discount.",
                "reward":   "Premium collected if stock stays above strike.",
                "avoid":    "On stocks you do NOT want to own if assigned.",
                "legs":     [{"type":"put","strike":K,"premium":mid_ex,"qty":1,"position":"short"}],
            },
            "Covered Call": {
                "when":     "Already long the stock (like your META or INTC positions). Want income. Expect sideways to slight upside.",
                "risk":     "Caps your upside if stock rallies above strike. Still exposed to downside on shares.",
                "reward":   "Premium collected. Reduces your cost basis.",
                "avoid":    "If you think the stock is about to make a big move up — you'll miss it.",
                "legs":     [{"type":"call","strike":K*1.05,"premium":mid_ex*0.4,"qty":1,"position":"short"}],
            },
        }

        g = guides[strat_pick]
        st.markdown(f"**When to use:** {g['when']}")
        st.markdown(f"**Max risk:** {g['risk']}  |  **Max reward:** {g['reward']}")
        st.markdown(f"**Avoid when:** {g['avoid']}")

        pnl = payoff_df(S, g["legs"])
        st.line_chart(pnl.set_index("Stock price"))
        st.caption(f"Payoff at expiry. Spot = ${S:.2f}, strike = ${K:.2f}. Horizontal axis = stock price range.")

        net = sum((l["premium"] if l["position"]=="long" else -l["premium"]) for l in g["legs"])
        st.markdown(f"Net cost/credit: **${net:+.3f}** per share = **${net*100:+.2f}** per contract.")

    # =======================================================================
    elif lesson == "8. Position sizing and risk management":
    # =======================================================================

        st.markdown("""
    **The rule that determines whether you survive long enough to get good:**

    Never risk more than 2–5% of your total portfolio on a single options trade.

    Options can go to zero. That is not a tail risk — it is a normal outcome on losing trades.
    If you size correctly, a string of losses doesn't wipe you out.
    """)

        nzdusd_l = fetch_nzdusd()
        port_nzd = st.number_input("Your total trading portfolio (NZD)", 1000.0, 500000.0, 5000.0, 500.0)
        risk_pct = st.slider("Max risk per trade (%)", 1, 10, 3)

        max_risk_nzd  = port_nzd * risk_pct / 100
        max_contracts = max(1, int(max_risk_nzd / (mid_ex * 100 / nzdusd_l)))

        st.markdown(f"""
    **Portfolio:** NZD {port_nzd:,.0f}
    **Max risk per trade ({risk_pct}%):** NZD {max_risk_nzd:,.0f}
    **INTC ATM call costs:** NZD {mid_ex*100/nzdusd_l:,.0f} per contract (your max loss per contract)
    **Max contracts:** {max_contracts}
    """)

        st.success(f"At {risk_pct}% risk, you can buy up to **{max_contracts} contract(s)** on INTC without breaking position sizing rules.")

        st.markdown("""
    **Exit rules — set these before you enter:**

    | Situation | Action |
    |---|---|
    | Option gains 50–100% | Take profit — the math says taking 50% winners consistently beats holding for 100% |
    | Option loses 50% | Cut the loss — the remaining value rarely recovers, and theta keeps eroding it |
    | 7 days to expiry | Close or roll — gamma and theta are extreme in the final week |
    | The thesis is wrong | Exit immediately — don't hold hoping it reverses |

    **The discipline gap:** Most losses in options come from not following exit rules, not from picking the wrong direction.
    """)

    # =======================================================================
    elif lesson == "9. The most common mistakes":
    # =======================================================================

        st.markdown("These are the moves that cost most beginners their first account.")

        mistakes = [
            {
                "title": "Buying far OTM weeklies",
                "why":   "They look cheap. $50 for a contract feels like a lottery ticket. They expire worthless 80%+ of the time. "
                         "The probability of a stock making a 15% move in 5 days is very low.",
                "fix":   "Start with ATM options, 30–45 DTE. Delta 0.40–0.60. More expensive but far more likely to have value at expiry.",
            },
            {
                "title": "Buying calls right before earnings",
                "why":   "IV inflates before earnings. You overpay. Even if the stock moves your way, IV crush can erase the gain. "
                         "See the IV crush lesson.",
                "fix":   "Either enter before IV inflates (1–2 weeks before earnings), or use a spread to reduce vega risk.",
            },
            {
                "title": "Ignoring theta on long holds",
                "why":   "Buying a 30 DTE call and holding it for 25 days while the stock goes sideways. "
                         "Theta has eaten most of your premium even though you were 'right' directionally.",
                "fix":   "Options need to move quickly. If the stock isn't moving in 7–10 days, reassess. Don't hold hoping.",
            },
            {
                "title": "No exit plan",
                "why":   "Entering with no defined profit target or stop loss. Watching a 60% winner turn into a 30% loser.",
                "fix":   "Set your exit levels before you enter: take 50–80% profit, cut at 50% loss. Use the position builder in the Options tab.",
            },
            {
                "title": "Oversizing — putting too much into one trade",
                "why":   "One bad trade wipes 30% of the account. Emotionally devastating. Leads to revenge trading.",
                "fix":   "Lesson 8 covers this. Max 2–5% of portfolio per trade.",
            },
            {
                "title": "Confusing 'cheap' with 'good value'",
                "why":   "A $0.20 option is not cheap if it needs a 25% move to profit. Price means nothing without context.",
                "fix":   "Always check the break-even price and the move needed. These are shown in the Chain & Position section.",
            },
        ]

        for m in mistakes:
            with st.expander(f"❌  {m['title']}"):
                st.markdown(f"**Why it happens:** {m['why']}")
                st.markdown(f"**Fix:** {m['fix']}")

        st.divider()
        st.markdown("""
    **The honest summary:**

    Options are not a shortcut to fast money. They are a tool. Used correctly — right sizing, right strategy for the IV environment, defined exits — they let you express a directional view with capped downside and leveraged upside.

    The screener tells you *what* to watch. The intraday scanner tells you *when* activity is building.
    Options let you act on that signal with less capital at risk than buying shares outright.

    That's the edge. Build it slowly.
    """)

    # =======================================================================
    elif lesson == "10. FOREX basics (different game from stocks)":
    # =======================================================================

        st.warning(
            "**Read this twice.** FOREX is leveraged trading. The same broker who "
            "lets you buy $100 of stock will let you control $10,000–$50,000 of "
            "currency with the same $100. That cuts both ways. Industry-published "
            "data shows **70–80% of retail FX traders lose money** in any given "
            "quarter. This lesson is education only — the FOREX tab gives you "
            "analysis, not advice."
        )

        st.markdown("""
    ### What FOREX is

    You're trading the **exchange rate between two currencies**. The pair `EURUSD`
    means "how many US dollars does it cost to buy 1 euro?" If EURUSD goes from
    1.0800 to 1.0850, the euro got stronger (or the dollar got weaker, same
    thing).

    You're always trading a **pair**: long one currency, short the other. Going
    long EURUSD = bet euro will rise relative to dollar.

    ### The four things every FX trader must know
    """)

        st.markdown("""
    | Concept | Plain English |
    |---|---|
    | **Pip** | The smallest standard price move. For most pairs (4 decimals) 1 pip = 0.0001. For JPY pairs (2 decimals) 1 pip = 0.01. EURUSD moving from 1.0800 to 1.0810 is a 10-pip move. |
    | **Lot** | The size of one standard trade. Standard lot = 100,000 units of the base currency. Mini lot = 10,000. Micro lot = 1,000. Most retail starts with micro lots. |
    | **Pip value** | $ profit/loss per pip. For a $100,000 standard lot of EURUSD, 1 pip = ~$10. For a micro lot, 1 pip = ~$0.10. |
    | **Leverage** | The broker lets you control a position much bigger than your account. 30:1 is the regulated retail max in EU/UK/AU; in the US it's 50:1 on majors. With 30:1 leverage, a 3.3% move against you wipes the account. |
    | **Spread** | The broker's cut. Difference between bid and ask. Tighter on majors (0.5–2 pips), wider on exotics (5–20+ pips). |
    | **Swap / Rollover** | Overnight financing — you pay or receive interest based on the rate differential between the two currencies. Long a high-yielder vs low-yielder = positive swap (the "carry trade"). |
    """)

        st.markdown("""
    ### The three tiers of pairs

    | Tier | Examples | Character |
    |---|---|---|
    | **Majors** | EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD | All include USD. Highest liquidity, tightest spreads. |
    | **Minors / Crosses** | EURGBP, EURJPY, GBPJPY, AUDJPY, EURAUD | Two non-USD majors. Slightly wider spreads. |
    | **Exotics** | USDTRY, USDZAR, USDMXN, USDPLN | One major + one emerging-market currency. Wide spreads, high volatility, gap risk. Avoid as a learner. |

    ### Why FOREX is different from stocks

    | Difference | Why it matters |
    |---|---|
    | **No central exchange** | Each broker has its own price feed. Slippage and spread vary. |
    | **No volume data** | Real volume is fragmented across thousands of dealers. RVOL, breakout-from-base, dollar-volume — all the volume-based signals from the Screener don't apply to FX. |
    | **24-hour trading** | Markets open Sunday 5pm ET, close Friday 5pm ET. Weekend gap risk is real — a Saudi oil announcement on Saturday can gap your stop on Sunday open. |
    | **Mostly mean-reverting** | Currencies range 70%+ of the time. Most equity momentum strategies fail in FX. The big trends happen on macro shifts (central bank pivots, war, inflation). |
    | **Macro-driven** | Central bank policy, interest rate differentials, and trade balances move FX. Earnings, contracts, and 8-Ks don't apply. |
    | **Carry trade is real** | If AUDJPY pays you ~3% annualized just to hold (interest differential), that's a real edge — but it's small per day and you lose it instantly if the pair drops. |

    ### What the FOREX tab in this app gives you

    - Major pairs at a glance: current price, pip move today, RSI, trend bias
    - Pair detail: chart, recent high/low, RSI, ATR, simple bullish/bearish/range verdict
    - **No "recommendation"** — FX doesn't have the same setup grammar as equities,
      and pretending it does would be misleading

    ### What it doesn't give you (and shouldn't)

    - Real-time tick data
    - Broker-specific spreads
    - Economic calendar (use [forexfactory.com](https://www.forexfactory.com) or
      [investing.com/economic-calendar](https://www.investing.com/economic-calendar))
    - Central bank statement parsing
    - Carry trade signals (needs interest rate data feed)

    ### A learner's roadmap

    1. **Trade on a demo account for 3+ months.** Most brokers offer them free.
    2. **Pick one major pair and master it.** EURUSD is the friendliest start —
       tightest spread, deepest liquidity, most public analysis.
    3. **Use 1:1 leverage initially.** Yes, that defeats much of FX's appeal —
       that's the point. Most retail blowups come from leverage, not bad calls.
    4. **Risk per trade ≤ 1% of account.** If you have $1,000, max risk per
       trade is $10. If your stop is 50 pips away, that means 0.02 micro lots —
       absurdly small, and that's correct for a learner.
    5. **Track every trade.** What pair, what setup, what stop, what target,
       what happened. The journal is more important than the strategy.

    ### The honest summary

    If your goal is **to learn how markets work**, FX is a fine sandbox because
    you can trade tiny size with low fees. If your goal is **to make money**,
    the data says you'll probably lose for the first 6–12 months. Most people
    who quit do so before they get good. Most people who get good keep their
    size small for years.

    The FOREX tab is a **screen** — not a signal. Use it to understand what's
    moving, then form your own view.
    """)
