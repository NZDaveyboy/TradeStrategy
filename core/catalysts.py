"""
core/catalysts.py — Catalyst layer for TradeStrategy (roadmap Phase 10).

Pulls earnings, news, analyst, and insider-trade data via yfinance and
computes a composite CatalystScore separate from TradeScore. The split
matters: TradeScore is *price/volume mechanics*; CatalystScore is *what's
happening to this company in the world right now*. Both signals can
disagree, and when they do, that's the most useful information.

Public API:
    get_next_earnings(ticker)             -> dict | None
    get_recent_earnings_history(ticker)   -> list[dict]
    get_recent_news(ticker, limit)        -> list[dict]
    get_analyst_actions(ticker, days)     -> dict
    get_insider_activity(ticker, days)    -> dict
    compute_catalyst_score(ticker, price) -> dict

Scoring philosophy:
    Baseline 50. Positive catalysts (beat trend, target raises, upgrades,
    consensus upside, news momentum, insider buying clusters) push score
    up; negative catalysts (miss, target cuts, downgrades, imminent binary
    event risk) push it down. Clamped to 0-100. Returns a `tags` list of
    plain-English explanations so the UI can render reasoning, not just
    a number.

    Insider note: BUYING is signal, SELLING is mostly noise. Insiders sell
    for many reasons (tax, diversification, 10b5-1 plans); they buy for
    only one (they think the stock will rise). The scoring reflects this
    asymmetry — strong reward for buyer clusters, small penalty for
    unusual selling.
"""

from __future__ import annotations

from datetime import datetime, date, timedelta, timezone
from typing import Any

import pandas as pd
import streamlit as st
import yfinance as yf


# ---------------------------------------------------------------------------
# Earnings
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_next_earnings(ticker: str) -> dict | None:
    """Return next earnings date + EPS/revenue estimate range.

    None for crypto, ETFs, or anything yfinance can't resolve.
    """
    if "-USD" in ticker or "=" in ticker:
        return None
    try:
        cal = yf.Ticker(ticker).calendar
    except Exception:
        return None
    if not cal or not isinstance(cal, dict):
        return None
    dates = cal.get("Earnings Date") or []
    if not dates:
        return None
    next_date = dates[0]
    if not isinstance(next_date, date):
        return None
    today = date.today()
    days_to = (next_date - today).days
    return {
        "date":             next_date.isoformat(),
        "days_to":          days_to,
        "eps_estimate":     cal.get("Earnings Average"),
        "eps_low":          cal.get("Earnings Low"),
        "eps_high":         cal.get("Earnings High"),
        "revenue_estimate": cal.get("Revenue Average"),
    }


@st.cache_data(ttl=86400)
def get_recent_earnings_history(ticker: str, n: int = 4) -> list[dict]:
    """Last `n` quarters of EPS surprises. Most-recent first.

    Each row: {quarter, eps_actual, eps_estimate, surprise_pct}
    """
    if "-USD" in ticker or "=" in ticker:
        return []
    try:
        eh = yf.Ticker(ticker).earnings_history
    except Exception:
        return []
    if eh is None or len(eh) == 0:
        return []
    try:
        eh_sorted = eh.sort_index(ascending=False).head(n)
    except Exception:
        return []
    out = []
    for idx, row in eh_sorted.iterrows():
        qstr = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
        try:
            out.append({
                "quarter":       qstr,
                "eps_actual":    float(row.get("epsActual")    or 0),
                "eps_estimate":  float(row.get("epsEstimate")  or 0),
                "surprise_pct":  float(row.get("surprisePercent") or 0) * 100,
            })
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

# Keyword vocabularies for headline sentiment classification.
# Deliberately bounded — these are the unambiguous financial-news signal
# words. Words used in many neutral contexts (growth, high, low) are
# excluded to keep false-positive rate low.
_BULLISH_KEYWORDS = {
    # Earnings / guidance
    "beat", "beats", "exceeds", "exceeded", "surpasses", "surpassed",
    "raises guidance", "raised guidance", "raises outlook", "guides higher",
    # Analyst
    "upgrade", "upgrades", "upgraded", "target raised", "price target raised",
    "buy rating", "outperform", "overweight",
    # Deals / wins
    "wins contract", "awarded", "partnership", "partners with",
    "acquires", "acquisition closes", "expands", "expansion",
    # Regulatory / approvals
    "fda approval", "approved", "clearance granted",
    # Price action
    "surge", "soars", "rallies", "record high", "all-time high",
    # Insider
    "insider buying", "insider purchase",
}

_BEARISH_KEYWORDS = {
    # Earnings / guidance
    "miss", "misses", "missed", "disappoints", "disappointed",
    "cuts guidance", "lowers guidance", "guides lower", "guides down",
    "warning", "warns", "profit warning",
    # Analyst
    "downgrade", "downgrades", "downgraded", "target cut", "target lowered",
    "price target cut", "sell rating", "underperform", "underweight",
    # Legal / regulatory
    "lawsuit", "sued", "fraud", "investigation", "probe", "subpoena",
    "fine", "fined", "settle", "settlement",
    # Operational distress
    "layoff", "layoffs", "restructuring", "bankruptcy", "default",
    "delist", "delisted", "halt", "halted", "trading halt",
    "ceo resigns", "ceo departs", "ceo steps down",
    # Price action
    "plunge", "plunges", "crash", "crashes", "slumps", "tumbles",
    "fresh low", "52-week low",
}

# Negators that flip a bearish keyword to neutral or bullish.
# Conservative — we only suppress, never flip to bullish, since the
# negation language is varied in financial press.
_NEGATIONS = {"dismissed", "resolved", "ended", "cleared", "denied", "drops"}


def _classify_news_sentiment(text: str) -> str:
    """Classify a headline (or headline+summary) as bullish/bearish/neutral.

    Pure substring matching against bounded keyword vocabularies. Cheap
    and noisy — designed to surface extreme cases (cluster of misses, or
    cluster of beats), not to score individual articles precisely.

    Returns "bullish", "bearish", or "neutral".
    """
    if not text:
        return "neutral"
    t = text.lower()

    has_bull = any(kw in t for kw in _BULLISH_KEYWORDS)
    has_bear = any(kw in t for kw in _BEARISH_KEYWORDS)

    # Suppress bearish reading if a negation appears (e.g. "lawsuit dismissed")
    if has_bear and any(neg in t for neg in _NEGATIONS):
        has_bear = False

    if has_bull and not has_bear:
        return "bullish"
    if has_bear and not has_bull:
        return "bearish"
    # If both fire, treat as neutral — too ambiguous to score directionally
    return "neutral"


@st.cache_data(ttl=1800)
def get_recent_news(ticker: str, limit: int = 10) -> list[dict]:
    """Recent Yahoo Finance news for the ticker.

    Each row: {title, summary, publisher, url, published_at}. Returns
    empty list for crypto/exotic tickers with no Yahoo news coverage.
    """
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    out: list[dict] = []
    for item in raw[:limit]:
        c = item.get("content") if isinstance(item, dict) else None
        if not isinstance(c, dict):
            c = item if isinstance(item, dict) else {}
        title = c.get("title") or ""
        if not title:
            continue
        summary  = (c.get("summary") or c.get("description") or "")[:300]
        provider = (c.get("provider") or {}).get("displayName", "")
        url      = (c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url", "")
        pub_at   = c.get("displayTime") or c.get("pubDate") or ""
        out.append({
            "title":        title,
            "summary":      summary,
            "publisher":    provider,
            "url":          url,
            "published_at": pub_at,
        })
    return out


# ---------------------------------------------------------------------------
# Analyst actions
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_analyst_actions(ticker: str, days: int = 90) -> dict:
    """Consensus rating + recent analyst actions in the last `days`.

    Returns dict with consensus fields and `recent_actions` list. Empty
    dict for tickers with no analyst coverage.
    """
    if "-USD" in ticker or "=" in ticker:
        return {}
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}

    consensus = {
        "consensus_label":      info.get("recommendationKey")        or "—",
        "consensus_mean":       info.get("recommendationMean")       or None,
        "num_analysts":         info.get("numberOfAnalystOpinions")  or None,
        "target_mean":          info.get("targetMeanPrice")          or None,
        "target_high":          info.get("targetHighPrice")          or None,
        "target_low":           info.get("targetLowPrice")           or None,
    }

    actions: list[dict] = []
    summary = {"upgrades": 0, "downgrades": 0, "target_raises": 0, "target_cuts": 0, "maintains": 0}

    try:
        ud = yf.Ticker(ticker).upgrades_downgrades
    except Exception:
        ud = None

    if ud is not None and len(ud) > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        for ts, row in ud.iterrows():
            try:
                ts_aware = ts if ts.tzinfo else ts.tz_localize("UTC")
            except Exception:
                continue
            if ts_aware < cutoff:
                continue
            action          = (row.get("Action") or "").lower()
            target_action   = (row.get("priceTargetAction") or "").lower()
            from_grade      = str(row.get("FromGrade") or "")
            to_grade        = str(row.get("ToGrade") or "")
            current_target  = row.get("currentPriceTarget")
            prior_target    = row.get("priorPriceTarget")

            actions.append({
                "date":           ts_aware.date().isoformat(),
                "firm":           row.get("Firm") or "",
                "action":         action,
                "from_grade":     from_grade,
                "to_grade":       to_grade,
                "target_action":  target_action,
                "target_old":     float(prior_target)   if prior_target   else None,
                "target_new":     float(current_target) if current_target else None,
            })

            # Categorize for summary counts
            # Note: "main" / "init" / "reit" Action with priceTargetAction
            # is how most modern grade changes appear in Yahoo's feed.
            if action in ("up", "upgrade"):
                summary["upgrades"] += 1
            elif action in ("down", "downgrade"):
                summary["downgrades"] += 1
            elif action == "main" or action == "reit":
                summary["maintains"] += 1

            if "raise" in target_action or "increase" in target_action:
                summary["target_raises"] += 1
            elif "lower" in target_action or "decrease" in target_action or "cut" in target_action:
                summary["target_cuts"] += 1

    consensus["recent_actions"] = actions
    consensus["recent_summary"] = summary
    consensus["window_days"]    = days
    return consensus


# ---------------------------------------------------------------------------
# Insider activity
# ---------------------------------------------------------------------------

# Pattern classification from yfinance's `Text` field
_BUY_RE  = ("purchase",)
_SELL_RE = ("sale",)
_NEUTRAL_RE = ("conversion", "exercise", "award", "grant", "gift")


def _classify_insider(text: str) -> str:
    t = (text or "").lower()
    if any(x in t for x in _BUY_RE):
        return "buy"
    if any(x in t for x in _SELL_RE):
        return "sell"
    if any(x in t for x in _NEUTRAL_RE):
        return "neutral"
    return "other"


@st.cache_data(ttl=3600)
def get_insider_activity(ticker: str, days: int = 90) -> dict:
    """Aggregate recent insider trades for `ticker` from yfinance.

    Returns dict with:
      buyer_count        — unique insiders who bought in window
      seller_count       — unique insiders who sold in window
      total_buy_value    — sum $ of purchase transactions
      total_sell_value   — sum $ of sale transactions
      net_value          — buy − sell
      transactions       — list of recent rows (date, insider, position, kind, shares, value)
      net_6m_summary     — dict from yfinance's insider_purchases aggregate
      window_days        — echoed back for transparency

    Empty dict for crypto, ETFs, FX. Never raises.
    """
    if "-USD" in ticker or "=" in ticker:
        return {}

    out: dict = {
        "buyer_count":      0,
        "seller_count":     0,
        "total_buy_value":  0.0,
        "total_sell_value": 0.0,
        "net_value":        0.0,
        "transactions":     [],
        "net_6m_summary":   {},
        "window_days":      days,
    }

    # ── Pull transactions DataFrame ────────────────────────────────────────
    try:
        df = yf.Ticker(ticker).insider_transactions
    except Exception:
        df = None

    if df is not None and len(df) > 0:
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
        try:
            df = df.copy()
            df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")
            df = df[df["Start Date"] >= cutoff]
        except Exception:
            pass

        buyers, sellers = set(), set()
        for _, row in df.iterrows():
            kind   = _classify_insider(str(row.get("Text") or ""))
            value  = row.get("Value")
            shares = row.get("Shares")
            insider = str(row.get("Insider") or "").strip()
            position = str(row.get("Position") or "").strip()
            sd     = row.get("Start Date")
            sd_str = sd.strftime("%Y-%m-%d") if hasattr(sd, "strftime") else str(sd)

            try:
                value_f = float(value) if pd.notna(value) else 0.0
            except Exception:
                value_f = 0.0
            try:
                shares_f = float(shares) if pd.notna(shares) else 0.0
            except Exception:
                shares_f = 0.0

            out["transactions"].append({
                "date":     sd_str,
                "insider":  insider,
                "position": position,
                "kind":     kind,
                "shares":   shares_f,
                "value":    value_f,
            })

            if kind == "buy":
                if insider:
                    buyers.add(insider)
                out["total_buy_value"] += value_f
            elif kind == "sell":
                if insider:
                    sellers.add(insider)
                out["total_sell_value"] += value_f

        out["buyer_count"]  = len(buyers)
        out["seller_count"] = len(sellers)
        out["net_value"]    = out["total_buy_value"] - out["total_sell_value"]

    # ── Pull yfinance's 6m aggregate for cross-reference ───────────────────
    try:
        ip = yf.Ticker(ticker).insider_purchases
    except Exception:
        ip = None
    if ip is not None and len(ip) > 0:
        try:
            summary = {}
            for _, r in ip.iterrows():
                label  = str(r.get("Insider Purchases Last 6m") or "").strip()
                shares = r.get("Shares")
                if label and pd.notna(shares):
                    try:
                        summary[label] = float(shares)
                    except Exception:
                        pass
            out["net_6m_summary"] = summary
        except Exception:
            pass

    return out


# ---------------------------------------------------------------------------
# Composite Catalyst Score
# ---------------------------------------------------------------------------

def compute_catalyst_score(ticker: str, price: float | None = None) -> dict:
    """Compute a 0-100 CatalystScore + components + plain-English tags.

    Baseline 50. Positive catalysts push up, negative push down.
    Returns dict with score, components, tags. Never raises — returns
    score=None when nothing catalyst-relevant can be fetched.
    """
    earnings_next = get_next_earnings(ticker)
    earnings_hist = get_recent_earnings_history(ticker, n=4)
    news          = get_recent_news(ticker, limit=15)
    analyst       = get_analyst_actions(ticker, days=90)
    insider       = get_insider_activity(ticker, days=90)

    # No catalyst data at all → return neutral
    has_insider_data = bool(insider.get("transactions") or insider.get("net_6m_summary"))
    if not any([earnings_next, earnings_hist, news, analyst.get("num_analysts"), has_insider_data]):
        return {
            "score":      None,
            "components": {},
            "tags":       ["No catalyst data available for this ticker."],
            "data": {
                "earnings_next": earnings_next,
                "earnings_history": earnings_hist,
                "news": news,
                "analyst": analyst,
                "insider": insider,
            },
        }

    components: dict[str, float] = {}
    tags: list[str] = []
    score = 50.0

    # ---- Earnings beat/miss trend (-25..+25) ----
    if earnings_hist:
        recent = earnings_hist[0]
        sp = recent.get("surprise_pct", 0)
        if sp >= 10:
            components["last_beat"] = +15
            tags.append(f"Strong earnings beat last quarter (+{sp:.1f}% surprise)")
        elif sp >= 2:
            components["last_beat"] = +8
            tags.append(f"Modest earnings beat last quarter (+{sp:.1f}% surprise)")
        elif sp <= -10:
            components["last_beat"] = -15
            tags.append(f"Big earnings miss last quarter ({sp:.1f}% surprise)")
        elif sp <= -2:
            components["last_beat"] = -8
            tags.append(f"Earnings miss last quarter ({sp:.1f}% surprise)")
        else:
            components["last_beat"] = 0

        # Trend across last 4 quarters
        beats = sum(1 for q in earnings_hist if q.get("surprise_pct", 0) > 0)
        misses = sum(1 for q in earnings_hist if q.get("surprise_pct", 0) < 0)
        if beats >= 4 and len(earnings_hist) >= 4:
            components["beat_streak"] = +10
            tags.append("4-quarter beat streak")
        elif misses >= 2 and len(earnings_hist) >= 3:
            components["miss_pattern"] = -10
            tags.append(f"{misses} of last {len(earnings_hist)} quarters were misses")

    # ---- Imminent earnings risk (binary event) (-15..0) ----
    if earnings_next and earnings_next.get("days_to") is not None:
        d = earnings_next["days_to"]
        if 0 <= d <= 7:
            components["earnings_imminent"] = -15
            tags.append(f"⚠ Earnings in {d} days — binary event risk")
        elif 8 <= d <= 14:
            components["earnings_imminent"] = -8
            tags.append(f"Earnings in {d} days — be aware of binary event window")
        elif 15 <= d <= 30:
            components["earnings_proximity"] = -3
            tags.append(f"Earnings in {d} days")
        elif d < 0 and d >= -7:
            components["earnings_just_passed"] = +3
            tags.append(f"Earnings reported {abs(d)} days ago — post-earnings window")

    # ---- Analyst momentum (-25..+25) ----
    if analyst.get("recent_summary"):
        s = analyst["recent_summary"]
        net_grade = s["upgrades"] - s["downgrades"]
        net_target = s["target_raises"] - s["target_cuts"]

        if net_grade >= 3:
            components["upgrade_momentum"] = +15
            tags.append(f"{s['upgrades']} upgrades vs {s['downgrades']} downgrades in 90d")
        elif net_grade >= 1:
            components["upgrade_momentum"] = +8
        elif net_grade <= -3:
            components["downgrade_momentum"] = -15
            tags.append(f"{s['downgrades']} downgrades vs {s['upgrades']} upgrades in 90d")
        elif net_grade <= -1:
            components["downgrade_momentum"] = -8

        if net_target >= 3:
            components["target_raises"] = +12
            tags.append(f"{s['target_raises']} price target raises in 90d")
        elif net_target >= 1:
            components["target_raises"] = +5
        elif net_target <= -3:
            components["target_cuts"] = -12
            tags.append(f"{s['target_cuts']} price target cuts in 90d")
        elif net_target <= -1:
            components["target_cuts"] = -5

    # ---- Consensus target upside vs current price (-10..+15) ----
    tgt = analyst.get("target_mean")
    n_an = analyst.get("num_analysts") or 0
    if tgt and price and price > 0 and n_an >= 3:
        upside_pct = (tgt - price) / price * 100
        if upside_pct >= 30:
            components["target_upside"] = +15
            tags.append(
                f"Mean analyst target ${tgt:.2f} = {upside_pct:+.0f}% upside "
                f"(consensus from {n_an} analysts)"
            )
        elif upside_pct >= 10:
            components["target_upside"] = +8
            tags.append(f"Mean target ${tgt:.2f} = {upside_pct:+.0f}% upside")
        elif upside_pct <= -15:
            components["target_downside"] = -10
            tags.append(
                f"Mean target ${tgt:.2f} is {abs(upside_pct):.0f}% BELOW current price — "
                f"market priced ahead of analysts"
            )
        elif upside_pct <= -5:
            components["target_downside"] = -5

    # ---- Consensus rating label (-8..+8) ----
    label = (analyst.get("consensus_label") or "").lower()
    if label in ("strong_buy", "buy"):
        components["consensus_label"] = +8
        tags.append(f"Consensus rating: **{label.replace('_', ' ')}** ({n_an} analysts)")
    elif label in ("sell", "strong_sell"):
        components["consensus_label"] = -8
        tags.append(f"Consensus rating: **{label.replace('_', ' ')}** ({n_an} analysts)")
    elif label == "hold" and n_an >= 5:
        tags.append(f"Consensus rating: hold ({n_an} analysts)")

    # ---- Insider activity (-5..+15) — buying is signal, selling is noise ----
    # Asymmetric scoring: 3+ insider buyers in 90d with meaningful $ is one
    # of the strongest single signals in the catalyst layer. Selling is
    # mostly noise (10b5-1 plans, diversification, tax) unless extreme.
    if insider:
        buyers       = insider.get("buyer_count", 0)
        sellers      = insider.get("seller_count", 0)
        buy_value    = insider.get("total_buy_value", 0.0) or 0.0
        sell_value   = insider.get("total_sell_value", 0.0) or 0.0

        # Cluster bonus — multiple buyers with non-trivial $ is the gold signal
        if buyers >= 3 and buy_value >= 250_000:
            components["insider_cluster"] = +15
            tags.append(
                f"🟢 Insider buy cluster: {buyers} buyers, "
                f"${buy_value/1e3:,.0f}k total in 90d"
            )
        elif buyers >= 2 and buy_value >= 100_000:
            components["insider_buying"] = +8
            tags.append(f"Insider buying: {buyers} buyers, ${buy_value/1e3:,.0f}k in 90d")
        elif buyers >= 1 and buy_value >= 500_000:
            # Big single buy — CEO/CFO putting their own money in at scale is
            # informative even from one person, especially at a depressed price
            components["insider_big_single_buy"] = +6
            tags.append(f"🟢 Single insider purchased ${buy_value/1e3:,.0f}k in 90d")
        elif buyers >= 1 and buy_value >= 50_000:
            components["insider_single_buy"] = +3
            tags.append(f"1 insider purchase, ${buy_value/1e3:,.0f}k in 90d")

        # Selling — only penalise if it's lopsided (sellers >> buyers, big $)
        if sellers >= 3 and buyers == 0 and sell_value >= 500_000:
            components["insider_selling"] = -5
            tags.append(
                f"Heavy insider selling: {sellers} sellers, "
                f"${sell_value/1e3:,.0f}k, no buyers"
            )

    # ---- News intensity + sentiment (-8..+5) ----
    # Intensity = number of recent items (coverage signal, direction-neutral)
    # Sentiment = bullish vs bearish keyword balance (direction signal,
    # asymmetric: bearish keywords are more reliable than bullish)
    if news:
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        recent_count = 0
        bullish_count = 0
        bearish_count = 0
        for n in news:
            ts = n.get("published_at") or ""
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt < cutoff:
                    continue
            except Exception:
                continue
            recent_count += 1
            blob = (n.get("title") or "") + " " + (n.get("summary") or "")
            sentiment = _classify_news_sentiment(blob)
            if sentiment == "bullish":
                bullish_count += 1
            elif sentiment == "bearish":
                bearish_count += 1

        # Coverage component
        if recent_count >= 5:
            components["news_intensity"] = +5
            tags.append(f"{recent_count} news items in last 7 days — active coverage")
        elif recent_count >= 2:
            components["news_intensity"] = +2

        # Sentiment component (asymmetric — bearish news is more reliable)
        if bearish_count >= 3 and bullish_count == 0:
            components["news_sentiment"] = -8
            tags.append(f"🔴 {bearish_count} bearish headlines in last 7d, no bullish")
        elif bearish_count >= 1 and bullish_count == 0:
            components["news_sentiment"] = -3
            tags.append(f"Bearish-leaning recent news ({bearish_count} bearish vs 0 bullish)")
        elif bullish_count >= 3 and bearish_count == 0:
            components["news_sentiment"] = +5
            tags.append(f"🟢 {bullish_count} bullish headlines in last 7d, no bearish")
        elif bullish_count >= 2 and bearish_count == 0:
            components["news_sentiment"] = +3
            tags.append(f"Bullish-leaning recent news ({bullish_count} bullish vs 0 bearish)")
        # mixed (both fire) → 0, no tag — too ambiguous to score

    # ---- Combine ----
    score += sum(components.values())
    score = max(0.0, min(100.0, score))

    return {
        "score":      round(score, 1),
        "components": components,
        "tags":       tags,
        "data": {
            "earnings_next":    earnings_next,
            "earnings_history": earnings_hist,
            "news":             news,
            "analyst":          analyst,
            "insider":          insider,
        },
    }
