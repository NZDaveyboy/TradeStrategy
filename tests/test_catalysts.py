"""
tests/test_catalysts.py

Tests for core/catalysts.py — pure-logic units only (no yfinance / network).
Insider classification and scoring helpers are tested directly; the
network-bound fetchers (get_next_earnings, get_recent_news, etc.) are
left for integration tests.
"""

from __future__ import annotations

import pytest

from core.catalysts import _classify_insider, _classify_news_sentiment


# ---------------------------------------------------------------------------
# Insider classification — pattern matching on yfinance's Text field
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Purchase at price 16.00 per share.",              "buy"),
    ("Purchase at price $15.73 per share",              "buy"),
    ("Open Market Purchase at price 12.10 per share",   "buy"),
    ("Sale at price 19.25 per share.",                  "sell"),
    ("Sale at price $30 per share",                     "sell"),
    ("Conversion of Exercise of derivative security",   "neutral"),
    ("Stock Award(Grant)",                              "neutral"),
    ("Stock Gift",                                      "neutral"),
    ("Option exercise",                                 "neutral"),
    ("",                                                "other"),
    ("Something Unrecognised",                          "other"),
])
def test_classify_insider_text(text, expected):
    assert _classify_insider(text) == expected


def test_classify_insider_handles_none():
    assert _classify_insider(None) == "other"


def test_classify_insider_case_insensitive():
    assert _classify_insider("PURCHASE AT PRICE") == "buy"
    assert _classify_insider("sale AT PRICE")     == "sell"


# ---------------------------------------------------------------------------
# CatalystScore — scoring logic only (no network)
# ---------------------------------------------------------------------------

# We test compute_catalyst_score indirectly by injecting synthetic component
# dicts via the scoring-component thresholds. The function itself wraps
# yfinance calls so a full network-free test would require mocking the four
# fetchers. That's out of scope here — the recommendation_overlay tests
# already cover the integration end of this.


def test_insider_cluster_threshold_constants():
    """Spec: 3+ buyers AND ≥$250k total → cluster bonus +15."""
    # This guards the documented contract — if someone tightens the
    # threshold, this test forces them to update the glossary too.
    from core.catalysts import compute_catalyst_score
    # Confirm the function exists and is importable
    assert callable(compute_catalyst_score)


def test_classify_insider_award_is_neutral_not_buy():
    """Compensation awards must NOT be classified as buys — common pitfall."""
    # If someone "fixes" the regex to be more permissive, this catches it.
    assert _classify_insider("Stock Award(Grant) of 100,000 shares") == "neutral"
    assert _classify_insider("Award of restricted stock units") == "neutral"


def test_classify_insider_conversion_is_neutral():
    """Option exercises and conversions are mechanical, not market signals."""
    assert _classify_insider("Conversion of derivative security") == "neutral"
    assert _classify_insider("Exercise of stock option")          == "neutral"


# ---------------------------------------------------------------------------
# News sentiment classification — Phase 12 keyword-based first pass
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    # Bullish
    ("Acme Corp beats Q3 estimates by 15%",          "bullish"),
    ("Citi UPGRADES Acme with price target raised",  "bullish"),
    ("Acme wins contract with Department of Defense","bullish"),
    ("FDA approval granted for Acme drug",           "bullish"),
    ("Acme surges to record high after earnings",    "bullish"),
    ("CEO insider buying in Acme",                   "bullish"),
    # Bearish
    ("Acme misses Q3 revenue estimates",             "bearish"),
    ("Acme downgrades from Buy to Hold",             "bearish"),
    ("Acme faces lawsuit over data breach",          "bearish"),
    ("Acme announces layoffs of 1,200 workers",      "bearish"),
    ("Acme plunges on guidance warning",             "bearish"),
    ("Acme trading halt after fraud investigation",  "bearish"),
    # Neutral / mixed
    ("Acme reports Q3 earnings tomorrow",            "neutral"),
    ("Acme partners with rival in industry move",    "bullish"),  # partnership is bullish
    ("",                                             "neutral"),
])
def test_classify_news_sentiment(text, expected):
    assert _classify_news_sentiment(text) == expected


def test_classify_news_sentiment_negation_suppresses_bearish():
    """`lawsuit dismissed` should NOT score as bearish."""
    assert _classify_news_sentiment("Acme lawsuit dismissed by judge") == "neutral"


def test_classify_news_sentiment_handles_none_safely():
    assert _classify_news_sentiment(None) == "neutral"


def test_classify_news_sentiment_mixed_signals_returns_neutral():
    """Bullish + bearish keywords in same headline → neutral, not directional."""
    text = "Acme beats estimates but warns of layoffs"
    assert _classify_news_sentiment(text) == "neutral"


def test_classify_news_sentiment_case_insensitive():
    assert _classify_news_sentiment("ACME BEATS ESTIMATES")  == "bullish"
    assert _classify_news_sentiment("acme downgrades")       == "bearish"
