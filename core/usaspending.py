"""
core/usaspending.py — USAspending.gov catalyst feed for the Lookup tab.

Pulls a ticker's recent federal contract transactions. Auto-derives a fuzzy
recipient search string from yfinance's longName so we don't need to hand-
maintain a {ticker: search_string} map. Returns an empty list for tickers
that aren't federal contractors (SOFI, NVDA, etc.) — that's correct.

This is NOT a trading strategy. It's a catalyst signal: "did a public-co
contractor get a meaningful federal award recently?" Use alongside SEC
filings, earnings, and TradeScore — not as a standalone signal.

Public API:
    get_contractor_search_string(ticker) -> str | None
    get_recent_contracts(ticker, days=180, min_amount=1_000_000) -> list[dict]
"""

from __future__ import annotations

import re

import streamlit as st
import requests
import yfinance as yf


# USAspending public transaction-search endpoint. No auth, no API key needed.
# Returns per-action data (each modification is its own row), filtered to
# definitive contracts and contract-like vehicles.
_API_URL        = "https://api.usaspending.gov/api/v2/search/spending_by_transaction/"
_CONTRACT_TYPES = ["A", "B", "C", "D"]  # BPA call, PO, delivery order, definitive contract


# ---------------------------------------------------------------------------
# Ticker → search-string derivation
# ---------------------------------------------------------------------------

# Common corporate suffixes to strip from longName before sending to USAspending's
# fuzzy recipient_search_text filter. USAspending stores recipients in
# ALL-CAPS with their own punctuation, so the cleaner the input, the better
# the match.
_CORP_SUFFIX_RE = re.compile(
    r"\b("
    r"corporation|corp|incorporated|inc|company|co|"
    r"limited|ltd|llc|lp|llp|plc|"
    r"holdings|holding|group|technologies|technology|systems|industries|international|intl"
    r")\.?\b",
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[,.;:!?&]")


@st.cache_data(ttl=86400)
def get_contractor_search_string(ticker: str) -> str | None:
    """Derive a USAspending-friendly search string from a ticker.

    Strategy: pull longName from yfinance, strip corp suffixes, take the
    first 2-3 words. Returns None for crypto, ETFs, or anything we can't
    resolve to a company name.
    """
    if "-USD" in ticker:
        return None
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return None
    name = (info.get("longName") or info.get("shortName") or "").strip()
    if not name:
        return None
    cleaned = _PUNCT_RE.sub(" ", name)
    cleaned = _CORP_SUFFIX_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    # Take first 2 words — specific enough to avoid false matches, fuzzy
    # enough to catch subsidiaries (e.g. "Kratos Defense" matches both
    # "KRATOS DEFENSE & ROCKET SUPPORT" and "KRATOS UNMANNED AERIAL").
    # 3 words was too specific (broke on "Kratos Defense Security").
    return " ".join(cleaned.split()[:2])


# ---------------------------------------------------------------------------
# USAspending fetch
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_recent_contracts(
    ticker: str,
    days: int = 180,
    min_amount: float = 250_000.0,
    limit: int = 10,
) -> list[dict]:
    """Return up to `limit` most recent federal contract transactions for `ticker`.

    Each transaction is a dict with keys:
        action_date, amount, award_id, recipient, agency, modification, description

    Returns [] if the ticker has no contracts, isn't a federal contractor,
    or we can't derive a search string. Never raises.
    """
    search = get_contractor_search_string(ticker)
    if not search:
        return []

    from datetime import date, timedelta
    end = date.today()
    start = end - timedelta(days=days)

    payload = {
        "filters": {
            "recipient_search_text": [search],
            "time_period": [{"start_date": start.isoformat(), "end_date": end.isoformat()}],
            "award_type_codes": _CONTRACT_TYPES,
            "award_amounts": [{"lower_bound": min_amount}],
        },
        "fields": [
            "Action Date",
            "Transaction Amount",
            "Award ID",
            "Recipient Name",
            "Awarding Agency",
            "Mod",
        ],
        "page": 1,
        "limit": limit,
        "sort": "Action Date",
        "order": "desc",
    }

    try:
        r = requests.post(_API_URL, json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    out: list[dict] = []
    for row in data.get("results", []):
        amt = float(row.get("Transaction Amount") or 0.0)
        action_d = row.get("Action Date") or ""
        if amt < min_amount or not action_d:
            continue
        mod = str(row.get("Mod") or "0")
        out.append({
            "action_date":  action_d,
            "amount":       amt,
            "award_id":     row.get("Award ID") or "",
            "recipient":    row.get("Recipient Name") or "",
            "agency":       row.get("Awarding Agency") or "",
            "modification": mod,
            "kind":         "NEW" if mod in ("0", "", None) else f"MOD {mod}",
        })
    return out
