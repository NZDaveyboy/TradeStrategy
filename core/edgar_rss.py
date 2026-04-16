"""
core/edgar_rss.py — SEC EDGAR RSS watcher for early-signal filing detection.

Polls three EDGAR Atom feeds and returns EarlySignal objects for any filing
that matches either the theme watchlist or the current screener universe.

Usage:
    from core.edgar_rss import poll_early_signals
    signals = poll_early_signals(screener_tickers=["NVDA", "AMD", "AAPL"])

Rate-limit note: EDGAR requires a User-Agent header with contact info.
Edit CONTACT_EMAIL below to your own address before deploying.
"""

from __future__ import annotations

import json
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

# ---------------------------------------------------------------------------
# User-editable config
# ---------------------------------------------------------------------------

CONTACT_EMAIL: str = "tradestrategy@example.com"

# ---------------------------------------------------------------------------
# EDGAR Atom feed URLs — last 40 entries each, refreshed every 2 minutes
# ---------------------------------------------------------------------------

_FEED_URLS: dict[str, str] = {
    "8-K":   "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count=40&search_text=&output=atom",
    "S-1":   "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=S-1&dateb=&owner=include&count=40&search_text=&output=atom",
    "SC 13G": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=SC+13G&dateb=&owner=include&count=40&search_text=&output=atom",
}

_ATOM_NS = "http://www.w3.org/2005/Atom"

# ---------------------------------------------------------------------------
# Theme company-name → ticker mapping
# Fragment matching: key is a lowercase substring of the EDGAR company name.
# Only needs to cover the AI_INFRASTRUCTURE watchlist.
# ---------------------------------------------------------------------------

_THEME_NAME_TO_TICKER: dict[str, str] = {
    # Semiconductors / compute
    "nvidia":           "NVDA",
    "advanced micro":   "AMD",
    "broadcom":         "AVGO",
    "marvell":          "MRVL",
    "super micro":      "SMCI",
    "astera labs":      "ALAB",
    "asml":             "ASML",
    "taiwan semiconductor": "TSM",
    # Networking / interconnect
    "arista":           "ANET",
    "ciena":            "CIEN",
    "lumentum":         "LITE",
    "coherent":         "COHR",
    # Power / cooling
    "vertiv":           "VRT",
    "eaton":            "ETN",
    "hubbell":          "HUBB",
    "powell industries": "POWL",
    # Cloud hyperscalers
    "microsoft":        "MSFT",
    "amazon":           "AMZN",
    "alphabet":         "GOOGL",
    # Pure-play AI / quantum
    "palantir":         "PLTR",
    "ionq":             "IONQ",
    "rocket lab":       "RKLB",
    "bigbear":          "BBAI",
    # Data infrastructure
    "snowflake":        "SNOW",
    "mongodb":          "MDB",
    "cloudflare":       "NET",
}

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EarlySignal:
    ticker:      str        # matched ticker (upper-case)
    company:     str        # company name from EDGAR
    filing_type: str        # "8-K" | "S-1" | "SC 13G"
    filed_at:    datetime   # UTC datetime
    url:         str        # EDGAR filing index URL
    match_source: str       # "theme" | "screener"


# ---------------------------------------------------------------------------
# Module-level deduplication — reset only on interpreter restart
# ---------------------------------------------------------------------------

_seen_urls: set[str] = set()

# ---------------------------------------------------------------------------
# CIK → ticker reverse map (lazy-loaded once from EDGAR JSON)
# ---------------------------------------------------------------------------

_cik_ticker_map: dict[str, str] = {}
_cik_map_loaded: bool = False

_COMPANY_TICKERS_URL = (
    "https://www.sec.gov/files/company_tickers.json"
)


def _load_cik_map() -> None:
    global _cik_ticker_map, _cik_map_loaded
    if _cik_map_loaded:
        return
    try:
        req = urllib.request.Request(
            _COMPANY_TICKERS_URL,
            headers={"User-Agent": f"TradeStrategy {CONTACT_EMAIL}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data: dict = json.loads(resp.read())
        # data = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "..."}, ...}
        for entry in data.values():
            cik = str(entry.get("cik_str", "")).zfill(10)
            ticker = str(entry.get("ticker", "")).upper()
            if cik and ticker:
                _cik_ticker_map[cik] = ticker
    except Exception:
        pass  # network unavailable — map stays empty, name matching still works
    finally:
        _cik_map_loaded = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_dt(text: str | None) -> datetime:
    """Parse an ISO-8601 datetime string to UTC datetime. Returns epoch on failure."""
    if not text:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    text = text.strip().rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _match_theme(company_name: str) -> str | None:
    """Return ticker if company_name contains a known theme fragment, else None."""
    lower = company_name.lower()
    for fragment, ticker in _THEME_NAME_TO_TICKER.items():
        if fragment in lower:
            return ticker
    return None


def _match_screener(cik: str, screener_set: frozenset[str]) -> str | None:
    """Return ticker if CIK maps to a ticker in the screener universe, else None."""
    ticker = _cik_ticker_map.get(cik.zfill(10))
    if ticker and ticker.upper() in screener_set:
        return ticker.upper()
    return None


def _parse_feed(xml_bytes: bytes, filing_type: str, screener_set: frozenset[str]) -> list[EarlySignal]:
    """Parse one Atom feed and return new EarlySignal objects."""
    signals: list[EarlySignal] = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return signals

    for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
        url_el      = entry.find(f"{{{_ATOM_NS}}}link")
        url         = url_el.get("href", "") if url_el is not None else ""
        updated_el  = entry.find(f"{{{_ATOM_NS}}}updated")
        filed_at    = _parse_dt(updated_el.text if updated_el is not None else None)
        title_el    = entry.find(f"{{{_ATOM_NS}}}title")
        title       = title_el.text or "" if title_el is not None else ""

        # Extract company name: title format is typically "company-name (CIK XXXXXXXXXX) (type)"
        # or "FILING-TYPE (company-name) - CIK XXXXXXXXXX"
        # We'll try to grab text before first "(" as company name
        company = title.split("(")[0].strip() if "(" in title else title.strip()

        # Extract CIK from title text: look for 10-digit number after "CIK "
        cik = ""
        import re as _re
        cik_match = _re.search(r"(?:CIK\s*)?(\d{7,10})", title)
        if cik_match:
            cik = cik_match.group(1).zfill(10)

        if not url or url in _seen_urls:
            continue

        # Try theme name match first, then screener CIK match
        ticker = _match_theme(company)
        source = "theme"
        if ticker is None and screener_set and cik:
            ticker = _match_screener(cik, screener_set)
            source = "screener"

        if ticker is None:
            continue

        _seen_urls.add(url)
        signals.append(EarlySignal(
            ticker=ticker.upper(),
            company=company,
            filing_type=filing_type,
            filed_at=filed_at,
            url=url,
            match_source=source,
        ))

    return signals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def poll_early_signals(
    screener_tickers: Sequence[str] = (),
) -> list[EarlySignal]:
    """
    Poll all three EDGAR Atom feeds and return new EarlySignal objects.

    Args:
        screener_tickers: current screener universe (e.g. today's results).
                          Signals matching these via CIK map get match_source="screener".

    Returns:
        List of EarlySignal objects, deduplicated across calls (module-level seen set).
        Empty list if network is unavailable.
    """
    _load_cik_map()
    screener_set = frozenset(t.upper() for t in screener_tickers)
    all_signals: list[EarlySignal] = []

    for filing_type, url in _FEED_URLS.items():
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": f"TradeStrategy {CONTACT_EMAIL}"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_bytes = resp.read()
        except Exception:
            continue  # network failure — skip this feed

        signals = _parse_feed(xml_bytes, filing_type, screener_set)
        all_signals.extend(signals)

    # Sort by most recent first
    all_signals.sort(key=lambda s: s.filed_at, reverse=True)
    return all_signals


def reset_seen_urls() -> None:
    """Clear the deduplication set. Intended for tests only."""
    global _seen_urls
    _seen_urls = set()
