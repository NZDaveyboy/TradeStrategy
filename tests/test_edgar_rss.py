"""
tests/test_edgar_rss.py

Tests for core/edgar_rss.py — Atom feed parsing and signal matching.

No network calls. All tests use hand-built XML bytes.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.edgar_rss import (
    EarlySignal,
    _parse_dt,
    _parse_feed,
    _match_theme,
    reset_seen_urls,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atom_feed(entries: list[dict]) -> bytes:
    """Build a minimal Atom feed XML from a list of entry dicts."""
    ns = "http://www.w3.org/2005/Atom"
    parts = [f'<?xml version="1.0"?>\n<feed xmlns="{ns}">']
    for e in entries:
        parts.append(
            f'  <entry>'
            f'    <title>{e.get("title", "Unknown Corp (CIK 0001234567) (8-K)")}</title>'
            f'    <link href="{e.get("url", "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001234567")}"/>'
            f'    <updated>{e.get("updated", "2024-01-15T10:30:00")}</updated>'
            f'  </entry>'
        )
    parts.append("</feed>")
    return "\n".join(parts).encode()


# ---------------------------------------------------------------------------
# 1. _parse_dt handles valid and invalid inputs
# ---------------------------------------------------------------------------

def test_parse_dt_iso_with_seconds():
    dt = _parse_dt("2024-03-15T14:30:00")
    assert dt == datetime(2024, 3, 15, 14, 30, 0, tzinfo=timezone.utc)


def test_parse_dt_iso_date_only():
    dt = _parse_dt("2024-03-15")
    assert dt == datetime(2024, 3, 15, tzinfo=timezone.utc)


def test_parse_dt_none_returns_epoch():
    dt = _parse_dt(None)
    assert dt.year == 1970


def test_parse_dt_empty_string_returns_epoch():
    dt = _parse_dt("")
    assert dt.year == 1970


# ---------------------------------------------------------------------------
# 2. _match_theme returns correct tickers
# ---------------------------------------------------------------------------

def test_match_theme_nvidia():
    assert _match_theme("NVIDIA CORP") == "NVDA"


def test_match_theme_case_insensitive():
    assert _match_theme("Advanced Micro Devices Inc") == "AMD"


def test_match_theme_no_match_returns_none():
    assert _match_theme("Completely Unknown Corp LLC") is None


def test_match_theme_partial_fragment():
    assert _match_theme("Palantir Technologies Inc.") == "PLTR"


# ---------------------------------------------------------------------------
# 3. _parse_feed matches theme tickers correctly
# ---------------------------------------------------------------------------

def test_parse_feed_matches_theme_company():
    reset_seen_urls()
    xml = _atom_feed([{
        "title": "NVIDIA CORP (CIK 0001045810) (8-K)",
        "url":   "https://www.sec.gov/Archives/edgar/data/1045810/000104581024000001",
        "updated": "2024-03-15T10:00:00",
    }])
    signals = _parse_feed(xml, "8-K", frozenset())
    assert len(signals) == 1
    assert signals[0].ticker == "NVDA"
    assert signals[0].filing_type == "8-K"
    assert signals[0].match_source == "theme"


def test_parse_feed_deduplicates_same_url():
    reset_seen_urls()
    entry = {
        "title":   "NVIDIA CORP (CIK 0001045810) (8-K)",
        "url":     "https://www.sec.gov/Archives/edgar/data/1045810/000104581024000099",
        "updated": "2024-03-15T10:00:00",
    }
    xml = _atom_feed([entry])
    signals_first  = _parse_feed(xml, "8-K", frozenset())
    signals_second = _parse_feed(xml, "8-K", frozenset())
    assert len(signals_first)  == 1
    assert len(signals_second) == 0   # duplicate — already seen


def test_parse_feed_no_match_returns_empty():
    reset_seen_urls()
    xml = _atom_feed([{
        "title":   "TOTALLY UNKNOWN WIDGET CO (CIK 0009999999) (8-K)",
        "url":     "https://www.sec.gov/Archives/edgar/data/9999999/000999999924000001",
        "updated": "2024-03-15T10:00:00",
    }])
    signals = _parse_feed(xml, "8-K", frozenset())
    assert signals == []


def test_parse_feed_invalid_xml_returns_empty():
    reset_seen_urls()
    signals = _parse_feed(b"this is not xml at all <<<", "8-K", frozenset())
    assert signals == []
