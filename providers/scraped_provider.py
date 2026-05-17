"""
providers/scraped_provider.py — HTML-scraping ticker-discovery providers.

Separated from yfinance_provider.py in Phase 5 of the upgrade roadmap so
that the data-fetching layer (yfinance) and the discovery layer (Finviz
scraping) can evolve independently. yfinance is a structured-API
dependency; Finviz is an HTML-scraping dependency with a different
failure profile.

Currently contains:

  - FinvizDiscoveryProvider — implements TickerDiscoveryProvider by
    scraping Finviz's top-gainers screener page.

Future scraping-based providers (e.g. Reddit WSB ticker mentions, news
tickers) should live here too. Anything that hits a structured API goes
in its own provider module.
"""

from __future__ import annotations

import requests
from bs4 import BeautifulSoup

from providers.base import TickerDiscoveryProvider


class FinvizDiscoveryProvider(TickerDiscoveryProvider):
    """Top-gainers ticker discovery via Finviz screener.

    The scrape parses `<a href="quote.ashx?t=...">` anchors out of the
    HTML page. Finviz lays out tickers in a table; the anchor pattern
    is stable across page-layout versions because it's the linking
    contract for their internal quote pages.

    Failure modes:
      - Network error → returns [] with a printed warning. Caller treats
        as "no discovery this run" and falls back to curated lists.
      - HTML structure change → silently returns []. Add a structural
        test if this matters for production reliability.
    """

    _URL = "https://finviz.com/screener.ashx?v=111&s=ta_topgainers"
    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    def get_gainers(self, limit: int = 50) -> list[str]:
        try:
            resp = requests.get(self._URL, headers=self._HEADERS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            tickers: list[str] = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("quote.ashx?t="):
                    t = href.split("t=")[1].split("&")[0].strip().upper()
                    if t and t not in tickers:
                        tickers.append(t)
            print(f"Finviz: {len(tickers)} top gainers")
            return tickers[:limit]
        except Exception as e:
            print(f"Finviz fetch failed: {e}")
            return []
