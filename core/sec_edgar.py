"""
SEC EDGAR helpers for TradeStrategy.

get_cik(ticker)            -> 10-digit CIK string or None
get_recent_filings(ticker) -> list of recent filing dicts
"""

import streamlit as st
import requests

_HEADERS = {"User-Agent": "TradeStrategy personal/1.0 dave@example.com"}
_FORMS   = {"8-K", "10-Q", "10-K", "S-1", "4"}


@st.cache_data(ttl=86400)
def get_cik(ticker: str) -> str | None:
    if "-USD" in ticker:
        return None
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                return str(entry["cik_str"]).zfill(10)
        return None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_recent_filings(ticker: str, limit: int = 5) -> list[dict]:
    cik = get_cik(ticker)
    if not cik:
        return []
    try:
        resp = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data    = resp.json()
        recent  = data.get("filings", {}).get("recent", {})
        forms   = recent.get("form",                  [])
        dates   = recent.get("filingDate",            [])
        accnums = recent.get("accessionNumber",       [])
        docs    = recent.get("primaryDocument",       [])
        descs   = recent.get("primaryDocDescription", [])

        cik_int = int(cik)
        results = []
        for form, filed, acc, doc, desc in zip(forms, dates, accnums, docs, descs):
            if form not in _FORMS:
                continue
            acc_nodash = acc.replace("-", "")
            url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik_int}/{acc_nodash}/{doc}"
            )
            results.append({
                "form":        form,
                "filed":       filed,
                "description": desc or "",
                "url":         url,
            })
            if len(results) >= limit:
                break
        return results
    except Exception:
        return []
