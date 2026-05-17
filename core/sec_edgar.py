"""
SEC EDGAR helpers for TradeStrategy.

get_cik(ticker)            -> 10-digit CIK string or None
get_recent_filings(ticker) -> list of recent filing dicts
format_8k_items(items)     -> plain-English label string for an 8-K's items
"""

import streamlit as st
import requests

_HEADERS = {"User-Agent": "TradeStrategy personal/1.0 dave@example.com"}
_FORMS   = {"8-K", "10-Q", "10-K", "S-1", "4"}

# 8-K item codes → human label.
# Used to turn "1.01,7.01,9.01" into a readable "highlight reel" line.
# Sources: SEC 8-K filing instructions
# https://www.sec.gov/files/form8-k.pdf
_ITEM_LABELS: dict[str, str] = {
    "1.01": "📝 Material Agreement",        # contract, partnership, M&A entry
    "1.02": "❌ Material Agreement Ended",
    "1.03": "⚠️ Bankruptcy / Receivership",
    "1.04": "🛡️ Insurance / Recovery Event",
    "2.01": "🏢 Acquisition Completed",
    "2.02": "💰 Earnings Results",
    "2.03": "💳 Material Debt Obligation",
    "2.04": "⚠️ Triggering Event / Default",
    "2.05": "🚪 Exit / Disposal Costs",
    "2.06": "📉 Material Impairment",
    "3.01": "🚫 Delisting Notice",
    "3.02": "🌊 Unregistered Equity Sale",   # dilution risk
    "3.03": "🔧 Modification of Holder Rights",
    "4.01": "🧾 Auditor Change",
    "4.02": "⚠️ Prior Financials Unreliable",
    "5.01": "🔄 Change in Control",
    "5.02": "👤 Officer/Director Change",
    "5.03": "📜 Bylaws / Articles Amended",
    "5.05": "💼 Code of Ethics Change",
    "5.07": "🗳️ Shareholder Vote Results",
    "5.08": "🗓️ Shareholder Meeting Date",
    "7.01": "💬 Reg FD Disclosure",
    "8.01": "📰 Other Event",
    # 9.01 (Financial Statements and Exhibits) is technical noise — almost
    # every 8-K includes it. Suppressed from the highlight reel.
}

# Items considered "filler" — present on most 8-Ks but rarely informative
# on their own. Suppressed unless they're the only item.
_ITEM_FILLER = {"9.01"}


def format_8k_items(items: str | None) -> str:
    """Turn '1.01,7.01,9.01' into a readable label string.

    Strategy:
      - Split on comma, strip whitespace
      - Filter out filler items (9.01) unless they're alone
      - Map each remaining item to its plain-English label
      - Join with ' · '
    Returns '' if the input is empty or all items are unknown.
    """
    if not items:
        return ""
    raw = [s.strip() for s in str(items).split(",") if s.strip()]
    if not raw:
        return ""

    # Remove filler if there's anything else
    informative = [i for i in raw if i not in _ITEM_FILLER]
    if not informative:
        informative = raw  # all-filler — show what we have

    labels: list[str] = []
    for item in informative:
        label = _ITEM_LABELS.get(item)
        if label:
            labels.append(label)
        else:
            labels.append(f"Item {item}")
    return " · ".join(labels)


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
        items   = recent.get("items",                 [])

        cik_int = int(cik)
        results = []
        for i, (form, filed, acc, doc, desc) in enumerate(zip(forms, dates, accnums, docs, descs)):
            if form not in _FORMS:
                continue
            acc_nodash = acc.replace("-", "")
            url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik_int}/{acc_nodash}/{doc}"
            )
            item_codes = items[i] if i < len(items) else ""
            results.append({
                "form":        form,
                "filed":       filed,
                "description": desc or "",
                "url":         url,
                "items":       item_codes,
                "items_label": format_8k_items(item_codes) if form == "8-K" else "",
            })
            if len(results) >= limit:
                break
        return results
    except Exception:
        return []
