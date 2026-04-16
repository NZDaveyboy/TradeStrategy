"""
core/theme_watchlist.py — Curated theme watchlists for early-signal prioritisation.

To add a ticker to the permanent watchlist, edit AI_INFRASTRUCTURE below.
To add a new theme, add an entry to THEME_WATCHLISTS.

Session-level overrides (add/remove) are stored in st.session_state and
reset on app restart. They do not write to disk.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Default watchlists — edit these constants to update the watchlists
# ---------------------------------------------------------------------------

AI_INFRASTRUCTURE: list[str] = [
    # Semiconductors / compute
    "NVDA", "AMD", "AVGO", "MRVL", "SMCI", "ALAB", "ASML", "TSM",
    # Networking / interconnect
    "ANET", "CIEN", "LITE", "COHR",
    # Power / cooling
    "VRT", "ETN", "HUBB", "POWL",
    # Cloud hyperscalers (infrastructure layer only)
    "MSFT", "AMZN", "GOOGL",
    # Pure-play AI / quantum
    "PLTR", "IONQ", "RKLB", "BBAI",
    # Data infrastructure
    "SNOW", "MDB", "NET",
]

THEME_WATCHLISTS: dict[str, list[str]] = {
    "AI Infrastructure": AI_INFRASTRUCTURE,
}

# ---------------------------------------------------------------------------
# Streamlit availability (graceful fallback for tests / CLI use)
# ---------------------------------------------------------------------------

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    st = None  # type: ignore[assignment]
    _HAS_STREAMLIT = False

# Session state key helpers
_ADD_KEY = "watchlist_add_{theme}"
_REM_KEY = "watchlist_rem_{theme}"


# ---------------------------------------------------------------------------
# Core lookups (no session state — safe to call from any context)
# ---------------------------------------------------------------------------

def get_watchlist(theme: str = "AI Infrastructure") -> list[str]:
    """Return the default (non-session) ticker list for a theme."""
    return list(THEME_WATCHLISTS.get(theme, []))


def is_on_watchlist(ticker: str, theme: str = "AI Infrastructure") -> bool:
    """True if ticker is in the default watchlist for theme (case-insensitive)."""
    return ticker.upper() in {t.upper() for t in get_watchlist(theme)}


def all_watchlist_tickers() -> list[str]:
    """Deduplicated union of all default tickers across all themes."""
    seen: set[str] = set()
    result: list[str] = []
    for tickers in THEME_WATCHLISTS.values():
        for t in tickers:
            if t not in seen:
                seen.add(t)
                result.append(t)
    return result


# ---------------------------------------------------------------------------
# Session-level overrides — require Streamlit context
# ---------------------------------------------------------------------------

def get_session_watchlist(theme: str = "AI Infrastructure") -> list[str]:
    """
    Return the effective watchlist for a theme, merging default tickers
    with any session-level additions and minus any session-level removals.
    Falls back to get_watchlist() when Streamlit is not available.
    """
    base = get_watchlist(theme)
    if not _HAS_STREAMLIT or st is None:
        return base

    additions: list[str] = st.session_state.get(_ADD_KEY.format(theme=theme), [])
    removals:  set[str]  = st.session_state.get(_REM_KEY.format(theme=theme), set())
    merged = list(dict.fromkeys(base + additions))   # deduplicate, preserve order
    return [t for t in merged if t not in removals]


def add_to_session_watchlist(ticker: str, theme: str = "AI Infrastructure") -> None:
    """Add ticker to the session watchlist for theme. No-op if already present."""
    if not _HAS_STREAMLIT or st is None:
        return
    key  = _ADD_KEY.format(theme=theme)
    curr = st.session_state.get(key, [])
    if ticker.upper() not in {t.upper() for t in curr}:
        st.session_state[key] = curr + [ticker.upper()]
    # Also remove from the removals set if it was there
    rem_key = _REM_KEY.format(theme=theme)
    removals = st.session_state.get(rem_key, set())
    removals.discard(ticker.upper())
    st.session_state[rem_key] = removals


def remove_from_session_watchlist(ticker: str, theme: str = "AI Infrastructure") -> None:
    """Remove ticker from the effective session watchlist for theme."""
    if not _HAS_STREAMLIT or st is None:
        return
    # Remove from additions if present
    add_key = _ADD_KEY.format(theme=theme)
    curr    = st.session_state.get(add_key, [])
    st.session_state[add_key] = [t for t in curr if t.upper() != ticker.upper()]
    # Add to removals set (suppresses default list entry)
    rem_key  = _REM_KEY.format(theme=theme)
    removals = st.session_state.get(rem_key, set())
    removals.add(ticker.upper())
    st.session_state[rem_key] = removals
