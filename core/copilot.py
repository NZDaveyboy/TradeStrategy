"""
core/copilot.py — TradeStrategy AI Copilot.

A Claude-powered assistant that answers trading questions grounded in the
user's own screener data, recommendations, peer fundamentals, and catalysts.
No invented numbers — every metric comes from a tool call against the local
DB or live yfinance data.

Entry point: `ask_copilot(messages, db_path)` — handles the full tool-use
loop and returns the final assistant message text.

Environment:
  ANTHROPIC_API_KEY — required.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from core.db import get_connection
from core.recommendations import STRATEGY_DISPLAY, build_recommendation
from core.peers import PEER_MAP, fetch_peer_fundamentals_raw


MODEL_ID = "claude-sonnet-4-6"
MAX_TOOL_ITERATIONS = 5   # batch tools mean fewer round-trips are needed


SYSTEM_PROMPT = """You are the TradeStrategy Copilot — an analyst embedded in the user's personal trading dashboard.

You have tools that query the user's local screener DB and live market data. Every number you cite MUST come from a tool call. Never invent prices, P/E ratios, targets, or growth rates. If a tool returns null, say "unavailable" — don't fill from training data.

## Efficient tool use

For multi-ticker questions (portfolios, watchlists, comparisons), use the BATCH variants — `get_screener_rows`, `get_recommendations`, `get_catalysts` — they take a list and return one combined result. Don't call single-ticker tools in a loop.

Typical evaluation flow for a ticker:
1. `get_screener_row` (or `get_screener_rows` for many) — TradeScore, EMAs, regime
2. `get_recommendation` — entry/stop/target + warnings (counter-trend, IV, etc.)
3. `get_peer_comparison` — only if user asks about valuation
4. `get_catalyst` — only if earnings/news risk is in scope

## Style

Sharp, opinionated analyst voice. Bottom line first, then evidence. Cite specific numbers. Surface risks proactively (counter-trend, regime conflicts, catalyst risk). Markdown. Tables only when comparing 3+ things across 3+ dimensions.

## Rules

- Don't override the recommendation engine's verdict without explicit reasoning. If it says "watchlist", explain why, don't upgrade to "buy".
- Frame risk as % of capital or R-multiples, never dollar position sizes.
- Don't assume the user's account size, broker, or tax situation.
"""


# ---------------------------------------------------------------------------
# Tool definitions (Anthropic JSON Schema format)
# ---------------------------------------------------------------------------

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "get_screener_rows",
        "description": (
            "BATCH: get latest screener rows for one or many tickers — "
            "TradeScore, direction, EMAs, RSI, ATR, setup_type, regime. "
            "Use this for portfolios/watchlists instead of single-ticker calls."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array", "items": {"type": "string"},
                    "description": "1+ ticker symbols e.g. ['MSFT','NVDA']",
                },
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "get_recommendations",
        "description": (
            "BATCH: build full Recommendation (entry/stop/target/strategy/"
            "rationale/warnings) for one or many tickers. Use this for "
            "portfolios; single-ticker call is fine for one ticker."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "get_peer_comparison",
        "description": (
            "Fundamentals (P/E TTM, Forward P/E, PEG, profit margin, ROE, "
            "rev growth, market cap) for the ticker + its configured peers. "
            "Slow (live yfinance fetch). Only call when user asks about valuation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "get_catalysts",
        "description": (
            "BATCH: catalyst signals (next earnings, beats/misses, analyst "
            "rating changes, news tags, catalyst score 0-100) for one or "
            "many tickers. Only call when binary-event risk is in scope."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "list_top_setups",
        "description": (
            "Top N candidates from the most recent screener run, ranked by "
            "TradeScore. For 'what should I look at today' questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit":     {"type": "integer", "default": 10},
                "direction": {"type": "string", "enum": ["long", "short", "any"], "default": "any"},
            },
        },
    },
    {
        "name": "list_open_trades",
        "description": "User's currently open trades from the local tracker.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _latest_row(conn, ticker: str) -> dict | None:
    df = pd.read_sql(
        "SELECT * FROM results WHERE ticker = ? ORDER BY run_date DESC LIMIT 1",
        conn, params=(ticker.upper(),),
    )
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def _trim_screener_row(row: dict | None) -> dict | None:
    """Keep only the fields the model needs for narrative reasoning.
    Round floats to reduce token count. Drop nulls entirely."""
    if row is None:
        return None
    keep = (
        "ticker", "price", "change_pct", "rvol", "tradescore", "score",
        "direction", "setup_type", "ema9", "ema20", "ema200",
        "rsi", "atr", "vwap", "market_cap",
    )
    out: dict = {}
    for k in keep:
        v = row.get(k)
        if v is None or (isinstance(v, float) and v != v):  # null or NaN
            continue
        out[k] = round(v, 2) if isinstance(v, float) else v
    return out


def _trim_recommendation(rec) -> dict:
    """Compact representation of a Recommendation — full rationale is the
    biggest token sink, so truncate aggressively. The structured fields
    (entry/stop/target/category/warnings) carry the operational info."""
    rationale = rec.rationale or ""
    if len(rationale) > 220:
        rationale = rationale[:220].rsplit(" ", 1)[0] + "…"
    return {
        "ticker":          rec.ticker,
        "direction":       rec.direction,
        "setup_type":      rec.setup_type,
        "category":        rec.recommendation_category,
        "strategy":        STRATEGY_DISPLAY.get(rec.strategy_name, rec.strategy_name),
        "entry":           rec.entry_reference,
        "stop":            rec.invalidation_price,
        "target":          rec.target_price,
        "rr":              rec.risk_reward,
        "iv":              rec.iv_assessment,
        "warnings":        rec.warnings,
        "actionable":      rec.is_actionable,
        "rationale":       rationale,
    }


def _tool_get_screener_rows(db_path: str, tickers: list[str]) -> dict:
    conn = get_connection(db_path)
    try:
        rows: dict[str, Any] = {}
        missing: list[str] = []
        for t in tickers:
            row = _latest_row(conn, t)
            trimmed = _trim_screener_row(row)
            if trimmed is None:
                missing.append(t.upper())
            else:
                rows[t.upper()] = trimmed
    finally:
        conn.close()
    result: dict = {"rows": rows}
    if missing:
        result["missing"] = missing
    return result


def _tool_get_recommendations(db_path: str, tickers: list[str]) -> dict:
    conn = get_connection(db_path)
    try:
        recs: dict[str, Any] = {}
        missing: list[str] = []
        for t in tickers:
            row = _latest_row(conn, t)
            if row is None:
                missing.append(t.upper())
                continue
            recs[t.upper()] = _trim_recommendation(
                build_recommendation(row, iv_mode="fallback")
            )
    finally:
        conn.close()
    result: dict = {"recommendations": recs}
    if missing:
        result["missing"] = missing
    return result


def _tool_get_peer_comparison(ticker: str) -> dict:
    ticker = ticker.upper()
    peers = PEER_MAP.get(ticker)
    if not peers:
        return {"error": f"No configured peer set for {ticker}."}
    df = fetch_peer_fundamentals_raw((ticker,) + peers)
    return {"rows": df.where(df.notna(), None).to_dict(orient="records")}


def _tool_get_catalysts(tickers: list[str]) -> dict:
    try:
        from core.catalysts import compute_catalyst_score
    except Exception as e:
        return {"error": f"catalyst module unavailable: {e}"}
    out: dict[str, Any] = {}
    for t in tickers:
        try:
            r = compute_catalyst_score(t.upper())
            # Trim aggressively — drop verbose `data` sub-dict, keep score + tags only.
            out[t.upper()] = {
                "score": r.get("score"),
                "tags":  (r.get("tags") or [])[:6],  # cap tag count
            }
        except Exception as e:
            out[t.upper()] = {"error": f"{type(e).__name__}"}
    return {"catalysts": out}


def _tool_list_top_setups(db_path: str, limit: int = 10, direction: str = "any") -> dict:
    limit = max(1, min(int(limit or 10), 50))
    conn = get_connection(db_path)
    try:
        latest = pd.read_sql(
            "SELECT MAX(run_date) AS d FROM results", conn
        )["d"].iloc[0]
        if not latest:
            return {"error": "No screener data available."}

        params: list = [latest]
        where = "run_date = ?"
        if direction in ("long", "short"):
            where += " AND direction = ?"
            params.append(direction)
        params.append(limit)
        df = pd.read_sql(
            f"SELECT ticker, direction, setup_type, tradescore, price, "
            f"change_pct, rvol FROM results WHERE {where} "
            f"ORDER BY COALESCE(tradescore, 0) DESC LIMIT ?",
            conn, params=tuple(params),
        )
    finally:
        conn.close()
    return {"run_date": latest, "rows": df.to_dict(orient="records")}


def _tool_list_open_trades(db_path: str) -> dict:
    conn = get_connection(db_path)
    try:
        # The trades table may not exist on fresh installs
        try:
            df = pd.read_sql(
                "SELECT ticker, trade_type, entry_price, position_size, "
                "opened_at, notes FROM trades WHERE closed_at IS NULL",
                conn,
            )
        except Exception:
            return {"rows": [], "note": "No trades table yet."}
    finally:
        conn.close()
    return {"rows": df.to_dict(orient="records")}


def _dispatch_tool(name: str, tool_input: dict, db_path: str) -> str:
    """Execute a tool and return a JSON string. JSON keeps Claude's parsing
    deterministic vs free-form text."""
    try:
        if name == "get_screener_rows":
            tickers = tool_input.get("tickers") or []
            result = _tool_get_screener_rows(db_path, tickers)
        elif name == "get_recommendations":
            tickers = tool_input.get("tickers") or []
            result = _tool_get_recommendations(db_path, tickers)
        # Legacy single-ticker names — kept so a stale model turn doesn't error
        elif name == "get_screener_row":
            result = _tool_get_screener_rows(db_path, [tool_input.get("ticker", "")])
        elif name == "get_recommendation":
            result = _tool_get_recommendations(db_path, [tool_input.get("ticker", "")])
        elif name == "get_peer_comparison":
            result = _tool_get_peer_comparison(tool_input.get("ticker", ""))
        elif name == "get_catalysts":
            tickers = tool_input.get("tickers") or []
            result = _tool_get_catalysts(tickers)
        elif name == "get_catalyst":
            result = _tool_get_catalysts([tool_input.get("ticker", "")])
        elif name == "list_top_setups":
            result = _tool_list_top_setups(
                db_path,
                limit=tool_input.get("limit", 10),
                direction=tool_input.get("direction", "any"),
            )
        elif name == "list_open_trades":
            result = _tool_list_open_trades(db_path)
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as e:
        result = {"error": f"Tool {name} raised: {type(e).__name__}: {e}"}
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ask_copilot(
    messages: list[dict],
    db_path: str,
    api_key: str | None = None,
) -> tuple[str, list[dict]]:
    """Run a single user turn through Claude with full tool-use loop.

    Args:
      messages: conversation history as list of {"role": ..., "content": ...}.
                Claude API format. The last message should be the new user turn.
      db_path:  path to the TradeStrategy SQLite DB.
      api_key:  Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.

    Returns:
      (final_text, updated_messages) — the assistant's final text response,
      plus the full messages list including all tool-use round-trips. Caller
      should persist updated_messages to maintain conversation continuity.
    """
    import anthropic

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return ("⚠️ ANTHROPIC_API_KEY not set. Add it to your shell environment "
                "or to TradeStrategy/.env (then source the file) before using the Copilot.",
                messages)

    client = anthropic.Anthropic(api_key=key)

    # Use prompt caching on the system prompt + tool definitions — saves
    # 90% on input tokens for follow-up turns.
    system_block = [{
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }]

    convo = list(messages)

    for _ in range(MAX_TOOL_ITERATIONS):
        try:
            response = client.messages.create(
                model=MODEL_ID,
                max_tokens=1536,   # 2048 was generous; trim to reduce token budget pressure
                system=system_block,
                tools=TOOL_SPECS,
                messages=convo,
            )
        except anthropic.RateLimitError:
            return (
                "⚠️ **Rate limit hit.** Your Anthropic account exceeded its "
                "input-token-per-minute quota.\n\n"
                "Wait ~60 seconds and try again. For frequent use, raise your tier at "
                "[console.anthropic.com](https://console.anthropic.com/) → Billing → "
                "Plans (loading $5 of credit auto-upgrades to Tier 1, which lifts "
                "the limit to 50K input tokens/min).",
                convo,
            )

        # Append assistant turn to history (using API format)
        convo.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            # Pull the final text
            text_parts = [
                b.text for b in response.content if getattr(b, "type", "") == "text"
            ]
            return ("\n".join(text_parts).strip() or "_(empty response)_", convo)

        # Process all tool_use blocks; build a single tool_result user turn
        tool_results = []
        for block in response.content:
            if getattr(block, "type", "") != "tool_use":
                continue
            output = _dispatch_tool(block.name, block.input or {}, db_path)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })

        if not tool_results:
            # Defensive: stop_reason said tool_use but we found none
            text_parts = [
                b.text for b in response.content if getattr(b, "type", "") == "text"
            ]
            return ("\n".join(text_parts).strip() or "_(no tool calls found)_", convo)

        convo.append({"role": "user", "content": tool_results})

    return ("⚠️ Copilot exceeded maximum tool iterations. Try a more focused question.", convo)
