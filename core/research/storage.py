"""
core/research/storage.py — Persist and load sweep/walk-forward results in screener.db.

Two-table schema:
  research_runs    — one row per (param_set × fold); aggregate metrics
  research_results — one row per (run_id × ticker); per-ticker breakdown

Both tables are created on first use (CREATE TABLE IF NOT EXISTS).
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Generator

from core.db import get_connection

import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "screener.db")

_DDL_RUNS = """
CREATE TABLE IF NOT EXISTS research_runs (
    run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_label    TEXT    NOT NULL,
    param_json   TEXT    NOT NULL,
    train_start  TEXT,
    train_end    TEXT,
    test_start   TEXT,
    test_end     TEXT,
    fold         INTEGER,
    n_tickers    INTEGER,
    n_trades     INTEGER,
    win_rate     REAL,
    expectancy   REAL,
    sharpe       REAL,
    max_drawdown REAL,
    return_pct   REAL,
    created_at   TEXT    DEFAULT (datetime('now'))
)
"""

_DDL_RESULTS = """
CREATE TABLE IF NOT EXISTS research_results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES research_runs(run_id),
    ticker       TEXT    NOT NULL,
    n_signals    INTEGER,
    n_trades     INTEGER,
    win_rate     REAL,
    expectancy   REAL,
    sharpe       REAL,
    max_drawdown REAL,
    return_pct   REAL,
    error        TEXT
)
"""


@contextmanager
def _connect(db_path: str = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    conn = get_connection(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def ensure_schema(db_path: str = DB_PATH) -> None:
    """Create research tables if they don't exist."""
    with _connect(db_path) as conn:
        conn.execute(_DDL_RUNS)
        conn.execute(_DDL_RESULTS)


def save_run(
    *,
    db_path:      str   = DB_PATH,
    run_label:    str,
    param_json:   str,
    train_start:  str   | None = None,
    train_end:    str   | None = None,
    test_start:   str   | None = None,
    test_end:     str   | None = None,
    fold:         int   | None = None,
    n_tickers:    int   = 0,
    n_trades:     int   = 0,
    win_rate:     float | None = None,
    expectancy:   float | None = None,
    sharpe:       float | None = None,
    max_drawdown: float | None = None,
    return_pct:   float | None = None,
    ticker_rows:  list[dict]   | None = None,
) -> int:
    """
    Insert one row into research_runs and optionally per-ticker rows into research_results.

    Returns the new run_id.
    """
    ensure_schema(db_path)

    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO research_runs
                (run_label, param_json, train_start, train_end,
                 test_start, test_end, fold,
                 n_tickers, n_trades, win_rate, expectancy,
                 sharpe, max_drawdown, return_pct)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_label, param_json,
                train_start, train_end,
                test_start, test_end, fold,
                n_tickers, n_trades, win_rate, expectancy,
                sharpe, max_drawdown, return_pct,
            ),
        )
        run_id = cur.lastrowid

        if ticker_rows:
            conn.executemany(
                """
                INSERT INTO research_results
                    (run_id, ticker, n_signals, n_trades, win_rate,
                     expectancy, sharpe, max_drawdown, return_pct, error)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        run_id,
                        r.get("ticker"),
                        r.get("n_signals"),
                        r.get("n_trades"),
                        r.get("win_rate"),
                        r.get("expectancy"),
                        r.get("sharpe"),
                        r.get("max_drawdown"),
                        r.get("return_pct"),
                        r.get("error"),
                    )
                    for r in ticker_rows
                ],
            )

    return run_id


def load_runs(
    db_path:   str        = DB_PATH,
    run_label: str | None = None,
    fold:      int | None = None,
) -> pd.DataFrame:
    """
    Load rows from research_runs, optionally filtered by run_label or fold.

    Returns an empty DataFrame if the table doesn't exist yet.
    """
    if not os.path.exists(db_path):
        return pd.DataFrame()

    ensure_schema(db_path)

    clauses: list[str] = []
    params:  list      = []
    if run_label is not None:
        clauses.append("run_label = ?")
        params.append(run_label)
    if fold is not None:
        clauses.append("fold = ?")
        params.append(fold)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql   = f"SELECT * FROM research_runs {where} ORDER BY run_id"

    conn = get_connection(db_path)
    try:
        return pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()


def load_ticker_results(db_path: str = DB_PATH, run_id: int | None = None) -> pd.DataFrame:
    """Load per-ticker rows from research_results, optionally filtered by run_id."""
    if not os.path.exists(db_path):
        return pd.DataFrame()

    ensure_schema(db_path)

    where  = "WHERE run_id = ?" if run_id is not None else ""
    params = [run_id] if run_id is not None else []
    sql    = f"SELECT * FROM research_results {where} ORDER BY id"

    conn = get_connection(db_path)
    try:
        return pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()
