"""
core/db.py — Database connection factory.

Returns a libsql embedded-replica connection when TURSO_URL and
TURSO_AUTH_TOKEN are both set; falls back to plain sqlite3 otherwise.
Both paths are DBAPI2-compatible: pd.read_sql(), executemany(),
row_factory, and cursor.lastrowid work identically.

Credentials are read from os.environ first, then st.secrets (when the
Streamlit runtime is active). If neither source has both values, the
local SQLite path is used.
"""

from __future__ import annotations

import os
import sqlite3


def _turso_creds() -> tuple[str, str]:
    url   = os.environ.get("TURSO_URL",        "")
    token = os.environ.get("TURSO_AUTH_TOKEN", "")
    if not (url and token):
        try:
            import streamlit as st
            url   = url   or (st.secrets.get("TURSO_URL",        "") or "")
            token = token or (st.secrets.get("TURSO_AUTH_TOKEN", "") or "")
        except Exception:
            pass
    return url, token


def get_connection(db_path: str) -> sqlite3.Connection:
    """
    Return a DBAPI2-compatible connection to db_path.

    Turso path: opens a libsql embedded replica at db_path synced to the
    Turso cloud DB. Does NOT auto-sync — callers must call sync_if_turso()
    after writes, and (for the app startup path) once before the first read.

    SQLite path: plain sqlite3.connect(db_path).
    """
    url, token = _turso_creds()
    if url and token:
        import libsql_experimental as libsql  # lazy import — only when creds present
        return libsql.connect(db_path, sync_url=url, auth_token=token)
    return sqlite3.connect(db_path)


def sync_if_turso(conn) -> None:
    """
    Push local writes to Turso and pull any remote changes.

    Call after conn.commit() in every write path so GitHub Actions runs
    are immediately visible to Streamlit Cloud and vice versa.
    No-op for plain sqlite3 connections (no sync attribute).
    """
    sync_fn = getattr(conn, "sync", None)
    if sync_fn is not None:
        sync_fn()
