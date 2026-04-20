"""
tests/test_db.py — Unit tests for core/db.py.

All tests run against local SQLite (no network access). Turso connection
is only exercised when TURSO_URL + TURSO_AUTH_TOKEN are both present in
the environment, which is not the case in normal CI.
"""

from __future__ import annotations

import importlib
import os
import sqlite3
import sys


def _reload_db_module():
    """Force re-import of core.db so env-var changes in monkeypatch take effect."""
    if "core.db" in sys.modules:
        del sys.modules["core.db"]
    import core.db
    return core.db


# ---------------------------------------------------------------------------
# Fallback behaviour — local SQLite when env vars are absent
# ---------------------------------------------------------------------------

def test_returns_sqlite_when_no_env_vars(monkeypatch, tmp_path):
    monkeypatch.delenv("TURSO_URL",        raising=False)
    monkeypatch.delenv("TURSO_AUTH_TOKEN", raising=False)
    db = _reload_db_module()
    conn = db.get_connection(str(tmp_path / "test.db"))
    assert isinstance(conn, sqlite3.Connection)
    conn.close()


def test_returns_sqlite_when_only_url_set(monkeypatch, tmp_path):
    monkeypatch.setenv("TURSO_URL", "libsql://test.turso.io")
    monkeypatch.delenv("TURSO_AUTH_TOKEN", raising=False)
    db = _reload_db_module()
    conn = db.get_connection(str(tmp_path / "test.db"))
    assert isinstance(conn, sqlite3.Connection)
    conn.close()


def test_returns_sqlite_when_only_token_set(monkeypatch, tmp_path):
    monkeypatch.delenv("TURSO_URL", raising=False)
    monkeypatch.setenv("TURSO_AUTH_TOKEN", "test-token-abc")
    db = _reload_db_module()
    conn = db.get_connection(str(tmp_path / "test.db"))
    assert isinstance(conn, sqlite3.Connection)
    conn.close()


# ---------------------------------------------------------------------------
# sync_if_turso — no-op for plain sqlite3
# ---------------------------------------------------------------------------

def test_sync_if_turso_noop_for_sqlite(monkeypatch, tmp_path):
    monkeypatch.delenv("TURSO_URL",        raising=False)
    monkeypatch.delenv("TURSO_AUTH_TOKEN", raising=False)
    db = _reload_db_module()
    conn = db.get_connection(str(tmp_path / "test.db"))
    db.sync_if_turso(conn)  # must not raise
    conn.close()


def test_sync_if_turso_noop_for_arbitrary_object():
    """sync_if_turso must not raise if conn has no sync attribute."""
    db = _reload_db_module()

    class NoSyncConn:
        pass

    db.sync_if_turso(NoSyncConn())  # must not raise


def test_sync_if_turso_calls_sync_when_present():
    """sync_if_turso calls .sync() if it exists on the connection."""
    db = _reload_db_module()
    called = []

    class FakeTursoConn:
        def sync(self):
            called.append(True)

    db.sync_if_turso(FakeTursoConn())
    assert called == [True]


# ---------------------------------------------------------------------------
# DBAPI2 compatibility — standard operations work on the returned connection
# ---------------------------------------------------------------------------

def test_connection_supports_execute_and_fetchall(monkeypatch, tmp_path):
    monkeypatch.delenv("TURSO_URL",        raising=False)
    monkeypatch.delenv("TURSO_AUTH_TOKEN", raising=False)
    db = _reload_db_module()
    conn = db.get_connection(str(tmp_path / "test.db"))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
    conn.execute("INSERT INTO t (val) VALUES (?)", ("hello",))
    conn.commit()
    rows = conn.execute("SELECT val FROM t").fetchall()
    assert rows[0][0] == "hello"
    conn.close()


def test_connection_supports_executemany(monkeypatch, tmp_path):
    monkeypatch.delenv("TURSO_URL",        raising=False)
    monkeypatch.delenv("TURSO_AUTH_TOKEN", raising=False)
    db = _reload_db_module()
    conn = db.get_connection(str(tmp_path / "test.db"))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, val TEXT)")
    conn.executemany("INSERT INTO t (val) VALUES (?)", [("a",), ("b",), ("c",)])
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    assert count == 3
    conn.close()


def test_cursor_lastrowid(monkeypatch, tmp_path):
    monkeypatch.delenv("TURSO_URL",        raising=False)
    monkeypatch.delenv("TURSO_AUTH_TOKEN", raising=False)
    db = _reload_db_module()
    conn = db.get_connection(str(tmp_path / "test.db"))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, val TEXT)")
    cur = conn.execute("INSERT INTO t (val) VALUES (?)", ("x",))
    assert cur.lastrowid == 1
    conn.close()
