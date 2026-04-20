"""
tests/test_db.py — Unit tests for core/db.py.

get_connection() always returns a plain sqlite3.Connection.
"""

from __future__ import annotations

import sqlite3
import sys


def _reload_db_module():
    if "core.db" in sys.modules:
        del sys.modules["core.db"]
    import core.db
    return core.db


def test_returns_sqlite_connection(tmp_path):
    db = _reload_db_module()
    conn = db.get_connection(str(tmp_path / "test.db"))
    assert isinstance(conn, sqlite3.Connection)
    conn.close()


def test_connection_supports_execute_and_fetchall(tmp_path):
    db = _reload_db_module()
    conn = db.get_connection(str(tmp_path / "test.db"))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
    conn.execute("INSERT INTO t (val) VALUES (?)", ("hello",))
    conn.commit()
    rows = conn.execute("SELECT val FROM t").fetchall()
    assert rows[0][0] == "hello"
    conn.close()


def test_connection_supports_executemany(tmp_path):
    db = _reload_db_module()
    conn = db.get_connection(str(tmp_path / "test.db"))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, val TEXT)")
    conn.executemany("INSERT INTO t (val) VALUES (?)", [("a",), ("b",), ("c",)])
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    assert count == 3
    conn.close()


def test_cursor_lastrowid(tmp_path):
    db = _reload_db_module()
    conn = db.get_connection(str(tmp_path / "test.db"))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, val TEXT)")
    cur = conn.execute("INSERT INTO t (val) VALUES (?)", ("x",))
    assert cur.lastrowid == 1
    conn.close()
