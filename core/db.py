"""
core/db.py — Database connection factory.

Exports a no-arg `get_conn()` that returns a connection to the project's
screener.db. Tab modules import this directly instead of receiving a
factory through render kwargs.

`get_connection(path)` is preserved for callers that need a non-default DB.
"""

from __future__ import annotations

import os
import sqlite3


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "screener.db")


def get_connection(db_path: str) -> sqlite3.Connection:
    """Return a sqlite3 connection to db_path."""
    return sqlite3.connect(db_path)


def get_conn() -> sqlite3.Connection:
    """Return a sqlite3 connection to the project's screener.db."""
    return sqlite3.connect(DB_PATH)
