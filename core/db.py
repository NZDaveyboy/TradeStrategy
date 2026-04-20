"""
core/db.py — Database connection factory.

Returns a plain sqlite3 connection. The function signature is preserved so
all nine call sites remain unchanged.
"""

from __future__ import annotations

import sqlite3


def get_connection(db_path: str) -> sqlite3.Connection:
    """Return a sqlite3 connection to db_path."""
    return sqlite3.connect(db_path)
