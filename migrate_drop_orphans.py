"""
migrate_drop_orphans.py — One-shot schema migration to drop orphaned tables.

After the Trade Tracker tab was removed (Phase 11-ish), the `trades` and
`crypto_holdings` tables in screener.db became orphaned. They aren't
referenced by any current code path but still take up disk and confuse
new readers of the schema.

This script drops them. **Run manually when you're ready** — it is NOT
auto-invoked. Idempotent: safe to run multiple times.

Usage:
    python3 migrate_drop_orphans.py            # show what would be dropped
    python3 migrate_drop_orphans.py --execute  # actually drop them

The DB lives at `screener.db` in the repo root.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


ORPHANED_TABLES = ("trades", "crypto_holdings")
DEFAULT_DB_PATH = Path(__file__).parent / "screener.db"


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _row_count(conn: sqlite3.Connection, name: str) -> int:
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually drop the tables (otherwise dry-run).",
    )
    parser.add_argument(
        "--db", default=str(DEFAULT_DB_PATH),
        help=f"SQLite DB path (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db_path))
    try:
        found = []
        for t in ORPHANED_TABLES:
            if _table_exists(conn, t):
                n = _row_count(conn, t)
                found.append((t, n))

        if not found:
            print("No orphaned tables found. Nothing to do.")
            return 0

        print(f"DB: {db_path}")
        print(f"Found {len(found)} orphaned table(s):")
        for t, n in found:
            print(f"  - {t}  ({n} rows)")

        if not args.execute:
            print("\nDry-run mode. Re-run with `--execute` to actually drop them.")
            return 0

        print("\nDropping…")
        for t, _n in found:
            conn.execute(f"DROP TABLE IF EXISTS {t}")
            print(f"  ✓ dropped {t}")
        conn.commit()

        # Reclaim disk space
        print("\nVacuuming DB to reclaim disk space…")
        conn.execute("VACUUM")
        print("Done.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
