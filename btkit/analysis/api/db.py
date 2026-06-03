"""DuckDB connection management for the btkit API."""
import os
import duckdb
from pathlib import Path

DB_PATH: str = os.getenv(
    "BTKIT_DB",
    str(Path.home() / "dev/backtest/db/es_options_backtests.db"),
)


def query(sql: str, params: list | None = None) -> tuple[list[str], list[tuple]]:
    """Execute *sql* and return (column_names, rows).

    Opens a fresh read-only connection per call so the file lock is released
    between requests, allowing concurrent backtest writes to the same file.
    """
    with duckdb.connect(DB_PATH, read_only=True) as con:
        c = con.cursor()
        if params:
            c.execute(sql, params)
        else:
            c.execute(sql)
        cols = [d[0] for d in c.description]
        rows = c.fetchall()
    return cols, rows
