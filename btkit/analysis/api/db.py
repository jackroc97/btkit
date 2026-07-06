"""DuckDB connection management for the btkit API."""

import json
import os
from pathlib import Path

import duckdb

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


def execute(sql: str, params: list | None = None) -> None:
    """Execute a write statement (INSERT / UPDATE / DELETE).

    Uses a short-lived writable connection. Mutations from the dashboard are
    infrequent (tag apply/remove), so a new connection per call is acceptable.
    """
    with duckdb.connect(DB_PATH, read_only=False) as con:
        if params:
            con.execute(sql, params)
        else:
            con.execute(sql)


# ── Cache helpers ─────────────────────────────────────────────────────────────


def cache_get(cache_key: str, fingerprint: str) -> str | None:
    """Return cached JSON string if fingerprint matches, else None."""
    try:
        with duckdb.connect(DB_PATH, read_only=True) as con:
            row = con.execute(
                "SELECT result_json FROM api_cache WHERE cache_key = ? AND fingerprint = ?",
                [cache_key, fingerprint],
            ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def cache_set(cache_key: str, result_json: str, fingerprint: str) -> None:
    """Upsert a cache entry (DELETE + INSERT for DuckDB compatibility)."""
    try:
        with duckdb.connect(DB_PATH, read_only=False) as con:
            con.execute("DELETE FROM api_cache WHERE cache_key = ?", [cache_key])
            con.execute(
                "INSERT INTO api_cache (cache_key, result_json, fingerprint) VALUES (?, ?, ?)",
                [cache_key, result_json, fingerprint],
            )
    except Exception:
        pass


# ── Preference helpers ────────────────────────────────────────────────────────


def pref_get(pref_key: str) -> object | None:
    """Return the stored preference value (parsed from JSON), or None."""
    try:
        with duckdb.connect(DB_PATH, read_only=True) as con:
            row = con.execute(
                "SELECT value_json FROM ui_preference WHERE pref_key = ?",
                [pref_key],
            ).fetchone()
        return json.loads(row[0]) if row else None
    except Exception:
        return None


def pref_set(pref_key: str, value: object) -> None:
    """Upsert a preference value (DELETE + INSERT for DuckDB compatibility)."""
    try:
        with duckdb.connect(DB_PATH, read_only=False) as con:
            con.execute("DELETE FROM ui_preference WHERE pref_key = ?", [pref_key])
            con.execute(
                "INSERT INTO ui_preference (pref_key, value_json) VALUES (?, ?)",
                [pref_key, json.dumps(value)],
            )
    except Exception:
        pass
