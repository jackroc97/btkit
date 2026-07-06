"""Read-only connection to the input (market data) database."""

from __future__ import annotations

import os

import duckdb


def connect() -> duckdb.DuckDBPyConnection | None:
    """Return a read-only connection, or None if BTKIT_INPUT_DB is unset or missing."""
    path = os.environ.get("BTKIT_INPUT_DB")
    if not path:
        return None
    if not os.path.exists(path):
        return None
    return duckdb.connect(path, read_only=True)
