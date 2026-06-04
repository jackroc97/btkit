"""
Migrate idx_option_greeks_lookup from (underlying_id, ts_event, dte)
to (underlying_id, dte, ts_event).

The fixed column order lets DuckDB use the index for per-underlying queries
with a tight DTE range (e.g. 0–2 DTE), which is the hot path in
greeks_for_all_legs() and greeks_for_strike_legs().

Usage:
    python scripts/migrate_option_greeks_index.py /path/to/input.db
"""

import sys
import time

import duckdb


def migrate(db_path: str) -> None:
    print(f"Connecting to {db_path}")
    con = duckdb.connect(db_path)

    print("Dropping idx_option_greeks_lookup ...")
    t0 = time.perf_counter()
    con.execute("DROP INDEX IF EXISTS idx_option_greeks_lookup")
    print(f"  dropped in {time.perf_counter() - t0:.1f}s")

    print("Creating idx_option_greeks_lookup (underlying_id, dte, ts_event) ...")
    t0 = time.perf_counter()
    con.execute(
        "CREATE INDEX idx_option_greeks_lookup "
        "ON option_greeks (underlying_id, dte, ts_event)"
    )
    print(f"  created in {time.perf_counter() - t0:.1f}s")

    con.close()
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <db_path>")
        sys.exit(1)
    migrate(sys.argv[1])
