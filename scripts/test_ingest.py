"""Quick smoke test for the ingest + Greeks pipeline."""
import os
import sys
import duckdb

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from btkit.pipeline.builder import DatabaseBuilder

db_path = "/tmp/btkit_test_ingest.db"
if os.path.exists(db_path):
    os.remove(db_path)

print("=== Building database ===")
builder = DatabaseBuilder(
    raw_data_path="tests/fixtures/data",
    db_path=db_path,
    indicator_scripts=["tests/fixtures/indicators.py"],
)
builder.build()

print("\n=== Verifying tables ===")
con = duckdb.connect(db_path, read_only=True)
for table in ("underlying_bars", "option_bars", "option_greeks", "indicator_definition", "indicator_bars"):
    n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"{table}: {n:,} rows")

print("\n-- Sample option_greeks --")
for row in con.execute(
    "SELECT ts_event, instrument_id, dte, iv, delta, gamma, theta, vega "
    "FROM option_greeks WHERE iv IS NOT NULL LIMIT 5"
).fetchall():
    print(" ", row)

con.close()
print("\nOK")
