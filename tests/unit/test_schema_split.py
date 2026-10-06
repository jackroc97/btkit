"""
Tests for splitting the input schema into tables and deferred indexes.

DuckDB's ART indexes are memory-resident and cannot spill, so maintaining them
incrementally across the ES bulk load (~70M option bars, ~69M greeks rows, ~713M
indicator bars) exhausted memory on a 17 GB machine — "failed to pin" during both
ingest and greeks, with temp_directory at 0 bytes and 180 GiB of disk free. The
large-table indexes are therefore created after the load.

The risk this guards against is the opposite failure: a database that loads
happily but ends up without its indexes, which would be a silent and severe query
regression rather than a crash.
"""

from __future__ import annotations

import duckdb

from btkit.db.input_db import (
    INPUT_DEFERRED_INDEXES_SQL,
    INPUT_SCHEMA_SQL,
    INPUT_TABLES_SQL,
)

# Indexes on the three largest tables, created only after the data is loaded.
DEFERRED = {
    "idx_option_bars_lookup",
    "idx_option_bars_instrument",
    "idx_option_greeks_lookup",
    "idx_indicator_bars",
}

# Small indexes the build itself queries through: the greeks ASOF join against
# underlying_bars and the per-underlying indicator passes.
UPFRONT = {
    "idx_underlying_bars",
    "idx_underlying_bars_expiry",
    "idx_indicator_def_unique",
}


def _indexes(con) -> set[str]:
    return {r[0] for r in con.execute("SELECT index_name FROM duckdb_indexes()").fetchall()}


def _tables(con) -> set[str]:
    return {
        r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()
    }


class TestTablesOnly:
    def test_creates_every_table(self):
        con = duckdb.connect(":memory:")
        con.execute(INPUT_TABLES_SQL)
        assert _tables(con) == {
            "contract",
            "underlying_bars",
            "option_bars",
            "option_greeks",
            "indicator_definition",
            "indicator_bars",
        }

    def test_creates_the_small_indexes_upfront(self):
        con = duckdb.connect(":memory:")
        con.execute(INPUT_TABLES_SQL)
        assert UPFRONT <= _indexes(con)

    def test_omits_the_large_table_indexes(self):
        con = duckdb.connect(":memory:")
        con.execute(INPUT_TABLES_SQL)
        assert _indexes(con) & DEFERRED == set()


class TestDeferredIndexes:
    def test_creates_exactly_the_deferred_set(self):
        con = duckdb.connect(":memory:")
        con.execute(INPUT_TABLES_SQL)
        before = _indexes(con)
        con.execute(INPUT_DEFERRED_INDEXES_SQL)
        assert _indexes(con) - before == DEFERRED

    def test_is_idempotent(self):
        """An --append build runs it against a database that already has them."""
        con = duckdb.connect(":memory:")
        con.execute(INPUT_TABLES_SQL)
        con.execute(INPUT_DEFERRED_INDEXES_SQL)
        con.execute(INPUT_DEFERRED_INDEXES_SQL)
        assert DEFERRED <= _indexes(con)

    def test_applies_to_a_loaded_table(self):
        """Index creation must work on populated tables, not just empty ones."""
        con = duckdb.connect(":memory:")
        con.execute(INPUT_TABLES_SQL)
        con.execute("INSERT INTO indicator_definition VALUES (1, 'x', 100, 'ESH6', NULL, 'src')")
        con.execute(
            "INSERT INTO indicator_bars "
            "SELECT now()::TIMESTAMPTZ + INTERVAL (i) MINUTE, 1, i::DOUBLE FROM range(500) t(i)"
        )
        con.execute(INPUT_DEFERRED_INDEXES_SQL)
        assert "idx_indicator_bars" in _indexes(con)
        assert con.execute("SELECT count(*) FROM indicator_bars").fetchone()[0] == 500


class TestCombinedSchema:
    def test_matches_tables_plus_deferred(self):
        """INPUT_SCHEMA_SQL is what tests and InputDatabase use; it must still
        produce the finished shape in one step."""
        full = duckdb.connect(":memory:")
        full.execute(INPUT_SCHEMA_SQL)

        staged = duckdb.connect(":memory:")
        staged.execute(INPUT_TABLES_SQL)
        staged.execute(INPUT_DEFERRED_INDEXES_SQL)

        assert _indexes(full) == _indexes(staged)
        assert _tables(full) == _tables(staged)

    def test_no_index_is_lost_in_the_split(self):
        con = duckdb.connect(":memory:")
        con.execute(INPUT_SCHEMA_SQL)
        assert _indexes(con) == DEFERRED | UPFRONT

    def test_contract_constraints_survive_in_tables_sql(self):
        """The contract primary key is what makes a contract_id hash collision fail
        loudly, so it must be present during the load, not deferred."""
        con = duckdb.connect(":memory:")
        con.execute(INPUT_TABLES_SQL)
        con.execute("INSERT INTO contract VALUES (1, 'A', NULL, 1, 1.0, 'P', 50, 'P')")
        try:
            con.execute("INSERT INTO contract VALUES (1, 'B', NULL, 1, 1.0, 'P', 50, 'P')")
        except duckdb.ConstraintException:
            pass
        else:
            raise AssertionError("duplicate contract_id should violate the primary key")
