"""Unit tests for the study table methods on OutputDatabase."""

from __future__ import annotations

import pytest

from btkit.db.output_db import OutputDatabase


@pytest.fixture()
def odb(tmp_path):
    db_path = str(tmp_path / "out.db")
    with OutputDatabase(db_path) as db:
        db.create_schema()
        yield db


class TestWriteStudy:
    def test_returns_integer_id(self, odb):
        sid = odb.write_study("my_study", "yaml_text", 10)
        assert isinstance(sid, int)
        assert sid >= 1

    def test_sequential_ids(self, odb):
        id1 = odb.write_study("s1", "y1", 5)
        id2 = odb.write_study("s2", "y2", 3)
        assert id2 == id1 + 1

    def test_study_row_stored(self, odb):
        odb.write_study("hello", "yaml_content", 7)
        row = odb._con.execute(
            "SELECT name, strategy_yaml, total_combinations FROM study"
        ).fetchone()
        assert row[0] == "hello"
        assert row[1] == "yaml_content"
        assert row[2] == 7

    def test_finished_at_is_null_before_finalize(self, odb):
        sid = odb.write_study("s", "y", 1)
        row = odb._con.execute("SELECT finished_at FROM study WHERE id = ?", [sid]).fetchone()
        assert row[0] is None


class TestFinalizeStudy:
    def test_sets_finished_at(self, odb):
        sid = odb.write_study("s", "y", 1)
        odb.finalize_study(sid)
        row = odb._con.execute("SELECT finished_at FROM study WHERE id = ?", [sid]).fetchone()
        assert row[0] is not None


class TestBacktestStudyId:
    def test_study_id_written(self, odb):
        from datetime import UTC, datetime

        sid = odb.write_study("s", "y", 2)
        bid = odb.write_backtest(
            {
                "study_id": sid,
                "combination_id": 1,
                "strategy_name": "test",
                "strategy_version": "1.0",
                "strategy_params": {},
                "initial_equity": 100_000.0,
                "slippage_pct": 0.0,
                "fee_per_contract": 0.0,
                "created_at": datetime.now(UTC),
            }
        )
        row = odb._con.execute(
            "SELECT study_id, combination_id FROM backtest WHERE id = ?", [bid]
        ).fetchone()
        assert row[0] == sid
        assert row[1] == 1

    def test_scalar_run_has_null_study_id(self, odb):
        from datetime import UTC, datetime

        bid = odb.write_backtest(
            {
                "strategy_name": "test",
                "strategy_params": {},
                "initial_equity": 100_000.0,
                "slippage_pct": 0.0,
                "fee_per_contract": 0.0,
                "created_at": datetime.now(UTC),
            }
        )
        row = odb._con.execute("SELECT study_id FROM backtest WHERE id = ?", [bid]).fetchone()
        assert row[0] is None


class TestMigration:
    def test_auto_migrates_legacy_matrix_id(self, tmp_path):
        """An existing DB with matrix_id column is auto-migrated to study_id."""
        import duckdb

        db_path = str(tmp_path / "legacy.db")
        # Create a legacy DB manually with matrix_id
        con = duckdb.connect(db_path)
        con.execute("""
            CREATE TABLE backtest (
                id INTEGER PRIMARY KEY,
                matrix_id INTEGER,
                combination_id INTEGER,
                strategy_name VARCHAR NOT NULL,
                strategy_version VARCHAR,
                strategy_params JSON NOT NULL,
                initial_equity DOUBLE NOT NULL,
                slippage_pct DOUBLE NOT NULL,
                fee_per_contract DOUBLE NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                status VARCHAR,
                duration_s DOUBLE,
                warnings JSON,
                error_message VARCHAR,
                error_traceback VARCHAR
            )
        """)
        con.close()

        # Opening via OutputDatabase should trigger migration
        with OutputDatabase(db_path) as odb:
            odb.create_schema()

        # Verify the column was renamed
        con = duckdb.connect(db_path, read_only=True)
        cols = {
            row[0]
            for row in con.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'backtest'"
            ).fetchall()
        }
        con.close()
        assert "study_id" in cols
        assert "matrix_id" not in cols
