"""Unit tests for OutputMerger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from btkit.db.output_db import OutputDatabase
from btkit.study.merger import OutputMerger


def _make_worker_db(path: str, backtest_rows: int = 1, positions_per_backtest: int = 2) -> None:
    """Helpers: populate a worker DB with synthetic rows."""
    with OutputDatabase(path) as odb:
        odb.create_schema()
        for i in range(backtest_rows):
            bid = odb.write_backtest({
                "study_id": 1,
                "combination_id": i + 1,
                "strategy_name": "test",
                "strategy_params": {},
                "initial_equity": 100_000.0,
                "slippage_pct": 0.0,
                "fee_per_contract": 0.0,
                "created_at": datetime.now(UTC),
            })
            odb.finalize_backtest(
                bid, status="completed", duration_s=1.0, warnings=[]
            )
            import polars as pl
            positions = pl.DataFrame({
                "entry_id": list(range(positions_per_backtest)),
                "trade_name": ["t1"] * positions_per_backtest,
                "open_time": [datetime.now(UTC)] * positions_per_backtest,
                "exit_time": [datetime.now(UTC)] * positions_per_backtest,
                "exit_reason": ["tp"] * positions_per_backtest,
                "open_mark": [1.0] * positions_per_backtest,
                "exit_mark": [0.5] * positions_per_backtest,
                "worst_mark": [1.5] * positions_per_backtest,
                "slippage_cost": [0.0] * positions_per_backtest,
                "fee_cost": [0.0] * positions_per_backtest,
                "net_pnl": [50.0] * positions_per_backtest,
            })
            legs = pl.DataFrame({
                "entry_id": list(range(positions_per_backtest)),
                "instrument_id": [1001] * positions_per_backtest,
                "symbol": ["ESZ26 P4500"] * positions_per_backtest,
                "expiration": [datetime.now(UTC).date()] * positions_per_backtest,
                "strike_price": [4500.0] * positions_per_backtest,
                "right": ["P"] * positions_per_backtest,
                "action": ["STO"] * positions_per_backtest,
                "quantity": [1] * positions_per_backtest,
                "multiplier": [50] * positions_per_backtest,
                "open_price": [1.0] * positions_per_backtest,
                "exit_price": [0.5] * positions_per_backtest,
                "entry_delta": [-0.25] * positions_per_backtest,
                "entry_iv": [0.20] * positions_per_backtest,
                "entry_gamma": [0.01] * positions_per_backtest,
                "entry_theta": [-0.05] * positions_per_backtest,
                "entry_vega": [2.0] * positions_per_backtest,
                "entry_dte": [21] * positions_per_backtest,
            })
            odb.write_results(bid, positions, legs)


class TestMergeBasic:
    def test_single_worker_merged(self, tmp_path):
        worker = str(tmp_path / "w1.db")
        output = str(tmp_path / "out.db")
        _make_worker_db(worker, backtest_rows=1, positions_per_backtest=2)

        with OutputDatabase(output) as odb:
            odb.create_schema()

        OutputMerger().merge([worker], output, cleanup=False)

        with OutputDatabase(output) as odb:
            bt_count = odb._con.execute("SELECT COUNT(*) FROM backtest").fetchone()[0]
            pos_count = odb._con.execute("SELECT COUNT(*) FROM position").fetchone()[0]
            pl_count = odb._con.execute("SELECT COUNT(*) FROM position_leg").fetchone()[0]
        assert bt_count == 1
        assert pos_count == 2
        assert pl_count == 2

    def test_two_workers_merged(self, tmp_path):
        w1 = str(tmp_path / "w1.db")
        w2 = str(tmp_path / "w2.db")
        output = str(tmp_path / "out.db")
        _make_worker_db(w1, backtest_rows=1, positions_per_backtest=2)
        _make_worker_db(w2, backtest_rows=1, positions_per_backtest=3)

        with OutputDatabase(output) as odb:
            odb.create_schema()

        OutputMerger().merge([w1, w2], output, cleanup=False)

        with OutputDatabase(output) as odb:
            bt_count = odb._con.execute("SELECT COUNT(*) FROM backtest").fetchone()[0]
            pos_count = odb._con.execute("SELECT COUNT(*) FROM position").fetchone()[0]
        assert bt_count == 2
        assert pos_count == 5


class TestIdResequencing:
    def test_backtest_ids_unique(self, tmp_path):
        w1 = str(tmp_path / "w1.db")
        w2 = str(tmp_path / "w2.db")
        output = str(tmp_path / "out.db")
        _make_worker_db(w1, backtest_rows=2)
        _make_worker_db(w2, backtest_rows=2)

        with OutputDatabase(output) as odb:
            odb.create_schema()

        OutputMerger().merge([w1, w2], output, cleanup=False)

        with OutputDatabase(output) as odb:
            ids = [r[0] for r in odb._con.execute("SELECT id FROM backtest ORDER BY id").fetchall()]
        assert ids == list(range(1, len(ids) + 1))
        assert len(ids) == len(set(ids))

    def test_position_ids_unique(self, tmp_path):
        w1 = str(tmp_path / "w1.db")
        w2 = str(tmp_path / "w2.db")
        output = str(tmp_path / "out.db")
        _make_worker_db(w1, positions_per_backtest=3)
        _make_worker_db(w2, positions_per_backtest=3)

        with OutputDatabase(output) as odb:
            odb.create_schema()

        OutputMerger().merge([w1, w2], output, cleanup=False)

        with OutputDatabase(output) as odb:
            ids = [r[0] for r in odb._con.execute("SELECT id FROM position ORDER BY id").fetchall()]
        assert len(ids) == len(set(ids))

    def test_position_leg_ids_unique(self, tmp_path):
        w1 = str(tmp_path / "w1.db")
        w2 = str(tmp_path / "w2.db")
        output = str(tmp_path / "out.db")
        _make_worker_db(w1, positions_per_backtest=2)
        _make_worker_db(w2, positions_per_backtest=2)

        with OutputDatabase(output) as odb:
            odb.create_schema()

        OutputMerger().merge([w1, w2], output, cleanup=False)

        with OutputDatabase(output) as odb:
            ids = [r[0] for r in odb._con.execute("SELECT id FROM position_leg ORDER BY id").fetchall()]
        assert len(ids) == len(set(ids))

    def test_position_backtest_id_fk_preserved(self, tmp_path):
        w1 = str(tmp_path / "w1.db")
        w2 = str(tmp_path / "w2.db")
        output = str(tmp_path / "out.db")
        _make_worker_db(w1, backtest_rows=1, positions_per_backtest=2)
        _make_worker_db(w2, backtest_rows=1, positions_per_backtest=2)

        with OutputDatabase(output) as odb:
            odb.create_schema()

        OutputMerger().merge([w1, w2], output, cleanup=False)

        with OutputDatabase(output) as odb:
            # Every position.backtest_id must reference an existing backtest.id
            orphans = odb._con.execute("""
                SELECT COUNT(*) FROM position p
                LEFT JOIN backtest b ON p.backtest_id = b.id
                WHERE b.id IS NULL
            """).fetchone()[0]
        assert orphans == 0

    def test_position_leg_position_id_fk_preserved(self, tmp_path):
        w1 = str(tmp_path / "w1.db")
        w2 = str(tmp_path / "w2.db")
        output = str(tmp_path / "out.db")
        _make_worker_db(w1, positions_per_backtest=2)
        _make_worker_db(w2, positions_per_backtest=2)

        with OutputDatabase(output) as odb:
            odb.create_schema()

        OutputMerger().merge([w1, w2], output, cleanup=False)

        with OutputDatabase(output) as odb:
            orphans = odb._con.execute("""
                SELECT COUNT(*) FROM position_leg pl
                LEFT JOIN position p ON pl.position_id = p.id
                WHERE p.id IS NULL
            """).fetchone()[0]
        assert orphans == 0

    def test_study_id_and_combination_id_unchanged(self, tmp_path):
        w1 = str(tmp_path / "w1.db")
        output = str(tmp_path / "out.db")
        _make_worker_db(w1, backtest_rows=1)

        with OutputDatabase(output) as odb:
            odb.create_schema()

        OutputMerger().merge([w1], output, cleanup=False)

        with OutputDatabase(output) as odb:
            row = odb._con.execute(
                "SELECT study_id, combination_id FROM backtest"
            ).fetchone()
        assert row[0] == 1   # study_id preserved
        assert row[1] == 1   # combination_id preserved


class TestEdgeCases:
    def test_missing_worker_file_skipped(self, tmp_path):
        output = str(tmp_path / "out.db")
        with OutputDatabase(output) as odb:
            odb.create_schema()
        # Should not raise even if the path doesn't exist
        OutputMerger().merge(
            [str(tmp_path / "nonexistent.db")], output, cleanup=False
        )

    def test_cleanup_removes_tmp_dir(self, tmp_path):
        w1 = str(tmp_path / "worker_dir" / "w1.db")
        Path(w1).parent.mkdir()
        output = str(tmp_path / "out.db")
        _make_worker_db(w1, positions_per_backtest=1)

        with OutputDatabase(output) as odb:
            odb.create_schema()

        OutputMerger().merge([w1], output, cleanup=True, tmp_dir=str(Path(w1).parent))
        assert not Path(w1).parent.exists()
