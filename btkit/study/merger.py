"""
OutputMerger — consolidates per-worker output databases into one.

Each worker writes to an isolated DuckDB file. After all workers complete,
OutputMerger reads them in sequence and INSERTs all rows into the final output
database, re-sequencing primary keys to be globally unique while preserving
study_id and combination_id.

FK chain:  study → backtest → position → position_leg

Re-sequencing order per worker:
    1. backtest rows: id += bt_offset
    2. position rows: id += pos_offset, backtest_id += bt_offset
    3. position_leg rows: id += pl_offset, position_id += pos_offset

Uses DuckDB ATTACH to avoid loading worker data through Python memory.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import duckdb


class OutputMerger:
    """Merge per-worker output DBs into one consolidated output database."""

    def merge(
        self,
        worker_db_paths: list[str],
        output_db_path: str,
        cleanup: bool = True,
        tmp_dir: str | None = None,
    ) -> None:
        """
        Merge worker databases into output_db_path (must already exist with schema).

        cleanup=True removes worker files (or the entire tmp_dir) after a
        successful merge.
        """
        con = duckdb.connect(output_db_path)
        try:
            for worker_path in worker_db_paths:
                if not Path(worker_path).exists():
                    continue  # worker wrote nothing (no entry signals fired)

                con.execute(f"ATTACH '{worker_path}' AS w (READ_ONLY)")
                try:
                    self._merge_one(con)
                finally:
                    con.execute("DETACH w")
        finally:
            con.close()

        if cleanup:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                for p in worker_db_paths:
                    try:
                        Path(p).unlink()
                    except FileNotFoundError:
                        pass

    # ------------------------------------------------------------------
    # Per-worker merge (con already has 'w' attached as READ_ONLY)
    # ------------------------------------------------------------------

    def _merge_one(self, con: duckdb.DuckDBPyConnection) -> None:
        # Check the worker has any rows to merge.
        bt_count = con.execute("SELECT COUNT(*) FROM w.backtest").fetchone()[0]
        if bt_count == 0:
            return

        # ── 1. backtest ────────────────────────────────────────────────
        bt_offset = con.execute(
            "SELECT COALESCE(MAX(id), 0) FROM backtest"
        ).fetchone()[0]

        con.execute(f"""
            INSERT INTO backtest
            SELECT
                id + {bt_offset},
                study_id,
                combination_id,
                strategy_name,
                strategy_version,
                strategy_params,
                initial_equity,
                slippage_pct,
                fee_per_contract,
                created_at,
                status,
                duration_s,
                warnings,
                error_message,
                error_traceback
            FROM w.backtest
        """)

        # ── 2. position ────────────────────────────────────────────────
        pos_count = con.execute("SELECT COUNT(*) FROM w.position").fetchone()[0]
        if pos_count == 0:
            return

        pos_offset = con.execute(
            "SELECT COALESCE(MAX(id), 0) FROM position"
        ).fetchone()[0]

        con.execute(f"""
            INSERT INTO position
            SELECT
                id          + {pos_offset},
                backtest_id + {bt_offset},
                trade_name,
                open_time,
                exit_time,
                exit_reason,
                open_mark,
                exit_mark,
                worst_mark,
                slippage_cost,
                fee_cost,
                net_pnl
            FROM w.position
        """)

        # ── 3. position_leg ────────────────────────────────────────────
        pl_count = con.execute("SELECT COUNT(*) FROM w.position_leg").fetchone()[0]
        if pl_count == 0:
            return

        pl_offset = con.execute(
            "SELECT COALESCE(MAX(id), 0) FROM position_leg"
        ).fetchone()[0]

        con.execute(f"""
            INSERT INTO position_leg
            SELECT
                id          + {pl_offset},
                position_id + {pos_offset},
                instrument_id,
                symbol,
                expiration,
                strike_price,
                "right",
                action,
                quantity,
                multiplier,
                open_price,
                exit_price,
                entry_delta,
                entry_iv,
                entry_gamma,
                entry_theta,
                entry_vega,
                entry_dte
            FROM w.position_leg
        """)
