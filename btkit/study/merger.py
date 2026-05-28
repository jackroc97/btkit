"""
OutputMerger — consolidates output databases into one.

Used internally to merge per-worker temp DBs after a study run, and
exposed as a general-purpose utility via `btkit db merge` for combining
any set of output databases.

FK chain:  study → backtest → position → position_leg

Re-sequencing order per source:
    1. study rows:       id += study_offset
    2. backtest rows:    id += bt_offset; study_id += study_offset (if set)
    3. position rows:    id += pos_offset; backtest_id += bt_offset
    4. position_leg rows: id += pl_offset; position_id += pos_offset

Uses DuckDB ATTACH to avoid loading source data through Python memory.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import duckdb


class OutputMerger:
    """Merge one or more source output DBs into a consolidated target database."""

    def merge(
        self,
        source_db_paths: list[str],
        target_db_path: str,
        cleanup: bool = False,
        tmp_dir: str | None = None,
    ) -> None:
        """
        Merge source databases into target_db_path.

        target_db_path must already exist with the btkit schema applied
        (i.e. OutputDatabase.create_schema() has been called on it).

        cleanup=True removes source files (or the entire tmp_dir if given)
        after a successful merge — used by StudyRunner for temp worker DBs.
        """
        con = duckdb.connect(target_db_path)
        try:
            for source_path in source_db_paths:
                if not Path(source_path).exists():
                    continue

                con.execute(f"ATTACH '{source_path}' AS w (READ_ONLY)")
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
                for p in source_db_paths:
                    try:
                        Path(p).unlink()
                    except FileNotFoundError:
                        pass

    # ------------------------------------------------------------------
    # Per-source merge (con already has 'w' attached as READ_ONLY)
    # ------------------------------------------------------------------

    def _merge_one(self, con: duckdb.DuckDBPyConnection) -> None:
        # ── 0. study ───────────────────────────────────────────────────
        # Worker DBs have no study rows (StudyRunner pre-creates them in
        # the target). General merges may have study rows — handle both.
        study_offset = con.execute(
            "SELECT COALESCE(MAX(id), 0) FROM study"
        ).fetchone()[0]

        src_study_count = con.execute("SELECT COUNT(*) FROM w.study").fetchone()[0]
        if src_study_count > 0:
            con.execute(f"""
                INSERT INTO study
                SELECT
                    id + {study_offset},
                    name,
                    strategy_yaml,
                    total_combinations,
                    created_at,
                    finished_at,
                    note
                FROM w.study
            """)

        # ── 1. backtest ────────────────────────────────────────────────
        bt_count = con.execute("SELECT COUNT(*) FROM w.backtest").fetchone()[0]
        if bt_count == 0:
            return

        bt_offset = con.execute(
            "SELECT COALESCE(MAX(id), 0) FROM backtest"
        ).fetchone()[0]

        con.execute(f"""
            INSERT INTO backtest
            SELECT
                id + {bt_offset},
                CASE WHEN study_id IS NULL THEN NULL
                     ELSE study_id + {study_offset} END,
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
                error_traceback,
                note
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
