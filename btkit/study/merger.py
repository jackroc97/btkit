"""
OutputMerger — consolidates output databases into one.

Used internally to merge per-worker temp DBs after a study run, and
exposed as a general-purpose utility via `btkit db merge` for combining
any set of output databases.

FK chain:  study → backtest → position → position_leg
                                       → position_continuation
           tag ──────────────↗ (via backtest_tag)

Re-sequencing order per source:
    1. study rows:                 id += study_offset
    2. backtest rows:              id += bt_offset; study_id += study_offset (if set)
    3. position rows:              id += pos_offset; backtest_id += bt_offset
    4. position_leg rows:          id += pl_offset; position_id += pos_offset
    5. position_continuation rows: id += pc_offset; position_id += pos_offset
    6. tag rows:                   merged by name; conflicts keep target color, log warning
    7. backtest_tag rows:          re-sequenced with bt_offset and resolved tag ids

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

        # Worker DBs have no study rows — their study_id is pre-created in the
        # target (so no offset needed). Only add study_offset when the source
        # actually contains study rows (general-purpose DB merge).
        if src_study_count > 0:
            bt_study_id_sql = f"study_id + {study_offset}"
        else:
            bt_study_id_sql = "study_id"

        con.execute(f"""
            INSERT INTO backtest
            SELECT
                id + {bt_offset},
                CASE WHEN study_id IS NULL THEN NULL
                     ELSE {bt_study_id_sql} END,
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
                net_pnl,
                target_name
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

        # ── 4. position_continuation ──────────────────────────────────
        pc_count = con.execute(
            "SELECT COUNT(*) FROM w.position_continuation"
        ).fetchone()[0]
        if pc_count > 0:
            pc_offset = con.execute(
                "SELECT COALESCE(MAX(id), 0) FROM position_continuation"
            ).fetchone()[0]
            con.execute(f"""
                INSERT INTO position_continuation
                SELECT
                    id          + {pc_offset},
                    position_id + {pos_offset},
                    continuation_entry_price,
                    continuation_exit_time,
                    continuation_exit_price,
                    continuation_exit_reason,
                    continuation_pnl
                FROM w.position_continuation
            """)

        # ── 5. tag + backtest_tag ──────────────────────────────────────
        # Tags are merged by name. If a tag name already exists in the target,
        # the target's color wins (no overwrite). New tag names are inserted
        # with re-sequenced IDs. backtest_tag rows are then written using the
        # resolved target tag IDs.
        src_tags = con.execute("SELECT id, name, color FROM w.tag").fetchall()
        if not src_tags:
            return

        # Build a mapping: source tag id → target tag id
        tag_id_map: dict[int, int] = {}
        for src_id, name, color in src_tags:
            existing = con.execute(
                "SELECT id FROM tag WHERE name = ?", [name]
            ).fetchone()
            if existing:
                tag_id_map[src_id] = existing[0]
            else:
                next_tag_id = con.execute(
                    "SELECT COALESCE(MAX(id), 0) + 1 FROM tag"
                ).fetchone()[0]
                con.execute(
                    "INSERT INTO tag (id, name, color) VALUES (?, ?, ?)",
                    [next_tag_id, name, color],
                )
                tag_id_map[src_id] = next_tag_id

        src_bt_tags = con.execute(
            "SELECT backtest_id, tag_id FROM w.backtest_tag"
        ).fetchall()
        for src_bt_id, src_tag_id in src_bt_tags:
            target_bt_id = src_bt_id + bt_offset
            target_tag_id = tag_id_map.get(src_tag_id)
            if target_tag_id is None:
                continue
            existing = con.execute(
                "SELECT 1 FROM backtest_tag WHERE backtest_id = ? AND tag_id = ?",
                [target_bt_id, target_tag_id],
            ).fetchone()
            if not existing:
                con.execute(
                    "INSERT INTO backtest_tag (backtest_id, tag_id) VALUES (?, ?)",
                    [target_bt_id, target_tag_id],
                )
