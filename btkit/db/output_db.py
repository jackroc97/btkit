"""
OutputDatabase — write access to a btkit backtest output database.

One OutputDatabase file is created per backtest run (scalar) or study run
(all combinations merged into one file). Schema is created on first use via
create_schema(), which also auto-migrates legacy DBs that still have the old
matrix_id column name.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import duckdb
import polars as pl


class OutputDatabase:
    def __init__(self, db_path: str) -> None:
        self._con = duckdb.connect(db_path)

    def create_schema(self) -> None:
        """Create output tables if they do not already exist."""
        self._maybe_migrate()
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS study (
                id                  INTEGER PRIMARY KEY,
                name                VARCHAR         NOT NULL,
                strategy_yaml       TEXT            NOT NULL,
                total_combinations  INTEGER         NOT NULL,
                created_at          TIMESTAMPTZ     NOT NULL,
                finished_at         TIMESTAMPTZ,
                note                VARCHAR
            )
        """)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS backtest (
                id                INTEGER PRIMARY KEY,
                study_id          INTEGER,
                combination_id    INTEGER,
                strategy_name     VARCHAR         NOT NULL,
                strategy_version  VARCHAR,
                strategy_params   JSON            NOT NULL,
                initial_equity    DOUBLE          NOT NULL,
                slippage_pct      DOUBLE          NOT NULL,
                fee_per_contract  DOUBLE          NOT NULL,
                created_at        TIMESTAMPTZ     NOT NULL,
                status            VARCHAR,
                duration_s        DOUBLE,
                warnings          JSON,
                error_message     VARCHAR,
                error_traceback   VARCHAR,
                note              VARCHAR
            )
        """)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS position (
                id              INTEGER PRIMARY KEY,
                backtest_id     INTEGER         NOT NULL REFERENCES backtest(id),
                trade_name      VARCHAR         NOT NULL,
                open_time       TIMESTAMPTZ     NOT NULL,
                exit_time       TIMESTAMPTZ,
                exit_reason     VARCHAR,
                open_mark       DOUBLE          NOT NULL,
                exit_mark       DOUBLE,
                worst_mark      DOUBLE,
                slippage_cost   DOUBLE,
                fee_cost        DOUBLE,
                net_pnl         DOUBLE
            )
        """)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS position_leg (
                id              INTEGER PRIMARY KEY,
                position_id     INTEGER         NOT NULL REFERENCES position(id),
                instrument_id   INTEGER         NOT NULL,
                symbol          VARCHAR         NOT NULL,
                expiration      DATE,
                strike_price    DOUBLE,
                "right"         VARCHAR(1),
                action          VARCHAR(3)      NOT NULL,
                quantity        INTEGER         NOT NULL,
                multiplier      INTEGER         NOT NULL,
                open_price      DOUBLE          NOT NULL,
                exit_price      DOUBLE,
                entry_delta     DOUBLE,
                entry_iv        DOUBLE,
                entry_gamma     DOUBLE,
                entry_theta     DOUBLE,
                entry_vega      DOUBLE,
                entry_dte       INTEGER
            )
        """)

    # ------------------------------------------------------------------
    # Study
    # ------------------------------------------------------------------

    def write_study(
        self,
        name: str,
        strategy_yaml: str,
        total_combinations: int,
        note: str | None = None,
    ) -> int:
        """Insert a study row and return the generated id."""
        next_id = self._next_id("study")
        self._con.execute(
            """
            INSERT INTO study (id, name, strategy_yaml, total_combinations, created_at, note)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [next_id, name, strategy_yaml, total_combinations, datetime.now(UTC), note],
        )
        return next_id

    def finalize_study(self, study_id: int) -> None:
        """Set finished_at on a study row."""
        self._con.execute(
            "UPDATE study SET finished_at = ? WHERE id = ?",
            [datetime.now(UTC), study_id],
        )

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def write_backtest(self, metadata: dict) -> int:
        """
        Insert a backtest record and return the generated id.
        metadata keys must match backtest table columns (excluding id).
        """
        next_id = self._next_id("backtest")
        self._con.execute(
            """
            INSERT INTO backtest (
                id, study_id, combination_id, strategy_name, strategy_version,
                strategy_params, initial_equity, slippage_pct, fee_per_contract,
                created_at, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                next_id,
                metadata.get("study_id"),
                metadata.get("combination_id"),
                metadata["strategy_name"],
                metadata.get("strategy_version"),
                json.dumps(metadata["strategy_params"]),
                metadata["initial_equity"],
                metadata["slippage_pct"],
                metadata["fee_per_contract"],
                metadata.get("created_at", datetime.now(UTC)),
                metadata.get("note"),
            ],
        )
        return next_id

    def finalize_backtest(
        self,
        backtest_id: int,
        *,
        status: str,
        duration_s: float,
        warnings: list[dict],
        error_message: str | None = None,
        error_traceback: str | None = None,
    ) -> None:
        """Update a backtest record with run outcome after completion or failure."""
        self._con.execute(
            """
            UPDATE backtest SET
                status          = ?,
                duration_s      = ?,
                warnings        = ?,
                error_message   = ?,
                error_traceback = ?
            WHERE id = ?
            """,
            [
                status,
                duration_s,
                json.dumps(warnings) if warnings else None,
                error_message,
                error_traceback,
                backtest_id,
            ],
        )

    def write_results(
        self,
        backtest_id: int,
        positions: pl.DataFrame,
        legs: pl.DataFrame,
    ) -> None:
        """
        Write position and position_leg rows for a completed backtest in a single
        transaction.

        positions must contain columns:
            entry_id, trade_name, open_time, exit_time, exit_reason,
            open_mark, exit_mark, worst_mark, slippage_cost, fee_cost, net_pnl

        legs must contain columns:
            entry_id, instrument_id, symbol, expiration, strike_price, right,
            action, quantity, multiplier, open_price, exit_price

        entry_id links legs to their parent position and is not written to the DB.
        """
        if positions.is_empty():
            return

        # Assign sequential position ids.
        pos_start = self._next_id("position")
        pos_ids = list(range(pos_start, pos_start + len(positions)))
        positions = positions.with_columns(
            pl.Series("id", pos_ids),
            pl.lit(backtest_id).alias("backtest_id"),
        )

        # Build entry_id → position_id map for legs.
        id_map = dict(zip(positions["entry_id"].to_list(), pos_ids, strict=False))

        # Write positions.
        pos_rows = positions.select(
            [
                "id",
                "backtest_id",
                "trade_name",
                "open_time",
                "exit_time",
                "exit_reason",
                "open_mark",
                "exit_mark",
                "worst_mark",
                "slippage_cost",
                "fee_cost",
                "net_pnl",
            ]
        )
        self._con.register("_positions", pos_rows)
        self._con.execute("INSERT INTO position SELECT * FROM _positions")
        self._con.unregister("_positions")

        # Write legs.
        if not legs.is_empty():
            leg_start = self._next_id("position_leg")
            leg_ids = list(range(leg_start, leg_start + len(legs)))
            position_ids = [id_map[eid] for eid in legs["entry_id"].to_list()]
            legs = legs.with_columns(
                pl.Series("id", leg_ids),
                pl.Series("position_id", position_ids),
            )
            leg_rows = legs.select(
                [
                    "id",
                    "position_id",
                    "instrument_id",
                    "symbol",
                    "expiration",
                    "strike_price",
                    "right",
                    "action",
                    "quantity",
                    "multiplier",
                    "open_price",
                    "exit_price",
                    "entry_delta",
                    "entry_iv",
                    "entry_gamma",
                    "entry_theta",
                    "entry_vega",
                    "entry_dte",
                ]
            )
            self._con.register("_legs", leg_rows)
            self._con.execute("INSERT INTO position_leg SELECT * FROM _legs")
            self._con.unregister("_legs")

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_backtest(self, backtest_id: int) -> None:
        """Permanently delete a backtest and all its positions and legs."""
        self._con.execute(
            "DELETE FROM position_leg WHERE position_id IN "
            "(SELECT id FROM position WHERE backtest_id = ?)",
            [backtest_id],
        )
        self._con.execute("DELETE FROM position WHERE backtest_id = ?", [backtest_id])
        self._con.execute("DELETE FROM backtest WHERE id = ?", [backtest_id])

    def delete_study(self, study_id: int) -> None:
        """Permanently delete a study and all its backtests, positions, and legs."""
        self._con.execute(
            "DELETE FROM position_leg WHERE position_id IN "
            "(SELECT id FROM position WHERE backtest_id IN "
            "(SELECT id FROM backtest WHERE study_id = ?))",
            [study_id],
        )
        self._con.execute(
            "DELETE FROM position WHERE backtest_id IN "
            "(SELECT id FROM backtest WHERE study_id = ?)",
            [study_id],
        )
        self._con.execute("DELETE FROM backtest WHERE study_id = ?", [study_id])
        self._con.execute("DELETE FROM study WHERE id = ?", [study_id])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _next_id(self, table: str) -> int:
        row = self._con.execute(f"SELECT COALESCE(MAX(id), 0) + 1 FROM {table}").fetchone()
        return row[0]

    def _maybe_migrate(self) -> None:
        """Apply incremental schema migrations to existing output DBs."""
        tables = {
            row[0]
            for row in self._con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
        }
        if "backtest" not in tables:
            return
        bt_cols = {
            row[0]
            for row in self._con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'backtest'"
            ).fetchall()
        }
        if "matrix_id" in bt_cols and "study_id" not in bt_cols:
            self._con.execute("ALTER TABLE backtest RENAME COLUMN matrix_id TO study_id")
        if "note" not in bt_cols:
            self._con.execute("ALTER TABLE backtest ADD COLUMN note VARCHAR")

        if "study" in tables:
            st_cols = {
                row[0]
                for row in self._con.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'study'"
                ).fetchall()
            }
            if "note" not in st_cols:
                self._con.execute("ALTER TABLE study ADD COLUMN note VARCHAR")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> OutputDatabase:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
