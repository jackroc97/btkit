"""
OutputDatabase — write access to a btkit backtest output database.

One OutputDatabase file is created per backtest run (or per matrix run, with
results for all combinations in a single file after merging). Schema is created
on first use via create_schema().
"""

from __future__ import annotations

import duckdb
import polars as pl


class OutputDatabase:
    def __init__(self, db_path: str) -> None:
        self._con = duckdb.connect(db_path)

    def create_schema(self) -> None:
        """Create output tables if they do not already exist."""
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS backtest (
                id                INTEGER PRIMARY KEY,
                matrix_id         INTEGER,
                combination_id    INTEGER,
                strategy_name     VARCHAR         NOT NULL,
                strategy_version  VARCHAR,
                strategy_params   JSON            NOT NULL,
                initial_equity    DOUBLE          NOT NULL,
                slippage_pct      DOUBLE          NOT NULL,
                fee_per_contract  DOUBLE          NOT NULL,
                created_at        TIMESTAMPTZ     NOT NULL
            )
        """)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS position (
                id              INTEGER PRIMARY KEY,
                backtest_id     INTEGER         NOT NULL REFERENCES backtest(id),
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
                right           VARCHAR(1),
                action          VARCHAR(3)      NOT NULL,
                quantity        INTEGER         NOT NULL,
                multiplier      INTEGER         NOT NULL,
                open_price      DOUBLE          NOT NULL,
                exit_price      DOUBLE
            )
        """)

    def write_backtest(self, metadata: dict) -> int:
        """
        Insert a backtest record and return the generated id.
        metadata keys must match the backtest table columns (excluding id).
        """
        raise NotImplementedError

    def write_results(
        self,
        backtest_id: int,
        positions: pl.DataFrame,
        legs: pl.DataFrame,
    ) -> None:
        """
        Write position and position_leg rows for a completed backtest.
        positions must match the position table schema (excluding id and backtest_id).
        legs must match the position_leg schema (excluding id; position_id provided).
        """
        raise NotImplementedError

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> OutputDatabase:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
