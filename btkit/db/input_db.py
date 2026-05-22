"""
InputDatabase — read-only access to the btkit input database.

All query methods return Polars DataFrames. A single DuckDB connection is kept
open for the lifetime of the object; call close() or use as a context manager.

The input database is built by pipeline.builder.DatabaseBuilder and is treated
as immutable during backtest runs. Multiple BacktestEngine instances (in parallel
matrix runs) each open their own InputDatabase connection to the same file.
"""

from __future__ import annotations

from datetime import datetime

import duckdb
import polars as pl


class InputDatabase:
    def __init__(self, db_path: str) -> None:
        self._con = duckdb.connect(db_path, read_only=True)

    # ------------------------------------------------------------------
    # Underlying bars
    # ------------------------------------------------------------------

    def underlying_bars(
        self,
        instrument_id: int,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        1-minute OHLCV bars for the given underlying instrument over [start, end].
        Returns columns: ts_event, instrument_id, symbol, open, high, low, close, volume.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Option bars
    # ------------------------------------------------------------------

    def option_bars(
        self,
        instrument_id: int,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        1-minute OHLCV bars for a single option instrument. Includes pre-joined
        definition metadata: expiration, strike_price, right, multiplier.
        """
        raise NotImplementedError

    def option_bars_for_legs(
        self,
        instrument_ids: list[int],
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        Batch-loads bars for a set of option instrument IDs over [start, end].
        Used by ExitScanner to load all open-position legs in a single query.
        Returns the same schema as option_bars plus instrument_id.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Greeks / leg selection
    # ------------------------------------------------------------------

    def greeks_for_entry(
        self,
        underlying_id: int,
        ts_events: list[datetime],
        right: str,
        target_delta: float,
        target_dte: int,
        delta_tolerance: float = 0.05,
        dte_tolerance: int = 5,
    ) -> pl.DataFrame:
        """
        For each timestamp in ts_events, returns candidate options near the
        desired delta and DTE. The caller (EntryScanner._select_legs) picks
        the best match per timestamp by minimising |actual_delta - target_delta|.

        Returns columns: ts_event, instrument_id, underlying_id, dte, iv, delta,
        gamma, theta, vega — joined with option_bars for strike, expiration, right,
        multiplier, and close price.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    def indicators(
        self,
        underlying_id: int,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        Returns a wide DataFrame with one column per indicator name, indexed
        by ts_event. Internally executes a DuckDB PIVOT over indicator_bars
        joined with indicator_definition. Callers always receive a wide DataFrame
        — the tall storage format is transparent.

        Returns columns: ts_event, <indicator_name_1>, <indicator_name_2>, ...
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> InputDatabase:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
