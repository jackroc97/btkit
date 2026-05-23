"""
InputDatabase — read-only access to the btkit input database.

All query methods return Polars DataFrames. A single DuckDB connection is kept
open for the lifetime of the object; call close() or use as a context manager.

The input database schema is created by DatabaseBuilder. InputDatabase opens
the file read-only — it cannot modify the database.
"""

from __future__ import annotations

from datetime import datetime

import duckdb
import polars as pl

# SQL that creates the input database schema. Called by DatabaseBuilder, not here.
INPUT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS underlying_bars (
    ts_event        TIMESTAMPTZ     NOT NULL,
    instrument_id   INTEGER         NOT NULL,
    symbol          VARCHAR         NOT NULL,
    open            DOUBLE          NOT NULL,
    high            DOUBLE          NOT NULL,
    low             DOUBLE          NOT NULL,
    close           DOUBLE          NOT NULL,
    volume          BIGINT
);
CREATE INDEX IF NOT EXISTS idx_underlying_bars
    ON underlying_bars (instrument_id, ts_event);

CREATE TABLE IF NOT EXISTS option_bars (
    ts_event        TIMESTAMPTZ     NOT NULL,
    instrument_id   INTEGER         NOT NULL,
    underlying_id   INTEGER         NOT NULL,
    symbol          VARCHAR         NOT NULL,
    expiration      DATE            NOT NULL,
    strike_price    DOUBLE          NOT NULL,
    "right"         VARCHAR(1)      NOT NULL,
    multiplier      INTEGER         NOT NULL,
    open            DOUBLE,
    high            DOUBLE,
    low             DOUBLE,
    close           DOUBLE,
    volume          BIGINT
);
CREATE INDEX IF NOT EXISTS idx_option_bars_lookup
    ON option_bars (underlying_id, "right", expiration, strike_price, ts_event);
CREATE INDEX IF NOT EXISTS idx_option_bars_instrument
    ON option_bars (instrument_id, ts_event);

CREATE TABLE IF NOT EXISTS option_greeks (
    ts_event        TIMESTAMPTZ     NOT NULL,
    instrument_id   INTEGER         NOT NULL,
    underlying_id   INTEGER         NOT NULL,
    dte             INTEGER         NOT NULL,
    T               DOUBLE          NOT NULL,
    iv              DOUBLE,
    delta           DOUBLE,
    gamma           DOUBLE,
    theta           DOUBLE,
    vega            DOUBLE
);
CREATE INDEX IF NOT EXISTS idx_option_greeks_lookup
    ON option_greeks (underlying_id, ts_event, dte);

CREATE TABLE IF NOT EXISTS indicator_definition (
    id                  INTEGER PRIMARY KEY,
    name                VARCHAR     NOT NULL,
    underlying_id       INTEGER     NOT NULL,
    underlying_symbol   VARCHAR     NOT NULL,
    params              JSON,
    script_source       TEXT        NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_indicator_def_unique
    ON indicator_definition (name, underlying_id);

CREATE TABLE IF NOT EXISTS indicator_bars (
    ts_event        TIMESTAMPTZ     NOT NULL,
    indicator_id    INTEGER         NOT NULL REFERENCES indicator_definition(id),
    value           DOUBLE
);
CREATE INDEX IF NOT EXISTS idx_indicator_bars
    ON indicator_bars (indicator_id, ts_event);
"""


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
        1-minute OHLCV bars for the given underlying over [start, end].
        Returns columns: ts_event, instrument_id, symbol, open, high, low, close, volume.
        """
        return self._con.execute(
            """
            SELECT ts_event, instrument_id, symbol, open, high, low, close, volume
            FROM underlying_bars
            WHERE instrument_id = ?
              AND ts_event >= ? AND ts_event <= ?
            ORDER BY ts_event
            """,
            [instrument_id, start, end],
        ).pl()

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
        1-minute OHLCV bars for a single option instrument including pre-joined
        definition metadata: expiration, strike_price, right, multiplier.
        """
        return self._con.execute(
            """
            SELECT ts_event, instrument_id, underlying_id, symbol,
                   expiration, strike_price, "right", multiplier,
                   open, high, low, close, volume
            FROM option_bars
            WHERE instrument_id = ?
              AND ts_event >= ? AND ts_event <= ?
            ORDER BY ts_event
            """,
            [instrument_id, start, end],
        ).pl()

    def option_bars_for_legs(
        self,
        instrument_ids: list[int],
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        Batch-loads bars for a set of option instrument IDs over [start, end].
        Used by ExitScanner to load all open-position legs in a single query.
        Returns the same columns as option_bars.
        """
        ids_df = pl.DataFrame({"instrument_id": instrument_ids})
        self._con.register("_leg_ids", ids_df)
        try:
            return self._con.execute(
                """
                SELECT ob.ts_event, ob.instrument_id, ob.underlying_id, ob.symbol,
                       ob.expiration, ob.strike_price, ob."right", ob.multiplier,
                       ob.open, ob.high, ob.low, ob.close, ob.volume
                FROM option_bars ob
                JOIN _leg_ids li ON ob.instrument_id = li.instrument_id
                WHERE ob.ts_event >= ? AND ob.ts_event <= ?
                ORDER BY ob.instrument_id, ob.ts_event
                """,
                [start, end],
            ).pl()
        finally:
            self._con.unregister("_leg_ids")

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
        For each timestamp in ts_events, returns all candidate options within
        delta_tolerance of target_delta and dte_tolerance of target_dte.

        EntryScanner._select_legs() picks the best match per timestamp by
        minimising |actual_delta - target_delta|.

        Returns columns: ts_event, instrument_id, underlying_id, dte, iv, delta,
        gamma, theta, vega, strike_price, expiration, right, multiplier, close.
        """
        ts_df = pl.DataFrame({"ts_event": ts_events})
        self._con.register("_entry_ts", ts_df)
        try:
            return self._con.execute(
                """
                SELECT og.ts_event, og.instrument_id, og.underlying_id,
                       og.dte, og.iv, og.delta, og.gamma, og.theta, og.vega,
                       ob.strike_price, ob.expiration, ob."right", ob.multiplier,
                       ob.symbol, ob.close
                FROM option_greeks og
                JOIN option_bars ob
                  ON og.instrument_id = ob.instrument_id
                 AND og.ts_event = ob.ts_event
                JOIN _entry_ts et ON og.ts_event = et.ts_event
                WHERE og.underlying_id = ?
                  AND ob."right" = ?
                  AND og.dte BETWEEN ? AND ?
                  AND og.delta BETWEEN ? AND ?
                  AND ob.close IS NOT NULL
                """,
                [
                    underlying_id,
                    right,
                    target_dte - dte_tolerance,
                    target_dte + dte_tolerance,
                    target_delta - delta_tolerance,
                    target_delta + delta_tolerance,
                ],
            ).pl()
        finally:
            self._con.unregister("_entry_ts")

    def greeks_for_all_legs(
        self,
        underlying_id: int,
        ts_events: list,
        leg_specs: list[dict],
        *,
        delta_tolerance: float = 0.10,
        dte_tolerance: int = 5,
    ) -> pl.DataFrame:
        """
        Batched replacement for calling greeks_for_entry() once per leg.

        leg_specs is a list of dicts with keys: name, right (C/P), target_delta,
        target_dte. Returns one DataFrame for all legs tagged with a leg_name
        column; caller partitions by leg_name and picks the best match per
        ts_event.

        Improvements over per-leg greeks_for_entry():
          - Single DB roundtrip regardless of leg count.
          - ts_event BETWEEN min/max filter lets the (underlying_id, ts_event, dte)
            index prune the scan before the join with _entry_ts.
        """
        ts_df = pl.DataFrame({"ts_event": pl.Series(ts_events, dtype=pl.Datetime("us", "UTC"))})
        params_df = pl.DataFrame({
            "leg_name":  [s["name"]  for s in leg_specs],
            "leg_right": [s["right"] for s in leg_specs],
            "delta_lo":  [float(s["target_delta"]) - delta_tolerance for s in leg_specs],
            "delta_hi":  [float(s["target_delta"]) + delta_tolerance for s in leg_specs],
            "dte_lo":    [int(s["target_dte"]) - dte_tolerance for s in leg_specs],
            "dte_hi":    [int(s["target_dte"]) + dte_tolerance for s in leg_specs],
        })
        self._con.register("_entry_ts", ts_df)
        self._con.register("_leg_params", params_df)
        ts_min = min(ts_events)
        ts_max = max(ts_events)
        try:
            return self._con.execute(
                """
                SELECT lp.leg_name, og.ts_event, og.instrument_id, og.underlying_id,
                       og.dte, og.iv, og.delta, og.gamma, og.theta, og.vega,
                       ob.strike_price, ob.expiration, ob."right", ob.multiplier,
                       ob.symbol, ob.close
                FROM option_greeks og
                JOIN option_bars ob
                  ON og.instrument_id = ob.instrument_id
                 AND og.ts_event      = ob.ts_event
                JOIN _entry_ts et ON og.ts_event = et.ts_event
                JOIN _leg_params lp
                  ON ob."right"  = lp.leg_right
                 AND og.dte     BETWEEN lp.dte_lo   AND lp.dte_hi
                 AND og.delta   BETWEEN lp.delta_lo AND lp.delta_hi
                WHERE og.underlying_id = ?
                  AND og.ts_event BETWEEN ? AND ?
                  AND ob.close IS NOT NULL
                """,
                [underlying_id, ts_min, ts_max],
            ).pl()
        finally:
            self._con.unregister("_entry_ts")
            self._con.unregister("_leg_params")

    def greeks_for_strike_legs(
        self,
        underlying_id: int,
        strike_targets: pl.DataFrame,
        *,
        strike_tolerance: float = 1.0,
    ) -> pl.DataFrame:
        """
        For each (ts_event, leg_name) in strike_targets, find the option whose
        strike_price is within strike_tolerance of target_strike and whose DTE
        falls in [dte_lo, dte_hi].

        strike_targets columns: ts_event, leg_name, right (C/P), target_strike,
        dte_lo, dte_hi.

        Returns the same column set as greeks_for_all_legs() so that
        EntryScanner._select_legs() can apply identical post-processing.
        Caller picks the closest-strike match per (ts_event, leg_name) in Polars.
        """
        ts_min = strike_targets["ts_event"].min()
        ts_max = strike_targets["ts_event"].max()
        self._con.register("_strike_targets", strike_targets)
        try:
            return self._con.execute(
                """
                SELECT st.leg_name, og.ts_event, og.instrument_id, og.underlying_id,
                       og.dte, og.iv, og.delta, og.gamma, og.theta, og.vega,
                       ob.strike_price, ob.expiration, ob."right", ob.multiplier,
                       ob.symbol, ob.close
                FROM option_greeks og
                JOIN option_bars ob
                  ON og.instrument_id = ob.instrument_id
                 AND og.ts_event      = ob.ts_event
                JOIN _strike_targets st
                  ON og.ts_event               = st.ts_event
                 AND ob."right"                = st.right
                 AND og.dte                    BETWEEN st.dte_lo AND st.dte_hi
                 AND abs(ob.strike_price - st.target_strike) <= ?
                WHERE og.underlying_id = ?
                  AND og.ts_event BETWEEN ? AND ?
                  AND ob.close IS NOT NULL
                """,
                [strike_tolerance, underlying_id, ts_min, ts_max],
            ).pl()
        finally:
            self._con.unregister("_strike_targets")

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
        by ts_event. Executes a DuckDB PIVOT over indicator_bars joined with
        indicator_definition. Returns empty DataFrame if no indicators exist.

        Columns: ts_event, <indicator_name_1>, <indicator_name_2>, ...
        """
        # Check whether any indicators exist for this underlying first.
        count = self._con.execute(
            "SELECT COUNT(*) FROM indicator_definition WHERE underlying_id = ?",
            [underlying_id],
        ).fetchone()[0]
        if count == 0:
            return pl.DataFrame({"ts_event": pl.Series([], dtype=pl.Datetime("us", "UTC"))})

        # DuckDB PIVOT cannot use parameters in its source subquery, so fetch
        # tall and pivot in Polars instead.
        tall = self._con.execute(
            """
            SELECT ib.ts_event, idef.name, ib.value
            FROM indicator_bars ib
            JOIN indicator_definition idef ON ib.indicator_id = idef.id
            WHERE idef.underlying_id = ?
              AND ib.ts_event >= ? AND ib.ts_event <= ?
            ORDER BY ib.ts_event
            """,
            [underlying_id, start, end],
        ).pl()
        if tall.is_empty():
            return pl.DataFrame({"ts_event": pl.Series([], dtype=pl.Datetime("us", "UTC"))})
        return tall.pivot(on="name", index="ts_event", values="value", aggregate_function="first")

    # ------------------------------------------------------------------
    # Instrument lookup
    # ------------------------------------------------------------------

    def instrument_id_for_symbol(self, root_symbol: str) -> int | None:
        """
        Returns the instrument_id for the given root_symbol from underlying_bars,
        or None if not found.
        """
        row = self._con.execute(
            "SELECT instrument_id FROM underlying_bars WHERE symbol LIKE ? LIMIT 1",
            [f"{root_symbol}%"],
        ).fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> InputDatabase:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
