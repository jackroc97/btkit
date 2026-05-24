"""
InputDatabase — read-only access to the btkit input database.

All query methods return Polars DataFrames. A single DuckDB connection is kept
open for the lifetime of the object; call close() or use as a context manager.

The input database schema is created by DatabaseBuilder. InputDatabase opens
the file read-only — it cannot modify the database.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import duckdb
import polars as pl

# SQL that creates the input database schema. Called by DatabaseBuilder, not here.
INPUT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS underlying_bars (
    ts_event        TIMESTAMPTZ     NOT NULL,
    instrument_id   INTEGER         NOT NULL,
    symbol          VARCHAR         NOT NULL,
    expiration      DATE,
    open            DOUBLE          NOT NULL,
    high            DOUBLE          NOT NULL,
    low             DOUBLE          NOT NULL,
    close           DOUBLE          NOT NULL,
    volume          BIGINT
);
CREATE INDEX IF NOT EXISTS idx_underlying_bars
    ON underlying_bars (instrument_id, ts_event);
CREATE INDEX IF NOT EXISTS idx_underlying_bars_expiry
    ON underlying_bars (symbol, expiration);

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
        ts_event_underlying: list[tuple],
        leg_specs: list[dict],
    ) -> pl.DataFrame:
        """
        Batched greeks lookup for all legs across all candidate timestamps.

        ts_event_underlying is a list of (ts_event, underlying_id) pairs — each
        candidate bar carries its own front-month underlying_id, so the query
        correctly filters options by the active futures contract at each point in
        time rather than using a single contract for the whole backtest.

        leg_specs is a list of dicts with keys: name, right (C/P), target_delta,
        target_dte, delta_tolerance, dte_tolerance. The tolerance values are
        per-leg, sourced from LegConfig.delta_tolerance / LegConfig.dte_tolerance.
        Returns one DataFrame for all legs tagged with a leg_name column; caller
        partitions by leg_name and picks the best match per ts_event.
        """
        ts_events = [t for t, _ in ts_event_underlying]
        underlying_ids = [u for _, u in ts_event_underlying]

        ts_df = pl.DataFrame(
            {
                "ts_event": pl.Series(ts_events, dtype=pl.Datetime("us", "UTC")),
                "underlying_id": pl.Series(underlying_ids, dtype=pl.Int64),
            }
        )
        params_df = pl.DataFrame(
            {
                "leg_name": [s["name"] for s in leg_specs],
                "leg_right": [s["right"] for s in leg_specs],
                "delta_lo": [
                    float(s["target_delta"]) - float(s["delta_tolerance"]) for s in leg_specs
                ],
                "delta_hi": [
                    float(s["target_delta"]) + float(s["delta_tolerance"]) for s in leg_specs
                ],
                "dte_lo": [int(s["target_dte"]) - int(s["dte_tolerance"]) for s in leg_specs],
                "dte_hi": [int(s["target_dte"]) + int(s["dte_tolerance"]) for s in leg_specs],
            }
        )
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
                JOIN _entry_ts et
                  ON og.ts_event      = et.ts_event
                 AND og.underlying_id = et.underlying_id
                JOIN _leg_params lp
                  ON ob."right"  = lp.leg_right
                 AND og.dte     BETWEEN lp.dte_lo   AND lp.dte_hi
                 AND og.delta   BETWEEN lp.delta_lo AND lp.delta_hi
                WHERE og.ts_event BETWEEN ? AND ?
                  AND ob.close IS NOT NULL
                """,
                [ts_min, ts_max],
            ).pl()
        finally:
            self._con.unregister("_entry_ts")
            self._con.unregister("_leg_params")

    def greeks_for_strike_legs(
        self,
        strike_targets: pl.DataFrame,
        *,
        strike_tolerance: float = 1.0,
    ) -> pl.DataFrame:
        """
        For each (ts_event, leg_name) in strike_targets, find the option whose
        strike_price is within strike_tolerance of target_strike and whose
        expiration matches the reference leg's exact expiration date.

        strike_targets columns: ts_event, underlying_id, leg_name, right (C/P),
        target_strike, reference_expiration (DATE).  The underlying_id column
        scopes the search to the correct front-month futures contract at each bar.

        Returns the same column set as greeks_for_all_legs().
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
                 AND og.underlying_id           = st.underlying_id
                 AND ob."right"                = st.right
                 AND ob.expiration             = st.reference_expiration
                 AND abs(ob.strike_price - st.target_strike) <= ?
                WHERE og.ts_event BETWEEN ? AND ?
                  AND ob.close IS NOT NULL
                """,
                [strike_tolerance, ts_min, ts_max],
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
    # Futures roll schedule
    # ------------------------------------------------------------------

    def front_future_schedule(
        self,
        root_symbol: str,
        start_date: date,
        end_date: date,
        roll_days: int = 7,
    ) -> pl.DataFrame:
        """
        Return a DataFrame mapping every calendar date in [start_date, end_date]
        to the instrument_id of the front-month futures contract for root_symbol.

        Roll logic: on any date D, the active contract is the one with the
        minimum expiration >= D + roll_days.  This rolls to the next contract
        `roll_days` before the front month expires, matching typical CME practice.

        Requires underlying_bars to have an expiration column (populated at ingest).
        Falls back to a single arbitrary contract if expiration data is absent.

        Returns columns: date (pl.Date), underlying_id (pl.Int64).
        """
        futures_df = self._con.execute(
            """
            SELECT DISTINCT instrument_id, expiration
            FROM underlying_bars
            WHERE symbol LIKE ? AND expiration IS NOT NULL
            ORDER BY expiration
            """,
            [f"{root_symbol}%"],
        ).pl()

        if futures_df.is_empty():
            # No expiration data — fall back to first match (legacy behaviour)
            row = self._con.execute(
                "SELECT instrument_id FROM underlying_bars WHERE symbol LIKE ? LIMIT 1",
                [f"{root_symbol}%"],
            ).fetchone()
            if row is None:
                return pl.DataFrame(
                    {
                        "date": pl.Series([], dtype=pl.Date),
                        "underlying_id": pl.Series([], dtype=pl.Int64),
                    }
                )
            uid = int(row[0])
            dates = [
                start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)
            ]
            return pl.DataFrame({"date": dates, "underlying_id": [uid] * len(dates)})

        expirations = list(
            zip(
                futures_df["instrument_id"].to_list(),
                futures_df["expiration"].to_list(),
                strict=False,
            )
        )

        dates: list[date] = []
        underlying_ids: list[int] = []
        d = start_date
        while d <= end_date:
            roll_cutoff = d + timedelta(days=roll_days)
            # Front month: earliest expiration that hasn't yet hit its roll date
            front_id = None
            for iid, exp in expirations:
                if exp >= roll_cutoff:
                    front_id = iid
                    break
            if front_id is None:
                # All contracts have rolled past — use the last one
                front_id = expirations[-1][0]
            dates.append(d)
            underlying_ids.append(front_id)
            d += timedelta(days=1)

        return pl.DataFrame(
            {
                "date": pl.Series(dates, dtype=pl.Date),
                "underlying_id": pl.Series(underlying_ids, dtype=pl.Int64),
            }
        )

    def front_future_id(
        self,
        root_symbol: str,
        as_of: date,
        roll_days: int = 7,
    ) -> int | None:
        """
        Return the instrument_id of the front-month futures contract for
        root_symbol as of a single date.  Convenience wrapper around
        front_future_schedule() used by engine and exit scanner for indicator
        lookups where a single underlying_id is sufficient.
        """
        schedule = self.front_future_schedule(root_symbol, as_of, as_of, roll_days)
        if schedule.is_empty():
            return None
        return int(schedule["underlying_id"][0])

    def underlying_bars_for_root(
        self,
        root_symbol: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        Load 1-minute bars from ALL futures contracts matching root_symbol over
        [start, end].  The returned DataFrame includes instrument_id so callers
        can filter to the front-month contract using a roll schedule.
        """
        return self._con.execute(
            """
            SELECT ts_event, instrument_id, symbol, open, high, low, close, volume
            FROM underlying_bars
            WHERE symbol LIKE ?
              AND ts_event >= ? AND ts_event <= ?
            ORDER BY instrument_id, ts_event
            """,
            [f"{root_symbol}%", start, end],
        ).pl()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> InputDatabase:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
