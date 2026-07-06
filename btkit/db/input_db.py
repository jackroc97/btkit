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
    ON option_greeks (underlying_id, dte, ts_event);

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
        Returns columns: ts_event, instrument_id, open, high, low, close, volume.

        Intentionally omits columns not consumed by _compute_position_marks /
        _adjust_leg_out_exits to reduce Arrow-transfer overhead. ORDER BY is
        also omitted — callers sort by [entry_id, ts_event] in Polars anyway.
        """
        id_list = ", ".join(str(int(x)) for x in instrument_ids)
        return self._con.execute(
            f"""
            SELECT ts_event, instrument_id, open, high, low, close, volume
            FROM option_bars
            WHERE instrument_id IN ({id_list})
              AND ts_event >= ? AND ts_event <= ?
            """,
            [start, end],
        ).pl()

    def option_greeks_for_legs(
        self,
        instrument_ids: list[int],
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        Batch-loads greeks for a set of option instrument IDs over [start, end].
        Used by ExitScanner to compute spread net vega for vega_exit conditions.
        Returns columns: ts_event, instrument_id, vega.
        """
        id_list = ", ".join(str(int(x)) for x in instrument_ids)
        return self._con.execute(
            f"""
            SELECT ts_event, instrument_id, vega
            FROM option_greeks
            WHERE instrument_id IN ({id_list})
              AND ts_event >= ? AND ts_event <= ?
            """,
            [start, end],
        ).pl()

    def underlying_ids_for_options(
        self,
        instrument_ids: list[int],
    ) -> dict[int, int]:
        """
        Return {option_instrument_id: underlying_instrument_id} for the given
        option instrument IDs.  A single query against option_bars resolves the
        exact futures contract each option settles against, as recorded by the
        data vendor — more reliable than a roll-schedule heuristic near roll dates.
        """
        if not instrument_ids:
            return {}
        id_list = ", ".join(str(int(x)) for x in instrument_ids)
        rows = self._con.execute(
            f"""
            SELECT DISTINCT instrument_id, underlying_id
            FROM option_bars
            WHERE instrument_id IN ({id_list})
            """
        ).fetchall()
        return {int(row[0]): int(row[1]) for row in rows}

    def settlement_closes_for_underlyings(
        self,
        underlying_ids: list[int],
        start: datetime,
        end: datetime,
        tz_str: str,
        close_time,
    ) -> pl.DataFrame:
        """
        Return [underlying_id, exp_date, settlement_close] — one row per
        (underlying, trading day) with the last bar close at or before
        close_time (local time in tz_str) on each day.

        Called once per scan() to pre-load all settlement prices for the
        entire backtest window, eliminating per-cohort DB queries.
        Returns an empty DataFrame when underlying_ids is empty or no bars exist.
        """
        if not underlying_ids:
            return pl.DataFrame(
                {
                    "underlying_id": pl.Series([], dtype=pl.Int64),
                    "exp_date": pl.Series([], dtype=pl.Date),
                    "settlement_close": pl.Series([], dtype=pl.Float64),
                }
            )
        close_minutes = close_time.hour * 60 + close_time.minute
        id_list = ", ".join(str(int(x)) for x in underlying_ids)
        bars = self._con.execute(
            f"""
            SELECT ts_event, instrument_id, close
            FROM underlying_bars
            WHERE instrument_id IN ({id_list})
              AND ts_event >= ? AND ts_event <= ?
            """,
            [start, end],
        ).pl()
        if bars.is_empty():
            return pl.DataFrame(
                {
                    "underlying_id": pl.Series([], dtype=pl.Int64),
                    "exp_date": pl.Series([], dtype=pl.Date),
                    "settlement_close": pl.Series([], dtype=pl.Float64),
                }
            )
        return (
            bars.with_columns(pl.col("ts_event").dt.convert_time_zone(tz_str).alias("ts_local"))
            .with_columns(
                [
                    pl.col("ts_local").dt.date().alias("exp_date"),
                    (
                        pl.col("ts_local").dt.hour().cast(pl.Int32) * 60
                        + pl.col("ts_local").dt.minute().cast(pl.Int32)
                    ).alias("_minutes"),
                ]
            )
            .filter(pl.col("_minutes") <= pl.lit(close_minutes))
            .sort(["instrument_id", "ts_event"])
            .group_by(["instrument_id", "exp_date"])
            .agg(pl.col("close").last().alias("settlement_close"))
            .rename({"instrument_id": "underlying_id"})
            .select(["underlying_id", "exp_date", "settlement_close"])
        )

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

    # ------------------------------------------------------------------
    # Audit filter helpers
    # ------------------------------------------------------------------

    def _audit_table_exists(self) -> bool:
        """Return True if the option_audit table is present in the database."""
        try:
            return (
                self._con.execute(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_name = 'option_audit'"
                ).fetchone()[0]
                > 0
            )
        except Exception:
            return False

    def _flagged_instrument_ids(self, flag_codes: frozenset[str]) -> frozenset[int]:
        """
        Return the set of instrument_ids that have at least one row in option_audit
        matching any of the given flag codes.

        Returns an empty frozenset if option_audit does not exist or is empty.
        """
        if not flag_codes or not self._audit_table_exists():
            return frozenset()
        placeholders = ", ".join(f"'{c}'" for c in flag_codes)
        try:
            rows = self._con.execute(
                f"SELECT DISTINCT instrument_id FROM option_audit "
                f"WHERE flag_code IN ({placeholders})"
            ).fetchall()
            return frozenset(int(r[0]) for r in rows)
        except Exception:
            return frozenset()

    def greeks_for_all_legs(
        self,
        ts_event_underlying: list[tuple],
        leg_specs: list[dict],
        audit_filter_codes: frozenset[str] | None = None,
    ) -> pl.DataFrame:
        """
        Batched greeks lookup for all legs across all candidate timestamps.

        ts_event_underlying is a list of (ts_event, underlying_id) pairs — each
        candidate bar carries its own front-month underlying_id, so the query
        correctly filters options by the active futures contract at each point in
        time rather than using a single contract for the whole backtest.

        leg_specs is a list of dicts with keys: name, right (C/P), target_delta,
        target_dte, delta_tolerance, dte_tolerance. The tolerance values are
        per-leg, sourced from LegConfig.delta.tolerance / LegConfig.dte_tolerance.
        Returns one DataFrame for all legs tagged with a leg_name column; caller
        partitions by leg_name and picks the best match per ts_event.

        Uses a two-phase approach so DuckDB can apply idx_option_greeks_lookup on
        scalar underlying_id predicates, then the greeks result drives the
        option_bars join as the build side.
        """
        ts_events = [t for t, _ in ts_event_underlying]
        ts_min = min(ts_events)
        ts_max = max(ts_events)

        # Global dte/delta bounds across all legs.
        dte_lo = min(int(s["target_dte"]) - int(s["dte_tolerance"]) for s in leg_specs)
        dte_hi = max(int(s["target_dte"]) + int(s["dte_tolerance"]) for s in leg_specs)
        delta_lo = min(float(s["target_delta"]) - float(s["delta_tolerance"]) for s in leg_specs)
        delta_hi = max(float(s["target_delta"]) + float(s["delta_tolerance"]) for s in leg_specs)

        # Build per-underlying time ranges.
        uid_ts_range: dict[int, list] = {}
        for ts, uid in ts_event_underlying:
            if uid not in uid_ts_range:
                uid_ts_range[uid] = [ts, ts]
            else:
                if ts < uid_ts_range[uid][0]:
                    uid_ts_range[uid][0] = ts
                if ts > uid_ts_range[uid][1]:
                    uid_ts_range[uid][1] = ts

        # --- Phase 1: option_greeks only, per underlying_id ---
        # Scalar `og.underlying_id = ?` lets DuckDB use idx_option_greeks_lookup.
        # No Arrow-scan join here — a plain WHERE range scan is ~2.5× faster than
        # the _uid_ts hash-join approach because DuckDB doesn't need to build or
        # probe a large ts_event hash table.
        greeks_frames: list[pl.DataFrame] = []
        for uid, (uid_min, uid_max) in uid_ts_range.items():
            df = self._con.execute(
                """
                SELECT og.ts_event,
                       og.instrument_id,
                       CAST(og.underlying_id AS BIGINT) AS underlying_id,
                       og.dte, og.iv, og.delta, og.gamma, og.theta, og.vega
                FROM option_greeks og
                WHERE og.underlying_id = ?
                  AND og.dte          BETWEEN ? AND ?
                  AND og.delta        BETWEEN ? AND ?
                  AND og.ts_event    >= ?
                  AND og.ts_event    <= ?
                """,
                [uid, dte_lo, dte_hi, delta_lo, delta_hi, uid_min, uid_max],
            ).pl()
            if not df.is_empty():
                greeks_frames.append(df)

        if not greeks_frames:
            return pl.DataFrame()

        greeks_df = pl.concat(greeks_frames)
        if greeks_df.is_empty():
            return pl.DataFrame()

        # --- Audit filter: exclude instruments flagged in option_audit ---
        if audit_filter_codes:
            flagged = self._flagged_instrument_ids(audit_filter_codes)
            if flagged:
                greeks_df = greeks_df.filter(~pl.col("instrument_id").is_in(flagged))
            if greeks_df.is_empty():
                return pl.DataFrame()

        # --- Phase 2: option_bars join with greeks as the build side ---
        self._con.register("_greeks_df", greeks_df)
        try:
            combined = self._con.execute(
                """
                SELECT g.ts_event, g.instrument_id, g.underlying_id,
                       g.dte, g.iv, g.delta, g.gamma, g.theta, g.vega,
                       ob.strike_price, ob.expiration, ob."right", ob.multiplier,
                       ob.symbol, ob.close
                FROM _greeks_df g
                JOIN option_bars ob
                  ON ob.instrument_id = g.instrument_id
                 AND ob.ts_event      = g.ts_event
                 AND ob.close IS NOT NULL
                WHERE ob.ts_event >= ? AND ob.ts_event <= ?
                """,
                [ts_min, ts_max],
            ).pl()
        finally:
            self._con.unregister("_greeks_df")

        if combined.is_empty():
            return pl.DataFrame()

        # --- Phase 3: per-leg delta / right filter, tag with leg_name ---
        leg_results: list[pl.DataFrame] = []
        for spec in leg_specs:
            dte_lo_l = int(spec["target_dte"]) - int(spec["dte_tolerance"])
            dte_hi_l = int(spec["target_dte"]) + int(spec["dte_tolerance"])
            d_lo = float(spec["target_delta"]) - float(spec["delta_tolerance"])
            d_hi = float(spec["target_delta"]) + float(spec["delta_tolerance"])

            leg_df = combined.filter(
                (pl.col("right") == spec["right"])
                & (pl.col("dte") >= dte_lo_l)
                & (pl.col("dte") <= dte_hi_l)
                & (pl.col("delta") >= d_lo)
                & (pl.col("delta") <= d_hi)
            ).with_columns(pl.lit(spec["name"]).alias("leg_name"))
            if not leg_df.is_empty():
                leg_results.append(leg_df)

        if not leg_results:
            return pl.DataFrame()

        return pl.concat(leg_results).select(
            [
                "leg_name",
                "ts_event",
                "instrument_id",
                "underlying_id",
                "dte",
                "iv",
                "delta",
                "gamma",
                "theta",
                "vega",
                "strike_price",
                "expiration",
                "right",
                "multiplier",
                "symbol",
                "close",
            ]
        )

    def greeks_for_strike_legs(
        self,
        strike_targets: pl.DataFrame,
        *,
        strike_tolerance: float = 1.0,
        audit_filter_codes: frozenset[str] | None = None,
    ) -> pl.DataFrame:
        """
        For each (ts_event, leg_name) in strike_targets, find the option whose
        strike_price is within strike_tolerance of target_strike and whose
        expiration matches the reference leg's exact expiration date.

        strike_targets columns: ts_event, underlying_id, leg_name, right (C/P),
        target_strike, reference_expiration (DATE).  The underlying_id column
        scopes the search to the correct front-month futures contract at each bar.

        Returns the same column set as greeks_for_all_legs().

        Uses the same two-phase per-underlying approach as greeks_for_all_legs()
        to avoid Arrow-scan hash-join overhead. DTE bounds are derived from the
        (ts_event, reference_expiration) pairs in strike_targets so the greeks
        scan is as tight as greeks_for_all_legs().
        """
        ts_min = strike_targets["ts_event"].min()
        ts_max = strike_targets["ts_event"].max()

        # Single query: the strike_targets Arrow scan (~150K rows) acts as the
        # build side; its hash fits comfortably in L3 cache. DuckDB probes
        # option_greeks and option_bars with zone-map pruning on ts_event.
        self._con.register("_strike_targets", strike_targets)
        try:
            result = self._con.execute(
                """
                SELECT st.leg_name,
                       og.ts_event,
                       og.instrument_id,
                       CAST(og.underlying_id AS BIGINT) AS underlying_id,
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
                 AND ob.close IS NOT NULL
                WHERE og.ts_event BETWEEN ? AND ?
                """,
                [strike_tolerance, ts_min, ts_max],
            ).pl()
        finally:
            self._con.unregister("_strike_targets")

        if result.is_empty():
            return pl.DataFrame()

        if audit_filter_codes:
            flagged = self._flagged_instrument_ids(audit_filter_codes)
            if flagged:
                result = result.filter(~pl.col("instrument_id").is_in(flagged))
            if result.is_empty():
                return pl.DataFrame()

        return result.select(
            [
                "leg_name",
                "ts_event",
                "instrument_id",
                "underlying_id",
                "dte",
                "iv",
                "delta",
                "gamma",
                "theta",
                "vega",
                "strike_price",
                "expiration",
                "right",
                "multiplier",
                "symbol",
                "close",
            ]
        )

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

        Returns columns: date (pl.Date), underlying_id (pl.Int64), expiry (pl.Date),
        next_underlying_id (pl.Int64, nullable).  The extra columns enable the entry
        scanner to include the next contract's options when target DTE extends beyond
        the front-month's remaining life.
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
                        "expiry": pl.Series([], dtype=pl.Date),
                        "next_underlying_id": pl.Series([], dtype=pl.Int64),
                    }
                )
            uid = int(row[0])
            dates = [
                start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)
            ]
            n = len(dates)
            return pl.DataFrame(
                {
                    "date": pl.Series(dates, dtype=pl.Date),
                    "underlying_id": pl.Series([uid] * n, dtype=pl.Int64),
                    "expiry": pl.Series([None] * n, dtype=pl.Date),
                    "next_underlying_id": pl.Series([None] * n, dtype=pl.Int64),
                }
            )

        expirations = list(
            zip(
                futures_df["instrument_id"].to_list(),
                futures_df["expiration"].to_list(),
                strict=False,
            )
        )

        dates: list[date] = []
        underlying_ids: list[int] = []
        expiry_dates: list[date | None] = []
        next_underlying_ids: list[int | None] = []
        d = start_date
        while d <= end_date:
            roll_cutoff = d + timedelta(days=roll_days)
            # Front month: earliest expiration that hasn't yet hit its roll date
            front_idx = None
            for i, (_iid, exp) in enumerate(expirations):
                if exp >= roll_cutoff:
                    front_idx = i
                    break
            if front_idx is None:
                # All contracts have rolled past — use the last one
                front_idx = len(expirations) - 1
            front_id = expirations[front_idx][0]
            front_exp = expirations[front_idx][1]
            next_id = expirations[front_idx + 1][0] if front_idx + 1 < len(expirations) else None
            dates.append(d)
            underlying_ids.append(front_id)
            expiry_dates.append(front_exp)
            next_underlying_ids.append(next_id)
            d += timedelta(days=1)

        return pl.DataFrame(
            {
                "date": pl.Series(dates, dtype=pl.Date),
                "underlying_id": pl.Series(underlying_ids, dtype=pl.Int64),
                "expiry": pl.Series(expiry_dates, dtype=pl.Date),
                "next_underlying_id": pl.Series(next_underlying_ids, dtype=pl.Int64),
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
