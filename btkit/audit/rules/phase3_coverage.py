"""
Phase 3 — Bar coverage checks.

Flags:
    BARS_TRUNCATED (hard) — (expiration − last_bar_date) / (expiration − first_bar_date) > 0.15.
        Bars end significantly before expiry: the engine will treat the last bar date as
        expiry and close the position prematurely, potentially at an incorrect price.

    BARS_SPARSE (soft) — bars per active trading day < 10.
        The option has very thin intraday coverage; greeks / fills may be based on stale data.

    NO_EXPIRY_BARS (soft) — the instrument has zero bar records on its expiration date.
        Distinct from BARS_TRUNCATED: a 45DTE option whose data runs through day 44 but
        has nothing on day 45 would not trip the 15% proportional threshold but would
        trip this flag. Relevant when the position survives to expiry.

All three flags are stored at (instrument_id, min_ts_event) — one row per instrument per
flag — because these are structural defects of the instrument as a whole, not of individual
bars. The entry-time filter uses instrument_id NOT IN (...) rather than a row-level join,
so a single sentinel row per instrument is sufficient.
"""

from __future__ import annotations

import duckdb
import polars as pl

from btkit.audit.schema import empty_audit_df

_TRUNCATED_THRESHOLD = 0.15
_SPARSE_THRESHOLD = 10.0


def run(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Return one audit row per (instrument_id, flag_code) for coverage violations."""
    results: list[pl.DataFrame] = []

    # BARS_TRUNCATED ----------------------------------------------------------------
    truncated = con.execute(
        """
        SELECT
            CAST(instrument_id AS BIGINT)                            AS instrument_id,
            MIN(ts_event)                                            AS min_ts,
            DATEDIFF('day', MAX(ts_event::date), expiration)        AS days_remaining,
            DATEDIFF('day', MIN(ts_event::date), expiration)        AS observable_life
        FROM option_bars
        GROUP BY instrument_id, expiration
        HAVING observable_life > 0
           AND days_remaining::DOUBLE / observable_life::DOUBLE > ?
        """,
        [_TRUNCATED_THRESHOLD],
    ).pl()

    if not truncated.is_empty():
        results.append(
            truncated.select(
                [
                    pl.col("instrument_id"),
                    pl.col("min_ts").alias("ts_event"),
                    pl.lit("BARS_TRUNCATED").alias("flag_code"),
                    pl.lit("hard").alias("flag_severity"),
                    (
                        pl.col("days_remaining").cast(pl.Float64)
                        / pl.col("observable_life").cast(pl.Float64)
                    ).alias("flag_value"),
                    pl.lit(_TRUNCATED_THRESHOLD).alias("threshold"),
                ]
            )
        )

    # BARS_SPARSE -------------------------------------------------------------------
    sparse = con.execute(
        """
        SELECT
            CAST(instrument_id AS BIGINT)                              AS instrument_id,
            MIN(ts_event)                                              AS min_ts,
            COUNT(*)::DOUBLE / COUNT(DISTINCT ts_event::date)::DOUBLE AS bars_per_day
        FROM option_bars
        GROUP BY instrument_id
        HAVING COUNT(DISTINCT ts_event::date) > 0
           AND COUNT(*)::DOUBLE / COUNT(DISTINCT ts_event::date)::DOUBLE < ?
        """,
        [_SPARSE_THRESHOLD],
    ).pl()

    if not sparse.is_empty():
        results.append(
            sparse.select(
                [
                    pl.col("instrument_id"),
                    pl.col("min_ts").alias("ts_event"),
                    pl.lit("BARS_SPARSE").alias("flag_code"),
                    pl.lit("soft").alias("flag_severity"),
                    pl.col("bars_per_day").alias("flag_value"),
                    pl.lit(_SPARSE_THRESHOLD).alias("threshold"),
                ]
            )
        )

    # NO_EXPIRY_BARS ----------------------------------------------------------------
    no_expiry = con.execute(
        """
        SELECT
            CAST(instrument_id AS BIGINT)                            AS instrument_id,
            MIN(ts_event)                                            AS min_ts,
            DATEDIFF('day', MAX(ts_event::date), expiration)::DOUBLE AS days_before_expiry
        FROM option_bars
        GROUP BY instrument_id, expiration
        HAVING COUNT(*) FILTER (WHERE ts_event::date = expiration) = 0
        """
    ).pl()

    if not no_expiry.is_empty():
        results.append(
            no_expiry.select(
                [
                    pl.col("instrument_id"),
                    pl.col("min_ts").alias("ts_event"),
                    pl.lit("NO_EXPIRY_BARS").alias("flag_code"),
                    pl.lit("soft").alias("flag_severity"),
                    pl.col("days_before_expiry").alias("flag_value"),
                    pl.lit(0.0).alias("threshold"),
                ]
            )
        )

    if not results:
        return empty_audit_df()

    combined = pl.concat(results)
    return combined.with_columns(pl.col("ts_event").dt.replace_time_zone("UTC"))
