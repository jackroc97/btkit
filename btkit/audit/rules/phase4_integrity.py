"""
Phase 4 — Basic integrity checks.

Flags (all hard):
    NEGATIVE_CLOSE        — option_bars.close < 0: impossible for options.
    NEGATIVE_DTE          — option_greeks.dte < 0: greeks computed after expiry.
    ZOMBIE_BAR            — option_bars.expiration < ts_event::date: bar dated after expiry.
    DELTA_SIGN_ERROR      — put delta > 0 or call delta < 0 (where delta is finite).
    DELTA_MAGNITUDE_ERROR — abs(delta) > 1.0 (where delta is finite).
"""

from __future__ import annotations

import duckdb
import polars as pl

from btkit.audit.schema import empty_audit_df


def run(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Return one audit row per flagged (instrument_id, ts_event) for all integrity checks."""
    results: list[pl.DataFrame] = []

    # NEGATIVE_CLOSE ---------------------------------------------------------------
    neg_close = con.execute(
        """
        SELECT CAST(instrument_id AS BIGINT) AS instrument_id,
               ts_event,
               close AS flag_value
        FROM option_bars
        WHERE close < 0
        """
    ).pl()

    if not neg_close.is_empty():
        results.append(
            neg_close.select([
                pl.col("instrument_id"),
                pl.col("ts_event"),
                pl.lit("NEGATIVE_CLOSE").alias("flag_code"),
                pl.lit("hard").alias("flag_severity"),
                pl.col("flag_value"),
                pl.lit(0.0).alias("threshold"),
            ])
        )

    # NEGATIVE_DTE -----------------------------------------------------------------
    neg_dte = con.execute(
        """
        SELECT CAST(instrument_id AS BIGINT) AS instrument_id,
               ts_event,
               CAST(dte AS DOUBLE) AS flag_value
        FROM option_greeks
        WHERE dte < 0
        """
    ).pl()

    if not neg_dte.is_empty():
        results.append(
            neg_dte.select([
                pl.col("instrument_id"),
                pl.col("ts_event"),
                pl.lit("NEGATIVE_DTE").alias("flag_code"),
                pl.lit("hard").alias("flag_severity"),
                pl.col("flag_value"),
                pl.lit(0.0).alias("threshold"),
            ])
        )

    # ZOMBIE_BAR -------------------------------------------------------------------
    zombie = con.execute(
        """
        SELECT CAST(instrument_id AS BIGINT) AS instrument_id,
               ts_event,
               CAST(DATEDIFF('day', expiration, ts_event::date) AS DOUBLE) AS flag_value
        FROM option_bars
        WHERE expiration < ts_event::date
        """
    ).pl()

    if not zombie.is_empty():
        results.append(
            zombie.select([
                pl.col("instrument_id"),
                pl.col("ts_event"),
                pl.lit("ZOMBIE_BAR").alias("flag_code"),
                pl.lit("hard").alias("flag_severity"),
                pl.col("flag_value"),
                pl.lit(0.0).alias("threshold"),
            ])
        )

    # DELTA_SIGN_ERROR and DELTA_MAGNITUDE_ERROR ------------------------------------
    delta_issues = con.execute(
        """
        SELECT
            CAST(og.instrument_id AS BIGINT) AS instrument_id,
            og.ts_event,
            og.delta,
            ob."right"
        FROM option_greeks og
        JOIN option_bars ob
          ON ob.instrument_id = og.instrument_id
         AND ob.ts_event      = og.ts_event
        WHERE og.delta IS NOT NULL
          AND NOT isnan(og.delta)
          AND NOT isinf(og.delta)
          AND (
              (ob."right" = 'P' AND og.delta > 0)
              OR (ob."right" = 'C' AND og.delta < 0)
              OR ABS(og.delta) > 1.0
          )
        """
    ).pl()

    if not delta_issues.is_empty():
        sign_mask = (
            ((delta_issues["right"] == "P") & (delta_issues["delta"] > 0))
            | ((delta_issues["right"] == "C") & (delta_issues["delta"] < 0))
        )
        sign_df = delta_issues.filter(sign_mask)
        if not sign_df.is_empty():
            results.append(
                sign_df.select([
                    pl.col("instrument_id"),
                    pl.col("ts_event"),
                    pl.lit("DELTA_SIGN_ERROR").alias("flag_code"),
                    pl.lit("hard").alias("flag_severity"),
                    pl.col("delta").alias("flag_value"),
                    pl.lit(0.0).alias("threshold"),
                ])
            )

        mag_df = delta_issues.filter(pl.col("delta").abs() > 1.0)
        if not mag_df.is_empty():
            results.append(
                mag_df.select([
                    pl.col("instrument_id"),
                    pl.col("ts_event"),
                    pl.lit("DELTA_MAGNITUDE_ERROR").alias("flag_code"),
                    pl.lit("hard").alias("flag_severity"),
                    pl.col("delta").abs().alias("flag_value"),
                    pl.lit(1.0).alias("threshold"),
                ])
            )

    if not results:
        return empty_audit_df()

    combined = pl.concat(results)
    return combined.with_columns(pl.col("ts_event").dt.replace_time_zone("UTC"))
