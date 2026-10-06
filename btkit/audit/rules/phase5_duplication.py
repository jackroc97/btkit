"""
Phase 5 — Row multiplicity and instrument identity.

Flags (all hard):
    DUPLICATE_BAR    — more than one option_bars row for one (instrument_id, ts_event).
                       Duplicated bars inflate the exit-scanner's rolling volume sum, so
                       the exit.liquidity volume gate passes on roughly double the real
                       volume, and they multiply rows through every per-leg join.

    DUPLICATE_GREEKS — more than one option_greeks row for one (instrument_id, ts_event).
                       Normally a consequence of DUPLICATE_BAR, since greeks are computed
                       per option bar.

    ID_DAY_COLLISION — one instrument_id carrying two genuinely DIFFERENT contracts
                       within a single UTC day: they differ in symbol, strike or
                       right, so at least one set of bars is misattributed.

    EXPIRY_RESCHEDULED (soft) — one instrument_id/day where the same contract
                       (identical symbol, strike and right) appears under more than
                       one expiration. This is an exchange reschedule, not a routing
                       error: when CME moves an expiration it republishes the
                       definition with the new date, and bars before and after the
                       announcement legitimately carry different expirations. It is
                       still worth knowing, because the superseded expiration gives a
                       wrong DTE. On the rebuilt ES database all 68 instances fall on
                       2018-12-03 and 2024-12-30 — the first being the US market
                       holiday for the national day of mourning, which moved
                       expirations from 2018-12-05 to 2018-12-04.

Why ID_DAY_COLLISION is scoped to a day
---------------------------------------
Databento assigns instrument_id from the publisher's numeric id (for GLBX.MDP3, tag 48
SecurityID) and only guarantees it is unique *within a given day*. An id legitimately
refers to different contracts in different periods, so carrying several definitions over
time is normal and is NOT flagged — the vast majority of ids do it. Only a collision
inside one day violates the guarantee, and it means some bars on that day are attributed
to the wrong contract.

See docs/data_duplication_fix.md for the root-cause analysis.
"""

from __future__ import annotations

import duckdb
import polars as pl

from btkit.audit.schema import empty_audit_df


def _tag(df: pl.DataFrame, code: str, threshold: float) -> pl.DataFrame:
    """Shape a (instrument_id, ts_event, flag_value) frame into the audit schema."""
    return df.select(
        [
            pl.col("instrument_id"),
            pl.col("ts_event"),
            pl.lit(code).alias("flag_code"),
            pl.lit("hard").alias("flag_severity"),
            pl.col("flag_value"),
            pl.lit(threshold).alias("threshold"),
        ]
    )


def run(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Return one audit row per flagged (instrument_id, ts_event) for all phase-5 checks."""
    results: list[pl.DataFrame] = []

    # DUPLICATE_BAR ----------------------------------------------------------------
    # One row per duplicated bar, so the flag count measures the defect directly and
    # can be used to verify a deduplication rebuild.
    dup_bars = con.execute(
        """
        SELECT CAST(instrument_id AS BIGINT) AS instrument_id,
               ts_event,
               CAST(count(*) AS DOUBLE) AS flag_value
        FROM option_bars
        GROUP BY 1, 2
        HAVING count(*) > 1
        """
    ).pl()

    if not dup_bars.is_empty():
        results.append(_tag(dup_bars, "DUPLICATE_BAR", 1.0))

    # DUPLICATE_GREEKS -------------------------------------------------------------
    dup_greeks = con.execute(
        """
        SELECT CAST(instrument_id AS BIGINT) AS instrument_id,
               ts_event,
               CAST(count(*) AS DOUBLE) AS flag_value
        FROM option_greeks
        GROUP BY 1, 2
        HAVING count(*) > 1
        """
    ).pl()

    if not dup_greeks.is_empty():
        results.append(_tag(dup_greeks, "DUPLICATE_GREEKS", 1.0))

    # ID_DAY_COLLISION / EXPIRY_RESCHEDULED ----------------------------------------
    # Stored as one row per (instrument_id, colliding day), anchored at that day's
    # earliest bar — distinct days therefore produce distinct primary keys.
    #
    # The two are separated by whether the colliding definitions describe different
    # instruments or the same one. Counting distinct (expiration, strike, right)
    # alone cannot tell them apart, and conflating them made a clean rebuild look
    # broken: 68 flags that were all exchange reschedules.
    id_day = con.execute(
        """
        SELECT CAST(instrument_id AS BIGINT) AS instrument_id,
               min(ts_event) AS ts_event,
               count(DISTINCT symbol) AS n_symbol,
               count(DISTINCT strike_price) AS n_strike,
               count(DISTINCT "right") AS n_right,
               CAST(count(DISTINCT expiration) AS DOUBLE) AS n_expiration
        FROM option_bars
        GROUP BY instrument_id, CAST(ts_event AT TIME ZONE 'UTC' AS DATE)
        HAVING count(DISTINCT concat(expiration, '|', strike_price, '|', "right")) > 1
        """
    ).pl()

    if not id_day.is_empty():
        differs = (pl.col("n_symbol") > 1) | (pl.col("n_strike") > 1) | (pl.col("n_right") > 1)
        collisions = id_day.filter(differs).with_columns(
            pl.max_horizontal("n_symbol", "n_strike", "n_right")
            .cast(pl.Float64)
            .alias("flag_value")
        )
        if not collisions.is_empty():
            results.append(_tag(collisions, "ID_DAY_COLLISION", 1.0))

        reschedules = id_day.filter(~differs).rename({"n_expiration": "flag_value"})
        if not reschedules.is_empty():
            results.append(
                _tag(reschedules, "EXPIRY_RESCHEDULED", 1.0).with_columns(
                    pl.lit("soft").alias("flag_severity")
                )
            )

    if not results:
        return empty_audit_df()

    combined = pl.concat(results)
    # convert_time_zone (not replace_time_zone) — DuckDB hands back TIMESTAMPTZ in the
    # session timezone, so the instant must be converted, not relabelled.
    return combined.with_columns(pl.col("ts_event").dt.convert_time_zone("UTC"))
