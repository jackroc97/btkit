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

    ID_DAY_COLLISION — one instrument_id carrying more than one contract definition
                       (expiration, strike, right) within a single UTC day.

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

    # ID_DAY_COLLISION -------------------------------------------------------------
    # Stored as one row per (instrument_id, colliding day), anchored at that day's
    # earliest bar — distinct days therefore produce distinct primary keys. The
    # definition key must be compared as text: comparing the tuple directly makes
    # count(DISTINCT ...) reject the row in DuckDB.
    id_day = con.execute(
        """
        SELECT CAST(instrument_id AS BIGINT) AS instrument_id,
               min(ts_event) AS ts_event,
               CAST(count(DISTINCT concat(expiration, '|', strike_price, '|', "right"))
                    AS DOUBLE) AS flag_value
        FROM option_bars
        GROUP BY instrument_id, CAST(ts_event AT TIME ZONE 'UTC' AS DATE)
        HAVING count(DISTINCT concat(expiration, '|', strike_price, '|', "right")) > 1
        """
    ).pl()

    if not id_day.is_empty():
        results.append(_tag(id_day, "ID_DAY_COLLISION", 1.0))

    if not results:
        return empty_audit_df()

    combined = pl.concat(results)
    # convert_time_zone (not replace_time_zone) — DuckDB hands back TIMESTAMPTZ in the
    # session timezone, so the instant must be converted, not relabelled.
    return combined.with_columns(pl.col("ts_event").dt.convert_time_zone("UTC"))
