"""
Phase 1 — Implied volatility distribution and sentinel detection.

Flags:
    IV_NAN      (soft) — isnan(iv): Black-76 IV computation failed (typically T=0 near expiry).
    IV_SENTINEL (soft) — iv = 10.0: greeks engine hit its bisection upper bound; option is
                          so deep ITM that IV is undefined.
    IV_HIGH     (soft) — iv > 2.0 and finite and not sentinel: IV above 200%, flagged for
                          review (may be legitimate during stress events).

All three checks run in a single SQL pass over option_greeks.
"""

from __future__ import annotations

import duckdb
import polars as pl

from btkit.audit.schema import empty_audit_df


def run(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Return one audit row per (instrument_id, ts_event) matching any IV flag."""
    result = con.execute("""
        SELECT
            CAST(instrument_id AS BIGINT) AS instrument_id,
            ts_event,
            CASE
                WHEN isnan(iv)                                    THEN 'IV_NAN'
                WHEN iv = 10.0                                    THEN 'IV_SENTINEL'
                WHEN iv > 2.0 AND NOT isnan(iv) AND iv != 10.0   THEN 'IV_HIGH'
            END                                                   AS flag_code,
            'soft'                                                AS flag_severity,
            iv                                                    AS flag_value,
            CAST(CASE
                WHEN isnan(iv)                                    THEN NULL
                WHEN iv = 10.0                                    THEN 10.0
                ELSE 2.0
            END AS DOUBLE)                                        AS threshold
        FROM option_greeks
        WHERE isnan(iv)
           OR iv = 10.0
           OR (iv > 2.0 AND NOT isnan(iv) AND iv != 10.0)
    """).pl()

    if result.is_empty():
        return empty_audit_df()

    return result.with_columns(pl.col("ts_event").dt.replace_time_zone("UTC"))
