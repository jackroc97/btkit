"""
Phase 2 — Black-76 delta consistency.

Flag:
    DELTA_INCONSISTENT (soft) — |reported_delta - theoretical_delta| > 0.10, where
        theoretical_delta is computed via the Black-76 model using the reported IV
        as the volatility input.

Performance design (mirrors GreeksCalculator):
    1. Materialise the expensive 3-way join once into a TEMP TABLE so the join
       cost is paid once, not once-per-date.
    2. Stream per-date batches from the cheap temp table.
    3. Pipeline numba _greeks() via ThreadPoolExecutor — the @njit(nogil=True)
       kernel releases the GIL, so threads scale to the physical core count.

progress_cb: optional callable(done: int, total: int) invoked after each date
    batch completes.  The runner uses this to drive an inline progress line.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

import duckdb
import numpy as np
import polars as pl

from btkit.audit.schema import empty_audit_df
from btkit.pipeline.greeks import _greeks

THRESHOLD = 0.10
_RISK_FREE_RATE = 0.01
_WORKERS = min(os.cpu_count() or 4, 8)


def run(
    con: duckdb.DuckDBPyConnection,
    progress_cb: Callable[[int, int], None] | None = None,
) -> pl.DataFrame:
    """
    Compute theoretical Black-76 delta for every eligible row, flag rows where
    |reported - theoretical| > THRESHOLD.

    progress_cb(done, total) is called after each date batch drains so callers
    can render an inline progress line.
    """
    # ------------------------------------------------------------------
    # Step 1 — materialise the 3-way join ONCE into a temp table.
    # Paying the join cost once instead of once-per-date is the dominant
    # performance win on a 49M-row greeks table.
    # ------------------------------------------------------------------
    con.execute("DROP TABLE IF EXISTS _phase2_pending")
    con.execute(
        """
        CREATE TEMP TABLE _phase2_pending AS
        SELECT
            CAST(og.instrument_id AS BIGINT) AS instrument_id,
            og.ts_event,
            og.delta        AS reported_delta,
            og.iv,
            og.T,
            ob."right",
            ob.strike_price,
            ub.close        AS underlying_close
        FROM option_greeks og
        JOIN option_bars ob
          ON ob.instrument_id = og.instrument_id
         AND ob.ts_event      = og.ts_event
        JOIN underlying_bars ub
          ON ub.instrument_id = og.underlying_id
         AND ub.ts_event      = og.ts_event
        WHERE og.delta      IS NOT NULL
          AND NOT isnan(og.delta)
          AND NOT isinf(og.delta)
          AND og.iv         IS NOT NULL
          AND NOT isnan(og.iv)
          AND NOT isinf(og.iv)
          AND og.iv > 0
          AND og.T > 0
          AND ub.close > 0
          AND ob.strike_price > 0
        ORDER BY DATE(og.ts_event), og.instrument_id
        """
    )

    total_rows = con.execute("SELECT COUNT(*) FROM _phase2_pending").fetchone()[0]
    if total_rows == 0:
        con.execute("DROP TABLE IF EXISTS _phase2_pending")
        return empty_audit_df()

    dates = [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT DATE(ts_event) FROM _phase2_pending ORDER BY 1"
        ).fetchall()
    ]
    n_dates = len(dates)

    # ------------------------------------------------------------------
    # Step 2/3 — stream per-date batches, pipeline numba computation.
    # ------------------------------------------------------------------
    results: list[pl.DataFrame] = []
    completed = 0
    pending: list[tuple[object, Future]] = []

    def _compute_batch(df: pl.DataFrame) -> pl.DataFrame | None:
        """Run on a thread pool worker — numba releases GIL so threads scale."""
        F = df["underlying_close"].to_numpy()
        K = df["strike_price"].to_numpy()
        T = df["T"].to_numpy()
        iv = df["iv"].to_numpy()
        is_call = (df["right"] == "C").cast(pl.Int8).to_numpy().astype(np.int64)
        r = np.full(len(df), _RISK_FREE_RATE)

        theoretical_delta, _, _, _ = _greeks(F, K, T, r, iv, is_call)

        reported = df["reported_delta"].to_numpy()
        diff = np.abs(reported - theoretical_delta)
        mask = (diff > THRESHOLD) & np.isfinite(theoretical_delta)

        if not mask.any():
            return None

        flagged = df.filter(pl.Series(mask))
        n = int(mask.sum())
        return pl.DataFrame(
            {
                "instrument_id": flagged["instrument_id"],
                "ts_event": flagged["ts_event"],
                "flag_code": pl.Series(["DELTA_INCONSISTENT"] * n, dtype=pl.Utf8),
                "flag_severity": pl.Series(["soft"] * n, dtype=pl.Utf8),
                "flag_value": pl.Series(diff[mask], dtype=pl.Float64),
                "threshold": pl.Series([THRESHOLD] * n, dtype=pl.Float64),
            }
        )

    def _drain_one() -> None:
        nonlocal completed
        _, fut = pending.pop(0)
        result = fut.result()
        if result is not None:
            results.append(result)
        completed += 1
        if progress_cb is not None:
            progress_cb(completed, n_dates)

    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        for d in dates:
            batch = con.execute(
                "SELECT * FROM _phase2_pending WHERE DATE(ts_event) = ? ORDER BY instrument_id",
                [d],
            ).pl()

            if not batch.is_empty():
                pending.append((d, pool.submit(_compute_batch, batch)))
            else:
                # Empty date (all rows filtered): count it immediately.
                completed += 1
                if progress_cb is not None:
                    progress_cb(completed, n_dates)

            # Drain the oldest future once the pipeline is saturated.
            if len(pending) >= _WORKERS * 2:
                _drain_one()

        while pending:
            _drain_one()

    con.execute("DROP TABLE IF EXISTS _phase2_pending")

    if not results:
        return empty_audit_df()

    combined = pl.concat(results)
    return combined.with_columns(pl.col("ts_event").dt.replace_time_zone("UTC"))
