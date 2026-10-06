"""
GreeksCalculator — batch Black-76 IV and Greeks computation using numba JIT.

Reads option_bars and writes iv, delta, gamma, theta, vega to option_greeks. The
spot price F is taken from the underlying future via a backward ASOF join (nearest
underlying bar at or before each option bar, within a staleness tolerance), so
options that traded a minute the future did not print still get greeks.

Takes a writable DuckDB connection so it can both read OHLCV tables and write
the option_greeks table within the same build transaction.

The numba-compiled _black76_* functions operate on numpy arrays and are called
per batch to avoid per-row Python overhead.

Black-76 model (options on futures):
    d1  = (ln(F/K) + 0.5*σ²*T) / (σ*√T)
    d2  = d1 - σ*√T
    df  = exp(-r*T)          # discount factor
    Call = df * (F*N(d1) - K*N(d2))
    Put  = df * (K*N(-d2) - F*N(-d1))

Greeks (annualised inputs, theta in per-day):
    delta (call) = df * N(d1)
    delta (put)  = df * (N(d1) - 1)
    gamma        = df * N'(d1) / (F * σ * √T)
    vega         = df * F * N'(d1) * √T  (per unit IV — divide by 100 for per-1%)
    theta (call) = df * (-F*N'(d1)*σ/(2√T) + r*F*N(d1) - r*K*N(d2)) / 365
    theta (put)  = df * (-F*N'(d1)*σ/(2√T) - r*F*N(-d1) + r*K*N(-d2)) / 365
"""

from __future__ import annotations

import math
import os
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import timedelta

import duckdb
import numpy as np
import polars as pl
from numba import njit

# Number of parallel workers for day-level Greeks computation.
# numba JIT releases the GIL, so threads scale well up to the physical core count.
_GREEKS_WORKERS = min(os.cpu_count() or 4, 8)

# Chunks accumulated before a single bulk INSERT into option_greeks.
# Each INSERT carries ~0.75 s of fixed DuckDB overhead regardless of size, so
# batching keeps variable cost dominant.
_BATCH_DAYS = 20

# Rows per streamed chunk out of _greek_pending. Bounds peak memory to roughly
# _GREEKS_CHUNK_ROWS × _GREEKS_WORKERS × 2 rows in flight (~250k rows at the
# default, a few tens of MB) while being large enough that per-chunk overhead and
# the numba call are amortised. Replaces the previous per-trading-day batching,
# which required one full scan of the pending table per day.
_GREEKS_CHUNK_ROWS = 250_000

# ---------------------------------------------------------------------------
# numba-compiled Black-76 kernels
# ---------------------------------------------------------------------------


@njit(cache=True)
def _norm_cdf(x: float) -> float:
    """Standard normal CDF via complementary error function."""
    return 0.5 * math.erfc(-x * 0.7071067811865476)  # 1/sqrt(2)


@njit(cache=True)
def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) * 0.3989422804014327  # 1/sqrt(2*pi)


@njit(cache=True)
def _black76_price(F: float, K: float, T: float, r: float, sigma: float, is_call: int) -> float:
    """Scalar Black-76 option price. is_call=1 for calls, 0 for puts."""
    if T <= 0.0 or sigma <= 0.0 or F <= 0.0 or K <= 0.0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    df = math.exp(-r * T)
    if is_call:
        return df * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
    else:
        return df * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))


@njit(cache=True, nogil=True)
def _implied_vol(
    F: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    market_price: np.ndarray,
    is_call: np.ndarray,  # 1 = call, 0 = put
) -> np.ndarray:
    """
    Vectorised implied volatility via bisection.
    Returns NaN for rows where IV cannot be found (e.g. zero-price options).
    """
    n = len(F)
    iv = np.empty(n, dtype=np.float64)
    for i in range(n):
        p = market_price[i]
        if p <= 0.0 or T[i] <= 0.0 or F[i] <= 0.0:
            iv[i] = np.nan
            continue
        lo, hi = 1e-4, 10.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            val = _black76_price(F[i], K[i], T[i], r[i], mid, is_call[i])
            if abs(val - p) < 1e-6:
                break
            if val < p:
                lo = mid
            else:
                hi = mid
        iv[i] = 0.5 * (lo + hi)
    return iv


@njit(cache=True, nogil=True)
def _greeks(
    F: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    is_call: np.ndarray,  # 1 = call, 0 = put
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised Black-76 Greeks.
    Returns (delta, gamma, theta, vega) per row.
    theta is in per-calendar-day units.
    vega is per unit IV (divide by 100 for per-1%).
    """
    n = len(F)
    delta = np.empty(n, dtype=np.float64)
    gamma = np.empty(n, dtype=np.float64)
    theta = np.empty(n, dtype=np.float64)
    vega = np.empty(n, dtype=np.float64)

    for i in range(n):
        if sigma[i] != sigma[i] or T[i] <= 0.0 or F[i] <= 0.0 or sigma[i] <= 0.0:
            delta[i] = np.nan
            gamma[i] = np.nan
            theta[i] = np.nan
            vega[i] = np.nan
            continue

        sqrtT = math.sqrt(T[i])
        d1 = (math.log(F[i] / K[i]) + 0.5 * sigma[i] * sigma[i] * T[i]) / (sigma[i] * sqrtT)
        d2 = d1 - sigma[i] * sqrtT
        df = math.exp(-r[i] * T[i])
        nd1 = _norm_cdf(d1)
        npd1 = _norm_pdf(d1)

        if is_call[i]:
            delta[i] = df * nd1
            theta_raw = df * (
                -F[i] * npd1 * sigma[i] / (2.0 * sqrtT)
                + r[i] * F[i] * nd1
                - r[i] * K[i] * _norm_cdf(d2)
            )
        else:
            delta[i] = df * (nd1 - 1.0)
            theta_raw = df * (
                -F[i] * npd1 * sigma[i] / (2.0 * sqrtT)
                - r[i] * F[i] * _norm_cdf(-d1)
                + r[i] * K[i] * _norm_cdf(-d2)
            )

        gamma[i] = df * npd1 / (F[i] * sigma[i] * sqrtT)
        theta[i] = theta_raw / 365.0
        vega[i] = df * F[i] * npd1 * sqrtT

    return delta, gamma, theta, vega


# ---------------------------------------------------------------------------
# GreeksCalculator
# ---------------------------------------------------------------------------


class GreeksCalculator:
    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        risk_free_rate: float = 0.01,
        batch_size: int = 50_000,
        underlying_max_staleness_minutes: int = 15,
    ) -> None:
        self.con = con
        self.risk_free_rate = risk_free_rate
        self.batch_size = batch_size
        # Max age of the nearest-prior underlying bar used as the spot price F for
        # an option bar. The underlying future does not print every minute the
        # option does; an ASOF join within this window supplies F from the last
        # underlying bar instead of dropping the option. 0 = require an exact
        # same-minute underlying bar (legacy behaviour).
        self.underlying_max_staleness_minutes = int(underlying_max_staleness_minutes)

    def run(
        self,
        skip_existing: bool = False,
        fill_range: tuple | None = None,
        instruments: list[int] | None = None,
    ) -> None:
        """
        Process all rows in option_bars one trading day at a time.

        Strategy:
          1. Materialize all pending rows (not yet in option_greeks when
             skip_existing=True) into a TEMP TABLE in a single pass — one
             NOT EXISTS scan instead of one per batch.
          2. Collect the distinct trading dates from that table.
          3. For each date, fetch rows with a simple equality filter,
             compute Greeks, and INSERT. O(n) total vs the previous O(n²)
             OFFSET approach.

        When skip_existing=False every option_bars row is processed.

        Scoping (interior "fill-the-gaps" builds): fill_range=(start_dt, end_dt)
        limits the materialisation to option bars in that window, and instruments
        (a list of underlying_ids) narrows it further. Greeks are per-bar with no
        trailing dependency, so scoping to the back-filled window is exact — and it
        bounds the NOT EXISTS scan to that window instead of the whole DB.
        """
        new_only_filter = (
            """
            AND NOT EXISTS (
                SELECT 1 FROM option_greeks og
                WHERE og.ts_event = ob.ts_event AND og.instrument_id = ob.instrument_id
            )
        """
            if skip_existing
            else ""
        )

        scope_filter = ""
        scope_params: list = []
        if fill_range is not None:
            scope_filter += "\n              AND ob.ts_event >= ? AND ob.ts_event <= ?"
            scope_params.extend([fill_range[0], fill_range[1]])
        if instruments:
            id_list = ",".join(str(int(i)) for i in instruments)
            scope_filter += f"\n              AND ob.underlying_id IN ({id_list})"

        # Step 1: Materialise pending rows once (expensive NOT EXISTS runs here,
        # not once per batch).
        #
        # The spot price F comes from the underlying future via a backward ASOF
        # join: for each option bar, the nearest underlying bar at or before the
        # option's ts_event. The future does not print every minute the option
        # does, so an exact-equality join silently drops those options and leaves
        # them with no greeks. The staleness filter bounds how old the borrowed
        # underlying bar may be (0 minutes ⇒ exact same-minute match, the legacy
        # behaviour); a small window keeps the match within the same session
        # because the only intraday gaps larger than it are hours-long.
        #
        # Duplicate option bars would compute and insert the same greeks twice:
        # NOT EXISTS only excludes rows already in option_greeks, and neither copy is
        # there yet. Ingest no longer admits duplicates, so the usual case needs no
        # dedup — and paying for one anyway is not an option at this scale. A
        # QUALIFY ROW_NUMBER() over the pending set is a window function over ~70M
        # rows, which cannot stream and exhausted a 12.7 GiB memory limit on the full
        # ES rebuild. So probe for duplicates first (an indexed early-out that stops
        # at the first one) and only then pay for the deduplicating plan.
        stale_minutes = self.underlying_max_staleness_minutes
        dedup_clause = ""
        if self._has_duplicate_bars():
            print("[greeks] duplicate option bars present — deduplicating pending rows")
            dedup_clause = """
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY ob.instrument_id, ob.ts_event ORDER BY ub.ts_event DESC
            ) = 1"""

        self.con.execute("DROP TABLE IF EXISTS _greek_pending")
        self.con.execute(
            f"""
            CREATE TEMP TABLE _greek_pending AS
            SELECT
                ob.ts_event,
                ob.instrument_id,
                ob.contract_id,
                ob.underlying_id,
                ob.expiration,
                ob.strike_price,
                ob."right",
                ob.close        AS option_close,
                ub.close        AS underlying_close,
                ub.ts_event     AS underlying_ts_event
            FROM option_bars ob
            ASOF JOIN underlying_bars ub
              ON ob.underlying_id = ub.instrument_id
             AND ob.ts_event >= ub.ts_event
            WHERE ob.close IS NOT NULL
              AND ob.ts_event - ub.ts_event <= INTERVAL '{stale_minutes} minutes'{scope_filter}
            {new_only_filter}{dedup_clause}
            ORDER BY ob.ts_event, ob.instrument_id
            """,
            scope_params,
        )

        total = self.con.execute("SELECT COUNT(*) FROM _greek_pending").fetchone()[0]
        if total == 0:
            self.con.execute("DROP TABLE IF EXISTS _greek_pending")
            return

        n_chunks = max(1, -(-total // _GREEKS_CHUNK_ROWS))  # ceil
        print(
            f"[greeks] Computing Greeks for {total:,} option bars "
            f"in {n_chunks} chunk(s) of up to {_GREEKS_CHUNK_ROWS:,}..."
        )

        written = 0
        completed = 0

        # Results accumulate here until _BATCH_DAYS chunks are ready, then flushed
        # with a single bulk INSERT to amortise DuckDB's per-statement fixed overhead.
        accumulated: list[pl.DataFrame] = []
        accumulated_rows = 0

        def _flush() -> None:
            nonlocal written, accumulated_rows
            if not accumulated:
                return
            batch = pl.concat(accumulated)
            self.con.register("_greeks_batch", batch)
            self.con.execute("INSERT INTO option_greeks SELECT * FROM _greeks_batch")
            self.con.unregister("_greeks_batch")
            written += len(batch)
            accumulated.clear()
            accumulated_rows = 0

        # Pipelined parallel execution: the main thread pulls fixed-size chunks from
        # DuckDB (sequential — not thread-safe) and immediately submits the numba
        # computation to the pool. Completed results are buffered; every _BATCH_DAYS
        # chunks the buffer is flushed with one bulk INSERT, amortising fixed overhead.
        #
        # numba @njit(nogil=True) releases the GIL so thread parallelism provides
        # near-linear speedup up to the physical core count.
        #
        # The chunks come from ONE streaming scan. This loop used to iterate distinct
        # trading dates and re-query `WHERE DATE(ts_event) = ?` per date, which is a
        # full scan of the pending table per day — DATE() wraps the column, so no
        # index can apply. At 5,038 days over 69.8M pending rows that is ~3.5e11 row
        # visits, and it dominated the runtime: measured at 16.5 days/min, roughly
        # 5 hours for the full ES history. Greeks are computed per row with no
        # cross-row dependency (see _compute_batch), so the day grouping was never
        # needed — arbitrary chunks give the same results in a single pass.
        pending: list[tuple[object, Future]] = []

        def _drain_one() -> None:
            nonlocal completed, accumulated_rows
            label, fut = pending.pop(0)
            result = fut.result()
            if not result.is_empty():
                accumulated.append(result)
                accumulated_rows += len(result)
            completed += 1
            print(
                f"\r[greeks] chunk {completed}/{n_chunks}  rows={written + accumulated_rows:,}",
                end="",
                flush=True,
            )
            if len(accumulated) >= _BATCH_DAYS:
                _flush()

        # to_arrow_reader streams the result; memory stays bounded by the chunk size
        # times the number of chunks in flight, not by the table size.
        reader = self.con.execute("SELECT * FROM _greek_pending").to_arrow_reader(
            _GREEKS_CHUNK_ROWS
        )

        with ThreadPoolExecutor(max_workers=_GREEKS_WORKERS) as pool:
            for idx, arrow_batch in enumerate(reader):
                batch_df = pl.from_arrow(arrow_batch)
                if batch_df.is_empty():
                    continue
                pending.append((idx, pool.submit(self._compute_batch, batch_df)))

                # Drain oldest result once the pipeline is full.
                if len(pending) >= _GREEKS_WORKERS * 2:
                    _drain_one()

            # Drain all remaining futures.
            while pending:
                _drain_one()

        _flush()  # write any chunks that didn't fill a complete batch
        self.con.execute("DROP TABLE IF EXISTS _greek_pending")
        print(f"\n[greeks] Wrote {written:,} option_greeks rows")

    def _has_duplicate_bars(self) -> bool:
        """True if any (instrument_id, ts_event) appears more than once in option_bars.

        Deliberately an early-out rather than a count: LIMIT 1 lets DuckDB stop at the
        first duplicate group, so a clean database pays only for a scan that finds
        nothing and a dirty one answers almost immediately.
        """
        row = self.con.execute(
            """
            SELECT 1 FROM option_bars
            GROUP BY instrument_id, ts_event
            HAVING count(*) > 1
            LIMIT 1
            """
        ).fetchone()
        return row is not None

    def _compute_batch(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Receive a batch DataFrame with columns:
            ts_event, instrument_id, contract_id, underlying_id, expiration,
            strike_price, right, option_close, underlying_close, underlying_ts_event

        Returns a DataFrame matching the option_greeks schema:
            ts_event, instrument_id, contract_id, underlying_id, dte, T,
            iv, delta, gamma, theta, vega, underlying_lag_s
        """
        today = df["ts_event"].dt.date()
        dte = (df["expiration"] - today).dt.total_days().cast(pl.Int32)

        # Seconds between the option bar and the underlying bar that supplied F
        # (0 when the same-minute underlying bar was used). Non-negative because
        # the ASOF join only borrows prior underlying bars.
        underlying_lag_s = (
            (df["ts_event"] - df["underlying_ts_event"]).dt.total_seconds().cast(pl.Int32)
        )

        # Use actual fractional time-to-expiry so 0DTE options get T > 0.
        # Integer-day DTE gives T=0 for same-day expiry, which produces NaN IV/greeks.
        # Treat expiration as 20:00 UTC (≈ 16:00 ET) on the expiration date.
        expiry_dt = (
            df["expiration"]
            .cast(pl.Datetime("us"))  # Date → midnight naive datetime
            .dt.replace_time_zone("UTC")  # mark as UTC
            + timedelta(hours=20)  # → 20:00 UTC on expiration date
        )
        T_sec = (
            (expiry_dt - df["ts_event"].dt.convert_time_zone("UTC"))
            .dt.total_seconds()
            .cast(pl.Float64)
        )
        T = (T_sec / (365.25 * 24 * 3600)).clip(lower_bound=0.0)

        is_call = (df["right"] == "C").cast(pl.Int8).to_numpy().astype(np.int64)
        F = df["underlying_close"].to_numpy()
        K = df["strike_price"].to_numpy()
        T_np = T.to_numpy()
        r_np = np.full(len(df), self.risk_free_rate)
        price = df["option_close"].fill_null(0.0).to_numpy()

        iv = _implied_vol(F, K, T_np, r_np, price, is_call)
        delta_arr, gamma_arr, theta_arr, vega_arr = _greeks(F, K, T_np, r_np, iv, is_call)

        return pl.DataFrame(
            {
                "ts_event": df["ts_event"],
                "instrument_id": df["instrument_id"],
                "contract_id": df["contract_id"],
                "underlying_id": df["underlying_id"],
                "dte": dte,
                "T": T,
                "iv": pl.Series(iv).cast(pl.Float64),
                "delta": pl.Series(delta_arr).cast(pl.Float64),
                "gamma": pl.Series(gamma_arr).cast(pl.Float64),
                "theta": pl.Series(theta_arr).cast(pl.Float64),
                "vega": pl.Series(vega_arr).cast(pl.Float64),
                "underlying_lag_s": underlying_lag_s,
            }
        )
