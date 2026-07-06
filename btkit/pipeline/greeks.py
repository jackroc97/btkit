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

# Days accumulated before a single bulk INSERT into option_greeks.
# Each INSERT carries ~0.75 s of fixed DuckDB overhead regardless of size.
# N=20 makes variable cost (20 × 37k rows / 100k rows·s⁻¹ ≈ 7.4 s) dominate
# fixed cost by 10×, capturing ~94% of the available throughput improvement.
_BATCH_DAYS = 20

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

    def run(self, skip_existing: bool = False) -> None:
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
        stale_minutes = self.underlying_max_staleness_minutes
        self.con.execute("DROP TABLE IF EXISTS _greek_pending")
        self.con.execute(
            f"""
            CREATE TEMP TABLE _greek_pending AS
            SELECT
                ob.ts_event,
                ob.instrument_id,
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
              AND ob.ts_event - ub.ts_event <= INTERVAL '{stale_minutes} minutes'
            {new_only_filter}
            ORDER BY ob.ts_event, ob.instrument_id
            """
        )

        total = self.con.execute("SELECT COUNT(*) FROM _greek_pending").fetchone()[0]
        if total == 0:
            self.con.execute("DROP TABLE IF EXISTS _greek_pending")
            return

        # Step 2: Distinct trading dates — drives per-day streaming.
        dates = [
            r[0]
            for r in self.con.execute(
                "SELECT DISTINCT DATE(ts_event) FROM _greek_pending ORDER BY 1"
            ).fetchall()
        ]

        print(f"[greeks] Computing Greeks for {total:,} option bars across {len(dates)} days...")

        written = 0
        completed = 0

        # Results accumulate here until _BATCH_DAYS days are ready, then flushed
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

        # Pipelined parallel execution: the main thread reads each day's batch from
        # DuckDB (sequential — not thread-safe) and immediately submits the numba
        # computation to the pool. Completed results are buffered; every _BATCH_DAYS
        # days the buffer is flushed with one bulk INSERT, amortising fixed overhead.
        #
        # numba @njit(nogil=True) releases the GIL so thread parallelism provides
        # near-linear speedup up to the physical core count.
        pending: list[tuple[object, Future]] = []

        def _drain_one() -> None:
            nonlocal completed, accumulated_rows
            date_d, fut = pending.pop(0)
            result = fut.result()
            if not result.is_empty():
                accumulated.append(result)
                accumulated_rows += len(result)
            completed += 1
            print(
                f"\r[greeks] {date_d}  {completed}/{len(dates)} days"
                f"  rows={written + accumulated_rows:,}",
                end="",
                flush=True,
            )
            if len(accumulated) >= _BATCH_DAYS:
                _flush()

        with ThreadPoolExecutor(max_workers=_GREEKS_WORKERS) as pool:
            for date in dates:
                batch_df = self.con.execute(
                    "SELECT * FROM _greek_pending "
                    "WHERE DATE(ts_event) = ? ORDER BY instrument_id, ts_event",
                    [date],
                ).pl()
                pending.append((date, pool.submit(self._compute_batch, batch_df)))

                # Drain oldest result once the pipeline is full.
                if len(pending) >= _GREEKS_WORKERS * 2:
                    _drain_one()

            # Drain all remaining futures.
            while pending:
                _drain_one()

        _flush()  # write any days that didn't fill a complete batch
        self.con.execute("DROP TABLE IF EXISTS _greek_pending")
        print(f"\n[greeks] Wrote {written:,} option_greeks rows")

    def _compute_batch(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Receive a batch DataFrame with columns:
            ts_event, instrument_id, underlying_id, expiration, strike_price,
            right, option_close, underlying_close, underlying_ts_event

        Returns a DataFrame matching the option_greeks schema:
            ts_event, instrument_id, underlying_id, dte, T,
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
