"""
GreeksCalculator — batch Black-76 IV and Greeks computation using numba JIT.

Reads option_bars (joined with underlying_bars for the underlying close price)
and writes iv, delta, gamma, theta, vega to option_greeks.

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

import duckdb
import numpy as np
import polars as pl
from numba import njit


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


@njit(cache=True)
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


@njit(cache=True)
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
    vega  = np.empty(n, dtype=np.float64)

    for i in range(n):
        if sigma[i] != sigma[i] or T[i] <= 0.0 or F[i] <= 0.0 or sigma[i] <= 0.0:
            delta[i] = np.nan
            gamma[i] = np.nan
            theta[i] = np.nan
            vega[i]  = np.nan
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
        vega[i]  = df * F[i] * npd1 * sqrtT

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
    ) -> None:
        self.con = con
        self.risk_free_rate = risk_free_rate
        self.batch_size = batch_size

    def run(self) -> None:
        """
        Process all rows in option_bars in batches of batch_size.
        For each batch: join with underlying_bars for the underlying close,
        call _compute_batch(), write results to option_greeks.
        """
        # Load options joined with underlying close in one query, ordered for
        # deterministic batching. underlying_bars and option_bars share ts_event.
        total = self.con.execute("SELECT COUNT(*) FROM option_bars").fetchone()[0]
        if total == 0:
            return

        print(f"[greeks] Computing Greeks for {total:,} option bars...")

        offset = 0
        written = 0
        while offset < total:
            batch_df = self.con.execute(
                """
                SELECT
                    ob.ts_event,
                    ob.instrument_id,
                    ob.underlying_id,
                    ob.expiration,
                    ob.strike_price,
                    ob."right",
                    ob.close        AS option_close,
                    ub.close        AS underlying_close
                FROM option_bars ob
                JOIN underlying_bars ub
                  ON ub.instrument_id = ob.underlying_id
                 AND ub.ts_event     = ob.ts_event
                ORDER BY ob.instrument_id, ob.ts_event
                LIMIT ? OFFSET ?
                """,
                [self.batch_size, offset],
            ).pl()

            if batch_df.is_empty():
                break

            result = self._compute_batch(batch_df)
            if not result.is_empty():
                self.con.register("_greeks_batch", result)
                self.con.execute("INSERT INTO option_greeks SELECT * FROM _greeks_batch")
                self.con.unregister("_greeks_batch")
                written += len(result)

            offset += self.batch_size

        print(f"[greeks] Wrote {written:,} option_greeks rows")

    def _compute_batch(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Receive a batch DataFrame with columns:
            ts_event, instrument_id, underlying_id, expiration, strike_price,
            right, option_close, underlying_close

        Returns a DataFrame matching the option_greeks schema:
            ts_event, instrument_id, underlying_id, dte, T,
            iv, delta, gamma, theta, vega
        """
        today = df["ts_event"].dt.date()
        dte = (df["expiration"] - today).dt.total_days().cast(pl.Int32)
        T = (dte.cast(pl.Float64) / 365.0).clip(lower_bound=0.0)

        is_call = (df["right"] == "C").cast(pl.Int8).to_numpy().astype(np.int64)
        F = df["underlying_close"].to_numpy()
        K = df["strike_price"].to_numpy()
        T_np = T.to_numpy()
        r_np = np.full(len(df), self.risk_free_rate)
        price = df["option_close"].fill_null(0.0).to_numpy()

        iv = _implied_vol(F, K, T_np, r_np, price, is_call)
        delta_arr, gamma_arr, theta_arr, vega_arr = _greeks(F, K, T_np, r_np, iv, is_call)

        return pl.DataFrame({
            "ts_event":      df["ts_event"],
            "instrument_id": df["instrument_id"],
            "underlying_id": df["underlying_id"],
            "dte":           dte,
            "T":             T,
            "iv":            pl.Series(iv).cast(pl.Float64),
            "delta":         pl.Series(delta_arr).cast(pl.Float64),
            "gamma":         pl.Series(gamma_arr).cast(pl.Float64),
            "theta":         pl.Series(theta_arr).cast(pl.Float64),
            "vega":          pl.Series(vega_arr).cast(pl.Float64),
        })
