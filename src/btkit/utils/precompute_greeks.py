import duckdb
import math
import numpy as np
import sys

from datetime import datetime
from numba import njit, prange


SQRT2PI = math.sqrt(2.0 * math.pi)

@njit(fastmath=True)
def _norm_cdf(x: float) -> float:
    # CDF using erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

@njit(fastmath=True)
def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT2PI

@njit(fastmath=True)
def black76_price_scalar(F: float, K: float, T: float, r: float, sigma: float, is_call: int) -> float:
    if T <= 0.0 or sigma <= 0.0 or F <= 0.0 or K <= 0.0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    df = math.exp(-r * T)
    if is_call == 1:
        return df * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
    else:
        return df * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))

@njit(fastmath=True, parallel=True)
def implied_vol_black76_numba(
    F_arr, K_arr, T_arr, price_arr, right_arr, r=0.01,
    initial_guess=0.3, tol=1e-8, max_iter=80,
    sigma_min=1e-6, sigma_max=5.0
):
    n = len(F_arr)
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        F = F_arr[i]; K = K_arr[i]; T = T_arr[i]; #r = r_arr[i]
        market_price = price_arr[i]
        is_call = 1 if right_arr[i] == 1 else -1

        # Basic validation
        if T <= 0.0 or market_price <= 0.0 or F <= 0.0 or K <= 0.0:
            out[i] = np.nan
            continue

        sigma = initial_guess
        if sigma < sigma_min:
            sigma = sigma_min
        if sigma > sigma_max:
            sigma = sigma_max

        converged = False
        for _ in range(max_iter):
            # theoretical price
            theo = black76_price_scalar(F, K, T, r, sigma, is_call)
            diff = theo - market_price
            if abs(diff) < tol:
                converged = True
                break

            sqrtT = math.sqrt(T)
            # compute vega analytically (Black-76 vega w.r.t sigma)
            d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
            pdf_d1 = _norm_pdf(d1)
            vega = math.exp(-r * T) * F * sqrtT * pdf_d1

            # guard
            if vega < 1e-12:
                break

            # Newton step
            sigma = sigma - diff / vega

            # enforce bounds
            if sigma <= 0:
                sigma = sigma_min
            elif sigma > sigma_max:
                sigma = sigma_max

        if converged:
            out[i] = sigma
            continue

        # fallback: simple bisection between sigma_min and sigma_max
        a = sigma_min
        b = sigma_max
        fa = black76_price_scalar(F, K, T, r, a, is_call) - market_price
        fb = black76_price_scalar(F, K, T, r, b, is_call) - market_price
        if fa * fb > 0:
            # cannot bracket, give up
            out[i] = np.nan
            continue
        # bisection loop
        for _ in range(60):
            m = 0.5 * (a + b)
            fm = black76_price_scalar(F, K, T, r, m, is_call) - market_price
            if abs(fm) < tol:
                out[i] = m
                break
            if fa * fm <= 0:
                b = m
                fb = fm
            else:
                a = m
                fa = fm
        else:
            # not converged in bisection
            out[i] = np.nan
    return out

@njit(fastmath=True, parallel=True)
def black76_delta_numba(F_arr, K_arr, T_arr, sigma_arr, right_arr, r=0.01):
    n = len(F_arr)
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        F = F_arr[i]; K = K_arr[i]; T = T_arr[i]; sigma = sigma_arr[i]; is_call = right_arr[i]
        if T <= 0.0 or math.isnan(sigma):
            out[i] = np.nan
            continue
        sqrtT = math.sqrt(T)
        d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
        if is_call == 1:
            out[i] = math.exp(-r * T) * _norm_cdf(d1)
        else:
            out[i] = -math.exp(-r * T) * _norm_cdf(-d1)
    return out
    
def precompute_greeks(database_path: str):
    SECONDS_PER_YEAR = 365.0 * 24 * 3600

    conn = duckdb.connect(database_path)

    # 1. Create a temporary table with flattened option definition data, option
    # close price, and underlying close price
    query = f"""
        CREATE TEMP TABLE options_data AS (
            WITH 
            option_definition AS (
                SELECT DISTINCT ON (raw_symbol)
                    raw_symbol,
                    instrument_id,
                    underlying_id,
                    activation,
                    ts_expiration,
                    strike_price,
                    instrument_class
                FROM definition
                WHERE instrument_class in ('C', 'P')
            ),

            option_ohlcv AS (
                SELECT 
                    d.raw_symbol,
                    d.instrument_id,
                    d.underlying_id,
                    d.activation,
                    d.ts_expiration,
                    d.strike_price,
                    d.instrument_class,
                    o.ts_event,
                    o.close,
                FROM option_definition d
                JOIN ohlcv o
                    ON (d.instrument_id = o.instrument_id)
                    AND (epoch(o.ts_event) BETWEEN epoch(d.activation) AND epoch(d.ts_expiration))
                ORDER BY raw_symbol
            )

            SELECT 
                ROW_NUMBER() OVER () AS id,
                opt.raw_symbol,
                opt.instrument_id,
                opt.underlying_id,
                opt.ts_expiration,
                opt.strike_price,
                opt.instrument_class AS option_right,
                opt.ts_event,
                opt.close AS option_close,
                und.close AS underlying_close,
                (epoch(opt.ts_expiration - opt.ts_event)) / {SECONDS_PER_YEAR} AS T,
                FLOOR(T * 365) AS dte,
                ABS(underlying_close - option_close) AS strike_distance
            FROM option_ohlcv opt
            JOIN ohlcv und
                ON (opt.underlying_id = und.instrument_id)
                AND (opt.ts_event = und.ts_event)
        )
    """
    conn.execute(query)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS option_greeks (
            ts_event BIGINT, 
            instrument_id INTEGER, 
            underlying_id INTEGER,
            strike_price DOUBLE, 
            T DOUBLE, 
            dte INTEGER,
            strike_distance DOUBLE,
            underlying_close DOUBLE, 
            option_close DOUBLE, 
            option_right VARCHAR,
            sigma DOUBLE,
            delta DOUBLE
        );             
    """)

    # 2. Process option data in batches
    time_total = 0
    last_id = 0
    batch_size = 10_000
    while True:
        t0 = datetime.now()
        df = conn.execute(f"""
            SELECT * FROM options_data
            WHERE id > {last_id}
            ORDER BY id
            LIMIT {batch_size}
        """).df()

        if df.empty:
            break

        # Create input vectors
        F = df["underlying_close"].to_numpy(dtype=np.float64)
        K = df["strike_price"].to_numpy(dtype=np.float64)
        T = df["T"].to_numpy(dtype=np.float64)
        option_price = df["option_close"].to_numpy(dtype=np.float64)
        option_right = np.where(df["option_right"].astype(str).str.upper().str.startswith('C'), 1, -1).astype(np.int64)

        # Calculate implied volatility and greeks
        sigma = implied_vol_black76_numba(F, K, T, option_price, option_right, initial_guess=0.4)
        delta = black76_delta_numba(F, K, T, sigma, option_right)

        # Combine results and write to output table
        output_df = df.copy()
        output_df["sigma"] = sigma
        output_df["delta"] = delta
        output_df = output_df[["ts_event", "instrument_id", "underlying_id", "strike_price", "T", "dte", "strike_distance", "underlying_close", "option_close", "option_right", "sigma", "delta"]]
        conn.register("output_df", output_df)
        conn.execute(f"INSERT INTO option_greeks SELECT * FROM output_df")
        conn.unregister("output_df")

        # Print status
        t1 = datetime.now() - t0
        print(f"IDs {df.id.min()} - {df.id.max()} in {t1.microseconds/1000} ms")
        time_total += t1.total_seconds()

        # Update last id
        last_id = df["id"].max()

    print(f"Processed data in {time_total} seconds")


if __name__ == "__main__":
    precompute_greeks(*sys.argv[1:])