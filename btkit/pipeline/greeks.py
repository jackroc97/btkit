"""
GreeksCalculator — batch Black-76 IV and Greeks computation using numba JIT.

Reads from option_bars (joined with underlying_bars for the underlying close
price) and writes iv, delta, gamma, theta, vega to option_greeks.

The numba-compiled _black76_* functions operate on numpy arrays and are called
per batch to avoid per-row Python overhead.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numba import njit

from btkit.db.input_db import InputDatabase


# ---------------------------------------------------------------------------
# numba-compiled Black-76 kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _implied_vol(
    F: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    market_price: np.ndarray,
    right: np.ndarray,   # 0 = call, 1 = put
) -> np.ndarray:
    """
    Vectorised implied volatility via bisection. Returns IV per row.
    right: 0 = call ('C'), 1 = put ('P').
    """
    raise NotImplementedError


@njit(cache=True)
def _greeks(
    F: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised Black-76 Greeks. Returns (delta, gamma, theta, vega) per row.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# GreeksCalculator
# ---------------------------------------------------------------------------

class GreeksCalculator:
    def __init__(
        self,
        db: InputDatabase,
        risk_free_rate: float = 0.01,
        batch_size: int = 50_000,
    ) -> None:
        self.db = db
        self.risk_free_rate = risk_free_rate
        self.batch_size = batch_size

    def run(self) -> None:
        """
        Process all rows in option_bars in batches of batch_size.
        For each batch: join with underlying_bars for the underlying close,
        call _compute_batch(), write results to option_greeks.
        """
        raise NotImplementedError

    def _compute_batch(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Receive a batch DataFrame with columns from option_bars joined with
        the underlying close. Extract numpy arrays, call numba kernels for
        implied_vol and greeks, return a DataFrame matching the option_greeks
        schema:
            ts_event, instrument_id, underlying_id, dte, T,
            iv, delta, gamma, theta, vega

        expiration, right, strike_price, underlying_close, and option_close
        are used as inputs but are NOT written to option_greeks — they already
        exist in option_bars.
        """
        raise NotImplementedError
