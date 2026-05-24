"""
Test indicator script — used by the MVP end-to-end test suite (see docs/mvp.md, Test 2).

Produces two simple moving averages whose values can be verified by hand:
    sma_20 — 20-bar simple moving average of close
    sma_5  — 5-bar simple moving average of close

The leading <window_size - 1> values will be null (rolling window not yet full).
"""

import polars as pl


def compute(df: pl.DataFrame) -> pl.DataFrame:
    """
    Receives underlying_bars columns:
        ts_event, instrument_id, symbol, open, high, low, close, volume

    Returns the same DataFrame with sma_20 and sma_5 appended.
    """
    return df.with_columns(
        [
            pl.col("close").rolling_mean(20).alias("sma_20"),
            pl.col("close").rolling_mean(5).alias("sma_5"),
        ]
    )
