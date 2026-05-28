"""Shared backtest utilities."""

from __future__ import annotations

import polars as pl


def tick_round_expr(expr: pl.Expr, tick_size: float) -> pl.Expr:
    """Round a Polars expression to the nearest tick.

    When tick_size is 0.0 the expression is returned unchanged, preserving
    the original continuous-price behaviour.  Uses (x / tick_size).round() *
    tick_size so the operation is fully vectorized with no Python loop.
    """
    if tick_size == 0.0:
        return expr
    return (expr / pl.lit(tick_size)).round(0) * pl.lit(tick_size)
