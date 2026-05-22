"""
StrategyMatrix — expands a parameterized StrategyDefinition into scalar instances.

DEFERRED: not part of the MVP. Matrix runs (sweep parameters, explicit combinations,
ProcessPoolExecutor parallelism) are a post-MVP feature.
"""

from __future__ import annotations

import polars as pl

from btkit.strategy.definition import StrategyDefinition


class StrategyMatrix:
    """
    Expands a parameterized StrategyDefinition into a list of fully-scalar
    instances, one per combination. Handles full-factorial sweep expansion
    and explicit combination modes.

    DEFERRED — not implemented for MVP.
    """

    def __init__(self, definition: StrategyDefinition) -> None:
        raise NotImplementedError("StrategyMatrix is deferred — not available in MVP")

    @property
    def combinations(self) -> list[StrategyDefinition]:
        """List of fully-scalar StrategyDefinition instances, one per combination."""
        raise NotImplementedError

    @property
    def params_df(self) -> pl.DataFrame:
        """Summary DataFrame of the parameter space. One row per combination."""
        raise NotImplementedError

    def _expand_sweeps(self) -> list[dict]:
        """Resolve sweep fields to cartesian product of scalar overrides."""
        raise NotImplementedError

    def _expand_combinations(self) -> list[dict]:
        """Resolve explicit combinations to the same flat-dict format as sweeps."""
        raise NotImplementedError
