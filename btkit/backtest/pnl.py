"""
PnLCalculator — Pass 3 of the vectorized backtest.

Pure arithmetic: joins entries and exits, applies slippage and fees, and
produces final position and leg records matching the output database schema.
No database access — operates entirely on in-memory Polars DataFrames.

Cost model (docs/fill_price_and_costs.md):
    gross_pnl    = open_mark - exit_mark
    slippage     = exit_mark * slippage_pct
    fees         = fee_per_contract * total_contracts * 2  (open + close)
    net_pnl      = gross_pnl - slippage - fees
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from btkit.strategy.definition import StrategyDefinition


@dataclass
class BacktestPositions:
    """Computed results ready to be written to the output database."""
    positions: pl.DataFrame   # matches position table schema
    legs: pl.DataFrame        # matches position_leg table schema


class PnLCalculator:
    def __init__(self, strategy: StrategyDefinition) -> None:
        self.strategy = strategy

    def compute(
        self,
        entries: pl.DataFrame,
        exits: pl.DataFrame,
    ) -> BacktestPositions:
        """
        Join entries + exits on entry_id and compute net P&L for each position.

        Steps:
            1. Join entries and exits on entry_id.
            2. gross_pnl = open_mark - exit_mark
            3. slippage_cost = exit_mark * costs.slippage_pct
            4. fee_cost = costs.fee_per_contract * total_contracts_in_position * 2
            5. net_pnl = gross_pnl - slippage_cost - fee_cost

        worst_mark is passed through from exits unchanged — it is already computed
        by ExitScanner during the bar scan.

        Returns BacktestPositions with DataFrames matching the output DB schema.
        """
        raise NotImplementedError
