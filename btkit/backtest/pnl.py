"""
PnLCalculator — Pass 3 of the vectorized backtest.

Pure arithmetic: joins entries and exits, applies slippage and fees, and
produces final position and leg records matching the output database schema.
No database access — operates entirely on in-memory Polars DataFrames.

Cost model (docs/fill_price_and_costs.md):
    gross_pnl    = open_mark - exit_mark
    slippage     = |exit_mark| * slippage_pct
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
        all_entries: dict[str, pl.DataFrame],
        all_exits: dict[str, pl.DataFrame],
    ) -> BacktestPositions:
        """
        Receives one entries DataFrame and one exits DataFrame per trade (keyed
        by trade name). Concatenates all trades, then for each position:

        Steps:
            1. Join entries + exits on entry_id.
            2. gross_pnl = open_mark - exit_mark
            3. slippage_cost = |exit_mark| * costs.slippage_pct
            4. fee_cost = costs.fee_per_contract * total_contracts_in_position * 2
            5. net_pnl = gross_pnl - slippage_cost - fee_cost

        trade_name is carried through from the entries DataFrames and written to
        the positions output so results from different trades are distinguishable.
        worst_mark is passed through from exits unchanged — it is already computed
        by ExitScanner during the bar scan.

        Returns BacktestPositions with DataFrames matching the output DB schema.
        """
        costs = self.strategy.costs
        positions_list: list[pl.DataFrame] = []
        legs_list: list[pl.DataFrame] = []

        for trade in self.strategy.trades:
            entries = all_entries.get(trade.name, pl.DataFrame())
            exits = all_exits.get(trade.name, pl.DataFrame())

            if entries.is_empty() or exits.is_empty():
                continue

            # -- Positions --
            total_contracts = sum(leg.quantity for leg in trade.legs)
            fee_cost_val = float(costs.fee_per_contract) * total_contracts * 2

            pos = entries.join(exits, on="entry_id", how="inner")
            pos = pos.with_columns([
                (pl.col("open_mark") - pl.col("exit_mark")).alias("gross_pnl"),
                (pl.col("exit_mark").abs() * pl.lit(float(costs.slippage_pct))).alias("slippage_cost"),
                pl.lit(fee_cost_val).alias("fee_cost"),
            ]).with_columns(
                (pl.col("gross_pnl") - pl.col("slippage_cost") - pl.col("fee_cost")).alias("net_pnl")
            )

            positions_list.append(
                pos.rename({"entry_time": "open_time"}).select([
                    "entry_id", "trade_name", "open_time", "exit_time", "exit_reason",
                    "open_mark", "exit_mark", "worst_mark",
                    "slippage_cost", "fee_cost", "net_pnl",
                ])
            )

            # -- Legs: one row per (entry, leg) --
            kept_ids = pos["entry_id"]
            for leg in trade.legs:
                action_code = "STO" if leg.action == "sell_to_open" else "BTO"
                leg_df = (
                    entries
                    .filter(pl.col("entry_id").is_in(kept_ids))
                    .select([
                        "entry_id",
                        pl.col(f"leg_{leg.name}_instrument_id").alias("instrument_id"),
                        pl.col(f"leg_{leg.name}_symbol").alias("symbol"),
                        pl.col(f"leg_{leg.name}_expiration").alias("expiration"),
                        pl.col(f"leg_{leg.name}_strike_price").alias("strike_price"),
                        pl.col(f"leg_{leg.name}_right").alias("right"),
                        pl.lit(action_code).alias("action"),
                        pl.lit(leg.quantity).cast(pl.Int32).alias("quantity"),
                        pl.col(f"leg_{leg.name}_multiplier").alias("multiplier"),
                        pl.col(f"leg_{leg.name}_close").alias("open_price"),
                        pl.lit(None).cast(pl.Float64).alias("exit_price"),
                    ])
                )
                legs_list.append(leg_df)

        if not positions_list:
            return BacktestPositions(positions=pl.DataFrame(), legs=pl.DataFrame())

        all_positions = pl.concat(positions_list)
        all_legs = pl.concat(legs_list) if legs_list else pl.DataFrame()

        return BacktestPositions(positions=all_positions, legs=all_legs)
