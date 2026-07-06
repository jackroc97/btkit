"""
PnLCalculator — Pass 3 of the vectorized backtest.

Pure arithmetic: joins entries and exits, applies slippage and fees, and
produces final position and leg records matching the output database schema.
No database access — operates entirely on in-memory Polars DataFrames.

Cost model:
    gross_pnl    = (open_mark - exit_mark) × multiplier                    [dollars]
    slippage     = |exit_mark| × multiplier × slippage_pct                  [dollars]
    fee          = (entry_fee + exit_or_expiration_fee) × total_contracts   [dollars]
    net_pnl      = gross_pnl - slippage - fee                               [dollars]

total_contracts = sum(leg.quantity for each leg in the trade).
Expiry exits use expiration_fee_per_contract; all other exits use exit_fee_per_contract.

open_mark, exit_mark, and worst_mark remain in per-point terms in the output.
All PnL columns (gross_pnl, slippage_cost, fee_cost, net_pnl) are in dollars.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from btkit.strategy.definition import StrategyDefinition


@dataclass
class BacktestPositions:
    """Computed results ready to be written to the output database."""

    positions: pl.DataFrame  # matches position table schema
    legs: pl.DataFrame       # matches position_leg table schema
    continuations: pl.DataFrame  # matches position_continuation table schema (may be empty)


class PnLCalculator:
    def __init__(self, strategy: StrategyDefinition) -> None:
        self.strategy = strategy

    def compute(
        self,
        all_entries: dict[str, pl.DataFrame],
        all_exits: dict[str, pl.DataFrame],
        all_continuations: dict[str, pl.DataFrame] | None = None,
    ) -> BacktestPositions:
        """
        Receives one entries DataFrame and one exits DataFrame per trade (keyed
        by trade name). Concatenates all trades, then for each position:

        Steps:
            1. Join entries + exits on entry_id.
            2. gross_pnl = (open_mark - exit_mark) * multiplier
            3. slippage_cost = |exit_mark| * multiplier * costs.slippage_pct
            4. fee_cost = (entry_fee + exit_or_expiration_fee) * total_contracts
            5. net_pnl = gross_pnl - slippage_cost - fee_cost

        trade_name is carried through from the entries DataFrames and written to
        the positions output so results from different trades are distinguishable.
        worst_mark is passed through from exits unchanged — it is already computed
        by ExitScanner during the bar scan.

        Returns BacktestPositions with DataFrames matching the output DB schema.
        """
        costs = self.strategy.costs
        fees = costs.effective_fees
        positions_list: list[pl.DataFrame] = []
        legs_list: list[pl.DataFrame] = []
        continuations_list: list[pl.DataFrame] = []

        for trade in self.strategy.trades:
            entries = all_entries.get(trade.name, pl.DataFrame())
            exits = all_exits.get(trade.name, pl.DataFrame())

            if entries.is_empty() or exits.is_empty():
                continue

            # -- Positions --
            # All legs share the same underlying multiplier; read it from the first leg.
            first_leg = trade.legs[0].name
            multiplier_col = f"leg_{first_leg}_multiplier"
            total_contracts = float(sum(leg.quantity for leg in trade.legs))

            entry_fee = fees.entry_fee_per_contract * total_contracts
            active_exit_fee = fees.exit_fee_per_contract * total_contracts
            expiry_exit_fee = fees.expiration_fee_per_contract * total_contracts

            pos = entries.join(exits, on="entry_id", how="inner")
            pos = pos.with_columns(
                [
                    # Dollar gross: per-point spread × multiplier
                    ((pl.col("open_mark") - pl.col("exit_mark")) * pl.col(multiplier_col)).alias(
                        "gross_pnl"
                    ),
                    # Dollar slippage: % of dollar exit value
                    (
                        pl.col("exit_mark").abs()
                        * pl.col(multiplier_col)
                        * pl.lit(float(costs.slippage_pct))
                    ).alias("slippage_cost"),
                    # Structured fee: entry + exit (expiry exits use expiration_fee_per_contract)
                    (
                        pl.lit(entry_fee)
                        + pl.when(pl.col("exit_reason") == pl.lit("expiry"))
                        .then(pl.lit(expiry_exit_fee))
                        .otherwise(pl.lit(active_exit_fee))
                    ).alias("fee_cost"),
                ]
            ).with_columns(
                (pl.col("gross_pnl") - pl.col("slippage_cost") - pl.col("fee_cost")).alias(
                    "net_pnl"
                )
            )

            # target_name: the winning conditional target for a `targets:` leg
            # (item 3), or null for legs selected any other way.
            if "_target_name" in pos.columns:
                pos = pos.with_columns(pl.col("_target_name").alias("target_name"))
            else:
                pos = pos.with_columns(pl.lit(None).cast(pl.Utf8).alias("target_name"))

            positions_list.append(
                pos.rename({"entry_time": "open_time"}).select(
                    [
                        "entry_id",
                        "trade_name",
                        "open_time",
                        "exit_time",
                        "exit_reason",
                        "open_mark",
                        "exit_mark",
                        "worst_mark",
                        "slippage_cost",
                        "fee_cost",
                        "net_pnl",
                        "target_name",
                    ]
                )
            )

            # -- Legs: one row per (entry, leg) --
            kept_id_list = pos["entry_id"].to_list()
            for leg in trade.legs:
                action_code = "STO" if leg.action == "sell_to_open" else "BTO"
                leg_df = entries.filter(pl.col("entry_id").is_in(kept_id_list)).select(
                    [
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
                        pl.col(f"leg_{leg.name}_delta").alias("entry_delta"),
                        pl.col(f"leg_{leg.name}_iv").alias("entry_iv"),
                        pl.col(f"leg_{leg.name}_gamma").alias("entry_gamma"),
                        pl.col(f"leg_{leg.name}_theta").alias("entry_theta"),
                        pl.col(f"leg_{leg.name}_vega").alias("entry_vega"),
                        pl.col(f"leg_{leg.name}_dte").cast(pl.Int32).alias("entry_dte"),
                    ]
                )
                legs_list.append(leg_df)

            # -- Continuations --
            cont_raw = (all_continuations or {}).get(trade.name, pl.DataFrame())
            if not cont_raw.is_empty():
                # Find the long leg for multiplier/quantity
                long_legs = [lg for lg in trade.legs if lg.action == "buy_to_open"]
                if long_legs:
                    ll = long_legs[0]
                    multiplier_col = f"leg_{ll.name}_multiplier"
                    ll_quantity = float(ll.quantity)
                    # Attach multiplier from entries
                    mult_df = entries.select(["entry_id", pl.col(multiplier_col).alias("_mult")])
                    cont = cont_raw.join(mult_df, on="entry_id", how="left")
                    cont = cont.with_columns(
                        (
                            (pl.col("continuation_exit_price") - pl.col("continuation_entry_price"))
                            * pl.col("_mult")
                            * pl.lit(ll_quantity)
                        ).alias("continuation_pnl")
                    ).drop("_mult")
                    continuations_list.append(cont.select([
                        "entry_id",
                        "continuation_entry_price",
                        "continuation_exit_time",
                        "continuation_exit_price",
                        "continuation_exit_reason",
                        "continuation_pnl",
                    ]))

        if not positions_list:
            return BacktestPositions(
                positions=pl.DataFrame(), legs=pl.DataFrame(), continuations=pl.DataFrame()
            )

        all_positions = pl.concat(positions_list)
        all_legs = pl.concat(legs_list) if legs_list else pl.DataFrame()
        all_conts = pl.concat(continuations_list) if continuations_list else pl.DataFrame()

        return BacktestPositions(positions=all_positions, legs=all_legs, continuations=all_conts)
