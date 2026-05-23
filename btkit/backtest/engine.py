"""
BacktestEngine — orchestrates a single backtest run.

Runs EntryScanner and ExitScanner for each trade, enforces the one-at-a-time
constraint per trade using real exit times, then runs PnLCalculator across all
trades combined. Receives a fully-scalar StrategyDefinition (all sweep fields
resolved to plain values).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import polars as pl

from btkit.backtest.entry import EntryScanner
from btkit.backtest.exit import ExitScanner
from btkit.backtest.pnl import PnLCalculator
from btkit.db.input_db import InputDatabase
from btkit.db.output_db import OutputDatabase
from btkit.strategy.definition import StrategyDefinition


class BacktestEngine:
    def __init__(
        self,
        input_db: InputDatabase,
        output_db: OutputDatabase,
        strategy: StrategyDefinition,
        initial_equity: float = 100_000.0,
    ) -> None:
        self.input_db = input_db
        self.output_db = output_db
        self.strategy = strategy
        self.initial_equity = initial_equity

    def run(self) -> int:
        """
        Execute the three-pass vectorized backtest across all trades.
        Returns the backtest_id written to the output database.

        strategy must be a fully-scalar StrategyDefinition (no SweepRange or
        list-valued fields). For matrix runs, MatrixRunner resolves each
        combination to a scalar definition before dispatching here.
        """
        backtest_id = self._write_backtest_record()
        all_entries: dict[str, pl.DataFrame] = {}
        all_exits: dict[str, pl.DataFrame] = {}
        for trade in self.strategy.trades:
            entries = EntryScanner(self.input_db, self.strategy, trade).scan()
            exits = ExitScanner(self.input_db, self.strategy, trade).scan(entries)
            entries, exits = self._enforce_one_at_a_time(entries, exits)
            all_entries[trade.name] = entries
            all_exits[trade.name] = exits
        positions = PnLCalculator(self.strategy).compute(all_entries, all_exits)
        self.output_db.write_results(backtest_id, positions.positions, positions.legs)
        return backtest_id

    def _enforce_one_at_a_time(
        self,
        entries: pl.DataFrame,
        exits: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Filters entries and exits so no new position opens before the previous
        one closes. Walks (entry_time, exit_time) pairs in chronological order
        and drops any entry whose entry_time falls before the previous exit_time.
        Returns the filtered (entries, exits) pair.
        """
        if entries.is_empty():
            return entries, exits

        combined = (
            entries.select(["entry_id", "entry_time"])
            .join(exits.select(["entry_id", "exit_time"]), on="entry_id", how="left")
            .sort("entry_time")
        )

        keep_ids: list[int] = []
        last_exit_time = None

        for row in combined.iter_rows(named=True):
            if last_exit_time is None or row["entry_time"] >= last_exit_time:
                keep_ids.append(row["entry_id"])
                last_exit_time = row["exit_time"]

        keep = pl.Series("entry_id", keep_ids)
        return (
            entries.filter(pl.col("entry_id").is_in(keep)),
            exits.filter(pl.col("entry_id").is_in(keep)),
        )

    def _write_backtest_record(self) -> int:
        """
        Insert a row into the backtest table and return the generated id.
        strategy_params is serialized to JSON for self-describing output.
        matrix_id and combination_id are NULL for single runs.
        """
        return self.output_db.write_backtest({
            "strategy_name": self.strategy.name,
            "strategy_version": self.strategy.version,
            "strategy_params": json.loads(self.strategy.model_dump_json()),
            "initial_equity": self.initial_equity,
            "slippage_pct": self.strategy.costs.slippage_pct,
            "fee_per_contract": self.strategy.costs.fee_per_contract,
            "created_at": datetime.now(timezone.utc),
        })
