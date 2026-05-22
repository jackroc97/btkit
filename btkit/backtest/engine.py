"""
BacktestEngine — orchestrates a single backtest run.

Wires together the three passes (EntryScanner → ExitScanner → PnLCalculator)
and writes results to the output database. Receives a fully-scalar
StrategyDefinition (all sweep fields resolved to plain values).
"""

from __future__ import annotations

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
        Execute the three-pass vectorized backtest.
        Returns the backtest_id written to the output database.

        strategy must be a fully-scalar StrategyDefinition (no SweepRange or
        list-valued fields). For matrix runs, MatrixRunner resolves each
        combination to a scalar definition before dispatching here.
        """
        backtest_id = self._write_backtest_record()
        entries = EntryScanner(self.input_db, self.strategy, self.initial_equity).scan()
        exits = ExitScanner(self.input_db, self.strategy).scan(entries)
        positions = PnLCalculator(self.strategy).compute(entries, exits)
        self.output_db.write_results(backtest_id, positions.positions, positions.legs)
        return backtest_id

    def _write_backtest_record(self) -> int:
        """
        Insert a row into the backtest table and return the generated id.
        strategy_params is serialized to JSON for self-describing output.
        matrix_id and combination_id are NULL for single runs.
        """
        raise NotImplementedError
