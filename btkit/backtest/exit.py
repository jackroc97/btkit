"""
ExitScanner — Pass 2 of the vectorized backtest.

For each entry produced by EntryScanner, scans forward through 1-minute bars
to find the first bar where an exit condition is met. Also tracks worst_mark
(Maximum Adverse Excursion) across the scan.

Exit priority order (see docs/strategy.md):
    1. Gap open past SL
    2. Gap open past TP
    3. Stop loss
    4. Take profit
    5. Indicator condition (exit.conditions, OR logic)
    6. DTE exit
    7. Expiry

Fill price rules: docs/fill_price_and_costs.md.
"""

from __future__ import annotations

import polars as pl

from btkit.db.input_db import InputDatabase
from btkit.strategy.definition import StrategyDefinition


class ExitScanner:
    def __init__(self, db: InputDatabase, strategy: StrategyDefinition) -> None:
        self.db = db
        self.strategy = strategy

    def scan(self, entries: pl.DataFrame) -> pl.DataFrame:
        """
        For each entry row, find the first exit event.

        Returns one row per entry with columns:
            entry_id, exit_time, exit_mark, worst_mark, exit_reason

        exit_reason: 'take_profit' | 'stop_loss' | 'condition' | 'dte_exit' | 'expiry'
        """
        option_bars, indicators = self._load_exit_data(entries)
        position_marks = self._compute_position_marks(option_bars, entries)
        return self._find_first_hit(position_marks, indicators, entries)

    def _load_exit_data(
        self,
        entries: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Batch-load all data needed to monitor open positions. Single DB read for
        all entries combined. Returns:
          - option_bars: all leg bars from the earliest entry_time to the latest
            possible exit (max expiration across all entries), covering all unique
            leg instrument IDs.
          - indicators: wide indicator DataFrame for the underlying over the same
            time window, for evaluating exit.conditions per bar.
        """
        raise NotImplementedError

    def _compute_position_marks(
        self,
        option_bars: pl.DataFrame,
        entries: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Compute position_mark per bar per entry as:
            position_mark = sum(leg_close * signed_quantity for each leg)

        Returns a long DataFrame: (entry_id, ts_event, position_mark,
        spread_open_mark) where spread_open_mark is computed from bar open
        prices for gap-open detection.

        Uses bar close prices rather than per-leg high/low for multi-leg
        positions — see docs/fill_price_and_costs.md for rationale.
        """
        raise NotImplementedError

    def _find_first_hit(
        self,
        position_marks: pl.DataFrame,
        indicators: pl.DataFrame,
        entries: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        For each entry, scan position_marks forward from entry_time and return
        the first bar satisfying any exit condition in priority order.

        Also computes worst_mark: the most adverse position_mark seen across all
        bars from entry_time to exit_time (free aggregation during the scan).

        Fill price determination per docs/fill_price_and_costs.md:
          - TP/SL mid-bar: fill at threshold price
          - Gap open past threshold: fill at spread_open_mark
          - SL takes priority when both TP and SL are crossed in the same bar

        For indicator exits: fill at bar close mark (no specific price threshold).
        """
        raise NotImplementedError
