"""
BacktestEngine — orchestrates a single backtest run.

Runs EntryScanner and ExitScanner for each trade, enforces the one-at-a-time
constraint per trade using real exit times, then runs PnLCalculator across all
trades combined. Receives a fully-scalar StrategyDefinition (all sweep fields
resolved to plain values).

Performance notes:
    - Indicators are fetched once for the full universe window + a dte buffer so
      that ExitScanner cohorts can reuse the in-memory DataFrame rather than
      issuing a separate DB query per trade.
    - entry_id is globally unique across trades via an incrementing offset, so
      positions from different trades never share an ID in the output database.
    - _enforce_one_at_a_time() uses numpy arrays for the sequential walk instead
      of Polars named-row iteration, which is faster for large entry counts.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
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
        """
        backtest_id = self._write_backtest_record()
        indicators = self._load_indicators_once()

        all_entries: dict[str, pl.DataFrame] = {}
        all_exits: dict[str, pl.DataFrame] = {}
        entry_id_offset = 0

        for trade in self.strategy.trades:
            entries = EntryScanner(
                self.input_db, self.strategy, trade, indicators=indicators
            ).scan(entry_id_offset)
            exits = ExitScanner(
                self.input_db, self.strategy, trade, indicators=indicators
            ).scan(entries)
            entries, exits = self._enforce_one_at_a_time(entries, exits)
            all_entries[trade.name] = entries
            all_exits[trade.name] = exits
            entry_id_offset += len(entries)

        positions = PnLCalculator(self.strategy).compute(all_entries, all_exits)
        self.output_db.write_results(backtest_id, positions.positions, positions.legs)
        return backtest_id

    def _load_indicators_once(self) -> pl.DataFrame:
        """
        Fetch indicators for the full universe window plus a DTE buffer.

        The buffer (max DTE across all legs + 10 days) ensures ExitScanner
        cohorts whose positions expire after universe.end_date still have
        indicator data without a separate DB fetch.

        Returns an empty DataFrame when no trades have conditions, so the DB
        is not queried at all for strategies without indicator-based filters.
        """
        needs_indicators = any(
            bool(trade.entry.conditions) or bool(trade.exit.conditions)
            for trade in self.strategy.trades
        )
        if not needs_indicators:
            return pl.DataFrame()

        trade0 = self.strategy.trades[0]
        underlying_id = self.input_db.instrument_id_for_symbol(
            trade0.instrument.root_symbol
        )
        if underlying_id is None:
            return pl.DataFrame()

        universe = self.strategy.universe
        tz = ZoneInfo(universe.session.timezone)
        start_dt = datetime(
            universe.start_date.year, universe.start_date.month, universe.start_date.day,
            tzinfo=tz,
        ).astimezone(timezone.utc)

        max_dte = max(
            int(leg.dte)
            for trade in self.strategy.trades
            for leg in trade.legs
        )
        base_end = datetime(
            universe.end_date.year, universe.end_date.month, universe.end_date.day,
            23, 59, 59, tzinfo=tz,
        ).astimezone(timezone.utc)
        end_dt = base_end + timedelta(days=max_dte + 10)

        return self.input_db.indicators(underlying_id, start_dt, end_dt)

    def _enforce_one_at_a_time(
        self,
        entries: pl.DataFrame,
        exits: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Filters entries and exits so no new position opens before the previous
        one closes. Walks (entry_time, exit_time) pairs in chronological order
        and drops any entry whose entry_time falls before the previous exit_time.

        Uses numpy arrays for the sequential walk — faster than Polars named-row
        iteration for large entry counts expected at multi-year scale.
        """
        if entries.is_empty():
            return entries, exits

        combined = (
            entries.select(["entry_id", "entry_time"])
            .join(exits.select(["entry_id", "exit_time"]), on="entry_id", how="left")
            .sort("entry_time")
        )

        # Cast to Int64 microseconds for fast numpy comparison.
        # Null exit_time (entry with no monitoring bars) becomes 0 (epoch =
        # 1970-01-01), which is before any entry_time, so the next entry is
        # always allowed — matching the original Python None-is-None semantics.
        # numpy NaT comparisons always return False, which would incorrectly
        # block all subsequent entries after an unexited position.
        entry_times = combined["entry_time"].cast(pl.Int64).to_numpy()
        exit_times  = combined["exit_time"].cast(pl.Int64).fill_null(0).to_numpy()
        entry_ids   = combined["entry_id"].to_numpy()

        keep = np.zeros(len(entry_times), dtype=bool)
        last_exit = None

        for i in range(len(entry_times)):
            if last_exit is None or entry_times[i] >= last_exit:
                keep[i] = True
                last_exit = exit_times[i]

        keep_ids = entry_ids[keep].tolist()

        return (
            entries.filter(pl.col("entry_id").is_in(keep_ids)),
            exits.filter(pl.col("entry_id").is_in(keep_ids)),
        )

    def _write_backtest_record(self) -> int:
        return self.output_db.write_backtest({
            "strategy_name":    self.strategy.name,
            "strategy_version": self.strategy.version,
            "strategy_params":  json.loads(self.strategy.model_dump_json()),
            "initial_equity":   self.initial_equity,
            "slippage_pct":     self.strategy.costs.slippage_pct,
            "fee_per_contract": self.strategy.costs.fee_per_contract,
            "created_at":       datetime.now(timezone.utc),
        })
