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
import time
import traceback
from datetime import UTC, datetime, time as dtime, timedelta
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
        study_id: int | None = None,
        combination_id: int | None = None,
        note: str | None = None,
    ) -> None:
        self.input_db = input_db
        self.output_db = output_db
        self.strategy = strategy
        self.initial_equity = initial_equity
        self.study_id = study_id
        self.combination_id = combination_id
        self.note = note

    def run(self) -> int:
        """
        Execute the three-pass vectorized backtest across all trades.
        Returns the backtest_id written to the output database.
        """
        backtest_id = self._write_backtest_record()
        warnings: list[dict] = []
        start = time.perf_counter()

        try:
            indicators = self._load_indicators_once()

            all_entries: dict[str, pl.DataFrame] = {}
            all_exits: dict[str, pl.DataFrame] = {}
            all_continuations: dict[str, pl.DataFrame] = {}
            entry_id_offset = 0

            for trade in self.strategy.trades:
                entry_scanner = EntryScanner(
                    self.input_db, self.strategy, trade, indicators=indicators
                )
                entries = entry_scanner.scan(entry_id_offset)
                warnings.extend(entry_scanner.warnings)
                n_scanned = len(entries)  # capture before enforcement drops rows

                exit_scanner = ExitScanner(
                    self.input_db, self.strategy, trade, indicators=indicators
                )
                exits = exit_scanner.scan(entries)
                warnings.extend(exit_scanner.warnings)

                # Build gate overrides when continuation is enabled: re-entry is
                # blocked until max(continuation_exit_time, next_trading_day_open).
                gate_overrides: dict[int, int] | None = None
                if trade.exit.on_sl_long_continuation and not exit_scanner.continuation_exits.is_empty():
                    gate_overrides = self._build_gate_overrides(
                        exits, exit_scanner.continuation_exits, trade
                    )

                entries, exits = self._enforce_one_at_a_time(entries, exits, gate_overrides)

                roll_exit_ids: set[int] = set()
                if trade.roll is not None and "exit_reason" in exits.columns:
                    roll_exit_ids = set(
                        exits.filter(pl.col("exit_reason") == "roll")["entry_id"].to_list()
                    )

                if trade.entry.max_entries_per_day is not None:
                    entries, exits = self._enforce_max_entries_per_day(
                        entries, exits, trade.entry.max_entries_per_day,
                        roll_exit_ids=roll_exit_ids if roll_exit_ids else None,
                    )
                if trade.entry.no_reentry_after_loss:
                    entries, exits = self._enforce_no_reentry_after_loss(entries, exits)
                all_entries[trade.name] = entries
                all_exits[trade.name] = exits

                # Retain only continuation rows whose spread position survived enforcement
                cont = exit_scanner.continuation_exits
                if not cont.is_empty():
                    keep_ids = set(entries["entry_id"].to_list())
                    cont = cont.filter(pl.col("entry_id").is_in(keep_ids))
                all_continuations[trade.name] = cont

                # Advance by pre-enforcement count so surviving entry_ids from this
                # trade never overlap with IDs assigned to the next trade.
                entry_id_offset += n_scanned

            positions = PnLCalculator(self.strategy).compute(
                all_entries, all_exits, all_continuations
            )
            self.output_db.write_results(
                backtest_id, positions.positions, positions.legs, positions.continuations
            )

            self.output_db.finalize_backtest(
                backtest_id,
                status="completed",
                duration_s=time.perf_counter() - start,
                warnings=warnings,
            )
        except Exception:
            self.output_db.finalize_backtest(
                backtest_id,
                status="error",
                duration_s=time.perf_counter() - start,
                warnings=warnings,
                error_message=str(traceback.format_exc().splitlines()[-1]),
                error_traceback=traceback.format_exc(),
            )
            raise

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

        universe = self.strategy.universe
        trade0 = self.strategy.trades[0]
        tz = ZoneInfo(universe.session.timezone)
        start_dt = datetime(
            universe.start_date.year,
            universe.start_date.month,
            universe.start_date.day,
            tzinfo=tz,
        ).astimezone(UTC)

        max_dte = max(int(leg.dte) for trade in self.strategy.trades for leg in trade.legs)
        base_end = datetime(
            universe.end_date.year,
            universe.end_date.month,
            universe.end_date.day,
            23,
            59,
            59,
            tzinfo=tz,
        ).astimezone(UTC)
        end_dt = base_end + timedelta(days=max_dte + 10)

        # Build a roll schedule extended by the DTE buffer so positions that
        # expire after universe.end_date still have indicator coverage.
        end_date_buffered = end_dt.astimezone(tz).date()
        schedule = self.input_db.front_future_schedule(
            trade0.instrument.root_symbol,
            universe.start_date,
            end_date_buffered,
            trade0.instrument.roll_days_before_expiry,
        )
        if schedule.is_empty():
            return pl.DataFrame()

        frames = [
            self.input_db.indicators(uid, start_dt, end_dt)
            for uid in schedule["underlying_id"].unique().to_list()
        ]
        non_empty = [f for f in frames if not f.is_empty() and len(f.columns) > 1]
        if not non_empty:
            return pl.DataFrame()
        if len(non_empty) == 1:
            return non_empty[0].sort("ts_event")
        return pl.concat(non_empty, how="diagonal").sort("ts_event")

    def _next_trading_day_open(self, after: datetime) -> datetime:
        """
        Return the datetime of the first moment of the trading session on the
        calendar day after `after`, respecting weekdays_only and skip_dates.
        """
        session = self.strategy.universe.session
        tz = ZoneInfo(session.timezone)
        current_local = after.astimezone(tz)
        next_date = current_local.date() + timedelta(days=1)
        skip = set(session.skip_dates or [])
        while True:
            if session.weekdays_only and next_date.weekday() >= 5:
                next_date += timedelta(days=1)
                continue
            if next_date in skip:
                next_date += timedelta(days=1)
                continue
            break
        start = session.start_time if session.start_time is not None else dtime(0, 0)
        gate_local = datetime(
            next_date.year, next_date.month, next_date.day,
            start.hour, start.minute, tzinfo=tz,
        )
        return gate_local.astimezone(UTC)

    def _build_gate_overrides(
        self,
        exits: pl.DataFrame,
        continuation_exits: pl.DataFrame,
        trade,
    ) -> dict[int, int]:
        """
        For SL exits that have a continuation, compute the effective gate time
        as max(continuation_exit_time, next_trading_day_open_after_sl_exit).
        Returns a dict mapping entry_id → Int64 microseconds.
        """
        sl_exits = exits.filter(
            pl.col("exit_reason").is_in(["stop_loss", "gap_sl"])
        ).select(["entry_id", "exit_time"])

        sl_with_cont = sl_exits.join(
            continuation_exits.select(["entry_id", "continuation_exit_time"]),
            on="entry_id",
            how="inner",
        )

        overrides: dict[int, int] = {}
        for row in sl_with_cont.iter_rows(named=True):
            eid = row["entry_id"]
            sl_exit_time: datetime = row["exit_time"]
            cont_exit_time: datetime = row["continuation_exit_time"]
            next_day = self._next_trading_day_open(sl_exit_time)
            gate = max(cont_exit_time, next_day)
            overrides[int(eid)] = int(gate.timestamp() * 1_000_000)
        return overrides

    def _enforce_one_at_a_time(
        self,
        entries: pl.DataFrame,
        exits: pl.DataFrame,
        gate_overrides: dict[int, int] | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Filters entries and exits so no new position opens before the previous
        one closes. Walks (entry_time, exit_time) pairs in chronological order
        and drops any entry whose entry_time falls before the previous exit_time.

        gate_overrides maps entry_id → effective gate time (Int64 µs). When
        provided, SL exits with active continuations use the later of the
        continuation exit time and the next-trading-day open instead of the
        raw spread exit_time.

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
        exit_times = combined["exit_time"].cast(pl.Int64).fill_null(0).to_numpy()
        entry_ids = combined["entry_id"].to_numpy()

        if gate_overrides:
            gate_times = np.array(
                [gate_overrides.get(int(eid), int(et)) for eid, et in zip(entry_ids, exit_times)],
                dtype=np.int64,
            )
        else:
            gate_times = exit_times

        keep = np.zeros(len(entry_times), dtype=bool)
        last_exit = None

        for i in range(len(entry_times)):
            if last_exit is None or entry_times[i] >= last_exit:
                keep[i] = True
                last_exit = gate_times[i]

        keep_ids = entry_ids[keep].tolist()

        return (
            entries.filter(pl.col("entry_id").is_in(keep_ids)),
            exits.filter(pl.col("entry_id").is_in(keep_ids)),
        )

    def _enforce_max_entries_per_day(
        self,
        entries: pl.DataFrame,
        exits: pl.DataFrame,
        max_entries: int,
        roll_exit_ids: set[int] | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        After one-at-a-time enforcement, further cap the number of positions
        opened per calendar day to max_entries. Entries are ranked chronologically
        within each day; only the first max_entries survive.

        When roll_exit_ids is provided, the re-entry immediately following each
        roll exit on the same calendar day is exempt from the daily cap — it is
        continuation of the same position, not a fresh trade.
        """
        if entries.is_empty():
            return entries, exits

        tz_str = self.strategy.universe.session.timezone

        # Identify roll re-entries: the first entry on the same day after each roll exit.
        roll_reentry_ids: set[int] = set()
        if roll_exit_ids:
            roll_exit_times = (
                exits.filter(pl.col("entry_id").is_in(roll_exit_ids))
                .join(entries.select(["entry_id", "entry_time"]), on="entry_id", how="left")
                .with_columns(
                    pl.col("exit_time").dt.convert_time_zone(tz_str).dt.date().alias("_roll_date")
                )
            )
            entries_sorted = entries.sort("entry_time").with_columns(
                pl.col("entry_time").dt.convert_time_zone(tz_str).dt.date().alias("_entry_date")
            )
            for row in roll_exit_times.iter_rows(named=True):
                roll_date = row["_roll_date"]
                roll_exit_t = row["exit_time"]
                candidate = (
                    entries_sorted
                    .filter(
                        (pl.col("_entry_date") == pl.lit(roll_date))
                        & (pl.col("entry_time") > pl.lit(roll_exit_t))
                    )
                    .head(1)
                )
                if not candidate.is_empty():
                    roll_reentry_ids.add(int(candidate["entry_id"][0]))

        entries_capped = (
            entries.sort("entry_time")
            .with_columns(
                pl.col("entry_time").dt.convert_time_zone(tz_str).dt.date().alias("_entry_date"),
                pl.col("entry_id").is_in(list(roll_reentry_ids)).alias("_is_roll_reentry"),
            )
            .with_columns(
                pl.when(pl.col("_is_roll_reentry"))
                .then(pl.lit(0))
                .otherwise(pl.col("entry_id").cum_count().over("_entry_date"))
                .alias("_day_rank")
            )
            .filter(
                pl.col("_is_roll_reentry") | (pl.col("_day_rank") <= max_entries)
            )
            .drop(["_entry_date", "_day_rank", "_is_roll_reentry"])
        )

        keep_ids = set(entries_capped["entry_id"].to_list())

        return (
            entries.filter(pl.col("entry_id").is_in(keep_ids)),
            exits.filter(pl.col("entry_id").is_in(keep_ids)),
        )

    def _enforce_no_reentry_after_loss(
        self,
        entries: pl.DataFrame,
        exits: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Drops same-day entries that follow a losing position on the same calendar date.
        A loss is defined as exit_reason in {'stop_loss', 'gap_sl'}.

        Walks the post-one-at-a-time positions in chronological order. Once a loss
        is recorded on date D, all subsequent entries on date D are dropped.
        """
        if entries.is_empty():
            return entries, exits

        tz_str = self.strategy.universe.session.timezone
        loss_reasons = {"stop_loss", "gap_sl"}

        combined = (
            entries.select(["entry_id", "entry_time"])
            .join(exits.select(["entry_id", "exit_reason"]), on="entry_id", how="left")
            .sort("entry_time")
            .with_columns(
                pl.col("entry_time")
                .dt.convert_time_zone(tz_str)
                .dt.date()
                .alias("_date")
            )
        )

        entry_dates = combined["_date"].to_list()
        exit_reasons = combined["exit_reason"].to_list()
        entry_ids = combined["entry_id"].to_numpy()

        keep = np.zeros(len(entry_ids), dtype=bool)
        loss_dates: set = set()

        for i in range(len(entry_ids)):
            if entry_dates[i] in loss_dates:
                continue
            keep[i] = True
            if exit_reasons[i] in loss_reasons:
                loss_dates.add(entry_dates[i])

        keep_ids = entry_ids[keep].tolist()
        return (
            entries.filter(pl.col("entry_id").is_in(keep_ids)),
            exits.filter(pl.col("entry_id").is_in(keep_ids)),
        )

    def _write_backtest_record(self) -> int:
        return self.output_db.write_backtest(
            {
                "strategy_name": self.strategy.name,
                "strategy_version": self.strategy.version,
                "strategy_params": json.loads(self.strategy.model_dump_json()),
                "initial_equity": self.initial_equity,
                "slippage_pct": self.strategy.costs.slippage_pct,
                "fee_per_contract": self.strategy.costs.fee_per_contract,
                "created_at": datetime.now(UTC),
                "study_id": self.study_id,
                "combination_id": self.combination_id,
                "note": self.note,
            }
        )
