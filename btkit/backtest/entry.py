"""
EntryScanner — Pass 1 of the vectorized backtest.

Scans the session to find every valid entry signal for a single TradeDefinition,
selects option legs for each, computes the opening spread mark, and evaluates
entry conditions. Returns a DataFrame where each row is a fully-specified entry
ready for ExitScanner.

Pipeline within scan():
    1. _apply_window_filters()   — time/session filter (cheap, no DB access)
    2. _select_legs()            — batched DuckDB query on option_greeks
    3. _compute_open_mark()      — spread mark + TP/SL price derivation
    4. _evaluate_conditions()    — conditions, min_credit/max_debit (vectorized)

Sign convention (consistent with docs/strategy.md):
    STO (sell_to_open): signed_qty = +1  (you receive premium)
    BTO (buy_to_open):  signed_qty = -1  (you pay premium)
    open_mark = Σ(leg_price × qty × signed_qty)
    sl_price  = open_mark + stop_loss     (exits when mark rises above threshold)
    tp_price  = open_mark - take_profit   (exits when mark falls below threshold)

The one-at-a-time constraint is NOT applied here. It is enforced by
BacktestEngine._enforce_one_at_a_time() after Pass 2 using real exit times.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import polars as pl

from btkit.db.input_db import InputDatabase
from btkit.strategy.definition import StrategyDefinition, TradeDefinition
from btkit.strategy.loader import parse_condition


class EntryScanner:
    def __init__(
        self,
        db: InputDatabase,
        strategy: StrategyDefinition,
        trade: TradeDefinition,
    ) -> None:
        self.db = db
        self.strategy = strategy
        self.trade = trade
        self._tz = ZoneInfo(strategy.universe.session.timezone)

    def scan(self) -> pl.DataFrame:
        """
        Run the full entry scan and return one row per valid entry.

        Returned columns:
            entry_id, trade_name, entry_time, open_mark, tp_price, sl_price,
            dte_exit (int, nullable),
            + per leg: leg_{name}_instrument_id, leg_{name}_open_price,
                       leg_{name}_multiplier, leg_{name}_strike,
                       leg_{name}_expiration, leg_{name}_right,
                       leg_{name}_action, leg_{name}_quantity,
                       leg_{name}_delta, leg_{name}_dte
        """
        underlying_id = self.db.instrument_id_for_symbol(
            self.trade.instrument.root_symbol
        )
        if underlying_id is None:
            return _empty_entries_df()

        universe = self.strategy.universe
        start_dt = datetime(
            universe.start_date.year, universe.start_date.month, universe.start_date.day,
            tzinfo=self._tz,
        ).astimezone(timezone.utc)
        end_dt = datetime(
            universe.end_date.year, universe.end_date.month, universe.end_date.day,
            23, 59, 59, tzinfo=self._tz,
        ).astimezone(timezone.utc)

        bars = self.db.underlying_bars(underlying_id, start_dt, end_dt)
        if bars.is_empty():
            return _empty_entries_df()

        indicators = self.db.indicators(underlying_id, start_dt, end_dt)

        candidates = self._apply_window_filters(bars)
        if candidates.is_empty():
            return _empty_entries_df()

        candidates = self._select_legs(candidates, underlying_id)
        if candidates.is_empty():
            return _empty_entries_df()

        candidates = self._compute_open_mark(candidates)
        candidates = self._evaluate_conditions(candidates, indicators)

        if candidates.is_empty():
            return _empty_entries_df()

        # Assign sequential entry_id (row index) and add trade name.
        return candidates.with_row_index("entry_id").with_columns(
            pl.lit(self.trade.name).alias("trade_name")
        )

    # ------------------------------------------------------------------
    # Step 1: Window + session filter
    # ------------------------------------------------------------------

    def _apply_window_filters(self, bars: pl.DataFrame) -> pl.DataFrame:
        """
        Fast first-pass filter on time alone — no DB access. Keeps only bars
        falling within: universe date range, session weekdays/skip_dates, and
        entry window (start_time to end_time in session timezone).
        """
        session = self.strategy.universe.session
        window = self.trade.entry.window
        tz_str = session.timezone

        # Convert ts_event (UTC) to session timezone for time-of-day comparisons.
        bars = bars.with_columns(
            pl.col("ts_event")
            .dt.convert_time_zone(tz_str)
            .alias("_ts_local")
        )

        # Filter: weekdays only
        if session.weekdays_only:
            bars = bars.filter(pl.col("_ts_local").dt.weekday() < 5)

        # Filter: skip_dates
        if session.skip_dates:
            bars = bars.filter(
                ~pl.col("_ts_local").dt.date().is_in(
                    [d for d in session.skip_dates]
                )
            )

        # Filter: entry window (time-of-day in session timezone).
        # Cast to Int32 before multiplication to prevent i8 overflow (3600 * 23 > 127).
        start_sec = window.start.hour * 3600 + window.start.minute * 60
        end_sec   = window.end.hour   * 3600 + window.end.minute   * 60
        _sec = (
            pl.col("_ts_local").dt.hour().cast(pl.Int32) * 3600
            + pl.col("_ts_local").dt.minute().cast(pl.Int32) * 60
        )
        bars = bars.filter((_sec >= start_sec) & (_sec <= end_sec))

        return bars.drop("_ts_local")

    # ------------------------------------------------------------------
    # Step 2: Leg selection
    # ------------------------------------------------------------------

    def _select_legs(
        self,
        candidates: pl.DataFrame,
        underlying_id: int,
    ) -> pl.DataFrame:
        """
        For each remaining candidate timestamp, issue a batched DuckDB query
        against option_greeks to find the best-matching option for each leg
        (minimise |actual_delta - target_delta| within dte_tolerance).
        Timestamps where any leg has no match within tolerance are dropped.
        """
        ts_events = candidates["ts_event"].to_list()

        result = candidates.select("ts_event")  # start narrow, join leg columns

        for leg in self.trade.legs:
            right = "C" if leg.right == "call" else "P"
            target_delta = float(leg.delta)
            target_dte   = int(leg.dte)

            leg_candidates = self.db.greeks_for_entry(
                underlying_id=underlying_id,
                ts_events=ts_events,
                right=right,
                target_delta=target_delta,
                target_dte=target_dte,
                delta_tolerance=0.10,
                dte_tolerance=5,
            )

            if leg_candidates.is_empty():
                return pl.DataFrame()  # no match for this leg → drop all

            # Pick best match per ts_event: minimum |actual_delta - target_delta|.
            best = (
                leg_candidates
                .with_columns(
                    (pl.col("delta") - target_delta).abs().alias("_delta_diff")
                )
                .sort(["ts_event", "_delta_diff"])
                .unique(subset=["ts_event"], keep="first")
                .drop("_delta_diff")
            )

            # Rename columns with leg prefix (except ts_event).
            rename_map = {
                col: f"leg_{leg.name}_{col}"
                for col in best.columns
                if col != "ts_event"
            }
            best = best.rename(rename_map)

            # Add leg metadata columns not returned by greeks_for_entry.
            best = best.with_columns([
                pl.lit(leg.action).alias(f"leg_{leg.name}_action"),
                pl.lit(leg.quantity).alias(f"leg_{leg.name}_quantity"),
            ])

            # Inner join: drops ts_events with no match for this leg.
            result = result.join(best, on="ts_event", how="inner")

        # Merge with bar OHLCV data (close, open for gap detection, etc.).
        bar_cols = candidates.select(["ts_event", "open", "high", "low", "close", "volume"])
        result = result.join(bar_cols, on="ts_event", how="left")

        return result

    # ------------------------------------------------------------------
    # Step 3: Open mark + TP/SL prices
    # ------------------------------------------------------------------

    def _compute_open_mark(self, entries: pl.DataFrame) -> pl.DataFrame:
        """
        Compute open_mark as the signed sum of leg prices:
            open_mark = Σ(leg_close × qty × sign)
        where sign = +1 for STO, -1 for BTO.

        Derive:
            tp_price = open_mark - take_profit
            sl_price = open_mark + stop_loss
        """
        # Build the open_mark expression as a sum of per-leg terms.
        mark_expr = pl.lit(0.0)
        for leg in self.trade.legs:
            sign = 1.0 if leg.action == "sell_to_open" else -1.0
            mark_expr = mark_expr + pl.col(f"leg_{leg.name}_close") * pl.lit(sign * leg.quantity)

        exit_cfg = self.trade.exit
        entries = entries.with_columns([
            mark_expr.alias("open_mark"),
        ])

        entries = entries.with_columns([
            (pl.col("open_mark") - pl.lit(float(exit_cfg.take_profit))).alias("tp_price"),
            (pl.col("open_mark") + pl.lit(float(exit_cfg.stop_loss))).alias("sl_price"),
            pl.lit(exit_cfg.dte_exit).cast(pl.Int32).alias("dte_exit"),
        ])

        # Rename ts_event to entry_time.
        entries = entries.rename({"ts_event": "entry_time"})

        return entries

    # ------------------------------------------------------------------
    # Step 4: Condition evaluation
    # ------------------------------------------------------------------

    def _evaluate_conditions(
        self,
        entries: pl.DataFrame,
        indicators: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Unified condition evaluation — runs after leg selection so all column
        namespaces are available.

        Steps (all vectorized):
            1. Join with indicators on entry_time
            2. Apply entry.conditions (AND logic — all must be true)
            3. Apply min_credit filter (open_mark >= min_credit)
            4. Apply max_debit filter (open_mark <= max_debit)
        """
        entry_cfg = self.trade.entry

        # Step 1: Join indicators (wide format, one column per indicator).
        if not indicators.is_empty() and "ts_event" in indicators.columns:
            # indicators uses ts_event as the key column
            entries = entries.join(
                indicators.rename({"ts_event": "entry_time"}),
                on="entry_time",
                how="left",
            )

        # Step 2: Condition expressions (AND logic).
        for cond_str in entry_cfg.conditions:
            expr = parse_condition(cond_str)
            # Condition may reference missing indicator columns → fill with False
            try:
                entries = entries.filter(expr)
            except Exception:
                # Column not found — condition cannot be evaluated → drop all
                return entries.clear()

        # Step 3: min_credit filter.
        if entry_cfg.min_credit is not None:
            entries = entries.filter(
                pl.col("open_mark") >= float(entry_cfg.min_credit)
            )

        # Step 4: max_debit filter.
        if entry_cfg.max_debit is not None:
            entries = entries.filter(
                pl.col("open_mark") <= float(entry_cfg.max_debit)
            )

        return entries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_entries_df() -> pl.DataFrame:
    """Return an empty DataFrame with the minimum required columns."""
    return pl.DataFrame({
        "entry_id":   pl.Series([], dtype=pl.UInt32),
        "trade_name": pl.Series([], dtype=pl.Utf8),
        "entry_time": pl.Series([], dtype=pl.Datetime("us", "UTC")),
        "open_mark":  pl.Series([], dtype=pl.Float64),
        "tp_price":   pl.Series([], dtype=pl.Float64),
        "sl_price":   pl.Series([], dtype=pl.Float64),
        "dte_exit":   pl.Series([], dtype=pl.Int32),
    })
