"""
EntryScanner — Pass 1 of the vectorized backtest.

Scans the session to find every valid entry signal for a single TradeDefinition,
selects option legs for each, computes the opening spread mark, and evaluates
entry conditions. Returns a DataFrame where each row is a fully-specified entry
ready for ExitScanner.

Pipeline within scan():
    1. _apply_window_filters()   — time/session filter (cheap, no DB access)
    2. _select_legs()            — single batched DuckDB query for all legs
    3. _compute_open_mark()      — spread mark + TP/SL price derivation
    4. _evaluate_conditions()    — conditions, min_credit/max_debit (vectorized)

Performance notes:
    - All legs are selected in a single DB roundtrip via greeks_for_all_legs().
    - A ts_event BETWEEN range filter lets the index prune the greeks scan before
      the join, dramatically reducing scan width at multi-year scale.
    - indicators may be pre-loaded by the engine and passed in to avoid a second
      DB fetch; if None and conditions reference indicators, they are fetched here.
    - Per-leg greek columns (iv, gamma, theta, vega, dte, delta) are dropped after
      condition evaluation — they are not needed by ExitScanner or PnLCalculator.

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

# Leg-level greek columns that are safe to drop after conditions are evaluated.
# ExitScanner and PnLCalculator do not use these.
_LEG_GREEK_SUFFIXES = ("iv", "gamma", "theta", "vega", "dte", "delta")


class EntryScanner:
    def __init__(
        self,
        db: InputDatabase,
        strategy: StrategyDefinition,
        trade: TradeDefinition,
        indicators: pl.DataFrame | None = None,
    ) -> None:
        self.db = db
        self.strategy = strategy
        self.trade = trade
        self._tz = ZoneInfo(strategy.universe.session.timezone)
        self._preloaded_indicators = indicators
        self.warnings: list[dict] = []

    def scan(self, entry_id_offset: int = 0) -> pl.DataFrame:
        """
        Run the full entry scan and return one row per valid entry.

        entry_id_offset is added to each row's entry_id so that IDs are globally
        unique across trades when BacktestEngine processes multiple trades.

        Returned columns:
            entry_id, trade_name, entry_time, open_mark, tp_price, sl_price,
            dte_exit (int, nullable),
            + per leg: leg_{name}_instrument_id, leg_{name}_open_price,
                       leg_{name}_multiplier, leg_{name}_strike_price,
                       leg_{name}_expiration, leg_{name}_right,
                       leg_{name}_action, leg_{name}_quantity,
                       leg_{name}_symbol, leg_{name}_close
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

        # Use pre-loaded indicators when available; fetch only if conditions need them.
        indicators = self._get_indicators(underlying_id, start_dt, end_dt)

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

        # Drop per-leg greek columns — not needed by ExitScanner or PnLCalculator.
        cols_to_drop = [
            f"leg_{leg.name}_{suffix}"
            for leg in self.trade.legs
            for suffix in _LEG_GREEK_SUFFIXES
            if f"leg_{leg.name}_{suffix}" in candidates.columns
        ]
        if cols_to_drop:
            candidates = candidates.drop(cols_to_drop)

        return (
            candidates
            .with_row_index("entry_id", offset=entry_id_offset)
            .with_columns(pl.lit(self.trade.name).alias("trade_name"))
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

        bars = bars.with_columns(
            pl.col("ts_event")
            .dt.convert_time_zone(tz_str)
            .alias("_ts_local")
        )

        if session.weekdays_only:
            bars = bars.filter(pl.col("_ts_local").dt.weekday() < 5)

        if session.skip_dates:
            bars = bars.filter(
                ~pl.col("_ts_local").dt.date().is_in(
                    [d for d in session.skip_dates]
                )
            )

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
    # Step 2: Leg selection — single batched DB query for all legs
    # ------------------------------------------------------------------

    def _select_legs(
        self,
        candidates: pl.DataFrame,
        underlying_id: int,
    ) -> pl.DataFrame:
        """
        For each remaining candidate timestamp, find the best-matching option
        for every leg simultaneously in a single DB query.

        Previously called greeks_for_entry() once per leg (N roundtrips). Now
        calls greeks_for_all_legs() once regardless of leg count. The best
        match per (ts_event, leg) is chosen in Polars by minimising
        |actual_delta - target_delta|. Timestamps where any leg has no match
        within tolerance are dropped via inner join.
        """
        ts_events = candidates["ts_event"].to_list()

        leg_specs = [
            {
                "name":         leg.name,
                "right":        "C" if leg.right == "call" else "P",
                "target_delta": float(leg.delta),
                "target_dte":   int(leg.dte),
            }
            for leg in self.trade.legs
        ]

        all_candidates = self.db.greeks_for_all_legs(
            underlying_id=underlying_id,
            ts_events=ts_events,
            leg_specs=leg_specs,
            delta_tolerance=0.10,
            dte_tolerance=5,
        )

        if all_candidates.is_empty():
            return pl.DataFrame()

        # Partition once by leg_name rather than filtering N times (O(rows) vs
        # O(rows × legs)). group_by returns string or tuple keys depending on
        # Polars version, so normalise to str before building the dict.
        partitions: dict[str, pl.DataFrame] = {}
        for key, group_df in all_candidates.group_by("leg_name"):
            leg_name = key[0] if isinstance(key, (list, tuple)) else key
            partitions[leg_name] = group_df

        result = candidates.select("ts_event")

        for leg in self.trade.legs:
            target_delta = float(leg.delta)
            leg_df = partitions.get(leg.name)

            if leg_df is None or leg_df.is_empty():
                return pl.DataFrame()

            best = (
                leg_df
                .with_columns(
                    (pl.col("delta") - target_delta).abs().alias("_delta_diff")
                )
                .sort(["ts_event", "_delta_diff"])
                .unique(subset=["ts_event"], keep="first")
                .drop(["_delta_diff", "leg_name"])
            )

            rename_map = {
                col: f"leg_{leg.name}_{col}"
                for col in best.columns
                if col != "ts_event"
            }
            best = best.rename(rename_map).with_columns([
                pl.lit(leg.action).alias(f"leg_{leg.name}_action"),
                pl.lit(leg.quantity).alias(f"leg_{leg.name}_quantity"),
            ])

            result = result.join(best, on="ts_event", how="inner")

        bar_cols = candidates.select(["ts_event", "open", "high", "low", "close", "volume"])
        result = result.join(bar_cols, on="ts_event", how="left")

        return result

    # ------------------------------------------------------------------
    # Step 3: Open mark + TP/SL prices
    # ------------------------------------------------------------------

    def _compute_open_mark(self, entries: pl.DataFrame) -> pl.DataFrame:
        mark_expr = pl.lit(0.0)
        for leg in self.trade.legs:
            sign = 1.0 if leg.action == "sell_to_open" else -1.0
            mark_expr = mark_expr + pl.col(f"leg_{leg.name}_close") * pl.lit(sign * leg.quantity)

        exit_cfg = self.trade.exit
        entries = entries.with_columns([mark_expr.alias("open_mark")])
        entries = entries.with_columns([
            (pl.col("open_mark") - pl.lit(float(exit_cfg.take_profit))).alias("tp_price"),
            (pl.col("open_mark") + pl.lit(float(exit_cfg.stop_loss))).alias("sl_price"),
            pl.lit(exit_cfg.dte_exit).cast(pl.Int32).alias("dte_exit"),
        ])

        return entries.rename({"ts_event": "entry_time"})

    # ------------------------------------------------------------------
    # Step 4: Condition evaluation
    # ------------------------------------------------------------------

    def _evaluate_conditions(
        self,
        entries: pl.DataFrame,
        indicators: pl.DataFrame,
    ) -> pl.DataFrame:
        entry_cfg = self.trade.entry

        if not indicators.is_empty() and "ts_event" in indicators.columns:
            entries = entries.join(
                indicators.rename({"ts_event": "entry_time"}),
                on="entry_time",
                how="left",
            )

        for cond_str in entry_cfg.conditions:
            expr = parse_condition(cond_str)
            try:
                entries = entries.filter(expr)
            except Exception as e:
                self.warnings.append({
                    "phase": "entry",
                    "trade": self.trade.name,
                    "type": "condition_error",
                    "condition": cond_str,
                    "error": str(e),
                })
                return entries.clear()

        if entry_cfg.min_credit is not None:
            entries = entries.filter(
                pl.col("open_mark") >= float(entry_cfg.min_credit)
            )

        if entry_cfg.max_debit is not None:
            entries = entries.filter(
                pl.col("open_mark") <= float(entry_cfg.max_debit)
            )

        return entries

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_indicators(
        self,
        underlying_id: int,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pl.DataFrame:
        """
        Return the indicators DataFrame to use for condition evaluation.

        If the engine pre-loaded indicators, use those. Otherwise, fetch from DB
        only when the trade actually has entry conditions that might reference
        indicator columns. Returns an empty DataFrame when indicators are
        definitely not needed.
        """
        if self._preloaded_indicators is not None:
            return self._preloaded_indicators

        if not self.trade.entry.conditions:
            return pl.DataFrame()

        return self.db.indicators(underlying_id, start_dt, end_dt)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_entries_df() -> pl.DataFrame:
    return pl.DataFrame({
        "entry_id":   pl.Series([], dtype=pl.UInt32),
        "trade_name": pl.Series([], dtype=pl.Utf8),
        "entry_time": pl.Series([], dtype=pl.Datetime("us", "UTC")),
        "open_mark":  pl.Series([], dtype=pl.Float64),
        "tp_price":   pl.Series([], dtype=pl.Float64),
        "sl_price":   pl.Series([], dtype=pl.Float64),
        "dte_exit":   pl.Series([], dtype=pl.Int32),
    })
