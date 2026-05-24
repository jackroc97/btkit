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
        root_symbol = self.trade.instrument.root_symbol
        roll_days   = self.trade.instrument.roll_days_before_expiry

        universe = self.strategy.universe
        start_dt = datetime(
            universe.start_date.year, universe.start_date.month, universe.start_date.day,
            tzinfo=self._tz,
        ).astimezone(timezone.utc)
        end_dt = datetime(
            universe.end_date.year, universe.end_date.month, universe.end_date.day,
            23, 59, 59, tzinfo=self._tz,
        ).astimezone(timezone.utc)

        # Build a roll schedule: date → underlying_id for the active front-month
        # futures contract.  This covers the full universe window so every
        # candidate bar gets the correct underlying regardless of roll events.
        schedule = self.db.front_future_schedule(
            root_symbol, universe.start_date, universe.end_date, roll_days
        )
        if schedule.is_empty():
            return _empty_entries_df()

        # Load bars from ALL matching futures contracts then keep only the
        # front-month bars for each date based on the roll schedule.
        all_bars = self.db.underlying_bars_for_root(root_symbol, start_dt, end_dt)
        if all_bars.is_empty():
            return _empty_entries_df()

        bars = (
            all_bars
            .with_columns(pl.col("ts_event").dt.date().alias("_date"))
            .join(schedule, left_on="_date", right_on="date", how="left")
            .filter(pl.col("instrument_id") == pl.col("underlying_id"))
            .drop(["_date", "underlying_id"])
        )
        if bars.is_empty():
            return _empty_entries_df()

        # Indicators: use the front-month at the start of the universe.
        start_underlying_id = self.db.front_future_id(
            root_symbol, universe.start_date, roll_days
        )
        indicators = self._get_indicators(start_underlying_id, start_dt, end_dt)

        candidates = self._apply_window_filters(bars)
        if candidates.is_empty():
            return _empty_entries_df()

        # Attach the active underlying_id to each candidate bar so _select_legs
        # can pass per-bar underlying IDs to the greeks queries.
        candidates = (
            candidates
            .with_columns(pl.col("ts_event").dt.date().alias("_date"))
            .join(schedule, left_on="_date", right_on="date", how="left")
            .drop("_date")
        )

        candidates = self._select_legs(candidates)
        if candidates.is_empty():
            return _empty_entries_df()

        candidates = self._compute_open_mark(candidates)
        candidates = self._evaluate_conditions(candidates, indicators)

        if candidates.is_empty():
            return _empty_entries_df()

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
    ) -> pl.DataFrame:
        """
        For each remaining candidate timestamp, find the best-matching option
        for every leg.

        candidates must contain an 'underlying_id' column (added by scan() from
        the roll schedule) so each bar's greeks query is scoped to the correct
        front-month futures contract.

        Legs are processed in two passes:
          Pass 1 — delta-selected legs: one batched greeks_for_all_legs() call,
            best match chosen by minimising |actual_delta - target_delta|.
          Pass 2 — strike-offset legs: for each such leg, compute the target
            strike from the reference leg's selected strike, then call
            greeks_for_strike_legs() and pick the closest available strike.

        Timestamps where any leg has no match are dropped via inner join.
        """
        ts_events      = candidates["ts_event"].to_list()
        underlying_ids = candidates["underlying_id"].to_list()
        ts_event_underlying = list(zip(ts_events, underlying_ids))

        delta_legs  = [leg for leg in self.trade.legs if leg.strike_offset is None]
        offset_legs = [leg for leg in self.trade.legs if leg.strike_offset is not None]

        result = candidates.select(["ts_event", "underlying_id"])

        # ------------------------------------------------------------------
        # Pass 1: delta-selected legs (single batched DB query)
        # ------------------------------------------------------------------
        if delta_legs:
            leg_specs = [
                {
                    "name":            leg.name,
                    "right":           "C" if leg.right == "call" else "P",
                    "target_delta":    float(leg.delta),
                    "target_dte":      int(leg.dte),
                    "delta_tolerance": float(leg.delta_tolerance),
                    "dte_tolerance":   int(leg.dte_tolerance),
                }
                for leg in delta_legs
            ]

            all_candidates = self.db.greeks_for_all_legs(
                ts_event_underlying=ts_event_underlying,
                leg_specs=leg_specs,
            )

            if all_candidates.is_empty():
                return pl.DataFrame()

            partitions: dict[str, pl.DataFrame] = {}
            for key, group_df in all_candidates.group_by("leg_name"):
                leg_name = key[0] if isinstance(key, (list, tuple)) else key
                partitions[leg_name] = group_df

            for leg in delta_legs:
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

        # ------------------------------------------------------------------
        # Pass 2: strike-offset legs (one DB query per leg)
        # ------------------------------------------------------------------
        for leg in offset_legs:
            ref_strike_col = f"leg_{leg.reference_leg}_strike_price"
            if ref_strike_col not in result.columns:
                return pl.DataFrame()

            right_char     = "C" if leg.right == "call" else "P"
            ref_expiry_col = f"leg_{leg.reference_leg}_expiration"

            strike_targets = result.select([
                "ts_event",
                "underlying_id",
                pl.lit(leg.name).alias("leg_name"),
                pl.lit(right_char).alias("right"),
                (pl.col(ref_strike_col) + pl.lit(float(leg.strike_offset))).alias("target_strike"),
                pl.col(ref_expiry_col).alias("reference_expiration"),
            ])

            offset_candidates = self.db.greeks_for_strike_legs(
                strike_targets=strike_targets,
            )

            if offset_candidates.is_empty():
                return pl.DataFrame()

            target_strikes = strike_targets.select(["ts_event", "target_strike"])
            best = (
                offset_candidates
                .join(target_strikes, on="ts_event", how="left")
                .with_columns(
                    (pl.col("strike_price") - pl.col("target_strike")).abs().alias("_strike_diff")
                )
                .sort(["ts_event", "_strike_diff"])
                .unique(subset=["ts_event"], keep="first")
                .drop(["_strike_diff", "leg_name", "underlying_id", "target_strike"])
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
        result = result.drop("underlying_id").join(bar_cols, on="ts_event", how="left")

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

        if exit_cfg.take_profit_pct is not None:
            # tp_price = open_mark × (1 - pct): exit when mark falls to this fraction
            tp_expr = (
                pl.col("open_mark") * pl.lit(1.0 - float(exit_cfg.take_profit_pct))
            ).alias("tp_price")
        else:
            tp_expr = (
                pl.col("open_mark") - pl.lit(float(exit_cfg.take_profit))
            ).alias("tp_price")

        entries = entries.with_columns([
            tp_expr,
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
        underlying_id: int | None,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pl.DataFrame:
        """
        Return the indicators DataFrame to use for condition evaluation.

        If the engine pre-loaded indicators, use those. Otherwise, fetch from DB
        only when the trade actually has entry conditions that might reference
        indicator columns. Returns an empty DataFrame when indicators are
        definitely not needed or underlying_id is unknown.
        """
        if self._preloaded_indicators is not None:
            return self._preloaded_indicators

        if not self.trade.entry.conditions or underlying_id is None:
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
