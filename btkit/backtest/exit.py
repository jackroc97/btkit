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
    6. Roll (close + re-open at fresh delta)
    7. Vega exit (spread net vega threshold)
    8. DTE exit
    9. Expiry

Fill price rules: docs/fill_price_and_costs.md.

Performance notes:
    - Entries are chunked by expiration tuple before calling _load_exit_data().
      Each chunk has a bounded set of instrument IDs and a short time window,
      keeping option_bars_for_legs queries tractable at multi-year scale.
    - indicators may be pre-loaded by the engine and passed in to avoid a second
      DB fetch per trade.
    - worst_mark is computed in a single pass via cum_max().over("entry_id")
      during _find_first_hit(), eliminating the previous second full scan of
      position_marks.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from btkit.backtest._util import tick_round_expr
from btkit.db.input_db import InputDatabase
from btkit.strategy.definition import (
    StopLossConfig,
    StrategyDefinition,
    TakeProfitConfig,
    TradeDefinition,
)
from btkit.strategy.loader import parse_condition


class ExitScanner:
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
        self._preloaded_indicators = indicators
        self.warnings: list[dict] = []
        self.continuation_exits: pl.DataFrame = _empty_continuation_df()
        # Settlement caches — populated once per scan() call, consumed by
        # _compute_settlement_marks() for every cohort.  Avoids O(cohorts ×
        # expiry_dates) individual underlying_bars queries.
        self._opt_to_underlying: dict[int, int] = {}
        self._settlement_closes_by_key: dict[tuple, float] = {}

    def scan(self, entries: pl.DataFrame) -> pl.DataFrame:
        """
        For each entry row, find the first exit event.

        Entries are chunked by their expiration tuple before processing. Within
        one expiration cycle, the set of unique leg instrument IDs is small and
        the monitoring window is bounded, keeping memory usage constant
        regardless of overall backtest length.

        Returns one row per entry with columns:
            entry_id, exit_time, exit_mark, worst_mark, exit_reason
        """
        if entries.is_empty():
            return _empty_exits_df()

        exp_cols = [f"leg_{leg.name}_expiration" for leg in self.trade.legs]
        entries_bucketed = entries.with_columns(
            pl.min_horizontal(exp_cols).dt.strftime("%Y-%m").alias("_exp_bucket")
        )

        # Pre-load all leg bars in a single DB query instead of one per bucket.
        # This cuts option_bars_for_legs call count from O(months) to 1, trading
        # a slightly larger result set (~67 MB for 4 years) for a 5× reduction in
        # query overhead. Per-bucket filtering then happens in Polars.
        all_ids: set[int] = set()
        for leg in self.trade.legs:
            all_ids.update(entries[f"leg_{leg.name}_instrument_id"].to_list())
        global_min_entry: datetime = entries["entry_time"].min()
        global_max_exp = None
        for leg in self.trade.legs:
            leg_max = entries[f"leg_{leg.name}_expiration"].max()
            if global_max_exp is None or leg_max > global_max_exp:
                global_max_exp = leg_max
        global_end_dt = datetime(
            global_max_exp.year,
            global_max_exp.month,
            global_max_exp.day,
            23,
            59,
            59,
            tzinfo=UTC,
        )
        all_option_bars = self.db.option_bars_for_legs(
            list(all_ids), global_min_entry, global_end_dt
        )

        # Pre-load settlement prices for the full backtest window in two queries:
        # one to resolve option → underlying mappings, one to fetch all settlement
        # closes. _compute_settlement_marks() then does pure dict lookups, avoiding
        # O(cohorts × expiry_dates) individual underlying_bars queries per scan().
        close_time = self.trade.instrument.expiry_close_time
        if close_time is not None:
            all_opt_ids: set[int] = set()
            for leg in self.trade.legs:
                all_opt_ids.update(entries[f"leg_{leg.name}_instrument_id"].to_list())
            self._opt_to_underlying = self.db.underlying_ids_for_options(list(all_opt_ids))
            unique_underlying_ids = list(set(self._opt_to_underlying.values()))
            tz_str_pre = self.strategy.universe.session.timezone
            settlement_df = self.db.settlement_closes_for_underlyings(
                unique_underlying_ids,
                global_min_entry,
                global_end_dt,
                tz_str_pre,
                close_time,
            )
            self._settlement_closes_by_key = {
                (int(row["underlying_id"]), row["exp_date"]): float(row["settlement_close"])
                for row in settlement_df.iter_rows(named=True)
            }
        else:
            self._opt_to_underlying = {}
            self._settlement_closes_by_key = {}

        chunks: list[pl.DataFrame] = []
        for _, cohort in entries_bucketed.group_by("_exp_bucket", maintain_order=True):
            cohort = cohort.drop("_exp_bucket")
            option_bars, indicators = self._load_exit_data(cohort, all_option_bars)
            if option_bars.is_empty():
                continue
            position_marks = self._compute_position_marks(option_bars, cohort)
            if position_marks.is_empty():
                continue
            exits_chunk = self._find_first_hit(position_marks, indicators, cohort)
            if not exits_chunk.is_empty():
                if self.trade.exit.leg_out:
                    exits_chunk = self._adjust_leg_out_exits(exits_chunk, cohort, option_bars)
                chunks.append(exits_chunk)

        if not chunks:
            return _empty_exits_df()

        all_exits = pl.concat(chunks)

        if self.trade.exit.on_sl_long_continuation:
            self.continuation_exits = self.scan_long_continuation(
                all_exits, entries, all_option_bars
            )

        return all_exits

    # ------------------------------------------------------------------
    # Step 1: Batch-load all data for a cohort of open positions
    # ------------------------------------------------------------------

    def _load_exit_data(
        self,
        entries: pl.DataFrame,
        cached_bars: pl.DataFrame | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Batch-load all data needed to monitor the open positions in this cohort.
        Returns:
          - option_bars: leg bars from earliest entry_time to latest expiration.
          - indicators: wide indicator DataFrame for the same window.

        If cached_bars is supplied (pre-loaded by scan()), this method filters it
        in Polars instead of issuing a new DB query — avoids O(months) round-trips.
        """
        instrument_ids: set[int] = set()
        for leg in self.trade.legs:
            ids = entries[f"leg_{leg.name}_instrument_id"].to_list()
            instrument_ids.update(ids)

        min_entry: datetime = entries["entry_time"].min()

        max_exp = None
        for leg in self.trade.legs:
            leg_max = entries[f"leg_{leg.name}_expiration"].max()
            if max_exp is None or leg_max > max_exp:
                max_exp = leg_max

        end_dt = datetime(
            max_exp.year,
            max_exp.month,
            max_exp.day,
            23,
            59,
            59,
            tzinfo=UTC,
        )

        if cached_bars is not None:
            # ts_event in cached_bars may differ from end_dt's timezone (UTC).
            # Convert end_dt to match so Polars accepts the comparison.
            ts_tz = cached_bars.schema["ts_event"].time_zone
            if ts_tz and ts_tz != "UTC":
                from zoneinfo import ZoneInfo

                end_dt_cmp = end_dt.astimezone(ZoneInfo(ts_tz))
            else:
                end_dt_cmp = end_dt
            option_bars = cached_bars.filter(
                pl.col("instrument_id").is_in(list(instrument_ids))
                & (pl.col("ts_event") >= min_entry)
                & (pl.col("ts_event") <= end_dt_cmp)
            )
        else:
            option_bars = self.db.option_bars_for_legs(list(instrument_ids), min_entry, end_dt)

        indicators = self._get_indicators(min_entry, end_dt)

        return option_bars, indicators

    # ------------------------------------------------------------------
    # Step 2: Compute position mark per bar per entry
    # ------------------------------------------------------------------

    def _compute_position_marks(
        self,
        option_bars: pl.DataFrame,
        entries: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        For each entry, compute position_mark and spread_open_mark at each
        subsequent bar using the legs selected at entry time.

        position_mark    = Σ(leg_close × signed_qty)  — used for TP/SL detection
        spread_open_mark = Σ(leg_open  × signed_qty)  — used for gap-open detection

        When liquidity config requires it, also produces:
          _min_leg_volume  — min contracts traded across legs (0 when any leg has no bar)
          _total_slippage  — Σ(|high - low| / 2) per leg — half bid-ask spread estimate

        Option bars are sparse (only when traded), so different legs rarely share
        the same ts_event. We full-outer-join all legs on (entry_id, ts_event) and
        forward-fill stale prices within each entry_id. Volume and spread columns
        are gap-filled with 0 (a missing bar means zero trading activity).
        """
        liq = self.trade.exit.liquidity
        need_volume = liq.needs_volume
        need_spread = liq.needs_spread
        need_staleness = liq.needs_staleness

        per_leg: list[pl.DataFrame] = []

        for leg in self.trade.legs:
            sign = 1.0 if leg.action == "sell_to_open" else -1.0
            signed_qty = sign * float(leg.quantity)

            leg_map = entries.select(
                [
                    "entry_id",
                    "entry_time",
                    pl.col(f"leg_{leg.name}_instrument_id").alias("instrument_id"),
                ]
            )

            # Build the per-leg select expression list
            bar_exprs: list[pl.Expr] = [
                pl.col("ts_event"),
                pl.col("instrument_id"),
                (pl.col("close") * pl.lit(signed_qty)).alias(f"_leg_{leg.name}_mark_close"),
                (pl.col("open") * pl.lit(signed_qty)).alias(f"_leg_{leg.name}_mark_open"),
            ]
            if need_volume:
                bar_exprs.append(pl.col("volume").alias(f"_leg_{leg.name}_volume"))
            if need_spread:
                # Half the high-low range is a per-leg slippage cost (unsigned)
                bar_exprs.append(
                    ((pl.col("high") - pl.col("low")) / 2.0).alias(f"_leg_{leg.name}_spread_half")
                )

            src_cols = ["ts_event", "instrument_id", "open", "close"]
            if need_volume:
                src_cols.append("volume")
            if need_spread:
                src_cols.extend(["high", "low"])

            leg_bars = (
                option_bars.select(src_cols)
                .join(leg_map, on="instrument_id", how="inner")
                .filter(pl.col("ts_event") > pl.col("entry_time"))
                .select(["entry_id"] + [e for e in bar_exprs if not isinstance(e, str)])
            )

            if leg_bars.is_empty():
                return pl.DataFrame()

            per_leg.append(leg_bars)

        if not per_leg:
            return pl.DataFrame()

        result = per_leg[0]
        for lb in per_leg[1:]:
            result = result.join(lb, on=["entry_id", "ts_event"], how="full", coalesce=True)

        close_cols = [f"_leg_{leg.name}_mark_close" for leg in self.trade.legs]
        open_cols = [f"_leg_{leg.name}_mark_open" for leg in self.trade.legs]
        vol_cols = [f"_leg_{leg.name}_volume" for leg in self.trade.legs] if need_volume else []
        spread_cols = (
            [f"_leg_{leg.name}_spread_half" for leg in self.trade.legs] if need_spread else []
        )

        # Before forward-filling prices, record the last timestamp where each leg
        # had a real bar. This is used by the staleness gate to suppress TP/SL when
        # a leg's price is stale beyond max_leg_stale_minutes.
        if need_staleness:
            last_bar_ts_exprs: list[pl.Expr] = []
            for leg in self.trade.legs:
                close_col = f"_leg_{leg.name}_mark_close"
                last_bar_ts_exprs.append(
                    pl.when(pl.col(close_col).is_not_null())
                    .then(pl.col("ts_event").dt.convert_time_zone("UTC"))
                    .otherwise(pl.lit(None).cast(pl.Datetime("us", "UTC")))
                    .forward_fill()
                    .over("entry_id")
                    .alias(f"_leg_{leg.name}_last_bar_ts")
                )
            result = result.sort(["entry_id", "ts_event"]).with_columns(last_bar_ts_exprs)

        fill_exprs: list[pl.Expr] = [
            # Prices: forward-fill stale values from the last real bar
            *[pl.col(c).forward_fill().over("entry_id") for c in close_cols],
            *[pl.col(c).forward_fill().over("entry_id") for c in open_cols],
            # Volume / spread: null means no bar was recorded → zero activity, not forward-filled
            *[pl.col(c).fill_null(0) for c in vol_cols],
            *[pl.col(c).fill_null(0) for c in spread_cols],
        ]

        result = result.sort(["entry_id", "ts_event"]).with_columns(fill_exprs)
        result = result.drop_nulls(subset=close_cols + open_cols)

        if result.is_empty():
            return pl.DataFrame()

        aggregate_exprs: list[pl.Expr] = [
            pl.sum_horizontal(close_cols).alias("position_mark"),
            pl.sum_horizontal(open_cols).alias("spread_open_mark"),
        ]
        if need_volume:
            # With leg_out, long legs fill independently after the short leg exits, so
            # sparse long-leg volume should not suppress the TP/SL trigger. Use only
            # short (STO) leg volumes; fall back to all legs if none are STO.
            if self.trade.exit.leg_out:
                short_vol_cols = [
                    f"_leg_{leg.name}_volume"
                    for leg in self.trade.legs
                    if leg.action == "sell_to_open"
                ]
                active_vol_cols = short_vol_cols if short_vol_cols else vol_cols
            else:
                active_vol_cols = vol_cols
            aggregate_exprs.append(
                pl.min_horizontal(active_vol_cols).fill_null(0).alias("_min_leg_volume")
            )
        if need_spread:
            aggregate_exprs.append(
                pl.sum_horizontal(spread_cols).fill_null(0).alias("_total_slippage")
            )
        if need_staleness:
            stale_exprs = [
                (
                    pl.col("ts_event").dt.convert_time_zone("UTC")
                    - pl.col(f"_leg_{leg.name}_last_bar_ts")
                )
                .dt.total_minutes()
                .cast(pl.Float64)
                for leg in self.trade.legs
            ]
            aggregate_exprs.append(
                pl.max_horizontal(stale_exprs).fill_null(0.0).alias("_max_leg_stale_minutes")
            )

        out_cols = ["entry_id", "ts_event", "position_mark", "spread_open_mark"]
        if need_volume:
            out_cols.append("_min_leg_volume")
        if need_spread:
            out_cols.append("_total_slippage")
        if need_staleness:
            out_cols.append("_max_leg_stale_minutes")

        return result.with_columns(aggregate_exprs).select(out_cols)

    # ------------------------------------------------------------------
    # Step 3: Find first exit hit per entry
    # ------------------------------------------------------------------

    def _parse_trigger_condition(self, condition: str | None, trigger: str) -> pl.Expr:
        """Parse an AND-gate condition for stop_loss/take_profit; lit(True) when absent."""
        if condition is None:
            return pl.lit(True)
        try:
            return parse_condition(condition)
        except Exception as e:
            self.warnings.append(
                {
                    "phase": "exit",
                    "trade": self.trade.name,
                    "type": "condition_error",
                    "trigger": trigger,
                    "condition": condition,
                    "error": str(e),
                }
            )
            return pl.lit(True)

    def _find_first_hit(
        self,
        position_marks: pl.DataFrame,
        indicators: pl.DataFrame,
        entries: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Tag each bar with exit flags, assign priority, pick earliest exit per
        entry, then compute worst_mark (max position_mark from entry to exit).

        Priority encoding: 1=gap_sl, 2=gap_tp, 3=stop_loss, 4=take_profit,
                           5=condition, 6=dte_exit, 7=expiry.

        worst_mark is computed in a single pass via cum_max().over("entry_id"),
        avoiding the previous second full scan of position_marks.

        Fill price (before slippage):
            gap_sl / gap_tp       → spread_open_mark
            stop_loss (close)     → sl_price
            take_profit (close)   → tp_price
            condition / dte / exp → position_mark (bar close)

        Liquidity adjustments (when liquidity config is non-default):
            volume gate      — price-triggered exits blocked when rolling volume
                               over the lookback window is below min_exit_volume.
                               Missing bars count as zero (gap-robust).
            pre-expiry lock  — price-triggered exits suppressed in the final N
                               minutes before the instrument's expiry close on
                               expiry day (DTE=0). Expiry/DTE exits unaffected.
            staleness gate   — price-triggered exits suppressed when any leg's
                               last real bar is older than max_leg_stale_minutes.
                               Prevents forward-filled stale prices from creating
                               artificially compressed spread marks that fire
                               spurious TP exits. Expiry/DTE exits unaffected.
            no-arb guard     — price-triggered exits suppressed when position_mark
                               violates the no-arbitrage bound implied by open_mark:
                               credit spreads (open_mark > 0) require mark ≥ 0;
                               debit spreads (open_mark < 0) require mark ≤ 0.
                               Catches stale-price artifacts that slip through the
                               staleness gate window. Bar-price exit marks and
                               fallback expiry marks are clipped to the same bound.
            spread slippage  — fill price worsened by Σ(|high-low|/2) per leg,
                               modelling the cost of crossing the bid-ask spread.
        """
        exit_cfg = self.trade.exit
        liq = exit_cfg.liquidity

        # Needed early for the post-expiration filter and DTE/expiry calculations.
        tz_str = self.strategy.universe.session.timezone

        # Compute intrinsic settlement values before building entry_meta so they
        # can be embedded as a per-entry column and flow through the scan.
        settlement_marks = self._compute_settlement_marks(entries, tz_str)

        exp_cols = [f"leg_{leg.name}_expiration" for leg in self.trade.legs]
        entry_meta = (
            entries.select(
                ["entry_id", "entry_time", "open_mark", "tp_price", "sl_price", "dte_exit"]
                + exp_cols
            )
            .with_columns(pl.min_horizontal(exp_cols).alias("trade_expiration"))
            .select(
                [
                    "entry_id",
                    "entry_time",
                    "open_mark",
                    "tp_price",
                    "sl_price",
                    "dte_exit",
                    "trade_expiration",
                ]
            )
            .join(settlement_marks, on="entry_id", how="left")
        )

        # Sort once; used by both forward-fill (already done) and cum_max below.
        m = position_marks.sort(["entry_id", "ts_event"]).join(
            entry_meta, on="entry_id", how="left"
        )

        # Discard any bar whose session date is strictly after the trade's expiration.
        # Input data sources can reuse instrument IDs across expiration cycles: the
        # same ID that represented an option expiring on day T may later appear with
        # bars for a completely different contract on day T+N. Without this guard
        # those phantom bars corrupt position marks, produce impossible exit_marks
        # (e.g. -167 on a 50-pt-wide spread), and push the fallback expiry exit
        # weeks past the true expiration date.
        m = m.filter(
            pl.col("ts_event").dt.convert_time_zone(tz_str).dt.date() <= pl.col("trade_expiration")
        )

        # Running worst mark per entry — single pass, no second scan needed.
        m = m.with_columns(
            pl.col("position_mark").cum_max().over("entry_id").alias("_running_worst")
        )

        if not indicators.is_empty() and len(indicators.columns) > 1:
            m = m.join(indicators, on="ts_event", how="left")

        # ── Spread vega ───────────────────────────────────────────────────
        # Compute net vega = Σ(signed_qty × vega) per (entry_id, ts_event).
        # Required when vega_exit or roll.vega is configured. Fetched from
        # option_greeks over the cohort window; one query per cohort.
        _need_vega = (
            exit_cfg.vega_exit is not None
            or (self.trade.roll is not None and self.trade.roll.vega is not None)
            or any("_spread_vega" in c or "open_vega" in c for c in exit_cfg.conditions)
            or (
                self.trade.roll is not None
                and any("_spread_vega" in c or "open_vega" in c for c in self.trade.roll.conditions)
            )
        )
        if _need_vega:
            cohort_start = m["ts_event"].min()
            cohort_end = m["ts_event"].max()
            instr_ids: list[int] = []
            for leg in self.trade.legs:
                col_name = f"leg_{leg.name}_instrument_id"
                if col_name in entries.columns:
                    instr_ids.extend(
                        int(v) for v in entries[col_name].to_list() if v is not None and int(v) > 0
                    )
            instr_ids = list(set(instr_ids))

            if instr_ids and cohort_start is not None and cohort_end is not None:
                greeks_df = self.db.option_greeks_for_legs(instr_ids, cohort_start, cohort_end)
                if not greeks_df.is_empty():
                    vega_parts: list[pl.DataFrame] = []
                    for leg in self.trade.legs:
                        sign = 1.0 if leg.action == "sell_to_open" else -1.0
                        signed_qty = sign * float(leg.quantity)
                        col_name = f"leg_{leg.name}_instrument_id"
                        if col_name not in entries.columns:
                            continue
                        leg_instr_map = entries.select(
                            [
                                "entry_id",
                                pl.col(col_name).alias("instrument_id"),
                            ]
                        )
                        part = greeks_df.join(
                            leg_instr_map, on="instrument_id", how="inner"
                        ).select(
                            [
                                "entry_id",
                                "ts_event",
                                (pl.col("vega") * pl.lit(signed_qty)).alias("_leg_vega_contrib"),
                            ]
                        )
                        vega_parts.append(part)

                    if vega_parts:
                        spread_vega = (
                            pl.concat(vega_parts)
                            .group_by(["entry_id", "ts_event"])
                            .agg(pl.col("_leg_vega_contrib").sum().alias("_spread_vega"))
                        )
                        m = m.join(spread_vega, on=["entry_id", "ts_event"], how="left")

                        # open_vega: spread vega at the first bar at or after entry_time.
                        # Exposed to exit conditions as a per-entry constant so users can
                        # write relative thresholds, e.g. "_spread_vega < 0.3 * open_vega".
                        open_vega_df = (
                            m.filter(
                                pl.col("_spread_vega").is_not_null()
                                & (pl.col("ts_event") >= pl.col("entry_time"))
                            )
                            .sort(["entry_id", "ts_event"])
                            .group_by("entry_id", maintain_order=True)
                            .agg(pl.col("_spread_vega").first().alias("open_vega"))
                        )
                        m = m.join(open_vega_df, on="entry_id", how="left")

        m = m.with_columns(
            (
                pl.col("trade_expiration").cast(pl.Date)
                - pl.col("ts_event").dt.convert_time_zone(tz_str).dt.date()
            )
            .dt.total_days()
            .cast(pl.Int32)
            .alias("_dte_now")
        )

        # Materialise local-time seconds once; used by session mask, expiry
        # exit, and the pre-expiry lock so we avoid repeated tz conversions.
        m = m.with_columns(
            (
                pl.col("ts_event").dt.convert_time_zone(tz_str).dt.hour().cast(pl.Int32) * 3600
                + pl.col("ts_event").dt.convert_time_zone(tz_str).dt.minute().cast(pl.Int32) * 60
            ).alias("_local_sec")
        )

        # Session mask: only allow price-triggered exits (SL/TP/gap) during
        # configured session hours so illiquid after-hours option prices don't
        # fire stops. DTE and expiry exits are intentionally unconstrained.
        # allow_after_hours_exits bypasses the time window while still
        # respecting the weekdays_only constraint.
        session = self.strategy.universe.session
        if (
            session.start_time is not None
            and session.end_time is not None
            and not self.trade.exit.allow_after_hours_exits
        ):
            sess_start_sec = session.start_time.hour * 3600 + session.start_time.minute * 60
            sess_end_sec = session.end_time.hour * 3600 + session.end_time.minute * 60
            _in_session = (pl.col("_local_sec") >= pl.lit(sess_start_sec)) & (
                pl.col("_local_sec") <= pl.lit(sess_end_sec)
            )
            if session.weekdays_only:
                _in_session = _in_session & (
                    pl.col("ts_event").dt.convert_time_zone(tz_str).dt.weekday() < 5
                )
        elif session.weekdays_only:
            _in_session = pl.col("ts_event").dt.convert_time_zone(tz_str).dt.weekday() < 5
        else:
            _in_session = pl.lit(True)

        # ── Feature 1: volume gate ────────────────────────────────────────
        # Time-based rolling sum of min-leg-volume over the lookback window.
        # Missing bars contribute 0 (gap-robust by construction: only rows
        # that exist in the DataFrame are summed).
        if liq.needs_volume and "_min_leg_volume" in m.columns:
            lookback_str = f"{liq.lookback_minutes}m"

            def _add_rolling_volume(df: pl.DataFrame) -> pl.DataFrame:
                return df.sort("ts_event").with_columns(
                    pl.col("_min_leg_volume")
                    .rolling_sum_by("ts_event", window_size=lookback_str)
                    .alias("_rolling_volume")
                )

            m = m.group_by("entry_id", maintain_order=True).map_groups(_add_rolling_volume)
            _liquid = pl.col("_rolling_volume") >= pl.lit(liq.min_exit_volume)
        else:
            _liquid = pl.lit(True)

        # ── Feature 2: pre-expiry lock ────────────────────────────────────
        # Suppress price-triggered exits in the final N minutes of expiry day.
        # Uses expiry_close_time if set, otherwise falls back to session end.
        if liq.pre_expiry_lock_minutes is not None:
            close_time = self.trade.instrument.expiry_close_time
            if close_time is not None:
                ref_close_sec = close_time.hour * 3600 + close_time.minute * 60
            else:
                ref_close_sec = (
                    session.end_time.hour * 3600 + session.end_time.minute * 60
                    if session.end_time is not None
                    else 16 * 3600  # fallback: 16:00 local
                )
            lock_start_sec = ref_close_sec - liq.pre_expiry_lock_minutes * 60
            _locked = (
                (pl.col("_dte_now") == 0)
                & (pl.col("_local_sec") >= pl.lit(lock_start_sec))
                & (pl.col("_local_sec") <= pl.lit(ref_close_sec))
            )
        else:
            _locked = pl.lit(False)

        # ── Feature 3: staleness gate ─────────────────────────────────────
        # Suppress price-triggered exits when any leg's last real bar is older
        # than max_leg_stale_minutes. Forward-filling stale prices creates
        # artificially compressed spread marks that trigger spurious TP exits.
        if liq.needs_staleness and "_max_leg_stale_minutes" in m.columns:
            _fresh = pl.col("_max_leg_stale_minutes") <= pl.lit(float(liq.max_leg_stale_minutes))
        else:
            _fresh = pl.lit(True)

        # ── No-arbitrage guard ────────────────────────────────────────────
        # A credit spread (open_mark > 0) cannot have a negative position mark;
        # a debit spread (open_mark < 0) cannot have a positive one. Either
        # condition means a stale leg price has crossed the no-arbitrage boundary.
        # Block price-triggered exits at such bars — the bar-price exit marks
        # and fallback marks are also clipped to the same bound below.
        _no_arb_ok = (
            pl.when(pl.col("open_mark") > 0)
            .then(pl.col("position_mark") >= 0)
            .when(pl.col("open_mark") < 0)
            .then(pl.col("position_mark") <= 0)
            .otherwise(pl.lit(True))
        )

        # Combined guard applied to all price-triggered exits
        _price_ok = _in_session & _liquid & ~_locked & _fresh & _no_arb_ok

        # Slippage term used in trigger conditions. When slippage_model=spread,
        # this shifts each trigger by the estimated bid-ask crossing cost so that
        # TP only fires when the executable spread (not just the midpoint) meets
        # the threshold. TP is tightened (midpoint must compress further); SL is
        # loosened (midpoint need not reach the full threshold before the fill
        # cost does). Zero when slippage_model=flat — no behaviour change.
        if liq.needs_spread and "_total_slippage" in m.columns:
            trigger_slippage = pl.col("_total_slippage")
        else:
            trigger_slippage = pl.lit(0.0)

        sl_cond = self._parse_trigger_condition(
            exit_cfg.stop_loss.condition
            if isinstance(exit_cfg.stop_loss, StopLossConfig)
            else None,
            "stop_loss",
        )
        tp_src = exit_cfg.take_profit
        tp_cond = self._parse_trigger_condition(
            tp_src.condition if isinstance(tp_src, TakeProfitConfig) else None,
            "take_profit",
        )

        m = m.with_columns(
            [
                (
                    ((pl.col("spread_open_mark") + trigger_slippage) >= pl.col("sl_price"))
                    & _price_ok
                    & sl_cond
                ).alias("_gap_sl"),
                (
                    ((pl.col("spread_open_mark") + trigger_slippage) <= pl.col("tp_price"))
                    & _price_ok
                    & tp_cond
                ).alias("_gap_tp"),
                (
                    ((pl.col("position_mark") + trigger_slippage) >= pl.col("sl_price"))
                    & _price_ok
                    & sl_cond
                ).alias("_sl"),
                (
                    ((pl.col("position_mark") + trigger_slippage) <= pl.col("tp_price"))
                    & _price_ok
                    & tp_cond
                ).alias("_tp"),
            ]
        )

        # ── TP confirmation bars ──────────────────────────────────────────
        # Require the TP condition to hold for N consecutive 1-min bars before
        # the exit fires. Filters out fleeting TP touches that a live system
        # with real scanning latency would miss. Only applies to close-based TP
        # (_tp); gap TP (_gap_tp) is instantaneous and skips confirmation.
        if isinstance(tp_src, TakeProfitConfig) and tp_src.confirmation_bars > 1:
            cb = tp_src.confirmation_bars

            def _apply_tp_confirmation(df: pl.DataFrame) -> pl.DataFrame:
                return df.sort("ts_event").with_columns(
                    (
                        pl.col("_tp").cast(pl.Int8).rolling_sum(window_size=cb).fill_null(0) >= cb
                    ).alias("_tp")
                )

            m = m.group_by("entry_id", maintain_order=True).map_groups(_apply_tp_confirmation)

        cond_expr = pl.lit(False)
        for cond_str in exit_cfg.conditions:
            try:
                cond_expr = cond_expr | parse_condition(cond_str)
            except Exception as e:
                self.warnings.append(
                    {
                        "phase": "exit",
                        "trade": self.trade.name,
                        "type": "condition_error",
                        "condition": cond_str,
                        "error": str(e),
                    }
                )
        try:
            m = m.with_columns(cond_expr.alias("_condition"))
        except Exception as e:
            self.warnings.append(
                {
                    "phase": "exit",
                    "trade": self.trade.name,
                    "type": "condition_eval_error",
                    "error": str(e),
                }
            )
            m = m.with_columns(pl.lit(False).alias("_condition"))

        if exit_cfg.dte_exit is not None:
            m = m.with_columns(
                (pl.col("_dte_now") <= pl.lit(int(exit_cfg.dte_exit))).alias("_dte_exit")
            )
        else:
            m = m.with_columns(pl.lit(False).alias("_dte_exit"))

        if exit_cfg.vega_exit is not None and "_spread_vega" in m.columns:
            m = m.with_columns(
                (pl.col("_spread_vega") < pl.lit(float(exit_cfg.vega_exit))).alias("_vega_exit")
            )
        else:
            m = m.with_columns(pl.lit(False).alias("_vega_exit"))

        roll_cfg = self.trade.roll
        if roll_cfg is not None:
            roll_window = roll_cfg.window or self.trade.entry.window
            roll_start_sec = roll_window.start.hour * 3600 + roll_window.start.minute * 60
            roll_end_sec = roll_window.end.hour * 3600 + roll_window.end.minute * 60
            _in_roll_window = (pl.col("_local_sec") >= pl.lit(roll_start_sec)) & (
                pl.col("_local_sec") <= pl.lit(roll_end_sec)
            )
            roll_trigger = pl.lit(False)
            if roll_cfg.dte is not None:
                roll_trigger = roll_trigger | (pl.col("_dte_now") <= pl.lit(int(roll_cfg.dte)))
            if roll_cfg.vega is not None and "_spread_vega" in m.columns:
                roll_trigger = roll_trigger | (
                    pl.col("_spread_vega") < pl.lit(float(roll_cfg.vega))
                )
            for cond_str in roll_cfg.conditions:
                roll_trigger = roll_trigger | parse_condition(cond_str)
            m = m.with_columns((roll_trigger & _in_roll_window).alias("_roll"))
        else:
            m = m.with_columns(pl.lit(False).alias("_roll"))

        if exit_cfg.expiry_exit:
            expiry_expr = pl.col("ts_event").dt.convert_time_zone(tz_str).dt.date() >= pl.col(
                "trade_expiration"
            )
            close_time = self.trade.instrument.expiry_close_time
            if close_time is not None:
                close_sec = close_time.hour * 3600 + close_time.minute * 60
                expiry_expr = expiry_expr & (pl.col("_local_sec") >= pl.lit(close_sec))
            m = m.with_columns(expiry_expr.alias("_expiry"))
        else:
            m = m.with_columns(pl.lit(False).alias("_expiry"))

        m = m.with_columns(
            pl.when(pl.col("_gap_sl"))
            .then(pl.lit(1))
            .when(pl.col("_gap_tp"))
            .then(pl.lit(2))
            .when(pl.col("_sl") & ~pl.col("_gap_sl") & ~pl.col("_gap_tp"))
            .then(pl.lit(3))
            .when(pl.col("_tp") & ~pl.col("_gap_sl") & ~pl.col("_gap_tp"))
            .then(pl.lit(4))
            .when(pl.col("_condition"))
            .then(pl.lit(5))
            .when(pl.col("_roll"))
            .then(pl.lit(6))
            .when(pl.col("_vega_exit"))
            .then(pl.lit(7))
            .when(pl.col("_dte_exit"))
            .then(pl.lit(8))
            .when(pl.col("_expiry"))
            .then(pl.lit(9))
            .otherwise(pl.lit(None).cast(pl.Int32))
            .alias("_priority")
        )

        exit_bars = m.filter(pl.col("_priority").is_not_null())
        first_exit = exit_bars.sort(["entry_id", "ts_event", "_priority"]).unique(
            subset=["entry_id"], keep="first"
        )

        # ── Feature 4: spread-adjusted fill price ─────────────────────────
        # Add the per-bar OHLC spread estimate to the fill price, modelling
        # the cost of crossing the bid-ask. Applied to all exit types.
        # Slippage is 0 for missing bars (fill_null(0) already applied in
        # _compute_position_marks).
        if liq.needs_spread and "_total_slippage" in first_exit.columns:
            slippage = pl.col("_total_slippage")
        else:
            slippage = pl.lit(0.0)

        tick = self.trade.instrument.tick_size

        # Raw exit mark: the estimated executable price at the exit bar.
        # Gap exits fill at the bar open mark; all others fill at the bar close
        # mark. Slippage (bid-ask crossing cost) is included in both, consistent
        # with the trigger conditions which also incorporate slippage — so there
        # is no double-counting. For flat slippage strategies slippage=0 and
        # this is identical to the previous threshold-based fill.
        #
        # Expiry exits use the underlying settlement price to compute each leg's
        # intrinsic value: max(0, S−K) for calls, max(0, K−S) for puts. This is
        # exact (no bar-price staleness) and carries zero slippage (options are
        # cash-settled at expiry; no market fill is required). When no underlying
        # settlement bar is available the fill falls back to bar close + slippage,
        # clipped to the no-arbitrage bound.
        #
        # The no-arbitrage guard above prevents TP/SL from firing on invalid marks,
        # so condition/DTE/expiry-no-settlement are the only paths that reach the
        # otherwise branch with a potentially stale mark. Clip defensively.
        _bar_mark = pl.col("position_mark") + slippage
        _bar_mark_clipped = (
            pl.when(pl.col("open_mark") > 0)
            .then(_bar_mark.clip(lower_bound=0.0))
            .when(pl.col("open_mark") < 0)
            .then(_bar_mark.clip(upper_bound=0.0))
            .otherwise(_bar_mark)
        )
        raw_exit_mark = (
            pl.when(pl.col("_priority").is_in([1, 2]))
            .then(pl.col("spread_open_mark") + slippage)
            .when((pl.col("_priority") == 9) & pl.col("settlement_mark").is_not_null())
            .then(pl.col("settlement_mark"))
            .otherwise(_bar_mark_clipped)
        )

        first_exit = first_exit.with_columns(
            [
                tick_round_expr(raw_exit_mark, tick).alias("exit_mark"),
                tick_round_expr(pl.col("_running_worst"), tick).alias("worst_mark"),
                pl.when(pl.col("_priority").is_in([1, 3]))
                .then(pl.lit("stop_loss"))
                .when(pl.col("_priority").is_in([2, 4]))
                .then(pl.lit("take_profit"))
                .when(pl.col("_priority") == 5)
                .then(pl.lit("condition"))
                .when(pl.col("_priority") == 6)
                .then(pl.lit("roll"))
                .when(pl.col("_priority") == 7)
                .then(pl.lit("vega_exit"))
                .when(pl.col("_priority") == 8)
                .then(pl.lit("dte_exit"))
                .otherwise(pl.lit("expiry"))
                .alias("exit_reason"),
            ]
        ).rename({"ts_event": "exit_time"})

        # --- Handle entries with no exit hit (force expiry at last available bar) ---
        # The most common reason for reaching here on a 0-DTE spread: the last
        # option bar is before expiry_close_time, so the expiry flag never fired.
        found_ids = first_exit["entry_id"].to_list()
        missing = m.filter(~pl.col("entry_id").is_in(found_ids))
        if not missing.is_empty():
            if liq.needs_spread and "_total_slippage" in missing.columns:
                bar_fallback_mark = pl.col("position_mark") + pl.col("_total_slippage")
            else:
                bar_fallback_mark = pl.col("position_mark")

            # Settlement mark wins when available: it is exact intrinsic value
            # with zero slippage (cash-settled, no market fill needed). When
            # unavailable, prefer the last fresh bar, clipped to the no-arbitrage
            # bound to guard against stale forward-fills at expiry.
            _bar_fallback_clipped = (
                pl.when(pl.col("open_mark") > 0)
                .then(bar_fallback_mark.clip(lower_bound=0.0))
                .when(pl.col("open_mark") < 0)
                .then(bar_fallback_mark.clip(upper_bound=0.0))
                .otherwise(bar_fallback_mark)
            )
            fallback_mark = (
                pl.when(pl.col("settlement_mark").is_not_null())
                .then(pl.col("settlement_mark"))
                .otherwise(_bar_fallback_clipped)
            )

            # Pick the representative bar row per entry:
            #  - settlement entries: any row works (mark is overridden); use last.
            #  - no-settlement entries: prefer last fresh row; fall back to last.
            if liq.needs_staleness and "_max_leg_stale_minutes" in missing.columns:
                fresh_limit = float(liq.max_leg_stale_minutes)
                has_settlement = set(
                    missing.filter(pl.col("settlement_mark").is_not_null())["entry_id"].to_list()
                )
                settlement_pool = (
                    missing.filter(pl.col("entry_id").is_in(has_settlement))
                    .sort(["entry_id", "ts_event"])
                    .unique(subset=["entry_id"], keep="last")
                )
                no_settlement = missing.filter(~pl.col("entry_id").is_in(has_settlement))
                fresh_pool = (
                    no_settlement.filter(pl.col("_max_leg_stale_minutes") <= pl.lit(fresh_limit))
                    .sort(["entry_id", "ts_event"])
                    .unique(subset=["entry_id"], keep="last")
                )
                stale_pool = (
                    no_settlement.filter(
                        ~pl.col("entry_id").is_in(fresh_pool["entry_id"].to_list())
                    )
                    .sort(["entry_id", "ts_event"])
                    .unique(subset=["entry_id"], keep="last")
                )
                parts = [p for p in [settlement_pool, fresh_pool, stale_pool] if not p.is_empty()]
                fallback_pool = (
                    pl.concat(parts)
                    if parts
                    else missing.sort(["entry_id", "ts_event"]).unique(
                        subset=["entry_id"], keep="last"
                    )
                )
            else:
                fallback_pool = missing.sort(["entry_id", "ts_event"]).unique(
                    subset=["entry_id"], keep="last"
                )

            last_bars = (
                fallback_pool.with_columns(
                    [
                        tick_round_expr(fallback_mark, tick).alias("exit_mark"),
                        pl.lit("expiry").alias("exit_reason"),
                        tick_round_expr(pl.col("_running_worst"), tick).alias("worst_mark"),
                    ]
                )
                .rename({"ts_event": "exit_time"})
                .select(["entry_id", "exit_time", "exit_mark", "exit_reason", "worst_mark"])
            )
            return pl.concat(
                [
                    first_exit.select(
                        ["entry_id", "exit_time", "exit_mark", "exit_reason", "worst_mark"]
                    ),
                    last_bars,
                ]
            )

        return first_exit.select(
            ["entry_id", "exit_time", "exit_mark", "exit_reason", "worst_mark"]
        )

    # ------------------------------------------------------------------
    # Leg-out post-processor
    # ------------------------------------------------------------------

    def _adjust_leg_out_exits(
        self,
        exits: pl.DataFrame,
        entries: pl.DataFrame,
        option_bars: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Replace forward-filled long-leg prices in exit_mark with the first real
        bar after exit_time (market-order semantics for non-expiry exits).

        The position scanner forward-fills stale long-leg prices into exit_mark.
        For TP/SL exits the long leg may not have traded in hours; its embedded
        price is stale. This replaces that component with the first actual traded
        bar. If no bar exists after exit_time the fill is 0 (illiquid — option
        cannot be sold into an empty market).

        Expiry exits are left unchanged: both legs' terminal values are already
        correctly embedded by the exit scanner.

        Adjustment per long leg (signed_qty = -1 × quantity):
            exit_mark' = exit_mark
                         - stale_close × signed_qty   [remove forward-fill component]
                         + fill_close  × signed_qty   [add market-order fill]
        """
        long_legs = [leg for leg in self.trade.legs if leg.action == "buy_to_open"]
        if not long_legs:
            return exits

        expiry_exits = exits.filter(pl.col("exit_reason") == "expiry")
        adjust_exits = exits.filter(pl.col("exit_reason") != "expiry")

        if adjust_exits.is_empty():
            return exits

        entry_lookup = entries.select(["entry_id", "entry_time"]).join(
            adjust_exits.select(["entry_id", "exit_time"]),
            on="entry_id",
            how="inner",
        )

        result = adjust_exits
        tick = self.trade.instrument.tick_size
        tz_str = self.strategy.universe.session.timezone

        for leg in long_legs:
            signed_qty = -1.0 * float(leg.quantity)

            leg_map = entries.select(
                [
                    "entry_id",
                    pl.col(f"leg_{leg.name}_instrument_id").alias("instrument_id"),
                    pl.col(f"leg_{leg.name}_expiration").alias("leg_expiration"),
                ]
            )

            # All bars for this leg per entry, annotated with entry/exit context.
            # Discard bars whose session date is after the leg's own expiration —
            # instrument IDs can be recycled for new contracts after expiry, and
            # without this guard those post-expiry bars corrupt the fill price.
            leg_bars = (
                option_bars.select(["ts_event", "instrument_id", "close"])
                .join(leg_map, on="instrument_id", how="inner")
                .join(entry_lookup, on="entry_id", how="inner")
                .filter(
                    pl.col("ts_event").dt.convert_time_zone(tz_str).dt.date()
                    <= pl.col("leg_expiration")
                )
            )

            if leg_bars.is_empty():
                continue

            # Last real bar in (entry_time, exit_time] — the stale forward-fill
            # price already embedded in exit_mark for this leg.
            last_before = (
                leg_bars.filter(
                    (pl.col("ts_event") > pl.col("entry_time"))
                    & (pl.col("ts_event") <= pl.col("exit_time"))
                )
                .sort(["entry_id", "ts_event"])
                .unique(subset=["entry_id"], keep="last")
                .select(["entry_id", pl.col("close").alias("_stale")])
            )

            # First real bar strictly after exit_time — market order fill price.
            # Empty when no bars exist (illiquid) → fill defaults to 0.
            first_after = (
                leg_bars.filter(pl.col("ts_event") > pl.col("exit_time"))
                .sort(["entry_id", "ts_event"])
                .unique(subset=["entry_id"], keep="first")
                .select(["entry_id", pl.col("close").alias("_fill")])
            )

            result = (
                result.join(last_before, on="entry_id", how="left")
                .join(first_after, on="entry_id", how="left")
                .with_columns(
                    tick_round_expr(
                        pl.col("exit_mark")
                        - pl.col("_stale").fill_null(0.0) * pl.lit(signed_qty)
                        + pl.col("_fill").fill_null(0.0) * pl.lit(signed_qty),
                        tick,
                    ).alias("exit_mark")
                )
                .drop(["_stale", "_fill"])
            )

        if expiry_exits.is_empty():
            return result

        return pl.concat([result, expiry_exits])

    # ------------------------------------------------------------------
    # Long-leg continuation scanner
    # ------------------------------------------------------------------

    def scan_long_continuation(
        self,
        exits: pl.DataFrame,
        entries: pl.DataFrame,
        option_bars: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        For every SL exit (stop_loss or gap_sl), track the long leg forward
        under a trailing stop and return the continuation result.

        The continuation entry price is the first real bar of the long leg
        strictly after the spread's SL exit_time (market-order semantics,
        matching _adjust_leg_out_exits). The trailing stop fires when the
        price pulls back long_trailing_stop_pct from the post-SL peak.
        If the trailing stop never fires the position runs to expiry and
        is valued at intrinsic settlement.

        Only single-long-leg trades are supported. Trades with multiple BTO
        legs emit a warning and return an empty DataFrame.

        Returns a DataFrame with columns:
            entry_id, continuation_entry_price, continuation_exit_time,
            continuation_exit_price, continuation_exit_reason
        """
        long_legs = [leg for leg in self.trade.legs if leg.action == "buy_to_open"]
        if not long_legs:
            return _empty_continuation_df()

        if len(long_legs) > 1:
            self.warnings.append(
                {
                    "type": "continuation_skipped",
                    "message": (
                        "on_sl_long_continuation with multiple long legs is not yet supported; "
                        "continuation tracking skipped"
                    ),
                }
            )
            return _empty_continuation_df()

        leg = long_legs[0]
        trail_pct = float(self.trade.exit.long_trailing_stop_pct)
        tz_str = self.strategy.universe.session.timezone

        sl_exits = exits.filter(pl.col("exit_reason").is_in(["stop_loss", "gap_sl"]))
        if sl_exits.is_empty():
            return _empty_continuation_df()

        # Per-entry context: instrument, expiration, strike, right
        leg_ctx = entries.select(
            [
                "entry_id",
                pl.col(f"leg_{leg.name}_instrument_id").alias("_instr_id"),
                pl.col(f"leg_{leg.name}_expiration").alias("_leg_exp"),
                pl.col(f"leg_{leg.name}_strike_price").alias("_strike"),
                pl.col(f"leg_{leg.name}_right").alias("_right"),
                pl.col(f"leg_{leg.name}_multiplier").cast(pl.Float64).alias("_multiplier"),
                pl.lit(float(leg.quantity)).alias("_quantity"),
            ]
        )

        sl_ctx = sl_exits.select(["entry_id", "exit_time"]).join(
            leg_ctx, on="entry_id", how="inner"
        )

        instr_ids = sl_ctx["_instr_id"].unique().to_list()

        # Bars for the long leg, joined with SL context, filtered to continuation window
        cont_bars = (
            option_bars.filter(pl.col("instrument_id").is_in(instr_ids))
            .select(["ts_event", "instrument_id", "close"])
            .join(sl_ctx.rename({"_instr_id": "instrument_id"}), on="instrument_id", how="inner")
            .filter(
                (pl.col("ts_event") > pl.col("exit_time"))
                & (pl.col("ts_event").dt.convert_time_zone(tz_str).dt.date() <= pl.col("_leg_exp"))
            )
            .sort(["entry_id", "ts_event"])
        )

        if cont_bars.is_empty():
            return _empty_continuation_df()

        # Continuation entry price = first real bar per entry after SL exit
        entry_prices = cont_bars.group_by("entry_id", maintain_order=True).agg(
            pl.col("close").first().alias("continuation_entry_price")
        )

        # Trailing stop: track running peak, fire when close <= peak * (1 - pct)
        monitored = cont_bars.with_columns(
            pl.col("close").cum_max().over("entry_id").alias("_peak")
        ).with_columns(
            (
                (pl.col("_peak") > 0.0)
                & (pl.col("close") <= pl.col("_peak") * pl.lit(1.0 - trail_pct))
            ).alias("_ts_hit")
        )

        # First trailing-stop bar per entry
        ts_exits = (
            monitored.filter(pl.col("_ts_hit"))
            .group_by("entry_id", maintain_order=True)
            .agg(
                [
                    pl.col("ts_event").first().alias("continuation_exit_time"),
                    pl.col("close").first().alias("continuation_exit_price"),
                ]
            )
            .with_columns(pl.lit("trailing_stop").alias("continuation_exit_reason"))
        )

        # Entries where trailing stop never fired → expiry_continuation
        ts_entry_ids = set(ts_exits["entry_id"].to_list())
        expiry_ctx = sl_ctx.filter(~pl.col("entry_id").is_in(ts_entry_ids))

        expiry_rows: list[dict] = []
        for row in expiry_ctx.iter_rows(named=True):
            eid = row["entry_id"]
            opt_id = row["_instr_id"]
            leg_exp = row["_leg_exp"]
            underlying_id = self._opt_to_underlying.get(opt_id)
            settlement = (
                self._settlement_closes_by_key.get((underlying_id, leg_exp))
                if underlying_id is not None
                else None
            )

            # Last bar before/at expiry as fallback when no settlement price
            last_bar = cont_bars.filter(pl.col("entry_id") == eid).sort("ts_event").tail(1)
            if last_bar.is_empty():
                continue

            last_ts = last_bar["ts_event"][0]
            last_close = float(last_bar["close"][0])

            if settlement is not None:
                right = str(row["_right"]).upper()
                strike = float(row["_strike"])
                if right.startswith("C"):
                    intr = max(0.0, settlement - strike)
                else:
                    intr = max(0.0, strike - settlement)
                exit_price = intr
            else:
                exit_price = last_close

            expiry_rows.append(
                {
                    "entry_id": eid,
                    "continuation_exit_time": last_ts,
                    "continuation_exit_price": exit_price,
                    "continuation_exit_reason": "expiry_continuation",
                }
            )

        result_parts: list[pl.DataFrame] = [ts_exits]
        if expiry_rows:
            result_parts.append(
                pl.DataFrame(expiry_rows).with_columns(
                    pl.col("entry_id").cast(pl.UInt32),
                )
            )

        continuation = pl.concat(result_parts).join(entry_prices, on="entry_id", how="left")
        return continuation.select(
            [
                "entry_id",
                "continuation_entry_price",
                "continuation_exit_time",
                "continuation_exit_price",
                "continuation_exit_reason",
            ]
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_settlement_marks(
        self,
        entries: pl.DataFrame,
        tz_str: str,
    ) -> pl.DataFrame:
        """
        Compute the intrinsic (settlement) value of each position at expiry.

        Returns a DataFrame with columns [entry_id, settlement_mark] where
        settlement_mark is null when no underlying bar is available.

        For each leg:
            call: max(0, S − strike)
            put:  max(0, strike − S)

        The sum of signed intrinsic values across all legs gives the spread's
        value at settlement — exact, with zero slippage (options are cash-settled
        at expiry; no market fill is required). S is the underlying close at or
        just before expiry_close_time on the trade's expiration date.
        """
        close_time = self.trade.instrument.expiry_close_time
        if close_time is None:
            return entries.select("entry_id").with_columns(
                pl.lit(None).cast(pl.Float64).alias("settlement_mark")
            )

        exp_cols = [f"leg_{leg.name}_expiration" for leg in self.trade.legs]

        # Per-entry trade expiration = min of leg expirations.
        entry_exps = (
            entries.select(["entry_id"] + exp_cols)
            .with_columns(pl.min_horizontal(exp_cols).alias("_trade_exp"))
            .select(["entry_id", "_trade_exp"])
        )

        unique_exps = entry_exps["_trade_exp"].unique().to_list()

        # Map each unique expiry date to a representative first-leg instrument ID,
        # then resolve underlying and settlement close from the caches pre-loaded
        # by scan().  No DB queries here — just dict lookups.
        first_leg = self.trade.legs[0]
        first_leg_id_col = f"leg_{first_leg.name}_instrument_id"
        first_leg_exp_col = f"leg_{first_leg.name}_expiration"

        exp_to_opt_id: dict = {
            row["exp"]: row["opt_id"]
            for row in (
                entries.select([first_leg_exp_col, first_leg_id_col])
                .rename({first_leg_exp_col: "exp", first_leg_id_col: "opt_id"})
                .unique(subset=["exp"], keep="first")
                .iter_rows(named=True)
            )
        }

        settlement_by_date: dict = {}
        for exp_date in unique_exps:
            opt_id = exp_to_opt_id.get(exp_date)
            underlying_id = self._opt_to_underlying.get(opt_id) if opt_id is not None else None
            if underlying_id is None:
                settlement_by_date[exp_date] = None
                continue
            settlement_by_date[exp_date] = self._settlement_closes_by_key.get(
                (underlying_id, exp_date)
            )

        # Attach settlement price per entry, then compute the intrinsic sum
        # vectorially in Polars.
        settlement_prices = [settlement_by_date.get(d) for d in entry_exps["_trade_exp"].to_list()]
        entry_with_s = entries.join(
            entry_exps.with_columns(
                pl.Series("_settlement", settlement_prices, dtype=pl.Float64)
            ).select(["entry_id", "_settlement"]),
            on="entry_id",
            how="left",
        )

        intrinsic_exprs: list[pl.Expr] = []
        for leg in self.trade.legs:
            signed_qty = (1.0 if leg.action == "sell_to_open" else -1.0) * float(leg.quantity)
            strike_col = f"leg_{leg.name}_strike_price"
            right_col = f"leg_{leg.name}_right"
            call_intr = (pl.col("_settlement") - pl.col(strike_col)).clip(lower_bound=0.0)
            put_intr = (pl.col(strike_col) - pl.col("_settlement")).clip(lower_bound=0.0)
            leg_intr = (
                pl.when(pl.col(right_col).str.to_uppercase().str.starts_with("C"))
                .then(call_intr)
                .otherwise(put_intr)
            ) * pl.lit(signed_qty)
            intrinsic_exprs.append(leg_intr.alias(f"_intr_{leg.name}"))

        intr_cols = [f"_intr_{leg.name}" for leg in self.trade.legs]

        return (
            entry_with_s.with_columns(intrinsic_exprs)
            .with_columns(
                pl.when(pl.col("_settlement").is_not_null())
                .then(pl.sum_horizontal(intr_cols))
                .otherwise(pl.lit(None).cast(pl.Float64))
                .alias("settlement_mark")
            )
            .select(["entry_id", "settlement_mark"])
        )

    def _get_indicators(
        self,
        min_entry: datetime,
        end_dt: datetime,
    ) -> pl.DataFrame:
        """
        Return the indicators DataFrame for this cohort's time window.

        If the engine pre-loaded indicators, use them as-is (the left join in
        _find_first_hit naturally limits to bars present in position_marks).
        Otherwise, fetch from DB only when exit conditions need indicators.
        """
        if self._preloaded_indicators is not None:
            return self._preloaded_indicators

        if not self.trade.exit.conditions:
            return pl.DataFrame()

        schedule = self.db.front_future_schedule(
            self.trade.instrument.root_symbol,
            min_entry.date(),
            end_dt.date(),
            self.trade.instrument.roll_days_before_expiry,
        )
        if schedule.is_empty():
            return pl.DataFrame()

        frames = [
            self.db.indicators(uid, min_entry, end_dt)
            for uid in schedule["underlying_id"].unique().to_list()
        ]
        non_empty = [f for f in frames if not f.is_empty() and len(f.columns) > 1]
        if not non_empty:
            return pl.DataFrame()
        if len(non_empty) == 1:
            return non_empty[0].sort("ts_event")
        return pl.concat(non_empty, how="diagonal").sort("ts_event")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_exits_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entry_id": pl.Series([], dtype=pl.UInt32),
            "exit_time": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "exit_mark": pl.Series([], dtype=pl.Float64),
            "worst_mark": pl.Series([], dtype=pl.Float64),
            "exit_reason": pl.Series([], dtype=pl.Utf8),
        }
    )


def _empty_continuation_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entry_id": pl.Series([], dtype=pl.UInt32),
            "continuation_entry_price": pl.Series([], dtype=pl.Float64),
            "continuation_exit_time": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "continuation_exit_price": pl.Series([], dtype=pl.Float64),
            "continuation_exit_reason": pl.Series([], dtype=pl.Utf8),
        }
    )
