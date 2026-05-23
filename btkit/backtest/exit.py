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

from datetime import datetime, timezone

import polars as pl

from btkit.db.input_db import InputDatabase
from btkit.strategy.definition import StrategyDefinition, TradeDefinition
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
        # Pre-loaded indicators from the engine (avoids a second DB fetch per trade).
        self._preloaded_indicators = indicators

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

        # Chunk by the calendar month of the earliest leg expiration. Monthly
        # buckets produce O(months) groups rather than O(unique_exp_tuples)
        # groups. The latter explodes when independent leg selection occasionally
        # picks different expiration cycles for different legs (e.g. short_put
        # selects May 16, long_put selects May 23), creating one group per
        # (exp1, exp2, ...) combination and defeating the batching goal.
        # Monthly buckets bound the time window and instrument-ID set without
        # the per-group query overhead of fine-grained tuple chunking.
        exp_cols = [f"leg_{leg.name}_expiration" for leg in self.trade.legs]
        entries_bucketed = entries.with_columns(
            pl.min_horizontal(exp_cols).dt.strftime("%Y-%m").alias("_exp_bucket")
        )

        chunks: list[pl.DataFrame] = []
        for _, cohort in entries_bucketed.group_by("_exp_bucket", maintain_order=True):
            cohort = cohort.drop("_exp_bucket")
            option_bars, indicators = self._load_exit_data(cohort)
            if option_bars.is_empty():
                continue
            position_marks = self._compute_position_marks(option_bars, cohort)
            if position_marks.is_empty():
                continue
            exits_chunk = self._find_first_hit(position_marks, indicators, cohort)
            if not exits_chunk.is_empty():
                chunks.append(exits_chunk)

        if not chunks:
            return _empty_exits_df()

        return pl.concat(chunks)

    # ------------------------------------------------------------------
    # Step 1: Batch-load all data for a cohort of open positions
    # ------------------------------------------------------------------

    def _load_exit_data(
        self,
        entries: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Batch-load all data needed to monitor the open positions in this cohort.
        Single DB read per cohort. Returns:
          - option_bars: all leg bars from the earliest entry_time to the latest
            possible exit (max expiration across cohort entries).
          - indicators: wide indicator DataFrame for the underlying over the same
            window (from pre-loaded data if available, otherwise fetched from DB).
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
            max_exp.year, max_exp.month, max_exp.day,
            23, 59, 59, tzinfo=timezone.utc,
        )

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

        Returns columns: entry_id, ts_event, position_mark, spread_open_mark.

        Option bars are sparse (only when traded), so different legs rarely share
        the same ts_event. We full-outer-join all legs on (entry_id, ts_event) and
        forward-fill stale prices within each entry_id.
        """
        per_leg: list[pl.DataFrame] = []

        for leg in self.trade.legs:
            sign = 1.0 if leg.action == "sell_to_open" else -1.0
            signed_qty = sign * float(leg.quantity)

            leg_map = entries.select([
                "entry_id",
                "entry_time",
                pl.col(f"leg_{leg.name}_instrument_id").alias("instrument_id"),
            ])

            leg_bars = (
                option_bars
                .select(["ts_event", "instrument_id", "open", "close"])
                .join(leg_map, on="instrument_id", how="inner")
                .filter(pl.col("ts_event") > pl.col("entry_time"))
                .select([
                    "entry_id",
                    "ts_event",
                    (pl.col("close") * pl.lit(signed_qty)).alias(f"_leg_{leg.name}_mark_close"),
                    (pl.col("open")  * pl.lit(signed_qty)).alias(f"_leg_{leg.name}_mark_open"),
                ])
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
        open_cols  = [f"_leg_{leg.name}_mark_open"  for leg in self.trade.legs]

        result = result.sort(["entry_id", "ts_event"]).with_columns([
            *[pl.col(c).forward_fill().over("entry_id") for c in close_cols],
            *[pl.col(c).forward_fill().over("entry_id") for c in open_cols],
        ])

        result = result.drop_nulls(subset=close_cols + open_cols)

        if result.is_empty():
            return pl.DataFrame()

        return result.with_columns([
            pl.sum_horizontal(close_cols).alias("position_mark"),
            pl.sum_horizontal(open_cols).alias("spread_open_mark"),
        ]).select(["entry_id", "ts_event", "position_mark", "spread_open_mark"])

    # ------------------------------------------------------------------
    # Step 3: Find first exit hit per entry
    # ------------------------------------------------------------------

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

        Fill price:
            gap_sl / gap_tp       → spread_open_mark
            stop_loss (close)     → sl_price
            take_profit (close)   → tp_price
            condition / dte / exp → position_mark (bar close)
        """
        exit_cfg = self.trade.exit

        exp_cols = [f"leg_{leg.name}_expiration" for leg in self.trade.legs]
        entry_meta = (
            entries
            .select(["entry_id", "entry_time", "tp_price", "sl_price", "dte_exit"] + exp_cols)
            .with_columns(pl.min_horizontal(exp_cols).alias("trade_expiration"))
            .select(["entry_id", "entry_time", "tp_price", "sl_price", "dte_exit", "trade_expiration"])
        )

        # Sort once; used by both forward-fill (already done) and cum_max below.
        m = (
            position_marks
            .sort(["entry_id", "ts_event"])
            .join(entry_meta, on="entry_id", how="left")
        )

        # Running worst mark per entry — single pass, no second scan needed.
        m = m.with_columns(
            pl.col("position_mark").cum_max().over("entry_id").alias("_running_worst")
        )

        if not indicators.is_empty() and len(indicators.columns) > 1:
            m = m.join(indicators, on="ts_event", how="left")

        m = m.with_columns(
            (pl.col("trade_expiration").cast(pl.Date) - pl.col("ts_event").dt.date())
            .dt.total_days().cast(pl.Int32).alias("_dte_now")
        )

        m = m.with_columns([
            (pl.col("spread_open_mark") >= pl.col("sl_price")).alias("_gap_sl"),
            (pl.col("spread_open_mark") <= pl.col("tp_price")).alias("_gap_tp"),
            (pl.col("position_mark")    >= pl.col("sl_price")).alias("_sl"),
            (pl.col("position_mark")    <= pl.col("tp_price")).alias("_tp"),
        ])

        cond_expr = pl.lit(False)
        for cond_str in exit_cfg.conditions:
            try:
                cond_expr = cond_expr | parse_condition(cond_str)
            except Exception:
                pass
        try:
            m = m.with_columns(cond_expr.alias("_condition"))
        except Exception:
            m = m.with_columns(pl.lit(False).alias("_condition"))

        if exit_cfg.dte_exit is not None:
            m = m.with_columns(
                (pl.col("_dte_now") <= pl.lit(int(exit_cfg.dte_exit))).alias("_dte_exit")
            )
        else:
            m = m.with_columns(pl.lit(False).alias("_dte_exit"))

        if exit_cfg.expiry_exit:
            m = m.with_columns(
                (pl.col("ts_event").dt.date() >= pl.col("trade_expiration")).alias("_expiry")
            )
        else:
            m = m.with_columns(pl.lit(False).alias("_expiry"))

        m = m.with_columns(
            pl.when(pl.col("_gap_sl")).then(pl.lit(1))
            .when(pl.col("_gap_tp")).then(pl.lit(2))
            .when(pl.col("_sl") & ~pl.col("_gap_sl") & ~pl.col("_gap_tp")).then(pl.lit(3))
            .when(pl.col("_tp") & ~pl.col("_gap_sl") & ~pl.col("_gap_tp")).then(pl.lit(4))
            .when(pl.col("_condition")).then(pl.lit(5))
            .when(pl.col("_dte_exit")).then(pl.lit(6))
            .when(pl.col("_expiry")).then(pl.lit(7))
            .otherwise(pl.lit(None).cast(pl.Int32)).alias("_priority")
        )

        exit_bars = m.filter(pl.col("_priority").is_not_null())
        first_exit = (
            exit_bars
            .sort(["entry_id", "ts_event", "_priority"])
            .unique(subset=["entry_id"], keep="first")
        )

        first_exit = first_exit.with_columns([
            pl.when(pl.col("_priority").is_in([1, 2])).then(pl.col("spread_open_mark"))
            .when(pl.col("_priority") == 3).then(pl.col("sl_price"))
            .when(pl.col("_priority") == 4).then(pl.col("tp_price"))
            .otherwise(pl.col("position_mark"))
            .alias("exit_mark"),

            pl.when(pl.col("_priority").is_in([1, 3])).then(pl.lit("stop_loss"))
            .when(pl.col("_priority").is_in([2, 4])).then(pl.lit("take_profit"))
            .when(pl.col("_priority") == 5).then(pl.lit("condition"))
            .when(pl.col("_priority") == 6).then(pl.lit("dte_exit"))
            .otherwise(pl.lit("expiry"))
            .alias("exit_reason"),
        ]).rename({"ts_event": "exit_time", "_running_worst": "worst_mark"})

        # --- Handle entries with no exit hit (force expiry at last available bar) ---
        found_ids = first_exit["entry_id"].to_list()
        missing = m.filter(~pl.col("entry_id").is_in(found_ids))
        if not missing.is_empty():
            last_bars = (
                missing
                .unique(subset=["entry_id"], keep="last")
                .with_columns([
                    pl.col("position_mark").alias("exit_mark"),
                    pl.lit("expiry").alias("exit_reason"),
                    pl.col("_running_worst").alias("worst_mark"),
                ])
                .rename({"ts_event": "exit_time"})
                .select(["entry_id", "exit_time", "exit_mark", "exit_reason", "worst_mark"])
            )
            return pl.concat([
                first_exit.select(["entry_id", "exit_time", "exit_mark", "exit_reason", "worst_mark"]),
                last_bars,
            ])

        return first_exit.select(
            ["entry_id", "exit_time", "exit_mark", "exit_reason", "worst_mark"]
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

        underlying_id = self.db.instrument_id_for_symbol(self.trade.instrument.root_symbol)
        if underlying_id is None:
            return pl.DataFrame()

        return self.db.indicators(underlying_id, min_entry, end_dt)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_exits_df() -> pl.DataFrame:
    return pl.DataFrame({
        "entry_id":    pl.Series([], dtype=pl.UInt32),
        "exit_time":   pl.Series([], dtype=pl.Datetime("us", "UTC")),
        "exit_mark":   pl.Series([], dtype=pl.Float64),
        "worst_mark":  pl.Series([], dtype=pl.Float64),
        "exit_reason": pl.Series([], dtype=pl.Utf8),
    })
