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
    ) -> None:
        self.db = db
        self.strategy = strategy
        self.trade = trade

    def scan(self, entries: pl.DataFrame) -> pl.DataFrame:
        """
        For each entry row, find the first exit event.

        Returns one row per entry with columns:
            entry_id, exit_time, exit_mark, worst_mark, exit_reason

        exit_reason: 'take_profit' | 'stop_loss' | 'condition' | 'dte_exit' | 'expiry'
        """
        if entries.is_empty():
            return _empty_exits_df()

        option_bars, indicators = self._load_exit_data(entries)
        if option_bars.is_empty():
            return _empty_exits_df()

        position_marks = self._compute_position_marks(option_bars, entries)
        if position_marks.is_empty():
            return _empty_exits_df()

        return self._find_first_hit(position_marks, indicators, entries)

    # ------------------------------------------------------------------
    # Step 1: Batch-load all data for open positions
    # ------------------------------------------------------------------

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

        underlying_id = self.db.instrument_id_for_symbol(self.trade.instrument.root_symbol)
        indicators = (
            self.db.indicators(underlying_id, min_entry, end_dt)
            if underlying_id is not None
            else pl.DataFrame()
        )

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
        forward-fill stale prices within each entry_id — standard practice for
        illiquid options monitored on a regular bar schedule.
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

        # Full outer join so every ts_event from any leg is represented.
        result = per_leg[0]
        for lb in per_leg[1:]:
            result = result.join(lb, on=["entry_id", "ts_event"], how="full", coalesce=True)

        # Forward-fill stale prices (option bars are sparse) within each entry.
        close_cols = [f"_leg_{leg.name}_mark_close" for leg in self.trade.legs]
        open_cols  = [f"_leg_{leg.name}_mark_open"  for leg in self.trade.legs]

        result = result.sort(["entry_id", "ts_event"]).with_columns([
            *[pl.col(c).forward_fill().over("entry_id") for c in close_cols],
            *[pl.col(c).forward_fill().over("entry_id") for c in open_cols],
        ])

        # Drop rows where any leg has no price yet (before first bar for that leg).
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

        Fill price:
            gap_sl / gap_tp       → spread_open_mark
            stop_loss (close)     → sl_price
            take_profit (close)   → tp_price
            condition / dte / exp → position_mark (bar close)
        """
        exit_cfg = self.trade.exit

        # --- Attach per-entry thresholds to every monitoring bar ---
        exp_cols = [f"leg_{leg.name}_expiration" for leg in self.trade.legs]
        entry_meta = (
            entries
            .select(["entry_id", "entry_time", "tp_price", "sl_price", "dte_exit"] + exp_cols)
            .with_columns(pl.min_horizontal(exp_cols).alias("trade_expiration"))
            .select(["entry_id", "entry_time", "tp_price", "sl_price", "dte_exit", "trade_expiration"])
        )

        m = position_marks.join(entry_meta, on="entry_id", how="left")

        # --- Join indicator columns (if any) ---
        if not indicators.is_empty() and len(indicators.columns) > 1:
            m = m.join(indicators, on="ts_event", how="left")

        # --- dte_now: days until option expiration ---
        m = m.with_columns(
            (pl.col("trade_expiration").cast(pl.Date) - pl.col("ts_event").dt.date())
            .dt.total_days().cast(pl.Int32).alias("_dte_now")
        )

        # --- Per-bar exit flags ---
        m = m.with_columns([
            (pl.col("spread_open_mark") >= pl.col("sl_price")).alias("_gap_sl"),
            (pl.col("spread_open_mark") <= pl.col("tp_price")).alias("_gap_tp"),
            (pl.col("position_mark")    >= pl.col("sl_price")).alias("_sl"),
            (pl.col("position_mark")    <= pl.col("tp_price")).alias("_tp"),
        ])

        # --- Indicator condition (OR logic across all exit conditions) ---
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

        # --- DTE exit flag ---
        if exit_cfg.dte_exit is not None:
            m = m.with_columns(
                (pl.col("_dte_now") <= pl.lit(int(exit_cfg.dte_exit))).alias("_dte_exit")
            )
        else:
            m = m.with_columns(pl.lit(False).alias("_dte_exit"))

        # --- Expiry exit flag ---
        if exit_cfg.expiry_exit:
            m = m.with_columns(
                (pl.col("ts_event").dt.date() >= pl.col("trade_expiration")).alias("_expiry")
            )
        else:
            m = m.with_columns(pl.lit(False).alias("_expiry"))

        # --- Assign priority (lowest = highest priority) ---
        # When both gap_sl and gap_tp are true simultaneously it's impossible
        # (sl_price > tp_price always), so the when/then chain is clean.
        # For close-based SL/TP: guard against double-counting with gap cases.
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

        # --- First exit bar per entry ---
        exit_bars = m.filter(pl.col("_priority").is_not_null())
        first_exit = (
            exit_bars
            .sort(["entry_id", "ts_event", "_priority"])
            .unique(subset=["entry_id"], keep="first")
        )

        # --- Determine exit_mark and exit_reason ---
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
        ]).rename({"ts_event": "exit_time"})

        # --- Handle entries with no exit hit (force expiry at last available bar) ---
        missing = position_marks.filter(
            ~pl.col("entry_id").is_in(first_exit["entry_id"])
        )
        if not missing.is_empty():
            last_bars = (
                missing
                .sort("ts_event")
                .unique(subset=["entry_id"], keep="last")
                .with_columns([
                    pl.col("position_mark").alias("exit_mark"),
                    pl.lit("expiry").alias("exit_reason"),
                ])
                .rename({"ts_event": "exit_time"})
                .select(["entry_id", "exit_time", "exit_mark", "exit_reason"])
            )
            first_exit_base = pl.concat([
                first_exit.select(["entry_id", "exit_time", "exit_mark", "exit_reason"]),
                last_bars,
            ])
        else:
            first_exit_base = first_exit.select(
                ["entry_id", "exit_time", "exit_mark", "exit_reason"]
            )

        # --- worst_mark = max(position_mark) from entry to exit per entry ---
        worst_marks = (
            position_marks
            .join(first_exit_base.select(["entry_id", "exit_time"]), on="entry_id", how="left")
            .filter(pl.col("ts_event") <= pl.col("exit_time"))
            .group_by("entry_id")
            .agg(pl.col("position_mark").max().alias("worst_mark"))
        )

        return first_exit_base.join(worst_marks, on="entry_id", how="left")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_exits_df() -> pl.DataFrame:
    """Return an empty DataFrame with the minimum required exit columns."""
    return pl.DataFrame({
        "entry_id":   pl.Series([], dtype=pl.UInt32),
        "exit_time":  pl.Series([], dtype=pl.Datetime("us", "UTC")),
        "exit_mark":  pl.Series([], dtype=pl.Float64),
        "worst_mark": pl.Series([], dtype=pl.Float64),
        "exit_reason": pl.Series([], dtype=pl.Utf8),
    })
