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

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from btkit.audit.schema import resolve_audit_filter
from btkit.backtest._util import tick_round_expr
from btkit.db.input_db import InputDatabase
from btkit.strategy.definition import (
    LegConfig,
    SimpleDeltaConfig,
    SteppedDeltaConfig,
    StopLossConfig,
    StrategyDefinition,
    TakeProfitConfig,
    TradeDefinition,
)
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
        self._audit_filter_codes: frozenset[str] = resolve_audit_filter(
            strategy.universe.audit_filter
        )

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
        roll_days = self.trade.instrument.roll_days_before_expiry

        # size_multiplier is reserved (position sizing not yet implemented); a
        # non-1.0 value is a no-op and warns so configs stay forward-compatible.
        for leg in self.trade.legs:
            if leg.targets is None:
                continue
            if any(t.size_multiplier != 1.0 for t in leg.targets.values()):
                self.warnings.append({
                    "phase": "entry",
                    "trade": self.trade.name,
                    "type": "size_multiplier_ignored",
                    "leg": leg.name,
                    "message": (
                        "target size_multiplier is reserved and currently a no-op; "
                        "quantity is unchanged"
                    ),
                })

        universe = self.strategy.universe
        start_dt = datetime(
            universe.start_date.year,
            universe.start_date.month,
            universe.start_date.day,
            tzinfo=self._tz,
        ).astimezone(UTC)
        end_dt = datetime(
            universe.end_date.year,
            universe.end_date.month,
            universe.end_date.day,
            23,
            59,
            59,
            tzinfo=self._tz,
        ).astimezone(UTC)

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
            all_bars.with_columns(pl.col("ts_event").dt.date().alias("_date"))
            .join(schedule, left_on="_date", right_on="date", how="left")
            .filter(pl.col("instrument_id") == pl.col("underlying_id"))
            .drop(["_date", "underlying_id"])
        )
        if bars.is_empty():
            return _empty_entries_df()

        # Indicators: query each distinct front-month contract so coverage is
        # continuous across quarterly rolls (not just the first contract).
        indicators = self._get_indicators_for_schedule(schedule, start_dt, end_dt)

        candidates = self._apply_window_filters(bars)
        if candidates.is_empty():
            return _empty_entries_df()

        # Attach the active underlying_id to each candidate bar so _select_legs
        # can pass per-bar underlying IDs to the greeks queries.
        candidates = (
            candidates.with_columns(pl.col("ts_event").dt.date().alias("_date"))
            .join(schedule, left_on="_date", right_on="date", how="left")
            .drop("_date")
        )

        # If any leg uses IV-stepped delta, merge indicator columns into candidates
        # now so _select_legs can resolve per-row effective delta/tolerance. A
        # session-scoped backward as-of join (not exact equality) ensures a coarse
        # step source — e.g. a daily regime signal — resolves for every intraday
        # candidate in the session, not only candidates landing on its timestamps.
        has_stepped_delta = any(
            isinstance(leg.delta, SteppedDeltaConfig)
            or leg.stepped is not None
            or leg.targets is not None
            for leg in self.trade.legs
        )
        if has_stepped_delta:
            candidates = self._asof_join_indicators(candidates, indicators, "ts_event")

        candidates = self._select_legs(candidates)
        if candidates.is_empty():
            return _empty_entries_df()

        candidates = self._compute_open_mark(candidates)
        candidates = self._evaluate_conditions(candidates, indicators)

        if candidates.is_empty():
            return _empty_entries_df()

        return candidates.with_row_index("entry_id", offset=entry_id_offset).with_columns(
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

        bars = bars.with_columns(pl.col("ts_event").dt.convert_time_zone(tz_str).alias("_ts_local"))

        if session.weekdays_only:
            bars = bars.filter(pl.col("_ts_local").dt.weekday() < 5)

        if session.skip_dates:
            bars = bars.filter(
                ~pl.col("_ts_local").dt.date().is_in([d for d in session.skip_dates])
            )

        # Cast to Int32 before multiplication to prevent i8 overflow (3600 * 23 > 127).
        start_sec = window.start.hour * 3600 + window.start.minute * 60
        end_sec = window.end.hour * 3600 + window.end.minute * 60
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
        ts_events = candidates["ts_event"].to_list()
        underlying_ids = candidates["underlying_id"].to_list()

        delta_legs = [leg for leg in self.trade.legs if leg.strike_offset is None]
        offset_legs = [leg for leg in self.trade.legs if leg.strike_offset is not None]

        # Build ts_event_underlying, extending with the next quarterly contract for
        # any bar where the front-month expiry is within the target DTE window.
        # Options at high DTE are listed under the next contract once the front-month
        # option chain runs out, so querying only the front contract misses them.
        _max_dte_needed: int | None = None
        if delta_legs and "expiry" in candidates.columns and "next_underlying_id" in candidates.columns:
            _dte_reach: list[int] = []
            for leg in delta_legs:
                if leg.stepped is not None:
                    _dte_reach.extend(
                        int(s.dte) + (s.dte_tolerance if s.dte_tolerance is not None else leg.dte_tolerance)
                        for s in leg.stepped.steps
                    )
                elif leg.targets is not None:
                    _dte_reach.extend(
                        int(t.dte) + (t.dte_tolerance if t.dte_tolerance is not None else leg.dte_tolerance)
                        for t in leg.targets.values()
                    )
                elif leg.dte is not None:
                    _dte_reach.append(int(leg.dte) + leg.dte_tolerance)
            _max_dte_needed = max(_dte_reach) if _dte_reach else None

        if _max_dte_needed is not None:
            expiry_col = candidates["expiry"].to_list()
            next_uid_col = candidates["next_underlying_id"].to_list()
            ts_event_underlying: list[tuple] = []
            for ts, uid, exp, nuid in zip(ts_events, underlying_ids, expiry_col, next_uid_col, strict=False):
                ts_event_underlying.append((ts, uid))
                if nuid is not None and exp is not None and (exp - ts.date()).days <= _max_dte_needed:
                    ts_event_underlying.append((ts, int(nuid)))
        else:
            ts_event_underlying = list(zip(ts_events, underlying_ids, strict=False))

        result = candidates.select(["ts_event", "underlying_id"])
        target_lookup: pl.DataFrame | None = None

        # ------------------------------------------------------------------
        # Pass 1: delta-selected legs (single batched DB query, or grouped
        #         dispatch when any leg uses IV-stepped delta)
        # ------------------------------------------------------------------
        if delta_legs:
            has_stepped = any(
                isinstance(leg.delta, SteppedDeltaConfig)
                or leg.stepped is not None
                or leg.targets is not None
                for leg in delta_legs
            )
            # The single leg (if any) whose winning target tags the position.
            targets_leg = next((leg for leg in delta_legs if leg.targets is not None), None)

            if has_stepped:
                # Resolve the effective (delta, delta_tol, dte, dte_tol) tuple per
                # row for each leg. Simple/stepped-delta legs contribute constant
                # dte; unified stepped legs (item 2) step dte and delta together;
                # targets legs (item 3) resolve the winning target by priority.
                eff_exprs = []
                name_size_exprs = []
                for leg in delta_legs:
                    if leg.targets is not None:
                        d_expr, t_expr, dte_expr, dtetol_expr, name_expr, size_expr = (
                            self._build_targets_exprs(leg)
                        )
                        name_size_exprs.append(name_expr.alias("_target_name"))
                        name_size_exprs.append(size_expr.alias("_size_mult"))
                    elif leg.stepped is not None:
                        d_expr, t_expr, dte_expr, dtetol_expr = self._build_stepped_leg_exprs(leg)
                    elif isinstance(leg.delta, SteppedDeltaConfig):
                        d_expr, t_expr = self._build_step_exprs(leg.delta)
                        dte_expr = pl.lit(int(leg.dte))
                        dtetol_expr = pl.lit(int(leg.dte_tolerance))
                    else:
                        assert isinstance(leg.delta, SimpleDeltaConfig)
                        d_expr = pl.lit(float(leg.delta.target))
                        t_expr = pl.lit(float(leg.delta.tolerance))
                        dte_expr = pl.lit(int(leg.dte))
                        dtetol_expr = pl.lit(int(leg.dte_tolerance))
                    eff_exprs.append(d_expr.alias(f"_eff_delta_{leg.name}"))
                    eff_exprs.append(t_expr.alias(f"_eff_tol_{leg.name}"))
                    eff_exprs.append(dte_expr.alias(f"_eff_dte_{leg.name}"))
                    eff_exprs.append(dtetol_expr.alias(f"_eff_dtetol_{leg.name}"))
                work = candidates.with_columns(eff_exprs + name_size_exprs)

                # Drop rows where the effective delta or dte resolved to null (no
                # step matched and there was no catch-all).
                for leg in delta_legs:
                    work = work.filter(
                        pl.col(f"_eff_delta_{leg.name}").is_not_null()
                        & pl.col(f"_eff_dte_{leg.name}").is_not_null()
                    )
                if work.is_empty():
                    return pl.DataFrame()

                # Per-candidate winning target name/size, tagged onto the final
                # result by exact ts_event (item 3 position attribution).
                if targets_leg is not None:
                    target_lookup = work.select(["ts_event", "_target_name", "_size_mult"])

                # Group by unique (eff_delta, eff_tol, eff_dte, eff_dtetol) tuples
                # across all legs and issue one batched greeks query per group.
                group_cols = [
                    c
                    for leg in delta_legs
                    for c in (
                        f"_eff_delta_{leg.name}",
                        f"_eff_tol_{leg.name}",
                        f"_eff_dte_{leg.name}",
                        f"_eff_dtetol_{leg.name}",
                    )
                ]
                unique_groups = work.select(group_cols).unique()

                # Select each group independently and concatenate. Best-matching
                # and joining PER GROUP is essential: greeks_for_all_legs is a
                # time-range query, so a single query returns options across the
                # whole window — globally best-matching would let one bucket's
                # options be picked for another bucket's candidates. Scoping the
                # match to each group's own candidates keeps buckets isolated.
                group_results: list[pl.DataFrame] = []
                for row in unique_groups.iter_rows(named=True):
                    mask = pl.lit(True)
                    for leg in delta_legs:
                        for col in (
                            f"_eff_delta_{leg.name}",
                            f"_eff_tol_{leg.name}",
                            f"_eff_dte_{leg.name}",
                            f"_eff_dtetol_{leg.name}",
                        ):
                            mask = mask & (pl.col(col) == pl.lit(row[col]))
                    grp = work.filter(mask)

                    grp_max_dte = max(
                        int(row[f"_eff_dte_{leg.name}"]) + int(row[f"_eff_dtetol_{leg.name}"])
                        for leg in delta_legs
                    )
                    if "expiry" in grp.columns and "next_underlying_id" in grp.columns:
                        teu: list[tuple] = []
                        for ts, uid, exp, nuid in zip(
                            grp["ts_event"].to_list(),
                            grp["underlying_id"].to_list(),
                            grp["expiry"].to_list(),
                            grp["next_underlying_id"].to_list(),
                            strict=False,
                        ):
                            teu.append((ts, uid))
                            if nuid is not None and exp is not None and (exp - ts.date()).days <= grp_max_dte:
                                teu.append((ts, int(nuid)))
                    else:
                        teu = list(zip(grp["ts_event"].to_list(), grp["underlying_id"].to_list(), strict=False))
                    specs = [
                        {
                            "name": leg.name,
                            "right": "C" if leg.right == "call" else "P",
                            "target_delta": float(row[f"_eff_delta_{leg.name}"]),
                            "target_dte": int(row[f"_eff_dte_{leg.name}"]),
                            "delta_tolerance": float(row[f"_eff_tol_{leg.name}"]),
                            "dte_tolerance": int(row[f"_eff_dtetol_{leg.name}"]),
                        }
                        for leg in delta_legs
                    ]
                    grp_opts = self.db.greeks_for_all_legs(
                        ts_event_underlying=teu,
                        leg_specs=specs,
                        audit_filter_codes=self._audit_filter_codes or None,
                    )
                    if grp_opts.is_empty():
                        continue

                    grp_res = grp.select(["ts_event", "underlying_id"])
                    ok = True
                    for leg in delta_legs:
                        leg_df = grp_opts.filter(pl.col("leg_name") == leg.name)
                        if leg_df.is_empty():
                            ok = False
                            break
                        grp_res = self._attach_delta_leg(
                            grp_res, leg_df, leg, float(row[f"_eff_delta_{leg.name}"])
                        )
                        if grp_res.is_empty():
                            ok = False
                            break
                    if ok and not grp_res.is_empty():
                        group_results.append(grp_res)

                if not group_results:
                    return pl.DataFrame()
                result = pl.concat(group_results, how="diagonal")
            else:
                # All simple delta — single batched DB query (fast path)
                leg_specs = [
                    {
                        "name": leg.name,
                        "right": "C" if leg.right == "call" else "P",
                        "target_delta": float(leg.delta.target),
                        "target_dte": int(leg.dte),
                        "delta_tolerance": float(leg.delta.tolerance),
                        "dte_tolerance": int(leg.dte_tolerance),
                    }
                    for leg in delta_legs
                ]
                all_candidates = self.db.greeks_for_all_legs(
                    ts_event_underlying=ts_event_underlying,
                    leg_specs=leg_specs,
                    audit_filter_codes=self._audit_filter_codes or None,
                )
                if all_candidates.is_empty():
                    return pl.DataFrame()

                partitions: dict[str, pl.DataFrame] = {}
                for key, group_df in all_candidates.group_by("leg_name"):
                    leg_name = key[0] if isinstance(key, (list, tuple)) else key
                    partitions[leg_name] = group_df

                for leg in delta_legs:
                    leg_df = partitions.get(leg.name)
                    if leg_df is None or leg_df.is_empty():
                        return pl.DataFrame()
                    result = self._attach_delta_leg(result, leg_df, leg, float(leg.delta.target))

        # ------------------------------------------------------------------
        # Pass 2: strike-offset legs (one DB query per leg)
        # ------------------------------------------------------------------
        for leg in offset_legs:
            ref_strike_col = f"leg_{leg.reference_leg}_strike_price"
            if ref_strike_col not in result.columns:
                return pl.DataFrame()

            right_char = "C" if leg.right == "call" else "P"
            ref_expiry_col = f"leg_{leg.reference_leg}_expiration"

            # Use the actual contract where the reference leg's option was found,
            # not the stale front-month id, so greeks_for_strike_legs queries
            # the right underlying when the option is on the next quarterly contract.
            ref_underlying_col = f"leg_{leg.reference_leg}_underlying_id"
            strike_targets = result.select(
                [
                    "ts_event",
                    pl.col(ref_underlying_col).alias("underlying_id"),
                    pl.lit(leg.name).alias("leg_name"),
                    pl.lit(right_char).alias("right"),
                    (pl.col(ref_strike_col) + pl.lit(float(leg.strike_offset))).alias(
                        "target_strike"
                    ),
                    pl.col(ref_expiry_col).alias("reference_expiration"),
                ]
            )

            offset_candidates = self.db.greeks_for_strike_legs(
                strike_targets=strike_targets,
                audit_filter_codes=self._audit_filter_codes or None,
            )

            if offset_candidates.is_empty():
                return pl.DataFrame()

            target_strikes = strike_targets.select(["ts_event", "target_strike"])
            best = (
                offset_candidates.join(target_strikes, on="ts_event", how="left")
                .with_columns(
                    (pl.col("strike_price") - pl.col("target_strike")).abs().alias("_strike_diff")
                )
                .sort(["ts_event", "_strike_diff"])
                .unique(subset=["ts_event"], keep="first")
                .drop(["_strike_diff", "leg_name", "underlying_id", "target_strike"])
            )

            rename_map = {col: f"leg_{leg.name}_{col}" for col in best.columns if col != "ts_event"}
            best = best.rename(rename_map).with_columns(
                [
                    pl.lit(leg.action).alias(f"leg_{leg.name}_action"),
                    pl.lit(leg.quantity).alias(f"leg_{leg.name}_quantity"),
                ]
            )

            result = result.join(best, on="ts_event", how="inner")

        bar_cols = candidates.select(["ts_event", "open", "high", "low", "close", "volume"])
        result = result.drop("underlying_id").join(bar_cols, on="ts_event", how="left")

        # Tag each surviving candidate with its winning target name (item 3).
        if target_lookup is not None:
            result = result.join(target_lookup, on="ts_event", how="left")

        return result

    def _attach_delta_leg(
        self,
        base: pl.DataFrame,
        leg_df: pl.DataFrame,
        leg: LegConfig,
        target_delta: float,
    ) -> pl.DataFrame:
        """
        Best-match one delta-selected leg to `target_delta` per ts_event, prefix
        its columns, and ASOF-join it onto `base` (a candidate frame with a
        ts_event column).

        The nearest-time match is capped by entry.time_tolerance seconds:
        time_tolerance=0 (default) requires an exact timestamp match; positive
        values absorb ingest-pipeline offsets without risking far-in-time price
        staleness.  Candidates with no match are dropped.
        """
        best_per_ts = (
            leg_df.with_columns(
                (pl.col("delta") - pl.lit(target_delta)).abs().alias("_delta_diff")
            )
            .sort(["ts_event", "_delta_diff"])
            .unique(subset=["ts_event"], keep="first")
            .drop(["_delta_diff", "leg_name"])
        )
        best_per_ts = best_per_ts.rename(
            {c: f"leg_{leg.name}_{c}" for c in best_per_ts.columns if c != "ts_event"}
        ).with_columns(pl.col("ts_event").dt.date().alias("_asof_date"))

        _tol = timedelta(seconds=self.trade.entry.time_tolerance)
        return (
            base.with_columns(pl.col("ts_event").dt.date().alias("_asof_date"))
            .sort("ts_event")
            .join_asof(
                best_per_ts.sort("ts_event"),
                on="ts_event",
                by="_asof_date",
                strategy="nearest",
                tolerance=_tol,
                check_sortedness=False,
            )
            .drop("_asof_date")
            .filter(pl.col(f"leg_{leg.name}_instrument_id").is_not_null())
            .with_columns([
                pl.lit(leg.action).alias(f"leg_{leg.name}_action"),
                pl.lit(leg.quantity).alias(f"leg_{leg.name}_quantity"),
            ])
        )

    # ------------------------------------------------------------------
    # Step 3: Open mark + TP/SL prices
    # ------------------------------------------------------------------

    def _compute_open_mark(self, entries: pl.DataFrame) -> pl.DataFrame:
        tick = self.trade.instrument.tick_size

        mark_expr = pl.lit(0.0)
        for leg in self.trade.legs:
            sign = 1.0 if leg.action == "sell_to_open" else -1.0
            mark_expr = mark_expr + pl.col(f"leg_{leg.name}_close") * pl.lit(sign * leg.quantity)

        exit_cfg = self.trade.exit
        entries = entries.with_columns(
            [tick_round_expr(mark_expr, tick).alias("open_mark")]
        )

        if isinstance(exit_cfg.take_profit, TakeProfitConfig):
            if exit_cfg.take_profit.pct is not None:
                tp_raw = pl.col("open_mark") * pl.lit(1.0 - float(exit_cfg.take_profit.pct))
            else:
                tp_raw = pl.col("open_mark") - pl.lit(float(exit_cfg.take_profit.price))
        elif exit_cfg.take_profit_pct is not None:
            # tp_price = open_mark × (1 - pct): exit when mark falls to this fraction
            tp_raw = pl.col("open_mark") * pl.lit(1.0 - float(exit_cfg.take_profit_pct))
        elif exit_cfg.take_profit is not None:
            tp_raw = pl.col("open_mark") - pl.lit(float(exit_cfg.take_profit))
        else:
            tp_raw = None

        tp_expr = (
            tick_round_expr(tp_raw, tick).alias("tp_price")
            if tp_raw is not None
            else pl.lit(None).cast(pl.Float64).alias("tp_price")
        )

        if exit_cfg.stop_loss is None:
            sl_expr = pl.lit(None).cast(pl.Float64).alias("sl_price")
        else:
            sl_price = (
                float(exit_cfg.stop_loss.price)
                if isinstance(exit_cfg.stop_loss, StopLossConfig)
                else float(exit_cfg.stop_loss)
            )
            sl_expr = tick_round_expr(
                pl.col("open_mark") + pl.lit(sl_price), tick
            ).alias("sl_price")

        entries = entries.with_columns(
            [
                tp_expr,
                sl_expr,
                pl.lit(exit_cfg.dte_exit).cast(pl.Int32).alias("dte_exit"),
            ]
        )

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

        # Session-scoped backward as-of merge: each entry receives the most recent
        # indicator value at or before its timestamp within the session, so coarse
        # (daily / 5-minute) signals gate all intraday entries rather than only
        # those aligned to an indicator timestamp.
        entries = self._asof_join_indicators(entries, indicators, "entry_time")

        for cond_str in entry_cfg.conditions:
            expr = parse_condition(cond_str)
            try:
                entries = entries.filter(expr)
            except Exception as e:
                self.warnings.append(
                    {
                        "phase": "entry",
                        "trade": self.trade.name,
                        "type": "condition_error",
                        "condition": cond_str,
                        "error": str(e),
                    }
                )
                return entries.clear()

        if entry_cfg.min_credit is not None:
            entries = entries.filter(pl.col("open_mark") >= float(entry_cfg.min_credit))

        if entry_cfg.max_debit is not None:
            entries = entries.filter(pl.col("open_mark") <= float(entry_cfg.max_debit))

        return entries

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _asof_join_indicators(
        self,
        frame: pl.DataFrame,
        indicators: pl.DataFrame,
        ts_col: str,
    ) -> pl.DataFrame:
        """
        Session-scoped backward as-of merge of indicator columns onto `frame`.

        Each row of `frame` receives the most recent value at or before its
        `ts_col` timestamp, per indicator column, within the same trading session
        (the session-local calendar date).  Values are never filled across a
        session boundary: a row that precedes the session's first indicator value
        gets null (and, when used in a condition or step, is skipped).

        Coarser-than-1-minute indicators — a daily regime signal, a 5-minute VES
        value — only land on a handful of the 1-minute entry grid's timestamps.
        An exact-equality join would drop every unaligned candidate silently; a
        backward as-of join instead carries the latest value forward within the
        session so it gates *all* the session's entries.

        Mixed-cadence indicators sharing one wide frame are handled by
        forward-filling each column within session before the as-of join, so a
        row still carries the latest value of a column that was absent on the
        exact matched indicator row.

        Only indicator columns not already present in `frame` are added.  `ts_col`
        is the frame's timestamp column ("ts_event" pre-rename, "entry_time"
        post-rename).
        """
        if indicators.is_empty() or "ts_event" not in indicators.columns:
            return frame

        new_cols = [
            c for c in indicators.columns if c != "ts_event" and c not in frame.columns
        ]
        if not new_cols:
            return frame

        tz_str = self.strategy.universe.session.timezone

        ind = (
            indicators.select(["ts_event"] + new_cols)
            .sort("ts_event")
            .with_columns(
                pl.col("ts_event").dt.convert_time_zone(tz_str).dt.date().alias("_session")
            )
            .with_columns([pl.col(c).forward_fill().over("_session") for c in new_cols])
        )
        if ts_col != "ts_event":
            ind = ind.rename({"ts_event": ts_col})

        return (
            frame.sort(ts_col)
            .with_columns(
                pl.col(ts_col).dt.convert_time_zone(tz_str).dt.date().alias("_session")
            )
            .join_asof(
                ind, on=ts_col, by="_session", strategy="backward", check_sortedness=False
            )
            .drop("_session")
        )

    def _get_indicators_for_schedule(
        self,
        schedule: pl.DataFrame,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pl.DataFrame:
        """
        Return the indicators DataFrame to use for condition evaluation.

        Queries db.indicators() for every distinct underlying_id in the roll
        schedule and concatenates the results, so indicator coverage is
        continuous across quarterly contract rolls. If the engine pre-loaded
        indicators those are used as-is (the caller already handled stitching).
        Returns an empty DataFrame when no conditions are configured or no
        indicators exist in the DB.
        """
        if self._preloaded_indicators is not None:
            return self._preloaded_indicators

        has_stepped_delta = any(
            isinstance(leg.delta, SteppedDeltaConfig)
            or leg.stepped is not None
            or leg.targets is not None
            for leg in self.trade.legs
        )
        if not self.trade.entry.conditions and not has_stepped_delta:
            return pl.DataFrame()

        frames = [
            self.db.indicators(uid, start_dt, end_dt)
            for uid in schedule["underlying_id"].unique().to_list()
        ]
        non_empty = [f for f in frames if not f.is_empty() and len(f.columns) > 1]
        if not non_empty:
            return pl.DataFrame()
        if len(non_empty) == 1:
            return non_empty[0].sort("ts_event")
        return pl.concat(non_empty, how="diagonal").sort("ts_event")

    def _build_step_exprs(
        self,
        stepped: SteppedDeltaConfig,
    ) -> tuple[pl.Expr, pl.Expr]:
        """
        Return (delta_expr, tolerance_expr) Polars when/then chains for a
        SteppedDeltaConfig.  Steps with `below` are evaluated in order; the
        first where source_col < below wins.  A catch-all step (no `below`)
        handles everything else.  If no step matches and there is no catch-all,
        both expressions resolve to null (rows will be dropped by the caller).
        """
        source = stepped.step_source
        with_below = [s for s in stepped.steps if s.below is not None]
        catch_all = next((s for s in stepped.steps if s.below is None), None)

        d_chain: pl.Expr | None = None
        t_chain: pl.Expr | None = None
        for s in with_below:
            cond = pl.col(source) < pl.lit(float(s.below))
            d_val = pl.lit(float(s.target))
            t_val = pl.lit(float(s.tolerance) if s.tolerance is not None else float(stepped.tolerance))
            if d_chain is None:
                d_chain = pl.when(cond).then(d_val)
                t_chain = pl.when(cond).then(t_val)
            else:
                d_chain = d_chain.when(cond).then(d_val)
                t_chain = t_chain.when(cond).then(t_val)

        if catch_all is not None:
            fallback_d = pl.lit(float(catch_all.target))
            fallback_t = pl.lit(float(catch_all.tolerance) if catch_all.tolerance is not None else float(stepped.tolerance))
        else:
            fallback_d = pl.lit(None).cast(pl.Float64)
            fallback_t = pl.lit(None).cast(pl.Float64)

        delta_expr = d_chain.otherwise(fallback_d) if d_chain is not None else fallback_d
        tol_expr = t_chain.otherwise(fallback_t) if t_chain is not None else fallback_t
        return delta_expr, tol_expr

    def _build_stepped_leg_exprs(
        self,
        leg: LegConfig,
    ) -> tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
        """
        Return (delta, delta_tol, dte, dte_tol) Polars when/then chains for a
        leg with a unified `stepped:` block (item 2).

        Steps with `below` are evaluated in declaration order; the first where
        `source < below` wins.  A trailing step without `below` is the catch-all.
        If no step matches and there is no catch-all, all four expressions
        resolve to null and the caller drops the row.

        Omitted per-step tolerances fall back to defaults: delta_tolerance →
        0.10; dte_tolerance → the leg's dte_tolerance (default 5).  `dte` and
        `delta` are required on every step by the schema.
        """
        stepped = leg.stepped
        source = stepped.source
        default_dtol = 0.10
        default_dtetol = int(leg.dte_tolerance)
        with_below = [s for s in stepped.steps if s.below is not None]
        catch_all = next((s for s in stepped.steps if s.below is None), None)

        d_chain: pl.Expr | None = None
        dtol_chain: pl.Expr | None = None
        dte_chain: pl.Expr | None = None
        dtetol_chain: pl.Expr | None = None
        for s in with_below:
            cond = pl.col(source) < pl.lit(float(s.below))
            d_val = pl.lit(float(s.delta))
            dtol_val = pl.lit(
                float(s.delta_tolerance) if s.delta_tolerance is not None else default_dtol
            )
            dte_val = pl.lit(int(s.dte))
            dtetol_val = pl.lit(
                int(s.dte_tolerance) if s.dte_tolerance is not None else default_dtetol
            )
            if d_chain is None:
                d_chain = pl.when(cond).then(d_val)
                dtol_chain = pl.when(cond).then(dtol_val)
                dte_chain = pl.when(cond).then(dte_val)
                dtetol_chain = pl.when(cond).then(dtetol_val)
            else:
                d_chain = d_chain.when(cond).then(d_val)
                dtol_chain = dtol_chain.when(cond).then(dtol_val)
                dte_chain = dte_chain.when(cond).then(dte_val)
                dtetol_chain = dtetol_chain.when(cond).then(dtetol_val)

        if catch_all is not None:
            ca_dtol = catch_all.delta_tolerance
            ca_dtetol = catch_all.dte_tolerance
            fb_d = pl.lit(float(catch_all.delta))
            fb_dtol = pl.lit(float(ca_dtol if ca_dtol is not None else default_dtol))
            fb_dte = pl.lit(int(catch_all.dte))
            fb_dtetol = pl.lit(int(ca_dtetol if ca_dtetol is not None else default_dtetol))
        else:
            fb_d = pl.lit(None).cast(pl.Float64)
            fb_dtol = pl.lit(None).cast(pl.Float64)
            fb_dte = pl.lit(None).cast(pl.Int64)
            fb_dtetol = pl.lit(None).cast(pl.Int64)

        d_expr = d_chain.otherwise(fb_d) if d_chain is not None else fb_d
        dtol_expr = dtol_chain.otherwise(fb_dtol) if dtol_chain is not None else fb_dtol
        dte_expr = dte_chain.otherwise(fb_dte) if dte_chain is not None else fb_dte
        dtetol_expr = dtetol_chain.otherwise(fb_dtetol) if dtetol_chain is not None else fb_dtetol
        return d_expr, dtol_expr, dte_expr, dtetol_expr

    def _build_targets_exprs(
        self,
        leg: LegConfig,
    ) -> tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
        """
        Return (delta, delta_tol, dte, dte_tol, target_name, size_mult) Polars
        when/then chains for a leg with a `targets:` map (item 3).

        Every non-`default` target's `condition` is evaluated per row; among
        those true the highest-`priority` target wins.  Determinism is achieved
        by building the when/then chain in descending priority order, so the
        first matching branch is always the highest-priority match.  The reserved
        `default` target (no condition/priority) is the otherwise branch; when
        absent the tuple resolves to null and the row is dropped (entry skipped).

        Omitted per-target tolerances fall back to defaults: delta_tolerance →
        0.10; dte_tolerance → the leg's dte_tolerance.
        """
        default_dtol = 0.10
        default_dtetol = int(leg.dte_tolerance)
        named = sorted(
            ((name, t) for name, t in leg.targets.items() if name != "default"),
            key=lambda kv: kv[1].priority,
            reverse=True,  # highest priority first → wins ties among true conditions
        )
        default = leg.targets.get("default")

        d_chain: pl.Expr | None = None
        dtol_chain: pl.Expr | None = None
        dte_chain: pl.Expr | None = None
        dtetol_chain: pl.Expr | None = None
        name_chain: pl.Expr | None = None
        size_chain: pl.Expr | None = None
        for name, t in named:
            cond = parse_condition(t.condition)
            d_val = pl.lit(float(t.delta))
            dtol_val = pl.lit(
                float(t.delta_tolerance if t.delta_tolerance is not None else default_dtol)
            )
            dte_val = pl.lit(int(t.dte))
            dtetol_val = pl.lit(
                int(t.dte_tolerance if t.dte_tolerance is not None else default_dtetol)
            )
            name_val = pl.lit(name)
            size_val = pl.lit(float(t.size_multiplier))
            if d_chain is None:
                d_chain = pl.when(cond).then(d_val)
                dtol_chain = pl.when(cond).then(dtol_val)
                dte_chain = pl.when(cond).then(dte_val)
                dtetol_chain = pl.when(cond).then(dtetol_val)
                name_chain = pl.when(cond).then(name_val)
                size_chain = pl.when(cond).then(size_val)
            else:
                d_chain = d_chain.when(cond).then(d_val)
                dtol_chain = dtol_chain.when(cond).then(dtol_val)
                dte_chain = dte_chain.when(cond).then(dte_val)
                dtetol_chain = dtetol_chain.when(cond).then(dtetol_val)
                name_chain = name_chain.when(cond).then(name_val)
                size_chain = size_chain.when(cond).then(size_val)

        if default is not None:
            df_dtol = default.delta_tolerance
            df_dtetol = default.dte_tolerance
            fb_d = pl.lit(float(default.delta))
            fb_dtol = pl.lit(float(df_dtol if df_dtol is not None else default_dtol))
            fb_dte = pl.lit(int(default.dte))
            fb_dtetol = pl.lit(int(df_dtetol if df_dtetol is not None else default_dtetol))
            fb_name = pl.lit("default")
            fb_size = pl.lit(float(default.size_multiplier))
        else:
            fb_d = pl.lit(None).cast(pl.Float64)
            fb_dtol = pl.lit(None).cast(pl.Float64)
            fb_dte = pl.lit(None).cast(pl.Int64)
            fb_dtetol = pl.lit(None).cast(pl.Int64)
            fb_name = pl.lit(None).cast(pl.Utf8)
            fb_size = pl.lit(None).cast(pl.Float64)

        return (
            d_chain.otherwise(fb_d) if d_chain is not None else fb_d,
            dtol_chain.otherwise(fb_dtol) if dtol_chain is not None else fb_dtol,
            dte_chain.otherwise(fb_dte) if dte_chain is not None else fb_dte,
            dtetol_chain.otherwise(fb_dtetol) if dtetol_chain is not None else fb_dtetol,
            name_chain.otherwise(fb_name) if name_chain is not None else fb_name,
            size_chain.otherwise(fb_size) if size_chain is not None else fb_size,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_entries_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entry_id": pl.Series([], dtype=pl.UInt32),
            "trade_name": pl.Series([], dtype=pl.Utf8),
            "entry_time": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "open_mark": pl.Series([], dtype=pl.Float64),
            "tp_price": pl.Series([], dtype=pl.Float64),
            "sl_price": pl.Series([], dtype=pl.Float64),
            "dte_exit": pl.Series([], dtype=pl.Int32),
        }
    )
