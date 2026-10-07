"""
Prototype: compare the current production bucket-equality query (live in
btkit/db/input_db.py) against a 5-minute-resample-based alternative for the
Pass-2 strike-offset tolerant match.

Design of the resample variant, exactly as described: primary leg selection
is unchanged (this script only touches the secondary/reference-leg matching
step). Both the anchor's own timestamp and every candidate's timestamp are
floored to a fixed 5-minute grid; the candidate pool is resampled to one bar
per (instrument_id, grid cell) using the LAST tick within that cell (the
standard "resample and take last" convention); matching is then EXACT
equality on the grid cell (no neighbor-bucket search, unlike the production
query) plus the existing right/expiration/strike-tolerance conditions.

This is NOT a correctness-parity comparison like the earlier ASOF and bucket
prototypes -- the resample variant is deliberately a different, coarser
matching semantic, not a faster way to compute the same answer. So instead of
asserting the two queries agree, this script measures how much they diverge:
for every anchor both queries manage to match, it records the entry
(close) price difference and the time difference between the resampled
candidate's real tick and the anchor's real timestamp -- including whether
that tick is AFTER the anchor's real time, which would mean the resampled
match used data not yet available at the actual moment of entry (look-ahead).
That risk was flagged when this idea was proposed and is exactly what this
script needs to quantify with real data before it could be considered as
anything more than a performance curiosity.

Usage:
    python scripts/compare_resample_strike_legs.py
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import date, timedelta
from datetime import time as dtime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl

from btkit.backtest.entry import EntryScanner
from btkit.db.input_db import InputDatabase
from btkit.strategy.definition import (
    EntryConfig,
    EntryWindowConfig,
    ExitConfig,
    InstrumentConfig,
    LegConfig,
    StrategyDefinition,
    TradeDefinition,
    UniverseConfig,
)

# ---------------------------------------------------------------------------
# Same real worst-case combination used throughout this investigation.
# ---------------------------------------------------------------------------

INPUT_DB = "/Users/jackrochester/dev/wealthy-option-backtest/db/input/es.db"

NARROW_START = date(2026, 1, 1)
NARROW_END = date(2026, 5, 31)

RESAMPLE_SECONDS = 300  # 5 minutes, per the request

STRATEGY = StrategyDefinition(
    name="compare_resample_strike_legs",
    universe=UniverseConfig(
        start_date=NARROW_START,
        end_date=NARROW_END,
        audit_filter=[
            "BARS_TRUNCATED",
            "NEGATIVE_CLOSE",
            "NEGATIVE_DTE",
            "ZOMBIE_BAR",
            "DELTA_SIGN_ERROR",
            "DELTA_MAGNITUDE_ERROR",
            "NO_EXPIRY_BARS",
        ],
    ),
    trades=[
        TradeDefinition(
            name="diag",
            instrument=InstrumentConfig(
                root_symbol="ES",
                asset_class="future",
                tick_size=0.05,
                expiry_close_time=dtime(16, 0),
            ),
            entry=EntryConfig(
                window=EntryWindowConfig(start=dtime(9, 45), end=dtime(14, 0)),
                min_credit=0.20,
                max_entries_per_day=2,
            ),
            legs=[
                LegConfig(
                    name="short_put",
                    right="put",
                    action="sell_to_open",
                    delta={"target": -0.05, "tolerance": 0.02},
                    dte={"target": 0, "tolerance": 0},
                    dte_skip_weekends=True,
                ),
                LegConfig(
                    name="long_put",
                    right="put",
                    action="buy_to_open",
                    reference_leg={
                        "name": "short_put",
                        "strike_offset": -50.0,
                        "entry_time_tolerance": 300,
                    },
                ),
            ],
            exit=ExitConfig(stop_loss=20.0, take_profit_pct=0.90),
        )
    ],
)

CSV_OUT = str(Path(__file__).parent / "resample_comparison_partitions.csv")
DIVERGENCE_CSV_OUT = str(Path(__file__).parent / "resample_comparison_divergences.csv")

# ---------------------------------------------------------------------------
# Query text
# ---------------------------------------------------------------------------

# Copied verbatim from the current production query in btkit/db/input_db.py
# (_TOLERANT_BUCKET_QUERY_TEMPLATE) -- this IS what ships today.
_CURRENT_BUCKET_QUERY_TEMPLATE = """
WITH combined AS (
    SELECT og.ts_event, og.instrument_id, og.underlying_id, og.dte, og.iv, og.delta,
           og.gamma, og.theta, og.vega, ob.strike_price, ob.expiration, ob."right",
           ob.multiplier, ob.symbol, ob.close,
           CAST(floor(epoch(og.ts_event) / ?) AS BIGINT) AS bucket
    FROM option_greeks og
    JOIN option_bars ob ON og.instrument_id = ob.instrument_id AND og.ts_event = ob.ts_event
    WHERE ob.close IS NOT NULL
      AND og.underlying_id = ?
      AND og.ts_event BETWEEN ? AND ?
      AND ob."right" = ?
      AND ob.strike_price BETWEEN ? AND ?
      AND ob.expiration IN ({expirations_sql})
),
anchor_buckets AS (
    SELECT st.ts_event, st.leg_name, st.anchor_ts_event, st.underlying_id, st."right",
           st.target_strike, st.reference_expiration,
           CAST(floor(epoch(st.anchor_ts_event) / ?) AS BIGINT) + off.bucket_offset AS bucket
    FROM _group_targets st
    CROSS JOIN (VALUES (-1),(0),(1)) AS off(bucket_offset)
),
matched AS (
    SELECT st.ts_event AS row_id, st.leg_name, c.ts_event AS _match_ts_event,
           c.instrument_id, c.underlying_id, c.dte, c.iv, c.delta, c.gamma, c.theta,
           c.vega, c.strike_price, c.expiration, c."right", c.multiplier, c.symbol,
           c.close,
           abs(epoch(c.ts_event) - epoch(st.anchor_ts_event)) AS time_diff,
           abs(c.strike_price - st.target_strike) AS strike_diff
    FROM anchor_buckets st
    JOIN combined c
      ON c.bucket = st.bucket
     AND c.underlying_id = st.underlying_id
     AND c."right" = st."right"
     AND c.expiration = st.reference_expiration
     AND abs(c.strike_price - st.target_strike) <= ?
)
SELECT leg_name, row_id AS ts_event, _match_ts_event, instrument_id, underlying_id,
       dte, iv, delta, gamma, theta, vega, strike_price, expiration, "right",
       multiplier, symbol, close
FROM matched
WHERE time_diff <= ?
QUALIFY ROW_NUMBER() OVER (PARTITION BY row_id ORDER BY time_diff, strike_diff) = 1
"""

# New: resample candidates to one bar per (instrument_id, grid cell) using the
# LAST tick in that cell, then match on EXACT grid-cell equality -- no
# neighbor-bucket search, no time_diff filter (a resampled bar's own tick can
# be anywhere within its 5-minute cell; that's the whole point being tested).
_RESAMPLE_QUERY_TEMPLATE = """
WITH combined AS (
    SELECT og.ts_event, og.instrument_id, og.underlying_id, og.dte, og.iv, og.delta,
           og.gamma, og.theta, og.vega, ob.strike_price, ob.expiration, ob."right",
           ob.multiplier, ob.symbol, ob.close
    FROM option_greeks og
    JOIN option_bars ob ON og.instrument_id = ob.instrument_id AND og.ts_event = ob.ts_event
    WHERE ob.close IS NOT NULL
      AND og.underlying_id = ?
      AND og.ts_event BETWEEN ? AND ?
      AND ob."right" = ?
      AND ob.strike_price BETWEEN ? AND ?
      AND ob.expiration IN ({expirations_sql})
),
resampled AS (
    SELECT *, CAST(floor(epoch(ts_event) / ?) AS BIGINT) AS bucket
    FROM combined
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY instrument_id, CAST(floor(epoch(ts_event) / ?) AS BIGINT)
        ORDER BY ts_event DESC
    ) = 1
),
anchor_buckets AS (
    SELECT st.ts_event, st.leg_name, st.anchor_ts_event, st.underlying_id, st."right",
           st.target_strike, st.reference_expiration,
           CAST(floor(epoch(st.anchor_ts_event) / ?) AS BIGINT) AS bucket
    FROM _group_targets st
),
matched AS (
    SELECT st.ts_event AS row_id, st.leg_name, st.anchor_ts_event, c.ts_event AS _match_ts_event,
           c.instrument_id, c.underlying_id, c.dte, c.iv, c.delta, c.gamma, c.theta,
           c.vega, c.strike_price, c.expiration, c."right", c.multiplier, c.symbol,
           c.close,
           abs(c.strike_price - st.target_strike) AS strike_diff
    FROM anchor_buckets st
    JOIN resampled c
      ON c.bucket = st.bucket
     AND c.underlying_id = st.underlying_id
     AND c."right" = st."right"
     AND c.expiration = st.reference_expiration
     AND abs(c.strike_price - st.target_strike) <= ?
)
SELECT leg_name, row_id AS ts_event, anchor_ts_event, _match_ts_event, instrument_id,
       underlying_id, dte, iv, delta, gamma, theta, vega, strike_price, expiration,
       "right", multiplier, symbol, close
FROM matched
QUALIFY ROW_NUMBER() OVER (PARTITION BY row_id ORDER BY strike_diff) = 1
"""


# ---------------------------------------------------------------------------
# Instrumented subclass
# ---------------------------------------------------------------------------


class ComparingInputDatabase(InputDatabase):
    def __init__(self, db_path: str) -> None:
        super().__init__(db_path)
        self.partition_stats: list[dict] = []
        self.divergences: list[dict] = []

    def _greeks_for_strike_legs_tolerant(
        self,
        strike_targets: pl.DataFrame,
        strike_tolerance: float,
        time_tolerance: int,
        audit_filter_codes,
    ) -> pl.DataFrame:
        _pad = timedelta(seconds=time_tolerance)
        bucket_width = float(time_tolerance) if time_tolerance > 0 else 1.0
        frames: list[pl.DataFrame] = []

        grouped = strike_targets.with_columns(
            pl.col("anchor_ts_event").dt.date().alias("_anchor_date")
        )
        n_partitions = grouped.select(["underlying_id", "_anchor_date"]).unique().height
        print(f"[compare] {n_partitions} (underlying_id, date) partitions")

        for (uid, anchor_date), group in grouped.group_by(["underlying_id", "_anchor_date"]):
            anchor_min = group["anchor_ts_event"].min() - _pad
            anchor_max = group["anchor_ts_event"].max() + _pad
            strike_min = float(group["target_strike"].min()) - strike_tolerance
            strike_max = float(group["target_strike"].max()) + strike_tolerance
            right = group["right"][0]
            expirations = group["reference_expiration"].unique().to_list()
            expirations_sql = ",".join(f"DATE '{d.isoformat()}'" for d in expirations)

            current_query = _CURRENT_BUCKET_QUERY_TEMPLATE.format(expirations_sql=expirations_sql)
            resample_query = _RESAMPLE_QUERY_TEMPLATE.format(expirations_sql=expirations_sql)

            current_params = [
                bucket_width, uid, anchor_min, anchor_max, right, strike_min, strike_max,
                bucket_width, strike_tolerance, time_tolerance,
            ]
            resample_params = [
                uid, anchor_min, anchor_max, right, strike_min, strike_max,
                RESAMPLE_SECONDS, RESAMPLE_SECONDS, RESAMPLE_SECONDS, strike_tolerance,
            ]

            self._con.register("_group_targets", group)
            try:
                t0 = time.perf_counter()
                current_df = self._con.execute(current_query, current_params).pl()
                current_elapsed = time.perf_counter() - t0

                t0 = time.perf_counter()
                resample_df = self._con.execute(resample_query, resample_params).pl()
                resample_elapsed = time.perf_counter() - t0
            finally:
                self._con.unregister("_group_targets")

            self._record_divergences(current_df, resample_df, uid, anchor_date)

            self.partition_stats.append(
                {
                    "underlying_id": uid,
                    "anchor_date": str(anchor_date),
                    "n_anchors": group.height,
                    "current_elapsed_s": current_elapsed,
                    "current_rows": current_df.height,
                    "resample_elapsed_s": resample_elapsed,
                    "resample_rows": resample_df.height,
                }
            )

            if not current_df.is_empty():
                frames.append(current_df)

        return self._finish_strike_legs(
            pl.concat(frames) if frames else pl.DataFrame(), audit_filter_codes
        )

    def _record_divergences(self, current_df, resample_df, uid, anchor_date):
        if current_df.is_empty() or resample_df.is_empty():
            return
        cur = {
            row["ts_event"]: (row["instrument_id"], row["close"], row["strike_price"])
            for row in current_df.iter_rows(named=True)
        }
        res = {
            row["ts_event"]: (row["instrument_id"], row["close"], row["anchor_ts_event"], row["_match_ts_event"])
            for row in resample_df.iter_rows(named=True)
        }
        both = set(cur) & set(res)
        for ts in both:
            cur_instrument, cur_close, cur_strike = cur[ts]
            res_instrument, res_close, anchor_ts, match_ts = res[ts]
            time_diff_s = (match_ts - anchor_ts).total_seconds()
            self.divergences.append(
                {
                    "underlying_id": uid,
                    "anchor_date": str(anchor_date),
                    "ts_event": str(ts),
                    "same_instrument": cur_instrument == res_instrument,
                    "price_diff": abs(cur_close - res_close),
                    "resample_time_diff_s": time_diff_s,
                    "resample_is_lookahead": time_diff_s > 0,
                }
            )
        self._last_only_current = len(set(cur) - set(res))
        self._last_only_resample = len(set(res) - set(cur))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


def main() -> None:
    if not Path(INPUT_DB).exists():
        print(f"ERROR: input database not found: {INPUT_DB}")
        sys.exit(1)

    print(f"btkit bucket vs {RESAMPLE_SECONDS//60}-min-resample comparison (Pass-2 tolerant match)")
    print(f"  Database: {INPUT_DB}")
    print(f"  Universe: {STRATEGY.universe.start_date} .. {STRATEGY.universe.end_date}")
    print()

    db = ComparingInputDatabase(INPUT_DB)
    trade = STRATEGY.trades[0]

    t0 = time.perf_counter()
    entries = EntryScanner(db, STRATEGY, trade).scan()
    total_elapsed = time.perf_counter() - t0

    print(f"\nscan() total wall time: {total_elapsed:.2f}s")
    print(f"entries found (current/bucket path, unaffected by this comparison): {len(entries)}")

    if not db.partition_stats:
        print("\nNo Pass-2 partitions were queried.")
        return

    current_times = [p["current_elapsed_s"] for p in db.partition_stats]
    resample_times = [p["resample_elapsed_s"] for p in db.partition_stats]
    print(f"\npartitions: {len(db.partition_stats)}")
    print(f"current (bucket, production):  sum={sum(current_times):.2f}s  "
          f"p50={_percentile(current_times, 0.5):.4f}s  max={max(current_times):.4f}s")
    print(f"resample ({RESAMPLE_SECONDS//60}min):              sum={sum(resample_times):.2f}s  "
          f"p50={_percentile(resample_times, 0.5):.4f}s  max={max(resample_times):.4f}s")
    ratio = sum(resample_times) / sum(current_times) if sum(current_times) else float("nan")
    print(f"resample/current total time ratio: {ratio:.2f}x  (< 1.0 means resample was faster)")

    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(db.partition_stats[0].keys()))
        w.writeheader()
        w.writerows(db.partition_stats)
    print(f"\nper-partition timing written to: {CSV_OUT}")

    print(f"\n{'=' * 70}")
    print("ENTRY PRICE / TIMING DIVERGENCE (anchors matched by both queries)")
    print("=" * 70)
    if not db.divergences:
        print("No anchors were matched by both queries -- nothing to compare.")
        return

    n = len(db.divergences)
    same_instrument = sum(1 for d in db.divergences if d["same_instrument"])
    price_diffs = [d["price_diff"] for d in db.divergences]
    time_diffs = [d["resample_time_diff_s"] for d in db.divergences]
    lookahead = [d for d in db.divergences if d["resample_is_lookahead"]]

    print(f"anchors compared: {n}")
    print(f"same instrument selected: {same_instrument}/{n} ({100*same_instrument/n:.1f}%)")
    print(
        f"entry price diff ($):  mean={sum(price_diffs)/n:.4f}  "
        f"p50={_percentile(price_diffs, 0.5):.4f}  "
        f"p95={_percentile(price_diffs, 0.95):.4f}  "
        f"max={max(price_diffs):.4f}"
    )
    print(
        f"resample tick vs anchor time (s):  mean={sum(time_diffs)/n:.1f}  "
        f"p50={_percentile(time_diffs, 0.5):.1f}  "
        f"min={min(time_diffs):.1f}  max={max(time_diffs):.1f}"
    )
    print(
        f"\n*** LOOK-AHEAD: resampled match used a tick AFTER the anchor's real "
        f"time in {len(lookahead)}/{n} cases ({100*len(lookahead)/n:.1f}%) ***"
    )
    if lookahead:
        la_diffs = [d["resample_time_diff_s"] for d in lookahead]
        print(
            f"    look-ahead magnitude (s): mean={sum(la_diffs)/len(la_diffs):.1f}  "
            f"max={max(la_diffs):.1f}"
        )

    with open(DIVERGENCE_CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(db.divergences[0].keys()))
        w.writeheader()
        w.writerows(db.divergences)
    print(f"\nfull per-anchor divergence detail written to: {DIVERGENCE_CSV_OUT}")


if __name__ == "__main__":
    main()
