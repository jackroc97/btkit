"""
Compare the current BETWEEN+QUALIFY per-partition query in
InputDatabase._greeks_for_strike_legs_tolerant against a bucketed-equality-join
prototype: round both anchor and candidate timestamps down to a fixed-width
time bucket, join on bucket equality (+ the existing right/expiration/strike
conditions) instead of a range predicate, then reduce with the same
nearest-time-then-nearest-strike QUALIFY as today.

Why this is a different bet than the ASOF prototype (see
compare_asof_strike_legs.py, which measured ~1.4x SLOWER than the current
query — DuckDB 1.5.5's ASOF JOIN still uses NESTED_LOOP_JOIN internally, and
needs two passes for bidirectional nearest-match): bucketing turns the time
condition into a plain equality, which is what makes the time_tolerance=0
exact-match path fast today (hash join). This prototype tests whether a real
equality-based join actually gets DuckDB to pick a HASH_JOIN for the tolerant
path too, not just in theory.

Correctness hazard unique to this approach (not present in the BETWEEN or ASOF
versions): flooring to a fixed grid means a candidate just on the other side of
a bucket boundary from its anchor can be very close in time yet land in a
different bucket. A naive "match my own bucket only" implementation would
silently drop it. This prototype's fix: match against the anchor's bucket AND
its immediate neighbor buckets (a small, fixed offset range), wide enough to
provably cover the full +/-time_tolerance window given bucket_width ==
time_tolerance (derivation: with bucket_width W == tolerance T, the window
[t-T, t+T] has width 2W and can therefore span at most 3 consecutive buckets,
so neighbor offsets -1..+1 are provably sufficient; this script uses -2..+2 for
margin against floating-point epoch() boundary edge cases). The final
`WHERE time_diff <= time_tolerance` filter is the actual correctness
guarantee regardless of bucket math — extra neighbor buckets can only add
candidates that get filtered back out, never cause a false match; the only way
bucket math can break correctness is by checking too FEW neighbors and
missing a true match, which is why this script's self-test specifically
targets that failure mode with boundary-adjacent synthetic cases (see
verify step in this file's __main__ self-test note, or the accompanying
selftest run).

This script must not modify InputDatabase, must not touch
_greeks_for_strike_legs_tolerant, and only measures — same constraints as
compare_asof_strike_legs.py.

Usage:
    python scripts/compare_bucket_strike_legs.py
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
# Same real worst-case combination used in diagnose_strike_legs_perf.py and
# compare_asof_strike_legs.py — narrow one-month universe first, per the same
# staged-verification discipline used for the ASOF prototype.
# ---------------------------------------------------------------------------

INPUT_DB = "/Users/jackrochester/dev/wealthy-option-backtest/db/input/es.db"

# Full 4-year universe, matching diagnose_strike_legs_perf.py's original
# real-scale run (911 partitions, ~31s for the BETWEEN-only loop). This
# script still runs the same per-(underlying_id, date) bounded loop that run
# already validated as memory-stable at this scale -- it just runs a second
# (bucket) query per partition alongside the first, in the same process, no
# worker subprocesses. Widened here after both the one-month and the
# worst-partition-targeted (2026-01..2026-05) windows passed parity with a
# clear, size-dependent speedup (up to 4.5x on the largest partitions).
NARROW_START = date(2022, 5, 1)
NARROW_END = date(2026, 6, 3)

STRATEGY = StrategyDefinition(
    name="compare_bucket_strike_legs",
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

N_EXPLAIN_SAMPLE = 5
CSV_OUT = str(Path(__file__).parent / "bucket_comparison_partitions.csv")

# Neighbor bucket offsets checked around each anchor's own bucket. Provably
# sufficient at -1..+1 when bucket_width == time_tolerance (see module
# docstring). Previously used -2..+2 as a margin against epoch() float
# boundary cases -- tightened to the proven minimum; see the boundary-case
# self-test run before this was trusted against real data.
_BUCKET_OFFSETS = (-1, 0, 1)


# ---------------------------------------------------------------------------
# Query text
# ---------------------------------------------------------------------------

_BETWEEN_QUERY = """
SELECT st.leg_name,
       st.ts_event,
       og.ts_event AS _match_ts_event,
       og.instrument_id,
       CAST(og.underlying_id AS BIGINT) AS underlying_id,
       og.dte, og.iv, og.delta, og.gamma, og.theta, og.vega,
       ob.strike_price, ob.expiration, ob."right", ob.multiplier,
       ob.symbol, ob.close
FROM option_greeks og
JOIN option_bars ob
  ON og.instrument_id = ob.instrument_id
 AND og.ts_event      = ob.ts_event
 AND ob."right"       = ?
 AND ob.strike_price BETWEEN ? AND ?
 AND ob.close IS NOT NULL
JOIN _group_targets st
  ON og.ts_event BETWEEN st.anchor_ts_event - INTERVAL (?) SECOND
                      AND st.anchor_ts_event + INTERVAL (?) SECOND
 AND ob.expiration = st.reference_expiration
 AND abs(ob.strike_price - st.target_strike) <= ?
WHERE og.underlying_id = ?
  AND og.ts_event BETWEEN ? AND ?
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY st.ts_event
    ORDER BY abs(epoch(og.ts_event) - epoch(st.anchor_ts_event)),
             abs(ob.strike_price - st.target_strike)
) = 1
"""

_OFFSET_VALUES_SQL = ",".join(f"({o})" for o in _BUCKET_OFFSETS)

# {expirations_sql} is filled in per-partition at execution time (the set of
# reference_expirations present varies by partition, unlike the bucket
# offsets, which are fixed). Inlined as SQL literals rather than a `= ANY(?)`
# bind parameter: EXPLAIN ANALYZE showed `= ANY(?)` compiles to an extra
# HASH_JOIN against an UNNEST of the list parameter, not a cheap inline
# filter -- see the comparison this replaced. {_OFFSET_VALUES_SQL} (the
# bucket-neighbor offsets) is baked in once here since it's fixed.
_BUCKET_QUERY_TEMPLATE = f"""
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
      AND ob.expiration IN ({{expirations_sql}})
),
anchor_buckets AS (
    SELECT st.ts_event, st.leg_name, st.anchor_ts_event, st.underlying_id, st."right",
           st.target_strike, st.reference_expiration,
           CAST(floor(epoch(st.anchor_ts_event) / ?) AS BIGINT) + off.bucket_offset AS bucket
    FROM _group_targets st
    CROSS JOIN (VALUES {_OFFSET_VALUES_SQL}) AS off(bucket_offset)
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


# ---------------------------------------------------------------------------
# Instrumented subclass
# ---------------------------------------------------------------------------


class ComparingInputDatabase(InputDatabase):
    def __init__(self, db_path: str) -> None:
        super().__init__(db_path)
        self.partition_stats: list[dict] = []
        self._partition_replay: dict[int, tuple] = {}
        self.mismatches: list[dict] = []

    def _greeks_for_strike_legs_tolerant(
        self,
        strike_targets: pl.DataFrame,
        strike_tolerance: float,
        time_tolerance: int,
        audit_filter_codes,
    ) -> pl.DataFrame:
        _pad = timedelta(seconds=time_tolerance)
        # bucket_width == time_tolerance: provably needs only +/-1 neighbor
        # bucket to cover the full +/-time_tolerance window (see module
        # docstring); guard against time_tolerance==0 (shouldn't happen on
        # this code path, but avoid a division by zero if it ever does).
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

            between_params = [
                right,
                strike_min,
                strike_max,
                time_tolerance,
                time_tolerance,
                strike_tolerance,
                uid,
                anchor_min,
                anchor_max,
            ]
            # EXPLAIN ANALYZE on real data showed the dominant cost isn't the
            # anchor-side neighbor-bucket expansion (cutting 5->3 buckets barely
            # moved timing) -- it's the size of `combined`, the candidate pool
            # built from the WHOLE partition's option_greeks/option_bars rows,
            # which every query variant has to build and which gets fully
            # probed regardless of bucket count. BETWEEN already narrows its
            # equivalent candidate scan with right= and strike BETWEEN filters
            # applied early (in its JOIN's ON clause); `combined` was missing
            # those same early filters, plus a same-partition expiration
            # restriction that's equally safe (any real match's expiration must
            # be one of the reference_expirations actually present here).
            expirations = group["reference_expiration"].unique().to_list()
            expirations_sql = ",".join(f"DATE '{d.isoformat()}'" for d in expirations)
            bucket_query = _BUCKET_QUERY_TEMPLATE.format(expirations_sql=expirations_sql)
            bucket_params = [
                bucket_width,
                uid,
                anchor_min,
                anchor_max,
                right,
                strike_min,
                strike_max,
                bucket_width,
                strike_tolerance,
                time_tolerance,
            ]

            self._con.register("_group_targets", group)
            try:
                t0 = time.perf_counter()
                between_df = self._con.execute(_BETWEEN_QUERY, between_params).pl()
                between_elapsed = time.perf_counter() - t0

                t0 = time.perf_counter()
                bucket_df = self._con.execute(bucket_query, bucket_params).pl()
                bucket_elapsed = time.perf_counter() - t0
            finally:
                self._con.unregister("_group_targets")

            match_ok, detail = self._compare(between_df, bucket_df)
            if not match_ok:
                self.mismatches.append(
                    {
                        "underlying_id": uid,
                        "anchor_date": str(anchor_date),
                        "detail": detail,
                    }
                )

            idx = len(self.partition_stats)
            self.partition_stats.append(
                {
                    "idx": idx,
                    "underlying_id": uid,
                    "anchor_date": str(anchor_date),
                    "n_anchors": group.height,
                    "between_elapsed_s": between_elapsed,
                    "between_rows": between_df.height,
                    "bucket_elapsed_s": bucket_elapsed,
                    "bucket_rows": bucket_df.height,
                    "parity_ok": match_ok,
                }
            )
            self._partition_replay[idx] = (group, between_params, bucket_query, bucket_params)

            if not between_df.is_empty():
                frames.append(between_df)

        return self._finish_strike_legs(
            pl.concat(frames) if frames else pl.DataFrame(), audit_filter_codes
        )

    @staticmethod
    def _compare(between_df: pl.DataFrame, bucket_df: pl.DataFrame) -> tuple[bool, str]:
        b = dict(
            zip(
                between_df["ts_event"].to_list(),
                between_df["instrument_id"].to_list(),
                strict=True,
            )
        )
        a = dict(
            zip(bucket_df["ts_event"].to_list(), bucket_df["instrument_id"].to_list(), strict=True)
        )
        if b == a:
            return True, ""
        only_between = set(b) - set(a)
        only_bucket = set(a) - set(b)
        differing = {k for k in set(b) & set(a) if b[k] != a[k]}
        return False, (
            f"only_in_between={len(only_between)} only_in_bucket={len(only_bucket)} "
            f"differing_match={len(differing)}"
        )

    def explain_both(self, idx: int) -> tuple[str, str]:
        group, between_params, bucket_query, bucket_params = self._partition_replay[idx]
        self._con.register("_group_targets", group)
        try:
            between_plan = self._con.execute(
                "EXPLAIN ANALYZE " + _BETWEEN_QUERY, between_params
            ).fetchall()
            bucket_plan = self._con.execute(
                "EXPLAIN ANALYZE " + bucket_query, bucket_params
            ).fetchall()
        finally:
            self._con.unregister("_group_targets")
        return (
            "\n".join(str(row) for row in between_plan),
            "\n".join(str(row) for row in bucket_plan),
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _percentile(xs: list[float], p: float) -> float:
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


def main() -> None:
    if not Path(INPUT_DB).exists():
        print(f"ERROR: input database not found: {INPUT_DB}")
        sys.exit(1)

    print("btkit bucket-join vs BETWEEN comparison (Pass-2 strike-offset tolerant match)")
    print(f"  Database: {INPUT_DB}")
    print(f"  Universe: {STRATEGY.universe.start_date} .. {STRATEGY.universe.end_date}")
    print(f"  Bucket neighbor offsets checked: {_BUCKET_OFFSETS}")
    print()

    db = ComparingInputDatabase(INPUT_DB)
    trade = STRATEGY.trades[0]

    t0 = time.perf_counter()
    entries = EntryScanner(db, STRATEGY, trade).scan()
    total_elapsed = time.perf_counter() - t0

    print(f"\nscan() total wall time: {total_elapsed:.2f}s")
    print(f"entries found: {len(entries)}")

    if not db.partition_stats:
        print("\nNo Pass-2 partitions were queried — check the leg config matches the")
        print("combination you're trying to compare.")
        return

    print(f"\npartitions compared: {len(db.partition_stats)}")
    if db.mismatches:
        print(f"\n*** PARITY FAILURES: {len(db.mismatches)} partition(s) disagree ***")
        for m in db.mismatches[:10]:
            print(f"  underlying_id={m['underlying_id']} date={m['anchor_date']}: {m['detail']}")
        print("\nSTOP — do not trust any bucket-query timing numbers until every partition")
        print("passes parity. This is a correctness bug in the bucket query, not evidence")
        print("about performance.")
    else:
        print("parity: ALL partitions match (same anchor -> same instrument_id)")

    between_times = [p["between_elapsed_s"] for p in db.partition_stats]
    bucket_times = [p["bucket_elapsed_s"] for p in db.partition_stats]
    print(f"\nBETWEEN (current):  sum={sum(between_times):.2f}s  "
          f"p50={_percentile(between_times, 0.5):.4f}s  max={max(between_times):.4f}s")
    print(f"BUCKET (prototype): sum={sum(bucket_times):.2f}s  "
          f"p50={_percentile(bucket_times, 0.5):.4f}s  max={max(bucket_times):.4f}s")
    ratio = sum(bucket_times) / sum(between_times) if sum(between_times) else float("nan")
    print(f"BUCKET/BETWEEN total time ratio: {ratio:.2f}x  (< 1.0 means bucket was faster)")

    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(db.partition_stats[0].keys()))
        w.writeheader()
        w.writerows(db.partition_stats)
    print(f"\nfull per-partition comparison written to: {CSV_OUT}")

    sample = sorted(db.partition_stats, key=lambda p: -p["n_anchors"])[:N_EXPLAIN_SAMPLE]
    for rank, p in enumerate(sample, 1):
        between_plan, bucket_plan = db.explain_both(p["idx"])
        print(f"\n{'=' * 70}")
        print(
            f"EXPLAIN ANALYZE — sample #{rank}: underlying_id={p['underlying_id']} "
            f"date={p['anchor_date']} anchors={p['n_anchors']}"
        )
        print("=" * 70)
        print("--- BETWEEN (current) ---")
        print(between_plan)
        print("--- BUCKET (prototype) ---")
        print(bucket_plan)


if __name__ == "__main__":
    main()
