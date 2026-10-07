"""
Compare the current BETWEEN+QUALIFY per-partition query in
InputDatabase._greeks_for_strike_legs_tolerant against a prototype ASOF JOIN-based
alternative — same partitioning (one query per (underlying_id, calendar date), the
already-validated-safe bound), same real data, run side by side.

This does NOT change production behavior: it only measures. The subclass below
still returns the real (BETWEEN-based) result from _greeks_for_strike_legs_tolerant
unchanged — the ASOF variant runs alongside, purely for comparison, and never
feeds into what EntryScanner.scan() actually returns.

Why this exists: prototyping ASOF JOIN in this DuckDB version (1.5.5) surfaced two
findings that argue against assuming it's simply faster (see the design doc / commit
history for the full trace):
  1. A real internal DuckDB binder bug triggers if a column referenced only in ASOF's
     extra (non-primary) join condition isn't also projected in the outer SELECT —
     worked around here since strike_price is projected anyway (production needs it).
  2. EXPLAIN shows DuckDB choosing a NESTED_LOOP_JOIN physical operator for ASOF in
     this shape — the SAME operator family implicated in the earlier single-query
     memory blowup incident. Operator names alone don't prove costs are identical,
     which is exactly why this script exists: to get real EXPLAIN ANALYZE numbers
     instead of guessing from either direction.

For every partition, this asserts row-for-row correctness parity between the two
query results (same anchor -> same matched instrument_id) — a mismatch is a bug in
the ASOF variant, not evidence to weigh against.

Usage:
    1. Point INPUT_DB at your real database (STRATEGY below is copied from the
       already-validated diagnose_strike_legs_perf.py — keep it in sync, or better,
       edit both from the same source of truth if you change the combination).
    2. NARROW_UNIVERSE_DAYS below limits the run to a short window by default (per
       the plan: prove correctness and get a real per-partition cost comparison at
       small scale FIRST — do not widen to the full universe until every partition
       tested passes the parity check and EXPLAIN ANALYZE shows ASOF is genuinely
       cheaper, not merely "didn't crash").
    3. Run:  python scripts/compare_asof_strike_legs.py
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
# EDIT THESE to match your real setup — copied from the validated
# diagnose_strike_legs_perf.py run; keep the leg/instrument/exit config in sync
# with that file if you change the combination being investigated.
# ---------------------------------------------------------------------------

INPUT_DB = "/Users/jackrochester/dev/wealthy-option-backtest/db/input/es.db"

# Per the plan: start narrow. Only widen toward the full universe
# (2022-05-01 .. 2026-06-03) after every partition below passes parity and
# EXPLAIN ANALYZE shows a real, not just plausible, win.
NARROW_START = date(2022, 5, 1)
NARROW_END = date(2022, 6, 1)  # one month

STRATEGY = StrategyDefinition(
    name="compare_asof_strike_legs",
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

# Number of largest partitions to also capture EXPLAIN ANALYZE for (verbose; keep small).
N_EXPLAIN_SAMPLE = 3
CSV_OUT = str(Path(__file__).parent / "asof_comparison_partitions.csv")


# ---------------------------------------------------------------------------
# Query text — BETWEEN variant is copied verbatim from
# InputDatabase._greeks_for_strike_legs_tolerant (btkit/db/input_db.py). Keep
# these two in sync if that method changes.
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

# ASOF variant — validated during planning: strike_price MUST be projected in the
# outer SELECT (a real DuckDB 1.5.5 internal binder bug otherwise crashes both
# EXPLAIN and EXPLAIN ANALYZE — see module docstring). The strike-tolerance
# condition MUST stay inside each ASOF JOIN's ON clause, not moved to a coarse
# outer WHERE — moving it out was tested and produces wrong matches whenever
# different anchors in the same partition have different target_strike values.
_ASOF_QUERY = """
WITH combined AS (
    SELECT og.ts_event, og.instrument_id, og.underlying_id, og.dte, og.iv, og.delta,
           og.gamma, og.theta, og.vega, ob.strike_price, ob.expiration, ob."right",
           ob.multiplier, ob.symbol, ob.close
    FROM option_greeks og
    JOIN option_bars ob ON og.instrument_id = ob.instrument_id AND og.ts_event = ob.ts_event
    WHERE ob.close IS NOT NULL
      AND og.underlying_id = ?
      AND og.ts_event BETWEEN ? AND ?
),
backward AS (
    SELECT st.ts_event AS row_id, st.leg_name, c.ts_event AS _match_ts_event,
           c.instrument_id, c.underlying_id, c.dte, c.iv, c.delta, c.gamma, c.theta,
           c.vega, c.strike_price, c.expiration, c."right", c.multiplier, c.symbol,
           c.close,
           abs(epoch(c.ts_event) - epoch(st.anchor_ts_event)) AS time_diff,
           abs(c.strike_price - st.target_strike) AS strike_diff
    FROM _group_targets st
    ASOF JOIN combined c
      ON c.underlying_id = st.underlying_id
     AND c."right" = st.right
     AND c.expiration = st.reference_expiration
     AND abs(c.strike_price - st.target_strike) <= ?
     AND c.ts_event <= st.anchor_ts_event
),
forward AS (
    SELECT st.ts_event AS row_id, st.leg_name, c.ts_event AS _match_ts_event,
           c.instrument_id, c.underlying_id, c.dte, c.iv, c.delta, c.gamma, c.theta,
           c.vega, c.strike_price, c.expiration, c."right", c.multiplier, c.symbol,
           c.close,
           abs(epoch(c.ts_event) - epoch(st.anchor_ts_event)) AS time_diff,
           abs(c.strike_price - st.target_strike) AS strike_diff
    FROM _group_targets st
    ASOF JOIN combined c
      ON c.underlying_id = st.underlying_id
     AND c."right" = st.right
     AND c.expiration = st.reference_expiration
     AND abs(c.strike_price - st.target_strike) <= ?
     AND c.ts_event >= st.anchor_ts_event
),
unioned AS (SELECT * FROM backward UNION ALL SELECT * FROM forward)
SELECT leg_name, row_id AS ts_event, _match_ts_event, instrument_id, underlying_id,
       dte, iv, delta, gamma, theta, vega, strike_price, expiration, "right",
       multiplier, symbol, close
FROM unioned
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
            asof_params = [
                uid,
                anchor_min,
                anchor_max,
                strike_tolerance,
                strike_tolerance,
                time_tolerance,
            ]

            self._con.register("_group_targets", group)
            try:
                t0 = time.perf_counter()
                between_df = self._con.execute(_BETWEEN_QUERY, between_params).pl()
                between_elapsed = time.perf_counter() - t0

                t0 = time.perf_counter()
                asof_df = self._con.execute(_ASOF_QUERY, asof_params).pl()
                asof_elapsed = time.perf_counter() - t0
            finally:
                self._con.unregister("_group_targets")

            match_ok, detail = self._compare(between_df, asof_df)
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
                    "asof_elapsed_s": asof_elapsed,
                    "asof_rows": asof_df.height,
                    "parity_ok": match_ok,
                }
            )
            self._partition_replay[idx] = (group, between_params, asof_params)

            if not between_df.is_empty():
                frames.append(between_df)

        # Return the REAL (BETWEEN-based) result unchanged — this comparison
        # never influences what scan() actually finds.
        return self._finish_strike_legs(
            pl.concat(frames) if frames else pl.DataFrame(), audit_filter_codes
        )

    @staticmethod
    def _compare(between_df: pl.DataFrame, asof_df: pl.DataFrame) -> tuple[bool, str]:
        b = dict(
            zip(
                between_df["ts_event"].to_list(),
                between_df["instrument_id"].to_list(),
                strict=True,
            )
        )
        a = dict(
            zip(asof_df["ts_event"].to_list(), asof_df["instrument_id"].to_list(), strict=True)
        )
        if b == a:
            return True, ""
        only_between = set(b) - set(a)
        only_asof = set(a) - set(b)
        differing = {k for k in set(b) & set(a) if b[k] != a[k]}
        return False, (
            f"only_in_between={len(only_between)} only_in_asof={len(only_asof)} "
            f"differing_match={len(differing)}"
        )

    def explain_both(self, idx: int) -> tuple[str, str]:
        group, between_params, asof_params = self._partition_replay[idx]
        self._con.register("_group_targets", group)
        try:
            between_plan = self._con.execute(
                "EXPLAIN ANALYZE " + _BETWEEN_QUERY, between_params
            ).fetchall()
            asof_plan = self._con.execute("EXPLAIN ANALYZE " + _ASOF_QUERY, asof_params).fetchall()
        finally:
            self._con.unregister("_group_targets")
        return (
            "\n".join(str(row) for row in between_plan),
            "\n".join(str(row) for row in asof_plan),
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

    print("btkit ASOF JOIN vs BETWEEN comparison (Pass-2 strike-offset tolerant match)")
    print(f"  Database: {INPUT_DB}")
    print(f"  Universe: {STRATEGY.universe.start_date} .. {STRATEGY.universe.end_date}  (narrow)")
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
        print(
            "\nSTOP — per the plan, do not trust any ASOF timing numbers until every"
        )
        print("partition passes parity. This is a correctness bug in the ASOF query, not")
        print("evidence about performance.")
    else:
        print("parity: ALL partitions match (same anchor -> same instrument_id)")

    between_times = [p["between_elapsed_s"] for p in db.partition_stats]
    asof_times = [p["asof_elapsed_s"] for p in db.partition_stats]
    print(f"\nBETWEEN (current):  sum={sum(between_times):.2f}s  "
          f"p50={_percentile(between_times, 0.5):.4f}s  max={max(between_times):.4f}s")
    print(f"ASOF (prototype):   sum={sum(asof_times):.2f}s  "
          f"p50={_percentile(asof_times, 0.5):.4f}s  max={max(asof_times):.4f}s")
    ratio = sum(asof_times) / sum(between_times) if sum(between_times) else float("nan")
    print(f"ASOF/BETWEEN total time ratio: {ratio:.2f}x  (< 1.0 means ASOF was faster)")

    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(db.partition_stats[0].keys()))
        w.writeheader()
        w.writerows(db.partition_stats)
    print(f"\nfull per-partition comparison written to: {CSV_OUT}")

    sample = sorted(db.partition_stats, key=lambda p: -p["n_anchors"])[:N_EXPLAIN_SAMPLE]
    for rank, p in enumerate(sample, 1):
        between_plan, asof_plan = db.explain_both(p["idx"])
        print(f"\n{'=' * 70}")
        print(
            f"EXPLAIN ANALYZE — sample #{rank}: underlying_id={p['underlying_id']} "
            f"date={p['anchor_date']} anchors={p['n_anchors']}"
        )
        print("=" * 70)
        print("--- BETWEEN (current) ---")
        print(between_plan)
        print("--- ASOF (prototype) ---")
        print(asof_plan)


if __name__ == "__main__":
    main()
