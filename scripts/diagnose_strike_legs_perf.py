"""
Diagnose real-scale performance of InputDatabase._greeks_for_strike_legs_tolerant
(the reference_leg.entry_time_tolerance code path in btkit/db/input_db.py).

This runs the REAL per-(underlying_id, calendar date) query loop — via a thin
subclass that mirrors the current loop body exactly, instrumented with timing —
driven by a REAL EntryScanner.scan() against your real input database, so the
strike_targets frame(s) it queries are exactly what production would build. No
synthetic data.

Reports:
  - How many (underlying_id, date) partitions the run actually touches, and the
    per-partition timing distribution (min/median/p95/max) — is time spread
    evenly across many small partitions, or dominated by a few slow ones?
  - EXPLAIN ANALYZE for the single slowest partition and a couple of
    representative ones, so you can see DuckDB's actual chosen query plan and
    where it reports time/rows being spent, instead of guessing from outside.
  - Writes full per-partition timing to a CSV next to this script for further
    inspection.

Usage:
    1. Edit the constants below to point at your real input database and the
       worst-case combination's parameters (the ones already identified:
       delta=-0.05, dte=0, strike_offset=-50.0, entry_time_tolerance=300) —
       fill in your real instrument/entry-window/exit/universe details.
    2. Run:  python scripts/diagnose_strike_legs_perf.py
    3. Share the printed report (and/or the CSV) back for interpretation —
       this script only measures, it doesn't change any production code path.
"""

from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from datetime import time as dtime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import csv

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
# EDIT THESE to match your real setup
# ---------------------------------------------------------------------------

INPUT_DB = "/Users/jackrochester/dev/wealthy-option-backtest/db/input/es.db"

# The worst-case combination already identified from the sweep, using the real
# es_put_credit_spread strategy's entry window / exit config / universe /
# instrument details / audit filter.
STRATEGY = StrategyDefinition(
    name="diagnose_strike_legs_perf",
    universe=UniverseConfig(
        start_date=date(2022, 5, 1),
        end_date=date(2026, 6, 3),
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
                    delta={"target": -0.05, "tolerance": 0.02},  # worst-case combination
                    dte={"target": 0, "tolerance": 0},  # worst-case combination
                    dte_skip_weekends=True,
                ),
                LegConfig(
                    name="long_put",
                    right="put",
                    action="buy_to_open",
                    reference_leg={
                        "name": "short_put",
                        "strike_offset": -50.0,  # worst-case combination
                        "entry_time_tolerance": 300,  # worst-case combination
                    },
                ),
            ],
            exit=ExitConfig(stop_loss=20.0, take_profit_pct=0.90),
        )
    ],
)

# Number of slowest partitions to re-run with EXPLAIN ANALYZE (verbose; keep small).
N_EXPLAIN_SLOWEST = 3
CSV_OUT = str(Path(__file__).parent / "strike_legs_perf_partitions.csv")


# ---------------------------------------------------------------------------
# Instrumented subclass — mirrors input_db.py's _greeks_for_strike_legs_tolerant
# loop body exactly, adding only timing capture and a stash of each partition's
# query inputs so slow ones can be re-run with EXPLAIN ANALYZE afterward.
# ---------------------------------------------------------------------------


class DiagnosticInputDatabase(InputDatabase):
    def __init__(self, db_path: str) -> None:
        super().__init__(db_path)
        self.partition_stats: list[dict] = []
        self._partition_replay: dict[int, tuple] = {}

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
        print(f"[diagnose] this Pass-2 call has {n_partitions} (underlying_id, date) partitions")

        call_t0 = time.perf_counter()
        for (uid, anchor_date), group in grouped.group_by(["underlying_id", "_anchor_date"]):
            anchor_min = group["anchor_ts_event"].min() - _pad
            anchor_max = group["anchor_ts_event"].max() + _pad
            strike_min = float(group["target_strike"].min()) - strike_tolerance
            strike_max = float(group["target_strike"].max()) + strike_tolerance
            right = group["right"][0]
            params = [
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

            self._con.register("_group_targets", group)
            t0 = time.perf_counter()
            try:
                df = self._con.execute(_QUERY, params).pl()
            finally:
                self._con.unregister("_group_targets")
            elapsed = time.perf_counter() - t0

            idx = len(self.partition_stats)
            self.partition_stats.append(
                {
                    "idx": idx,
                    "underlying_id": uid,
                    "anchor_date": str(anchor_date),
                    "n_anchors": group.height,
                    "elapsed_s": elapsed,
                    "n_rows": df.height,
                }
            )
            self._partition_replay[idx] = (group, params)

            if not df.is_empty():
                frames.append(df)

        print(f"[diagnose] total loop time: {time.perf_counter() - call_t0:.2f}s")
        return self._finish_strike_legs(
            pl.concat(frames) if frames else pl.DataFrame(), audit_filter_codes
        )

    def explain_partition(self, idx: int) -> str:
        group, params = self._partition_replay[idx]
        self._con.register("_group_targets", group)
        try:
            plan = self._con.execute("EXPLAIN ANALYZE " + _QUERY, params).fetchall()
        finally:
            self._con.unregister("_group_targets")
        return "\n".join(str(row) for row in plan)


_QUERY = """
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
        print("Edit INPUT_DB at the top of this script to point at your real database.")
        sys.exit(1)

    print("btkit Pass-2 (strike-offset, entry_time_tolerance) performance diagnostic")
    print(f"  Database: {INPUT_DB}")
    print(f"  Universe: {STRATEGY.universe.start_date} .. {STRATEGY.universe.end_date}")
    print()

    db = DiagnosticInputDatabase(INPUT_DB)
    trade = STRATEGY.trades[0]

    t0 = time.perf_counter()
    entries = EntryScanner(db, STRATEGY, trade).scan()
    total_elapsed = time.perf_counter() - t0

    print()
    print(f"scan() total wall time: {total_elapsed:.2f}s")
    print(f"entries found: {len(entries)}")

    if not db.partition_stats:
        print("\nNo Pass-2 partitions were queried (no matching entries, or an all-exact")
        print("bar_time_tolerance/entry_time_tolerance=0 path was used instead). Check the")
        print("leg config above matches the combination you're trying to diagnose.")
        return

    times = [p["elapsed_s"] for p in db.partition_stats]
    anchors = [p["n_anchors"] for p in db.partition_stats]
    print(f"\npartitions queried: {len(db.partition_stats)}")
    print(f"total anchors: {sum(anchors)}")
    print(
        f"per-partition time (s):  min={min(times):.4f}  "
        f"p50={_percentile(times, 0.5):.4f}  "
        f"p95={_percentile(times, 0.95):.4f}  "
        f"max={max(times):.4f}"
    )
    print(f"sum of per-partition times: {sum(times):.2f}s  (vs {total_elapsed:.2f}s wall time —")
    print("  a big gap suggests overhead outside the loop itself, e.g. Pass 1 or result concat)")

    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(db.partition_stats[0].keys()))
        w.writeheader()
        w.writerows(db.partition_stats)
    print(f"\nfull per-partition timing written to: {CSV_OUT}")

    slowest = sorted(db.partition_stats, key=lambda p: -p["elapsed_s"])[:N_EXPLAIN_SLOWEST]
    for rank, p in enumerate(slowest, 1):
        print(f"\n{'=' * 70}")
        print(
            f"EXPLAIN ANALYZE — slowest #{rank}: underlying_id={p['underlying_id']} "
            f"date={p['anchor_date']} anchors={p['n_anchors']} elapsed={p['elapsed_s']:.4f}s "
            f"rows={p['n_rows']}"
        )
        print("=" * 70)
        print(db.explain_partition(p["idx"]))


if __name__ == "__main__":
    main()
