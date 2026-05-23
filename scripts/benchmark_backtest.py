"""
Backtest performance benchmark — 1 month of ES data.

Measures wall-clock time for each phase of the backtest pipeline and reports
per-phase breakdowns and total throughput. Run with:

    python scripts/benchmark_backtest.py

Options (edit constants below):
    INPUT_DB   — pre-built input database (build with scripts/test_ingest.py)
    STRATEGY   — strategy YAML to run
    N_RUNS     — number of timed repetitions (first run excluded as warm-up)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl

from btkit.backtest.engine import BacktestEngine
from btkit.backtest.entry import EntryScanner
from btkit.backtest.exit import ExitScanner
from btkit.backtest.pnl import PnLCalculator
from btkit.db.input_db import InputDatabase
from btkit.db.output_db import OutputDatabase
from btkit.strategy.loader import load_strategy

INPUT_DB = "/tmp/btkit_test_ingest.db"
STRATEGY = "tests/fixtures/strategies/short_put_spread.yaml"
N_RUNS   = 5   # timed repetitions after warm-up


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

class Timer:
    """Simple context-manager timer."""
    def __init__(self, label: str) -> None:
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed = time.perf_counter() - self._start


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return (s[n // 2 - 1] + s[n // 2]) / 2 if n % 2 == 0 else s[n // 2]

def _fmt(s: float) -> str:
    if s >= 1.0:
        return f"{s:.3f}s"
    return f"{s*1000:.1f}ms"


# ---------------------------------------------------------------------------
# Phase-level benchmark
# ---------------------------------------------------------------------------

def run_phases(db: InputDatabase, strategy, trade) -> dict[str, float]:
    """Run all phases for one trade and return per-phase wall times."""
    times: dict[str, float] = {}

    with Timer("entry") as t:
        entries = EntryScanner(db, strategy, trade).scan()
    times["entry"] = t.elapsed

    with Timer("exit") as t:
        exits = ExitScanner(db, strategy, trade).scan(entries)
    times["exit"] = t.elapsed

    return times, entries, exits


def benchmark_phases(db: InputDatabase, strategy, n_runs: int) -> None:
    print(f"\n{'─'*60}")
    print(f"Per-phase timing  ({n_runs} runs, first excluded as warm-up)")
    print(f"{'─'*60}")

    phase_samples: dict[str, list[float]] = {}

    for i in range(n_runs + 1):
        run_times: dict[str, float] = {}
        all_entries: dict[str, pl.DataFrame] = {}
        all_exits:   dict[str, pl.DataFrame] = {}

        for trade in strategy.trades:
            times, entries, exits = run_phases(db, strategy, trade)
            for phase, t in times.items():
                key = f"{trade.name}.{phase}"
                run_times[key] = t
            all_entries[trade.name] = entries
            all_exits[trade.name]   = exits

        with Timer("pnl") as t:
            PnLCalculator(strategy).compute(all_entries, all_exits)
        run_times["pnl"] = t.elapsed

        if i == 0:
            continue  # warm-up

        for phase, elapsed in run_times.items():
            phase_samples.setdefault(phase, []).append(elapsed)

    for phase, samples in phase_samples.items():
        print(f"  {phase:<30}  median={_fmt(_median(samples))}  mean={_fmt(_mean(samples))}")


# ---------------------------------------------------------------------------
# Full engine benchmark (including one-at-a-time + DB writes)
# ---------------------------------------------------------------------------

def benchmark_engine(db: InputDatabase, strategy, n_runs: int) -> None:
    print(f"\n{'─'*60}")
    print(f"Full engine.run()  ({n_runs} runs, first excluded as warm-up)")
    print(f"{'─'*60}")

    samples: list[float] = []

    for i in range(n_runs + 1):
        with OutputDatabase(":memory:") as odb:
            odb.create_schema()
            engine = BacktestEngine(db, odb, strategy, initial_equity=100_000.0)
            with Timer("engine") as t:
                engine.run()
            if i > 0:
                samples.append(t.elapsed)

    if samples:
        print(f"  median: {_fmt(_median(samples))}")
        print(f"  mean:   {_fmt(_mean(samples))}")
        print(f"  min:    {_fmt(min(samples))}")
        print(f"  max:    {_fmt(max(samples))}")


# ---------------------------------------------------------------------------
# Entry breakdown
# ---------------------------------------------------------------------------

def benchmark_entry_steps(db: InputDatabase, strategy, n_runs: int) -> None:
    print(f"\n{'─'*60}")
    print(f"EntryScanner step breakdown  ({n_runs} runs, first excluded as warm-up)")
    print(f"{'─'*60}")

    trade = strategy.trades[0]

    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(strategy.universe.session.timezone)
    universe = strategy.universe
    start_dt = datetime(universe.start_date.year, universe.start_date.month, universe.start_date.day, tzinfo=tz).astimezone(timezone.utc)
    end_dt   = datetime(universe.end_date.year, universe.end_date.month, universe.end_date.day, 23, 59, 59, tzinfo=tz).astimezone(timezone.utc)
    underlying_id = db.instrument_id_for_symbol(trade.instrument.root_symbol)
    indicators    = db.indicators(underlying_id, start_dt, end_dt)

    step_samples: dict[str, list[float]] = {}

    for i in range(n_runs + 1):
        scanner = EntryScanner(db, strategy, trade)

        bars = db.underlying_bars(underlying_id, start_dt, end_dt)

        with Timer("1_window") as t:
            candidates = scanner._apply_window_filters(bars)
        with Timer("2_legs") as t2:
            candidates2 = scanner._select_legs(candidates, underlying_id)
        with Timer("3_mark") as t3:
            candidates3 = scanner._compute_open_mark(candidates2)
        with Timer("4_conditions") as t4:
            scanner._evaluate_conditions(candidates3, indicators)

        if i == 0:
            continue
        for name, timer in [("1_window_filter", t), ("2_leg_selection", t2), ("3_open_mark", t3), ("4_conditions", t4)]:
            step_samples.setdefault(name, []).append(timer.elapsed)

    for name, samples in step_samples.items():
        print(f"  {name:<25}  median={_fmt(_median(samples))}")


# ---------------------------------------------------------------------------
# Exit breakdown
# ---------------------------------------------------------------------------

def benchmark_exit_steps(db: InputDatabase, strategy, n_runs: int) -> None:
    print(f"\n{'─'*60}")
    print(f"ExitScanner step breakdown  ({n_runs} runs, first excluded as warm-up)")
    print(f"{'─'*60}")

    trade = strategy.trades[0]
    # Pre-compute entries once (not part of exit timing)
    entries = EntryScanner(db, strategy, trade).scan()

    step_samples: dict[str, list[float]] = {}

    for i in range(n_runs + 1):
        scanner = ExitScanner(db, strategy, trade)

        with Timer("1_load") as t1:
            option_bars, indicators = scanner._load_exit_data(entries)
        with Timer("2_marks") as t2:
            position_marks = scanner._compute_position_marks(option_bars, entries)
        with Timer("3_hit") as t3:
            scanner._find_first_hit(position_marks, indicators, entries)

        if i == 0:
            continue
        for name, timer in [("1_load_exit_data", t1), ("2_position_marks", t2), ("3_find_first_hit", t3)]:
            step_samples.setdefault(name, []).append(timer.elapsed)

    for name, samples in step_samples.items():
        print(f"  {name:<25}  median={_fmt(_median(samples))}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("btkit Backtest Benchmark")
    print(f"  Database:  {INPUT_DB}")
    print(f"  Strategy:  {STRATEGY}")
    print(f"  Runs:      {N_RUNS} timed + 1 warm-up")

    strategy = load_strategy(STRATEGY)
    print(f"  Trades:    {[t.name for t in strategy.trades]}")

    # Check DB exists
    from pathlib import Path
    if not Path(INPUT_DB).exists():
        print(f"\nERROR: Input database not found: {INPUT_DB}")
        print("Build it first with:  python scripts/test_ingest.py")
        sys.exit(1)

    with InputDatabase(INPUT_DB) as db:
        benchmark_entry_steps(db, strategy, N_RUNS)
        benchmark_exit_steps(db, strategy, N_RUNS)
        benchmark_phases(db, strategy, N_RUNS)
        benchmark_engine(db, strategy, N_RUNS)

    print()


if __name__ == "__main__":
    main()
