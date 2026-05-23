"""
Smoke test for the three-pass backtest pipeline (Phases 4-6).

Exercises EntryScanner → ExitScanner → PnLCalculator for each test fixture
strategy using the pre-built input database, then prints a summary of entries,
exits, and net PnL per trade.

Expected database: tests/fixtures/input.db (built by scripts/test_ingest.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from btkit.backtest.entry import EntryScanner
from btkit.backtest.exit import ExitScanner
from btkit.backtest.pnl import PnLCalculator
from btkit.db.input_db import InputDatabase
from btkit.strategy.loader import load_strategy

DB_PATH = "/tmp/btkit_test_ingest.db"
STRATEGIES_DIR = Path("tests/fixtures/strategies")


def run_strategy(path: Path, db: InputDatabase) -> None:
    strategy = load_strategy(path)
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy.name}")

    all_entries = {}
    all_exits = {}

    for trade in strategy.trades:
        entries = EntryScanner(db, strategy, trade).scan()
        print(f"  trade={trade.name!r}: {len(entries)} entries")

        exits = ExitScanner(db, strategy, trade).scan(entries)
        print(f"  trade={trade.name!r}: {len(exits)} exits")

        if not entries.is_empty() and not exits.is_empty():
            reason_counts = exits["exit_reason"].value_counts().sort("exit_reason")
            for row in reason_counts.iter_rows(named=True):
                print(f"    {row['exit_reason']}: {row['count']}")

        all_entries[trade.name] = entries
        all_exits[trade.name] = exits

    result = PnLCalculator(strategy).compute(all_entries, all_exits)
    if not result.positions.is_empty():
        pnl = result.positions["net_pnl"].sum()
        print(f"  total positions: {len(result.positions)}, net_pnl={pnl:.2f}")
        if not result.legs.is_empty():
            print(f"  total legs: {len(result.legs)}")
    else:
        print("  no positions")


def main() -> None:
    print(f"Using database: {DB_PATH}")
    with InputDatabase(DB_PATH) as db:
        for yaml_file in sorted(STRATEGIES_DIR.glob("*.yaml")):
            try:
                run_strategy(yaml_file, db)
            except Exception as exc:
                print(f"\nERROR in {yaml_file.name}: {exc}")
                import traceback
                traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
